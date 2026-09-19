"""Web 运行时状态：模型、checkpointer、检索服务、入库锁与设置持久化。"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite import SqliteSaver

from ..agent.llm import create_model, model_label
from ..config import ChatConfig, Settings, get_settings
from ..retrieval.service import RetrievalService
from ..trace import get_logger

WEB_SETTINGS_FILE = "web-settings.json"
MODEL_FIELDS = ("model", "temperature", "max_tokens", "timeout_s", "max_retries", "api_key")
SESSION_TITLE_LIMIT = 20


class SessionRegistry:
    """会话元数据表（标题/时间），与 checkpoint 同库，thread_id 隔离。"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    thread_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_session(self, thread_id: str, first_message: str) -> dict[str, Any]:
        now = datetime.now().isoformat(timespec="seconds")
        title = " ".join(first_message.split())[:SESSION_TITLE_LIMIT] or "新对话"
        with self._connect() as conn:
            row = conn.execute(
                "SELECT thread_id FROM chat_sessions WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO chat_sessions(thread_id, title, created_at, updated_at)"
                    " VALUES(?, ?, ?, ?)",
                    (thread_id, title, now, now),
                )
                return {"thread_id": thread_id, "title": title, "created": True}
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE thread_id = ?",
                (now, thread_id),
            )
            return {"thread_id": thread_id, "title": None, "created": False}

    def get(self, thread_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chat_sessions WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_sessions ORDER BY updated_at DESC, created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, thread_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM chat_sessions WHERE thread_id = ?", (thread_id,)
            )
            return cursor.rowcount > 0


def load_web_settings(base: ChatConfig, path: Path) -> ChatConfig:
    if not path.exists():
        return base
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        get_logger().warning("Web 设置文件解析失败，使用默认配置: %r", exc)
        return base
    changes: dict[str, Any] = {}
    for field in MODEL_FIELDS:
        if field not in data:
            continue
        value = data[field]
        try:
            if field in {"temperature"}:
                changes[field] = float(value)
            elif field in {"max_tokens", "timeout_s", "max_retries"}:
                changes[field] = int(value)
            else:
                changes[field] = str(value)
        except (TypeError, ValueError):
            get_logger().warning("Web 设置字段 %s 类型异常，忽略", field)
    return replace(base, **changes) if changes else base


def save_web_settings(config: ChatConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {field: getattr(config, field) for field in MODEL_FIELDS}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class RuntimeState:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.settings.paths.ensure_dirs()
        self.settings_path = self.settings.paths.outputs_dir / WEB_SETTINGS_FILE
        self.chat_config = load_web_settings(self.settings.chat, self.settings_path)
        self.checkpoint_db = self.settings.paths.data_dir / "checkpoints.db"
        self._checkpoint_conn = sqlite3.connect(
            self.checkpoint_db, check_same_thread=False
        )
        self.checkpointer = SqliteSaver(self._checkpoint_conn)
        self.sessions = SessionRegistry(self.checkpoint_db)
        self.retrieval = RetrievalService(self.settings)
        self.ingest_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.model_factory: Callable[[str, ChatConfig], BaseChatModel] = create_model
        self.model: BaseChatModel | None = None
        self.model_error: str | None = None
        self._build_model(log=False)

    def _build_model(self, log: bool = True) -> None:
        try:
            self.model = self.model_factory(self.chat_config.model, self.chat_config)
            self.model_error = None
            if log:
                get_logger().info("Web 模型就绪: %s", model_label(self.model))
        except Exception as exc:
            self.model = None
            self.model_error = str(exc)
            get_logger().warning("Web 模型初始化失败: %r", exc)

    def get_model(self) -> BaseChatModel:
        if self.model is None:
            raise RuntimeError(
                f"对话模型未就绪：{self.model_error or '请在设置页配置模型与 API Key'}"
            )
        return self.model

    def update_settings(self, changes: dict[str, Any]) -> ChatConfig:
        with self.model_lock:
            candidate = replace(self.chat_config, **changes)
            model = self.model_factory(candidate.model, candidate)
            self.chat_config = candidate
            self.model = model
            self.model_error = None
            save_web_settings(candidate, self.settings_path)
            get_logger().info("Web 设置已更新: model=%s", candidate.model)
            return candidate

    def settings_payload(self) -> dict[str, Any]:
        return {
            "model": self.chat_config.model,
            "temperature": self.chat_config.temperature,
            "max_tokens": self.chat_config.max_tokens,
            "timeout_s": self.chat_config.timeout_s,
            "max_retries": self.chat_config.max_retries,
            "api_key": self.chat_config.api_key,
            "model_ready": self.model is not None,
            "model_error": self.model_error,
        }
