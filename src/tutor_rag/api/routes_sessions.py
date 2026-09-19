"""会话管理：列表、历史回放、删除（SQLite checkpointer + chat_sessions 表）。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from ..trace import get_logger
from .state import RuntimeState

router = APIRouter()

_ROLE_MAP = {"human": "user", "ai": "assistant"}


def read_thread_messages(checkpointer, thread_id: str) -> list[dict[str, Any]]:
    config = {"configurable": {"thread_id": thread_id}}
    try:
        checkpoint_tuple = checkpointer.get_tuple(config)
    except Exception as exc:
        get_logger().warning("读取会话历史失败: %r", exc)
        return []
    if checkpoint_tuple is None:
        return []
    channel_values = (checkpoint_tuple.checkpoint or {}).get("channel_values", {})
    messages = channel_values.get("messages", [])
    history: list[dict[str, Any]] = []
    for message in messages:
        role = _ROLE_MAP.get(getattr(message, "type", ""))
        if role is None:
            continue
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            content = str(content)
        if not content.strip():
            continue
        history.append({"role": role, "content": content})
    return history


@router.get("/api/sessions")
def list_sessions(request: Request) -> dict:
    runtime: RuntimeState = request.app.state.runtime
    return {"items": runtime.sessions.list_sessions()}


@router.get("/api/sessions/{thread_id}/messages")
def get_session_messages(thread_id: str, request: Request) -> dict:
    runtime: RuntimeState = request.app.state.runtime
    session = runtime.sessions.get(thread_id)
    return {
        "thread_id": thread_id,
        "title": (session or {}).get("title", ""),
        "messages": read_thread_messages(runtime.checkpointer, thread_id),
    }


@router.delete("/api/sessions/{thread_id}")
def delete_session(thread_id: str, request: Request) -> dict:
    runtime: RuntimeState = request.app.state.runtime
    removed = runtime.sessions.delete(thread_id)
    delete_thread = getattr(runtime.checkpointer, "delete_thread", None)
    if delete_thread is not None:
        try:
            delete_thread(thread_id)
        except Exception as exc:
            get_logger().warning("删除会话 checkpoint 失败: %r", exc)
    get_logger().info("已删除会话: %s", thread_id)
    return {"ok": True, "thread_id": thread_id, "removed": removed}
