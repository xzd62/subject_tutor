"""长期记忆存储：每条记忆一个 md 文件（YAML front matter + 正文）。

front matter 必填字段：name（与文件名一致）、description、type；附加 updated_at。
工具（memory_read/memory_write）与 Web 记忆管理接口共用本模块。
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from .config import MemoryConfig
from .trace import get_logger

MEMORY_TYPES: tuple[str, ...] = ("学生画像", "错题与错因", "讲解偏好", "薄弱知识点")
WRITE_MODES: tuple[str, ...] = ("append", "replace")
NAME_MAX = 40

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
_INVALID_NAME_CHARS = '\\/:*?"<>|\r\n\t'


class MemoryError(ValueError):
    pass


def sanitize_name(name: str) -> str:
    cleaned = str(name or "").strip().strip(".")
    if not cleaned:
        raise MemoryError("记忆名称不能为空")
    if any(char in cleaned for char in _INVALID_NAME_CHARS):
        raise MemoryError("记忆名称不能包含路径分隔符或非法字符")
    if len(cleaned) > NAME_MAX:
        raise MemoryError(f"记忆名称不能超过 {NAME_MAX} 个字符")
    return cleaned


def validate_type(memory_type: str) -> str:
    value = str(memory_type or "").strip()
    if value not in MEMORY_TYPES:
        raise MemoryError(f"记忆类别必须是以下之一：{'、'.join(MEMORY_TYPES)}")
    return value


def validate_mode(mode: str) -> str:
    value = str(mode or "append").strip().lower()
    if value not in WRITE_MODES:
        raise MemoryError("mode 只能是 append 或 replace")
    return value


@dataclass
class MemoryMeta:
    name: str
    description: str
    type: str
    updated_at: str
    path: Path
    damaged: bool = False


@dataclass
class MemoryRecord:
    meta: MemoryMeta
    body: str
    raw: str


class MemoryStore:
    def __init__(self, directory: Path, config: MemoryConfig):
        self.directory = directory
        self.config = config
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def path_for(self, name: str) -> Path:
        return self.directory / f"{sanitize_name(name)}.md"

    def count(self) -> int:
        return len(list(self.directory.glob("*.md")))

    def list_metas(self) -> list[MemoryMeta]:
        metas = [self._read_meta(path) for path in sorted(self.directory.glob("*.md"))]
        metas.sort(key=lambda meta: (meta.updated_at or "", meta.name), reverse=True)
        return metas

    def read(self, name: str) -> MemoryRecord | None:
        path = self.path_for(name)
        if not path.is_file():
            return None
        raw = path.read_text(encoding="utf-8")
        front, body = self._split_front_matter(raw, path)
        meta = MemoryMeta(
            name=str(front.get("name") or path.stem),
            description=str(front.get("description") or ""),
            type=str(front.get("type") or ""),
            updated_at=str(front.get("updated_at") or ""),
            path=path,
        )
        return MemoryRecord(meta=meta, body=body, raw=raw)

    def read_text(self, name: str) -> str | None:
        record = self.read(name)
        if record is None:
            return None
        limit = self.config.read_max_chars
        if len(record.raw) > limit:
            return (
                record.raw[:limit]
                + f"\n\n[内容过长，已截断，完整内容共 {len(record.raw)} 字符，可在记忆管理中查看]"
            )
        return record.raw

    def write(
        self,
        name: str,
        memory_type: str,
        description: str,
        content: str,
        mode: str = "append",
        allow_existing: bool = True,
    ) -> tuple[MemoryMeta, bool]:
        clean_name = sanitize_name(name)
        clean_type = validate_type(memory_type)
        clean_mode = validate_mode(mode)
        clean_content = str(content or "").strip()
        if not clean_content:
            raise MemoryError("记忆正文不能为空")

        with self._lock:
            path = self.directory / f"{clean_name}.md"
            existing = self.read(clean_name) if path.is_file() else None
            if existing is not None and not allow_existing:
                raise MemoryError(f"记忆「{clean_name}」已存在")

            if existing is None or clean_mode == "replace":
                body = clean_content
            else:
                today = datetime.now().strftime("%Y-%m-%d")
                body = f"{existing.body.rstrip()}\n\n## {today}\n{clean_content}"

            description_text = str(description or "").strip()
            if not description_text and existing is not None:
                description_text = existing.meta.description
            front = {
                "name": clean_name,
                "description": description_text or "(无描述)",
                "type": clean_type,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            front_text = yaml.safe_dump(front, allow_unicode=True, sort_keys=False).strip()
            path.write_text(f"---\n{front_text}\n---\n\n{body}\n", encoding="utf-8")
            meta = MemoryMeta(
                name=clean_name,
                description=front["description"],
                type=clean_type,
                updated_at=front["updated_at"],
                path=path,
            )
            return meta, existing is None

    def delete(self, name: str) -> bool:
        with self._lock:
            path = self.path_for(name)
            if path.is_file():
                path.unlink()
                return True
            return False

    def build_index(self) -> str:
        limit = self.config.index_limit
        description_limit = self.config.description_limit
        metas = [meta for meta in self.list_metas() if not meta.damaged][:limit]
        if not metas:
            return ""
        lines = []
        for meta in metas:
            description = meta.description.strip()
            if len(description) > description_limit:
                description = description[:description_limit] + "..."
            lines.append(f"- {meta.name}（{meta.type}）：{description}")
        return "\n".join(lines)

    def _read_meta(self, path: Path) -> MemoryMeta:
        try:
            front, _body = self._split_front_matter(
                path.read_text(encoding="utf-8"), path
            )
        except Exception as exc:
            get_logger().warning("记忆文件解析失败: %s (%r)", path.name, exc)
            return MemoryMeta(
                name=path.stem,
                description="(front matter 解析失败，请在记忆管理中修复)",
                type="",
                updated_at="",
                path=path,
                damaged=True,
            )
        return MemoryMeta(
            name=str(front.get("name") or path.stem),
            description=str(front.get("description") or ""),
            type=str(front.get("type") or ""),
            updated_at=str(front.get("updated_at") or ""),
            path=path,
        )

    @staticmethod
    def _split_front_matter(raw: str, path: Path) -> tuple[dict, str]:
        text = raw.lstrip("\ufeff")
        match = _FRONT_MATTER_RE.match(text)
        if match is None:
            raise MemoryError(f"{path.name} 缺少 front matter")
        front = yaml.safe_load(match.group(1)) or {}
        if not isinstance(front, dict):
            raise MemoryError(f"{path.name} front matter 格式错误")
        return front, match.group(2).strip()
