"""长期记忆管理接口：学生可查看、修改、删除 AI 写入的记忆。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..memory import MEMORY_TYPES, MemoryError, MemoryMeta, MemoryStore
from ..trace import get_logger
from .schemas import MemoryCreatePayload, MemoryUpdatePayload
from .state import RuntimeState

router = APIRouter()


def _store(request: Request) -> MemoryStore:
    runtime: RuntimeState = request.app.state.runtime
    return MemoryStore(runtime.settings.paths.memories_dir, runtime.settings.memory)


def _meta_payload(meta: MemoryMeta) -> dict:
    return {
        "name": meta.name,
        "type": meta.type,
        "description": meta.description,
        "updated_at": meta.updated_at,
        "damaged": meta.damaged,
    }


@router.get("/api/memories")
def list_memories(request: Request) -> dict:
    store = _store(request)
    return {
        "items": [_meta_payload(meta) for meta in store.list_metas()],
        "types": list(MEMORY_TYPES),
    }


@router.get("/api/memories/{name}")
def get_memory(name: str, request: Request) -> dict:
    store = _store(request)
    record = store.read(name)
    if record is None:
        raise HTTPException(404, "记忆不存在")
    return {
        **_meta_payload(record.meta),
        "body": record.body,
        "raw": record.raw,
    }


@router.post("/api/memories")
def create_memory(payload: MemoryCreatePayload, request: Request) -> dict:
    store = _store(request)
    try:
        meta, _created = store.write(
            payload.name,
            payload.type,
            payload.description,
            payload.content,
            mode="replace",
            allow_existing=False,
        )
    except MemoryError as exc:
        raise HTTPException(400, str(exc)) from exc
    get_logger().info("手动创建记忆: %s", meta.name)
    return {"ok": True, **_meta_payload(meta)}


@router.put("/api/memories/{name}")
def update_memory(name: str, payload: MemoryUpdatePayload, request: Request) -> dict:
    store = _store(request)
    if store.read(name) is None:
        raise HTTPException(404, "记忆不存在")
    try:
        meta, _created = store.write(
            name,
            payload.type,
            payload.description,
            payload.content,
            mode="replace",
        )
    except MemoryError as exc:
        raise HTTPException(400, str(exc)) from exc
    get_logger().info("更新记忆: %s", meta.name)
    return {"ok": True, **_meta_payload(meta)}


@router.delete("/api/memories/{name}")
def delete_memory(name: str, request: Request) -> dict:
    store = _store(request)
    if not store.delete(name):
        raise HTTPException(404, "记忆不存在")
    get_logger().info("删除记忆: %s", name)
    return {"ok": True, "name": name}
