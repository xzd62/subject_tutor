"""语义缓存管理接口：统计、删除单条、清空。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from .state import RuntimeState

router = APIRouter()


@router.get("/api/cache/stats")
def cache_stats(request: Request) -> dict:
    runtime: RuntimeState = request.app.state.runtime
    return runtime.cache.stats_payload()


@router.delete("/api/cache/{cache_id}")
def cache_delete(cache_id: str, request: Request) -> dict:
    runtime: RuntimeState = request.app.state.runtime
    removed = runtime.cache.delete(cache_id)
    if not removed:
        raise HTTPException(404, "缓存不存在或已过期")
    return {"ok": True, "cache_id": cache_id}


@router.post("/api/cache/clear")
def cache_clear(request: Request) -> dict:
    runtime: RuntimeState = request.app.state.runtime
    removed = runtime.cache.clear()
    return {"ok": True, "removed": removed}
