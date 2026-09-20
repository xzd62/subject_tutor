"""FastAPI 应用：API 路由 + 前端静态托管（frontend/dist）。"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ..config import PROJECT_ROOT, Settings
from ..trace import get_logger, setup_logging
from .routes_books import router as books_router
from .routes_cache import router as cache_router
from .routes_chat import router as chat_router
from .routes_memories import router as memories_router
from .routes_sessions import router as sessions_router
from .routes_settings import router as settings_router
from .routes_system import router as system_router
from .scheduler import sweep_loop
from .state import RuntimeState


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(sweep_loop(app.state.runtime))
    get_logger().info("定时教材扫描任务已启动（启动补齐 + 每天 00:00）")
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def create_app(settings: Settings | None = None, runtime: RuntimeState | None = None) -> FastAPI:
    app = FastAPI(title="初中家教 API", version="0.6.0", lifespan=lifespan)
    app.state.runtime = runtime or RuntimeState(settings)
    setup_logging(app.state.runtime.settings.paths.logs_dir / "web.log")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_router)
    app.include_router(books_router)
    app.include_router(sessions_router)
    app.include_router(settings_router)
    app.include_router(system_router)
    app.include_router(memories_router)
    app.include_router(cache_router)

    @app.get("/api/health")
    def health(request: Request) -> dict:
        state: RuntimeState = request.app.state.runtime
        return {
            "status": "ok",
            "model_ready": state.model is not None,
            "model": state.chat_config.model,
            "model_error": state.model_error,
            "corpus_size": state.retrieval.corpus_size,
            "collection_chunks": state.retrieval.store.count(),
            "cache": state.cache.stats_payload(),
        }

    dist_dir = PROJECT_ROOT / "frontend" / "dist"
    if dist_dir.exists():

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str):
            candidate = dist_dir / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist_dir / "index.html")

    return app


app = create_app()
