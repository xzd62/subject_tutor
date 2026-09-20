"""对话 WebSocket：事件流 status / retrieval / token / done / error。"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..agent.factory import build_tutor_agent
from ..cache import memory_fingerprint, model_fingerprint, prompt_fingerprint, sha1_text
from ..memory import MemoryStore
from ..retrieval.types import RetrievalResult
from ..trace import get_logger
from .routes_sessions import read_thread_messages
from .state import RuntimeState

router = APIRouter()


def serialize_retrieval(result: RetrievalResult) -> dict[str, Any]:
    return {
        "query": result.query,
        "top_k": result.top_k,
        "degraded": result.degraded,
        "elapsed_ms": {key: round(value, 1) for key, value in result.elapsed_ms.items()},
        "hits": [
            {
                "chunk_id": chunk.chunk_id,
                "subject": chunk.subject,
                "grade": str(chunk.metadata.get("grade", "")),
                "source_file": chunk.source_file,
                "heading_path": chunk.heading_path,
                "rrf_score": round(chunk.rrf_score, 6),
                "vector_rank": chunk.vector_rank,
                "bm25_rank": chunk.bm25_rank,
                "preview": chunk.text.replace("\n", " ")[:80],
            }
            for chunk in result.fused
        ],
    }


def tool_status_event(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("tool") == "ask_subject_expert":
        subject = str((event.get("args") or {}).get("subject") or "")
        return {"type": "status", "stage": "subagent", "data": {"subject": subject}}
    return {"type": "status", "stage": "memory", "data": event}


@router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    runtime: RuntimeState = websocket.app.state.runtime
    get_logger().info("WebSocket 已连接")
    try:
        while True:
            payload = await websocket.receive_json()
            message_type = payload.get("type", "chat")
            thread_id = str(payload.get("thread_id") or "web")

            if message_type == "reset":
                delete = getattr(runtime.checkpointer, "delete_thread", None)
                if delete is not None:
                    delete(thread_id)
                runtime.sessions.delete(thread_id)
                await websocket.send_json({"type": "reset_ok"})
                continue

            message = str(payload.get("message") or "").strip()
            if not message:
                await websocket.send_json({"type": "error", "message": "消息不能为空"})
                continue
            runtime.sessions.ensure_session(thread_id, message)
            await _stream_reply(
                runtime,
                websocket,
                message,
                thread_id,
                use_cache=bool(payload.get("use_cache", True)),
            )
    except WebSocketDisconnect:
        get_logger().info("WebSocket 客户端断开")
    except Exception as exc:
        get_logger().warning("WebSocket 会话异常: %r", exc)
        try:
            await websocket.close()
        except Exception:
            pass


def chunk_text(text: str, size: int = 24) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)]


def _cache_fingerprints(runtime: RuntimeState, thread_id: str) -> dict[str, str]:
    store = MemoryStore(runtime.settings.paths.memories_dir, runtime.settings.memory)
    history = read_thread_messages(runtime.checkpointer, thread_id)
    last_user = next(
        (item["content"] for item in reversed(history) if item["role"] == "user"), ""
    )
    return {
        "model_fp": model_fingerprint(runtime.chat_config),
        "memory_fp": memory_fingerprint(store),
        "context_fp": sha1_text(last_user)[:16] if last_user.strip() else "",
        "prompt_fp": prompt_fingerprint(),
    }


async def _stream_reply(
    runtime: RuntimeState,
    websocket: WebSocket,
    message: str,
    thread_id: str,
    use_cache: bool = True,
) -> None:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def push(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    try:
        model = runtime.get_model()
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        return

    fingerprints: dict[str, str] = {}
    cache_active = use_cache and runtime.cache.enabled
    if cache_active:
        try:
            fingerprints = _cache_fingerprints(runtime, thread_id)
            hit = runtime.cache.lookup(message, **fingerprints)
        except Exception as exc:
            get_logger().warning("缓存查询异常，按未命中处理: %r", exc)
            hit = None
        if hit is not None:
            runtime.append_exchange(thread_id, message, hit.answer)
            await websocket.send_json(
                {
                    "type": "cache",
                    "data": {
                        "hit": True,
                        "hit_type": hit.hit_type,
                        "score": round(hit.score, 4),
                        "cache_id": hit.cache_id,
                        "sources": hit.sources,
                    },
                }
            )
            for piece in chunk_text(hit.answer):
                await websocket.send_json({"type": "token", "text": piece})
            await websocket.send_json({"type": "done"})
            return

    collected_sources: list[dict[str, Any]] = []

    def on_retrieval(result: RetrievalResult) -> None:
        data = serialize_retrieval(result)
        collected_sources[:] = data["hits"]
        push({"type": "retrieval", "data": data})

    agent = build_tutor_agent(
        settings=runtime.settings,
        model=model,
        checkpointer=runtime.checkpointer,
        retrieval=runtime.settings.retrieval.enabled,
        retrieval_service=runtime.retrieval,
        on_retrieval_start=lambda _query: push({"type": "status", "stage": "retrieving"}),
        on_retrieval=on_retrieval,
        on_tool_event=lambda event: push(tool_status_event(event)),
    )

    def run() -> None:
        answer_parts: list[str] = []
        try:
            push({"type": "status", "stage": "thinking"})
            for piece in agent.stream(message, thread_id):
                answer_parts.append(piece)
                push({"type": "token", "text": piece})
            answer = "".join(answer_parts).strip()
            if cache_active and answer:
                try:
                    runtime.cache.store(
                        message,
                        answer,
                        collected_sources,
                        **fingerprints,
                    )
                except Exception as exc:
                    get_logger().warning("缓存写入异常: %r", exc)
            push({"type": "done"})
        except Exception as exc:
            get_logger().warning("对话生成失败: %r", exc)
            push({"type": "error", "message": f"生成失败: {exc}"})

    task = asyncio.create_task(asyncio.to_thread(run))
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)
            if event["type"] in {"done", "error"}:
                break
    finally:
        await asyncio.gather(task, return_exceptions=True)
