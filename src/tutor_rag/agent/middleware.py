"""检索中间件：每轮模型调用前检索教材并临时注入上下文。

用 `@wrap_model_call` 修改的是单次模型请求（`request.override`），不会写入
checkpointer 的对话历史，因此上下文不会随轮次累积污染会话。
"""

from __future__ import annotations

from typing import Any, Callable

from langchain.agents.middleware import (
    ModelRequest,
    ToolCallRequest,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.messages import HumanMessage, SystemMessage

from ..memory import MemoryStore
from ..retrieval.service import RetrievalService
from ..retrieval.types import RetrievalResult
from ..trace import get_logger
from .prompts import (
    RETRIEVAL_CONTEXT_HEADER,
    RETRIEVAL_INSTRUCTIONS,
    memory_prompt_section,
)


def render_context(result: RetrievalResult) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(result.fused, start=1):
        metadata = chunk.metadata
        location = " ".join(
            str(part)
            for part in (
                f"{metadata.get('subject', '')}{metadata.get('grade', '')}",
                metadata.get("heading_path", ""),
            )
            if part
        )
        blocks.append(f"[{index}] {location}\n{chunk.text}")
    return "\n\n".join(blocks)


def _last_human_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def make_memory_prompt_middleware(store: MemoryStore, base_prompt: str):
    @dynamic_prompt
    def memory_prompt_middleware(request: ModelRequest) -> SystemMessage:
        section = memory_prompt_section(store.build_index())
        return SystemMessage(content=f"{base_prompt}\n\n{section}")

    return memory_prompt_middleware


def make_tool_event_middleware(on_event: Callable[[dict[str, Any]], None]):
    @wrap_tool_call
    def tool_event_middleware(request: ToolCallRequest, handler):
        tool_call = request.tool_call or {}
        name = str(tool_call.get("name") or "")
        if name.startswith("memory_") or name == "ask_subject_expert":
            on_event({"tool": name, "args": dict(tool_call.get("args") or {})})
        return handler(request)

    return tool_event_middleware


def make_retrieval_middleware(
    service: RetrievalService,
    top_k: int | None = None,
    enabled: bool = True,
    subject: str | None = None,
    on_start: Callable[[str], None] | None = None,
    on_result: Callable[[RetrievalResult], None] | None = None,
):
    @wrap_model_call
    def retrieval_middleware(request: ModelRequest, handler):
        if not enabled:
            return handler(request)
        messages = list(request.messages)
        if not messages or not isinstance(messages[-1], HumanMessage):
            return handler(request)
        query = _last_human_text(messages)
        if not query.strip():
            return handler(request)
        if on_start is not None:
            on_start(query)
        try:
            result = service.search(query, top_k=top_k, subject=subject)
        except Exception as exc:
            get_logger().warning("检索失败，跳过上下文注入: %r", exc)
            return handler(request)
        if on_result is not None:
            on_result(result)
        if not result.fused:
            get_logger().info("检索无命中，按无上下文回答: query=%r", query[:40])
            return handler(request)
        base = request.system_message.content if request.system_message else ""
        merged = (
            f"{base}\n\n{RETRIEVAL_CONTEXT_HEADER}\n{render_context(result)}"
            f"\n\n{RETRIEVAL_INSTRUCTIONS}"
        ).strip()
        get_logger().info(
            "注入教材片段 %d 条: top1=%s",
            len(result.fused),
            result.fused[0].chunk_id,
        )
        return handler(request.override(system_message=SystemMessage(content=merged)))

    return retrieval_middleware
