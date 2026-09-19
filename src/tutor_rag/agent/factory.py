"""Agent 组装：检索中间件 + 长期记忆工具 + 学科子智能体委派。"""

from __future__ import annotations

from typing import Any, Callable

from ..config import Settings, get_settings
from ..memory import MemoryStore
from ..retrieval.service import RetrievalService
from ..trace import get_logger
from .graph import TutorAgent
from .llm import build_chat_model
from .memory_tools import make_memory_tools
from .middleware import (
    make_memory_prompt_middleware,
    make_retrieval_middleware,
    make_tool_event_middleware,
)
from .prompts import DELEGATION_RULES, TUTOR_SYSTEM_PROMPT
from .subagents import make_subject_tools


def build_tutor_agent(
    settings: Settings | None = None,
    model: Any = None,
    checkpointer: Any = None,
    retrieval: bool = True,
    top_k: int | None = None,
    memory: bool = True,
    subagents: bool | None = None,
    subject_model_factory: Callable[[str], Any] | None = None,
    retrieval_service: RetrievalService | None = None,
    on_retrieval_start: Callable[[str], None] | None = None,
    on_retrieval: Callable[[Any], None] | None = None,
    on_tool_event: Callable[[dict[str, Any]], None] | None = None,
) -> TutorAgent:
    settings = settings or get_settings()
    resolved_model = model or build_chat_model()
    middleware: list[Any] = []

    service: RetrievalService | None = None
    if retrieval and settings.retrieval.enabled:
        service = retrieval_service or RetrievalService(settings)
        middleware.append(
            make_retrieval_middleware(
                service,
                top_k=top_k or settings.retrieval.top_k,
                on_start=on_retrieval_start,
                on_result=on_retrieval,
            )
        )

    subagents_enabled = (
        settings.subagents.enabled if subagents is None else subagents
    ) and settings.subagents.enabled
    delegation_ready = subagents_enabled and service is not None

    supervisor_prompt = TUTOR_SYSTEM_PROMPT
    if delegation_ready:
        supervisor_prompt = f"{TUTOR_SYSTEM_PROMPT}\n\n{DELEGATION_RULES}"

    tools: list[Any] = []
    store: MemoryStore | None = None
    if memory and settings.memory.enabled:
        store = MemoryStore(settings.paths.memories_dir, settings.memory)
        tools.extend(make_memory_tools(store))
        middleware.insert(0, make_memory_prompt_middleware(store, supervisor_prompt))

    if delegation_ready:
        model_factory = subject_model_factory or (lambda _subject: resolved_model)
        tools.extend(make_subject_tools(settings, model_factory, service))

    if on_tool_event is not None and tools:
        middleware.append(make_tool_event_middleware(on_tool_event))

    agent = TutorAgent(
        model=resolved_model,
        checkpointer=checkpointer,
        system_prompt=supervisor_prompt,
        middleware=middleware,
        tools=tools,
    )
    agent.retrieval_service = service
    agent.memory_store = store
    get_logger().info(
        "Agent 组装完成: 检索=%s 子智能体=%s 工具=%s 记忆条数=%d",
        bool(service),
        delegation_ready,
        [getattr(item, "name", type(item).__name__) for item in tools],
        store.count() if store else 0,
    )
    return agent
