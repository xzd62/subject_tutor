"""对话 Agent：create_agent + InMemorySaver。

- 多轮上下文存在进程内 checkpointer，按 thread_id 隔离，退出进程即丢（本期无持久记忆）；
- `switch_model()` 可在对话过程中热切换模型：图会重建，但 checkpointer 与 thread_id
  不变，因此历史消息保留，新模型无缝接着聊；
- 二期检索预留两种接法：做成 tool（模型自主决定检索）或 middleware（每轮自动注入）。
"""

from __future__ import annotations

import time
from typing import Any, Iterator, Sequence

from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from ..trace import get_logger
from .llm import build_chat_model, model_label
from .prompts import TUTOR_SYSTEM_PROMPT

DEFAULT_THREAD_ID = "cli"


class TutorAgent:
    def __init__(
        self,
        model: BaseChatModel | None = None,
        checkpointer: Any = None,
        system_prompt: str = TUTOR_SYSTEM_PROMPT,
        middleware: Sequence[Any] | None = None,
        tools: Sequence[Any] | None = None,
    ):
        self.checkpointer = checkpointer or InMemorySaver()
        self.system_prompt = system_prompt
        self.middleware = list(middleware or [])
        self.tools = list(tools or [])
        self.retrieval_service: Any = None
        self.memory_store: Any = None
        self.model = model or build_chat_model()
        self.graph = self._compile()

    def _compile(self):
        get_logger().info(
            "构建对话图: model=%s middleware=%d tools=%d",
            model_label(self.model),
            len(self.middleware),
            len(self.tools),
        )
        return create_agent(
            model=self.model,
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
            middleware=tuple(self.middleware),
            tools=tuple(self.tools) or None,
        )

    def switch_model(self, model: BaseChatModel) -> None:
        self.model = model
        self.graph = self._compile()

    def thread_config(self, thread_id: str = DEFAULT_THREAD_ID) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def invoke(self, text: str, thread_id: str = DEFAULT_THREAD_ID) -> AIMessage:
        start = time.perf_counter()
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=text)]},
            config=self.thread_config(thread_id),
        )
        message = result["messages"][-1]
        get_logger().info(
            "对话完成 thread=%s model=%s 耗时=%.2fs usage=%s",
            thread_id,
            model_label(self.model),
            time.perf_counter() - start,
            getattr(message, "usage_metadata", None),
        )
        return message

    def stream(self, text: str, thread_id: str = DEFAULT_THREAD_ID) -> Iterator[str]:
        start = time.perf_counter()
        first_token_at: float | None = None
        total_chars = 0
        for chunk, _metadata in self.graph.stream(
            {"messages": [HumanMessage(content=text)]},
            config=self.thread_config(thread_id),
            stream_mode="messages",
        ):
            if not isinstance(chunk, AIMessageChunk):
                continue
            content = getattr(chunk, "content", "")
            if not isinstance(content, str) or not content:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
            total_chars += len(content)
            yield content
        elapsed = time.perf_counter() - start
        first = (first_token_at - start) if first_token_at is not None else elapsed
        get_logger().info(
            "流式对话完成 thread=%s model=%s 首token=%.2fs 总耗时=%.2fs 字数=%d",
            thread_id,
            model_label(self.model),
            first,
            elapsed,
            total_chars,
        )
        if total_chars == 0:
            get_logger().warning(
                "本轮无文本输出：推理模型可能已耗尽 max_tokens（当前=%d），"
                "可调大 TUTOR_RAG_LLM_MAX_TOKENS",
                getattr(self.model, "max_tokens", None) or 0,
            )

    def history(self, thread_id: str = DEFAULT_THREAD_ID) -> list[Any]:
        state = self.graph.get_state(self.thread_config(thread_id))
        return list(state.values.get("messages", []))

    def reset(self, thread_id: str = DEFAULT_THREAD_ID) -> None:
        delete = getattr(self.checkpointer, "delete_thread", None)
        if delete is not None:
            delete(thread_id)
            get_logger().info("已清空会话 thread=%s", thread_id)
