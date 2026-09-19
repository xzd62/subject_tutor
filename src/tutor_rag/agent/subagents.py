"""学科子智能体与委派工具（协调者模式）。

- 9 个学科子代理：无状态、无记忆工具、无委派工具，只带“按本科目过滤”的教材检索中间件；
- 主管通过唯一的 ask_subject_expert(subject, task) 工具委派；
- 工具返回“子代理答案 + 参考教材章节”，便于主管汇总时标注来源。
"""

from __future__ import annotations

import time
from typing import Any, Callable, Literal

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool

from ..config import Settings
from ..retrieval.service import RetrievalService
from ..retrieval.types import RetrievalResult
from ..trace import get_logger
from .middleware import make_retrieval_middleware
from .prompts import SUBJECT_AGENT_PROMPT

SubjectName = Literal["语文", "数学", "英语", "物理", "化学", "生物", "政治", "历史", "地理"]

MAX_SOURCES = 5


def build_subject_agent(
    subject: str,
    model: Any,
    settings: Settings,
    retrieval_service: RetrievalService,
    on_retrieval: Callable[[RetrievalResult], None] | None = None,
):
    middleware = []
    if settings.retrieval.enabled:
        middleware.append(
            make_retrieval_middleware(
                retrieval_service,
                top_k=settings.subagents.top_k,
                subject=subject,
                on_result=on_retrieval,
            )
        )
    return create_agent(
        model=model,
        system_prompt=SUBJECT_AGENT_PROMPT.format(subject=subject),
        middleware=tuple(middleware),
    )


def _last_ai_message(messages: list[Any]) -> Any | None:
    for message in reversed(messages):
        if getattr(message, "type", "") == "ai":
            return message
    return None


def _empty_answer_note(message: Any | None) -> str:
    usage = getattr(message, "usage_metadata", None) or {}
    reasoning = (usage.get("output_token_details") or {}).get("reasoning")
    if reasoning:
        return (
            f"（本科目解答因达到 max_tokens 限制未能输出正文，"
            f"思考消耗 {reasoning} tokens，可在模型设置中调大 max_tokens 后重试）"
        )
    return "（本科目老师未生成有效内容）"


def _format_sources(results: list[RetrievalResult]) -> str:
    labels: list[str] = []
    for result in results:
        for chunk in result.fused:
            label = f"- {chunk.heading_path or chunk.source_file}（{chunk.source_file}）"
            if label not in labels:
                labels.append(label)
    if not labels:
        return ""
    return "\n\n参考教材：\n" + "\n".join(labels[:MAX_SOURCES])


def make_subject_tools(
    settings: Settings,
    model_factory: Callable[[str], Any],
    retrieval_service: RetrievalService,
) -> list:
    @tool
    def ask_subject_expert(subject: SubjectName, task: str) -> str:
        """把具体的学科问题委派给对应的学科老师，返回学科老师的解答和参考教材。

        subject: 学科（语文/数学/英语/物理/化学/生物/政治/历史/地理 之一）。
        task: 交给该学科老师的完整任务描述（题目、已知条件、期望的解答形式，越具体越好）。
        """
        collector: list[RetrievalResult] = []
        agent = build_subject_agent(
            subject,
            model=model_factory(subject),
            settings=settings,
            retrieval_service=retrieval_service,
            on_retrieval=collector.append,
        )
        started = time.perf_counter()
        try:
            result = agent.invoke({"messages": [HumanMessage(content=task)]})
        except Exception as exc:
            get_logger().warning("学科子代理 %s 执行失败: %r", subject, exc)
            return f"{subject}老师暂时无法回答（{exc}）"

        message = _last_ai_message(result.get("messages", []))
        content = getattr(message, "content", "")
        answer = content.strip() if isinstance(content, str) else ""
        get_logger().info(
            "学科子代理 %s 完成: 耗时=%.1fs 字数=%d usage=%s",
            subject,
            time.perf_counter() - started,
            len(answer),
            getattr(message, "usage_metadata", None),
        )
        if not answer:
            get_logger().warning(
                "学科子代理 %s 返回空内容: extra=%s",
                subject,
                list((getattr(message, "additional_kwargs", {}) or {}).keys())
                if message is not None
                else "no-ai-message",
            )
            return f"{subject}老师{_empty_answer_note(message)}"
        return answer + _format_sources(collector)

    return [ask_subject_expert]
