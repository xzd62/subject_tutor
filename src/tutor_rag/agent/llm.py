"""对话模型工厂。

- 用 LangChain `init_chat_model()` 统一接入各厂商（`provider:model` 写法）；
- 消息格式、流式、工具调用、结构化输出由 LangChain 归一化，业务代码不感知厂商差异；
- 换模型：改 `TUTOR_RAG_LLM_MODEL`、装对应集成包（如 langchain-openai）、配好 API key 即可；
- 降级链在 ChatModel 层实现：`create_agent` 需要 `bind_tools`，而
  `Runnable.with_fallbacks()` 不代理该方法，因此需要 FallbackChatModel。
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from ..config import ChatConfig, get_settings
from ..trace import get_logger


class FallbackChatModel(BaseChatModel):
    models: list[BaseChatModel] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fallback-chat"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FallbackChatModel":
        return FallbackChatModel(
            models=[model.bind_tools(tools, **kwargs) for model in self.models]
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_error: Exception | None = None
        for model in self.models:
            try:
                message = model.invoke(messages, stop=stop)
                return ChatResult(generations=[ChatGeneration(message=message)])
            except Exception as exc:
                last_error = exc
                get_logger().warning(
                    "模型降级: %s 调用失败 (%r)，尝试下一个", model_label(model), exc
                )
        if last_error is None:
            raise RuntimeError("FallbackChatModel 没有可用的模型")
        raise last_error

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        last_error: Exception | None = None
        for model in self.models:
            try:
                for chunk in model.stream(messages, stop=stop):
                    yield ChatGenerationChunk(message=chunk)
                return
            except Exception as exc:
                last_error = exc
                get_logger().warning(
                    "模型降级(流式): %s 调用失败 (%r)，尝试下一个", model_label(model), exc
                )
        if last_error is None:
            raise RuntimeError("FallbackChatModel 没有可用的模型")
        raise last_error


def model_label(model: BaseChatModel) -> str:
    name = getattr(model, "model_name", "") or getattr(model, "model", "")
    return str(name) if name else model._llm_type


def create_model(name: str, config: ChatConfig | None = None) -> BaseChatModel:
    config = config or get_settings().chat
    kwargs: dict[str, Any] = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout_s,
        "max_retries": config.max_retries,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    try:
        return init_chat_model(name, **kwargs)
    except ImportError as exc:
        raise RuntimeError(
            f"模型 {name!r} 的集成包未安装，请先安装对应的 langchain-* 包"
            f"（原始错误: {exc}）"
        ) from exc


def build_chat_model(
    config: ChatConfig | None = None,
    model_name: str | None = None,
) -> BaseChatModel:
    config = config or get_settings().chat
    primary_name = model_name or config.model
    models = [create_model(primary_name, config)]
    models.extend(create_model(name, config) for name in config.fallback_models)
    get_logger().info(
        "对话模型就绪: %s%s",
        primary_name,
        f"（降级链: {', '.join(config.fallback_models)}）" if config.fallback_models else "",
    )
    if len(models) == 1:
        return models[0]
    return FallbackChatModel(models=models)
