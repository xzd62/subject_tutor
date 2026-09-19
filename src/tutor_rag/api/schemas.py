"""API 请求/响应模型与预设。"""

from __future__ import annotations

from pydantic import BaseModel, Field

MODEL_PRESETS: list[dict] = [
    {
        "provider": "deepseek",
        "label": "DeepSeek",
        "models": ["deepseek:deepseek-flash", "deepseek:deepseek-v4-pro"],
        "key_env": "DEEPSEEK_API_KEY",
        "installed": True,
        "note": "deepseek-flash 通用；deepseek-v4-pro 推理更强",
    },
    {
        "provider": "openai",
        "label": "OpenAI",
        "models": ["openai:gpt-4o-mini"],
        "key_env": "OPENAI_API_KEY",
        "installed": False,
        "note": "需安装 langchain-openai",
    },
    {
        "provider": "ollama",
        "label": "Ollama（本地）",
        "models": ["ollama:qwen2.5:7b"],
        "key_env": "",
        "installed": False,
        "note": "需安装 langchain-ollama 并启动本地 Ollama 服务",
    },
]


class TextbookContentPayload(BaseModel):
    content: str = Field(min_length=1, max_length=2_000_000)


class MemoryCreatePayload(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    type: str = Field(min_length=1, max_length=40)
    description: str = Field(default="", max_length=200)
    content: str = Field(min_length=1, max_length=200_000)


class MemoryUpdatePayload(BaseModel):
    type: str = Field(min_length=1, max_length=40)
    description: str = Field(default="", max_length=200)
    content: str = Field(min_length=1, max_length=200_000)


class ModelSettingsPayload(BaseModel):
    model: str = Field(min_length=3, max_length=120)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=256, le=32768)
    timeout_s: int = Field(default=60, ge=5, le=600)
    max_retries: int = Field(default=2, ge=0, le=5)
    api_key: str = Field(default="", max_length=200)
