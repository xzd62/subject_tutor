"""模型设置：预设、读取/保存（本地 JSON）、连接测试。"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request

from ..trace import get_logger
from .schemas import MODEL_PRESETS, ModelSettingsPayload
from .state import RuntimeState

router = APIRouter()


@router.get("/api/settings")
def get_model_settings(request: Request) -> dict:
    runtime: RuntimeState = request.app.state.runtime
    return {
        "settings": runtime.settings_payload(),
        "presets": MODEL_PRESETS,
    }


@router.put("/api/settings")
def update_model_settings(payload: ModelSettingsPayload, request: Request) -> dict:
    runtime: RuntimeState = request.app.state.runtime
    if ":" not in payload.model:
        raise HTTPException(400, "模型格式应为 provider:model，例如 deepseek:deepseek-flash")
    try:
        runtime.update_settings(payload.model_dump())
    except Exception as exc:
        raise HTTPException(400, f"模型初始化失败: {exc}") from exc
    get_logger().info("设置页保存模型: %s", payload.model)
    return {"settings": runtime.settings_payload()}


@router.post("/api/settings/test")
def test_model_settings(request: Request) -> dict:
    runtime: RuntimeState = request.app.state.runtime
    try:
        model = runtime.get_model()
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    started = time.perf_counter()
    try:
        reply = model.invoke("请只回复：ok")
    except Exception as exc:
        raise HTTPException(400, f"调用失败: {exc}") from exc
    latency = time.perf_counter() - started
    content = getattr(reply, "content", "")
    return {
        "ok": True,
        "model": runtime.chat_config.model,
        "latency_s": round(latency, 2),
        "reply": str(content)[:50],
    }
