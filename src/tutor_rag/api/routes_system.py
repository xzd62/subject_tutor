"""系统状态：定时扫描结果。"""

from __future__ import annotations

from fastapi import APIRouter, Request

from .scheduler import load_schedule_state
from .state import RuntimeState

router = APIRouter()


@router.get("/api/schedule")
def get_schedule_state(request: Request) -> dict:
    runtime: RuntimeState = request.app.state.runtime
    return {"last_run": load_schedule_state(runtime)}
