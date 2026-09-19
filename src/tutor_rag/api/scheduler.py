"""定时兜底：启动补齐 + 每天 00:00 扫描教材变更并重建索引。

扫描复用流水线的 content_hash 变更检测（只重建变动的文档）；
每次执行结果写入 outputs/schedule-last.json，供前端展示与启动补齐判断。
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..pipeline import load_manifest, run_ingest
from ..trace import get_logger
from .state import RuntimeState

SCHEDULE_STATE_FILE = "schedule-last.json"
STARTUP_CATCHUP_HOURS = 20
STARTUP_DELAY_S = 30.0


def schedule_state_path(runtime: RuntimeState) -> Path:
    return runtime.settings.paths.outputs_dir / SCHEDULE_STATE_FILE


def load_schedule_state(runtime: RuntimeState) -> dict[str, Any]:
    path = schedule_state_path(runtime)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        get_logger().warning("扫描状态文件解析失败: %r", exc)
        return {}


def run_sweep(runtime: RuntimeState, trigger: str = "manual") -> dict[str, Any]:
    started = time.perf_counter()
    with runtime.ingest_lock:
        result = run_ingest(runtime.settings, embedder=runtime.retrieval.embedder)
        runtime.retrieval.refresh()

    manifest = load_manifest(runtime.settings.paths.manifest_path).get("documents", {})
    rebuilt = [
        manifest[doc_id].get("source_file", doc_id)
        for doc_id in result.ingested
        if doc_id in manifest
    ]
    summary = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "trigger": trigger,
        "rebuilt": rebuilt,
        "rebuilt_count": len(rebuilt),
        "skipped_count": len(result.skipped),
        "failed_count": len(result.failed),
        "elapsed_s": round(time.perf_counter() - started, 1),
        "corpus_size": runtime.retrieval.corpus_size,
        "collection_chunks": runtime.retrieval.store.count(),
    }
    path = schedule_state_path(runtime)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    get_logger().info(
        "教材扫描完成(%s): 重建 %d 本 / 跳过 %d 本 / 失败 %d 本，耗时 %.1fs",
        trigger,
        len(rebuilt),
        len(result.skipped),
        len(result.failed),
        summary["elapsed_s"],
    )
    return summary


def seconds_until_midnight(now: datetime | None = None) -> float:
    current = now or datetime.now()
    tomorrow = (current + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return max(1.0, (tomorrow - current).total_seconds())


def _needs_startup_catchup(runtime: RuntimeState) -> bool:
    state = load_schedule_state(runtime)
    last_time_text = state.get("time")
    if not last_time_text:
        return True
    try:
        last_time = datetime.fromisoformat(str(last_time_text))
    except ValueError:
        return True
    return datetime.now() - last_time >= timedelta(hours=STARTUP_CATCHUP_HOURS)


async def sweep_loop(runtime: RuntimeState) -> None:
    try:
        await asyncio.sleep(STARTUP_DELAY_S)
        if _needs_startup_catchup(runtime):
            await asyncio.to_thread(run_sweep, runtime, "startup")
        while True:
            await asyncio.sleep(seconds_until_midnight())
            await asyncio.to_thread(run_sweep, runtime, "nightly")
    except asyncio.CancelledError:
        get_logger().info("定时扫描任务已停止")
        raise
    except Exception as exc:
        get_logger().warning("定时扫描任务异常退出: %r", exc)
