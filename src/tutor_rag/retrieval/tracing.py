"""检索过程留痕：每轮 query 的三路候选、耗时、降级情况写入 JSONL。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .types import RetrievalResult


class RetrievalTracer:
    def __init__(self, directory: Path | None):
        self.directory = directory
        self.path: Path | None = None
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / f"trace-{datetime.now():%Y%m%d}.jsonl"

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def record(self, result: RetrievalResult) -> None:
        if self.path is None:
            return
        payload = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "query": result.query,
            "top_k": result.top_k,
            "degraded": result.degraded,
            "elapsed_ms": {
                key: round(value, 1) for key, value in result.elapsed_ms.items()
            },
            "vector": [chunk.to_trace() for chunk in result.vector_hits],
            "bm25": [chunk.to_trace() for chunk in result.bm25_hits],
            "fused": [chunk.to_trace() for chunk in result.fused],
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
