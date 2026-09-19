"""检索数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector_rank: int | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0

    @property
    def heading_path(self) -> str:
        return str(self.metadata.get("heading_path", ""))

    @property
    def subject(self) -> str:
        return str(self.metadata.get("subject", ""))

    @property
    def source_file(self) -> str:
        return str(self.metadata.get("source_file", ""))

    @property
    def sources(self) -> tuple[str, ...]:
        found: list[str] = []
        if self.vector_rank is not None:
            found.append("vector")
        if self.bm25_rank is not None:
            found.append("bm25")
        return tuple(found)

    def to_trace(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "subject": self.subject,
            "source_file": self.source_file,
            "heading_path": self.heading_path,
            "vector_rank": self.vector_rank,
            "vector_score": (
                round(self.vector_score, 4) if self.vector_score is not None else None
            ),
            "bm25_rank": self.bm25_rank,
            "bm25_score": (
                round(self.bm25_score, 4) if self.bm25_score is not None else None
            ),
            "rrf_score": round(self.rrf_score, 6),
            "preview": self.text.replace("\n", " ")[:60],
        }


@dataclass
class RetrievalResult:
    query: str
    top_k: int
    vector_hits: list[RetrievedChunk] = field(default_factory=list)
    bm25_hits: list[RetrievedChunk] = field(default_factory=list)
    fused: list[RetrievedChunk] = field(default_factory=list)
    elapsed_ms: dict[str, float] = field(default_factory=dict)
    degraded: str | None = None

    def by_mode(self, mode: str) -> list[RetrievedChunk]:
        if mode == "vector":
            return self.vector_hits
        if mode == "bm25":
            return self.bm25_hits
        return self.fused
