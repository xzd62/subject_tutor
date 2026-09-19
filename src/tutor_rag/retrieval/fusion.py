"""RRF（Reciprocal Rank Fusion）融合。

score(chunk) = Σ 1 / (rrf_k + rank_i)，rank 从 1 开始；
RRF 只用排名不用原始分，天然规避向量距离与 BM25 分数不可比的问题。
"""

from __future__ import annotations

from .types import RetrievedChunk

_MISSING_RANK = 10**9


def rrf_fuse(
    vector_hits: list[RetrievedChunk],
    bm25_hits: list[RetrievedChunk],
    rrf_k: int = 60,
    top_k: int = 3,
) -> list[RetrievedChunk]:
    registry: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}

    def collect(hits: list[RetrievedChunk], source: str) -> None:
        for rank, hit in enumerate(hits, start=1):
            target = registry.get(hit.chunk_id)
            if target is None:
                target = RetrievedChunk(
                    chunk_id=hit.chunk_id,
                    text=hit.text,
                    metadata=dict(hit.metadata),
                )
                registry[hit.chunk_id] = target
            if source == "vector":
                target.vector_rank = rank
                target.vector_score = hit.vector_score
            else:
                target.bm25_rank = rank
                target.bm25_score = hit.bm25_score
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    collect(vector_hits, "vector")
    collect(bm25_hits, "bm25")

    for chunk_id, score in scores.items():
        registry[chunk_id].rrf_score = score

    ordered = sorted(
        registry.values(),
        key=lambda item: (
            -item.rrf_score,
            min(item.vector_rank or _MISSING_RANK, item.bm25_rank or _MISSING_RANK),
        ),
    )
    return ordered[:top_k] if top_k > 0 else ordered
