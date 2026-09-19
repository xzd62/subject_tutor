"""检索编排：向量 + BM25 并行召回 -> RRF 融合 -> 留痕。

设计要点：
- 单路失败不影响另一路（降级继续，并写入 degraded 原因）；
- BM25 语料来自 Chroma，保证与向量库同源；
- 预留 rerank 扩展点：后续只需在 fuse 之后插入重排并在此处记录耗时。
"""

from __future__ import annotations

import time

from ..config import Settings, get_settings
from ..index import ChromaStore, Embedder
from ..trace import get_logger
from .fusion import rrf_fuse
from .keyword import BM25Retriever
from .tracing import RetrievalTracer
from .types import RetrievalResult
from .vector import VectorRetriever


class RetrievalService:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        store: ChromaStore | None = None,
        bm25: BM25Retriever | None = None,
        tracer: RetrievalTracer | None = None,
    ):
        self.settings = settings or get_settings()
        self.config = self.settings.retrieval
        self.paths = self.settings.paths
        self.embedder = embedder or Embedder(self.settings.embedding)
        self.store = store or ChromaStore(self.paths.chroma_dir, self.settings.chroma)
        self.vector = VectorRetriever(self.embedder, self.store)
        self.bm25 = bm25 or BM25Retriever.from_store(self.store)
        if tracer is None and self.config.trace:
            tracer = RetrievalTracer(self.paths.retrieval_dir)
        self.tracer = tracer

    @property
    def corpus_size(self) -> int:
        return self.bm25.size

    def refresh(self) -> None:
        self.store.reload()
        self.bm25 = BM25Retriever.from_store(self.store)
        get_logger().info("检索索引已刷新: corpus=%d 块", self.bm25.size)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        subject: str | None = None,
        rrf_k: int | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self.config.top_k
        rrf_k = rrf_k or self.config.rrf_k
        result = RetrievalResult(query=query, top_k=top_k)

        started = time.perf_counter()
        try:
            stage = time.perf_counter()
            result.vector_hits = self.vector.search(query, top_k=top_k, subject=subject)
            result.elapsed_ms["vector"] = (time.perf_counter() - stage) * 1000
        except Exception as exc:
            result.degraded = f"vector_failed: {exc!r}"
            get_logger().warning("向量检索失败，降级继续: %r", exc)

        try:
            stage = time.perf_counter()
            result.bm25_hits = self.bm25.search(query, top_k=top_k, subject=subject)
            result.elapsed_ms["bm25"] = (time.perf_counter() - stage) * 1000
        except Exception as exc:
            reason = f"bm25_failed: {exc!r}"
            result.degraded = f"{result.degraded}; {reason}" if result.degraded else reason
            get_logger().warning("BM25 检索失败，降级继续: %r", exc)

        stage = time.perf_counter()
        result.fused = rrf_fuse(
            result.vector_hits,
            result.bm25_hits,
            rrf_k=rrf_k,
            top_k=top_k,
        )
        result.elapsed_ms["fuse"] = (time.perf_counter() - stage) * 1000
        result.elapsed_ms["total"] = (time.perf_counter() - started) * 1000

        if self.tracer is not None:
            self.tracer.record(result)

        get_logger().info(
            "检索完成 query=%r top_k=%d vector=%d bm25=%d fused=%d 耗时=%.1fms 降级=%s",
            query[:40],
            top_k,
            len(result.vector_hits),
            len(result.bm25_hits),
            len(result.fused),
            result.elapsed_ms["total"],
            result.degraded or "无",
        )
        return result
