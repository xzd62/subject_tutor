"""向量检索：bge 向量化 + Chroma 余弦相似度。"""

from __future__ import annotations

from ..index import ChromaStore, Embedder
from .types import RetrievedChunk


class VectorRetriever:
    def __init__(self, embedder: Embedder, store: ChromaStore):
        self.embedder = embedder
        self.store = store

    def search(
        self,
        query: str,
        top_k: int = 3,
        subject: str | None = None,
    ) -> list[RetrievedChunk]:
        embedding = self.embedder.embed_query(query)
        hits = self.store.query(embedding, top_k=top_k, subject=subject)
        results: list[RetrievedChunk] = []
        for hit in hits:
            distance = hit.get("distance")
            similarity = (
                1.0 - float(distance) if isinstance(distance, (int, float)) else None
            )
            results.append(
                RetrievedChunk(
                    chunk_id=hit["chunk_id"],
                    text=hit.get("document") or "",
                    metadata=dict(hit.get("metadata") or {}),
                    vector_score=similarity,
                )
            )
        return results
