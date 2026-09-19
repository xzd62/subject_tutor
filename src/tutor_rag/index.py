"""向量化与 Chroma 持久化。

- Embedder：封装 LlamaIndex 的 HuggingFaceEmbedding（bge 系列，CLS pooling + 归一化），
  查询侧由本模块统一拼接 bge 推荐的 query instruction；
- ChromaStore：直接使用 chromadb 持久化客户端，按 chunk_id upsert、按 doc_id 删除，
  既保证幂等，也方便二期把同一 collection 挂给 LlamaIndex 检索器。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from .chunk import ChunkRecord
from .config import ChromaConfig, EmbeddingConfig
from .trace import get_logger

ProgressCallback = Callable[[int], None]

_PAGE_SIZE = 1000


class Embedder:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.device: str | None = None
        self._model = None
        self._lock = threading.RLock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                return self._load_model()
        return self._model

    def _load_model(self):
        if self._model is None:
            import torch
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            from transformers.utils import logging as transformers_logging

            transformers_logging.set_verbosity_error()
            transformers_logging.disable_progress_bar()
            device = self.config.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            get_logger().info(
                "加载嵌入模型 %s (device=%s, max_length=%d, batch=%d)",
                self.config.model_name,
                device,
                self.config.max_length,
                self.config.batch_size,
            )
            self._model = HuggingFaceEmbedding(
                model_name=self.config.model_name,
                max_length=self.config.max_length,
                embed_batch_size=self.config.batch_size,
                device=device,
                normalize=self.config.normalize,
            )
        return self._model

    def embed_documents(
        self, texts: Sequence[str], on_progress: ProgressCallback | None = None
    ) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:
            model = self.model
            batch_size = max(1, self.config.batch_size)
            vectors: list[list[float]] = []
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                vectors.extend(model.get_text_embedding_batch(batch, show_progress=False))
                if on_progress is not None:
                    on_progress(min(start + batch_size, len(texts)))
            return vectors

    def embed_query(self, text: str) -> list[float]:
        query = f"{self.config.query_instruction}{text}"
        with self._lock:
            return self.model.get_text_embedding(query)


class ChromaStore:
    def __init__(self, persist_dir: Path, config: ChromaConfig):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.config = config
        self.persist_dir = persist_dir
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": config.distance},
        )

    def upsert_chunks(
        self,
        records: Sequence[ChunkRecord],
        embeddings: Sequence[Sequence[float]],
        on_progress: ProgressCallback | None = None,
    ) -> int:
        if len(records) != len(embeddings):
            raise ValueError(
                f"记录数与向量数不一致: {len(records)} != {len(embeddings)}"
            )
        batch_size = max(1, self.config.upsert_batch_size)
        written = 0
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            self.collection.upsert(
                ids=[record.chunk_id for record in batch],
                embeddings=[list(vector) for vector in embeddings[start : start + batch_size]],
                documents=[record.embed_text for record in batch],
                metadatas=[record.metadata for record in batch],
            )
            written += len(batch)
            if on_progress is not None:
                on_progress(written)
        return written

    def reload(self) -> None:
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance},
        )

    def delete_document(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        return self.collection.count()

    def count_document(self, doc_id: str) -> int:
        result = self.collection.get(where={"doc_id": doc_id}, include=[])
        return len(result.get("ids") or [])

    def count_by_doc(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for metadata in self._iter_metadatas():
            doc_id = metadata.get("doc_id")
            if doc_id:
                counts[doc_id] = counts.get(doc_id, 0) + 1
        return counts

    def doc_ids(self) -> set[str]:
        return set(self.count_by_doc())

    def _iter_metadatas(self) -> Iterator[dict[str, Any]]:
        offset = 0
        while True:
            batch = self.collection.get(include=["metadatas"], limit=_PAGE_SIZE, offset=offset)
            metadatas = batch.get("metadatas") or []
            if not metadatas:
                break
            for metadata in metadatas:
                if metadata:
                    yield metadata
            offset += len(metadatas)
            if len(metadatas) < _PAGE_SIZE:
                break

    def export_all(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        offset = 0
        while True:
            batch = self.collection.get(
                include=["documents", "metadatas"], limit=_PAGE_SIZE, offset=offset
            )
            ids = batch.get("ids") or []
            if not ids:
                break
            documents = batch.get("documents") or []
            metadatas = batch.get("metadatas") or []
            for index, chunk_id in enumerate(ids):
                records.append(
                    {
                        "chunk_id": chunk_id,
                        "text": documents[index] if index < len(documents) else "",
                        "metadata": metadatas[index] if index < len(metadatas) else {},
                    }
                )
            offset += len(ids)
            if len(ids) < _PAGE_SIZE:
                break
        return records

    def query(
        self,
        embedding: Sequence[float],
        top_k: int = 5,
        subject: str | None = None,
    ) -> list[dict[str, Any]]:
        where = {"subject": subject} if subject else None
        result = self.collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for position, chunk_id in enumerate(ids):
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "document": documents[position] if position < len(documents) else "",
                    "metadata": metadatas[position] if position < len(metadatas) else {},
                    "distance": distances[position] if position < len(distances) else None,
                }
            )
        return hits

    def reset(self) -> None:
        self.client.delete_collection(self.config.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance},
        )

    def to_llama_index_vector_store(self):
        from llama_index.vector_stores.chroma import ChromaVectorStore

        return ChromaVectorStore(chroma_collection=self.collection)


def format_embedding_summary(model_name: str, device: str | None, dim: int) -> str:
    return f"模型={model_name} device={device or 'n/a'} 维度={dim}"
