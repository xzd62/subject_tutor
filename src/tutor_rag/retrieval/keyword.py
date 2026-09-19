"""BM25 关键词检索。

- 语料直接从 Chroma collection 导出（与向量库同源，避免 chunks.jsonl 过期不一致）；
- 中文分词用 jieba（搜索引擎模式，兼顾长词与子词），jieba 不可用时退化为字符二元组。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..index import ChromaStore
from .types import RetrievedChunk

_ASCII_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def bigram_tokenize(text: str) -> list[str]:
    tokens = [word.lower() for word in _ASCII_WORD_RE.findall(text)]
    for run in _CJK_RUN_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


_jieba_ready = False


def tokenize(text: str) -> list[str]:
    global _jieba_ready
    if not text.strip():
        return []
    try:
        import jieba
    except ImportError:
        return bigram_tokenize(text)
    if not _jieba_ready:
        jieba.setLogLevel(logging.ERROR)
        _jieba_ready = True
    return [token for token in jieba.lcut_for_search(text) if token.strip()]


class BM25Retriever:
    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        self._tokenized = [tokenize(str(chunk.get("text") or "")) for chunk in chunks]
        self._bm25 = None
        if self._tokenized and any(self._tokenized):
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self._tokenized)

    @classmethod
    def from_store(cls, store: ChromaStore) -> "BM25Retriever":
        return cls(store.export_all())

    @property
    def size(self) -> int:
        return len(self.chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
        subject: str | None = None,
    ) -> list[RetrievedChunk]:
        if self._bm25 is None or top_k <= 0:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        candidates: list[tuple[float, int]] = []
        for index, score in enumerate(scores):
            if score <= 0:
                continue
            metadata = self.chunks[index].get("metadata") or {}
            if subject and metadata.get("subject") != subject:
                continue
            candidates.append((float(score), index))
        candidates.sort(key=lambda item: (-item[0], item[1]))

        results: list[RetrievedChunk] = []
        for score, index in candidates[:top_k]:
            chunk = self.chunks[index]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk["chunk_id"],
                    text=chunk.get("text") or "",
                    metadata=dict(chunk.get("metadata") or {}),
                    bm25_score=score,
                )
            )
        return results
