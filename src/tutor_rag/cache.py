"""Redis 语义缓存：精确匹配 + RediSearch 向量近邻。

设计要点：
- 两层命中：精确键（归一化问题+指纹的 sha1）优先，语义层用 HNSW KNN（余弦相似度 ≥ 阈值）；
- 数字/公式保护：含数字或数学符号的问题跳过语义层（防止 x²-5x+6 与 x²-5x+7 互相串答案）；
- 指纹隔离：模型/提示词/记忆/上一轮问题任一变化即不可命中，避免跨上下文串答案；
- Redis 不可用时自动降级（返回未命中并记一次警告），不阻塞主流程。
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .config import CacheConfig
from .trace import get_logger

_FULLWIDTH = str.maketrans(
    "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ＋－＝×÷（）",
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-=×÷()",
)
_PUNCT_RE = re.compile(
    r"[\s，。！？、；：“”‘’《》【】,.!?;:()\[\]{}<>\"'`~@#$%^&*_+=\-|/\\]+"
)
_NUMERIC_RE = re.compile(
    r"[0-9]|[+\-×÷√^=<>]|\$|\\(?:frac|sqrt|begin|times|le|ge|pm)"
    r"|\u00b2|\u00b3|\u2070|[\u2080-\u209f]"
)


def normalize_question(text: str) -> str:
    value = str(text or "").translate(_FULLWIDTH).strip().lower()
    return _PUNCT_RE.sub("", value)


def is_numeric_sensitive(text: str) -> bool:
    return bool(_NUMERIC_RE.search(str(text or "")))


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


_TAG_SPECIAL = set(',.<>{}[]"\':;!@#$%^&*()-+=~| \\')


def escape_tag(value: str) -> str:
    return "".join(
        f"\\{char}" if char in _TAG_SPECIAL else char for char in str(value)
    )


def model_fingerprint(chat_config: Any) -> str:
    raw = "|".join(
        [
            str(chat_config.model),
            str(chat_config.temperature),
            str(chat_config.max_tokens),
            str(chat_config.timeout_s),
            str(chat_config.max_retries),
        ]
    )
    return sha1_text(raw)[:16]


def prompt_fingerprint() -> str:
    from .agent.prompts import (
        DELEGATION_RULES,
        MEMORY_RULES,
        RETRIEVAL_INSTRUCTIONS,
        SUBJECT_AGENT_PROMPT,
        TUTOR_SYSTEM_PROMPT,
    )

    raw = "|".join(
        [TUTOR_SYSTEM_PROMPT, DELEGATION_RULES, MEMORY_RULES, RETRIEVAL_INSTRUCTIONS, SUBJECT_AGENT_PROMPT]
    )
    return sha1_text(raw)[:16]


def memory_fingerprint(store: Any) -> str:
    if store is None:
        return "none"
    parts = [f"{meta.name}|{meta.updated_at}" for meta in store.list_metas() if not meta.damaged]
    return sha1_text("\n".join(sorted(parts)))[:16]


@dataclass
class CacheHit:
    cache_id: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    score: float = 1.0
    hit_type: str = "exact"


class CacheBackend(Protocol):
    def ping(self) -> bool: ...

    def exact_get(self, key: str) -> str | None: ...

    def exact_set(self, key: str, value: str, ttl: int) -> None: ...

    def exact_delete(self, key: str) -> None: ...

    def hash_set(self, cache_id: str, mapping: dict[str, Any], ttl: int) -> None: ...

    def hash_get(self, cache_id: str) -> dict[str, Any] | None: ...

    def hash_delete(self, cache_id: str) -> None: ...

    def knn(
        self, embedding: Sequence[float], filters: dict[str, str], top_k: int = 1
    ) -> tuple[str, float] | None: ...

    def count(self) -> int: ...

    def clear(self) -> int: ...


class RedisBackend:
    def __init__(self, config: CacheConfig):
        import redis as redis_lib

        self.config = config
        self.namespace = config.namespace
        self.prefix = f"{config.namespace}:cache:"
        self.exact_prefix = f"{config.namespace}:exact:"
        self.index = f"{config.namespace}:cache:idx"
        self.client = redis_lib.Redis.from_url(
            config.redis_url,
            socket_connect_timeout=1.5,
            socket_timeout=3.0,
            protocol=2,
        )
        self._index_ready = False

    def ping(self) -> bool:
        return bool(self.client.ping())

    def _ensure_index(self) -> None:
        if self._index_ready:
            return
        try:
            self.client.execute_command("FT.INFO", self.index)
        except Exception:
            self.client.execute_command(
                "FT.CREATE",
                self.index,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                self.prefix,
                "SCHEMA",
                "question",
                "TEXT",
                "answer",
                "TEXT",
                "sources",
                "TEXT",
                "created_at",
                "NUMERIC",
                "model_fp",
                "TAG",
                "memory_fp",
                "TAG",
                "context_fp",
                "TAG",
                "prompt_fp",
                "TAG",
                "embedding",
                "VECTOR",
                "HNSW",
                "6",
                "TYPE",
                "FLOAT32",
                "DIM",
                str(self.config.vector_dim),
                "DISTANCE_METRIC",
                "COSINE",
            )
            get_logger().info("已创建 Redis 向量索引: %s", self.index)
        self._index_ready = True

    def exact_get(self, key: str) -> str | None:
        value = self.client.get(key)
        return value.decode("utf-8") if isinstance(value, bytes) else value

    def exact_set(self, key: str, value: str, ttl: int) -> None:
        self.client.set(key, value, ex=ttl)

    def exact_delete(self, key: str) -> None:
        self.client.delete(key)

    def hash_set(self, cache_id: str, mapping: dict[str, Any], ttl: int) -> None:
        key = f"{self.prefix}{cache_id}"
        payload: dict[str, Any] = {}
        for field_name, value in mapping.items():
            if value is None:
                continue
            if field_name == "embedding" and not isinstance(value, (bytes, bytearray)):
                vector = [float(item) for item in value]
                value = struct.pack(f"<{len(vector)}f", *vector)
            payload[field_name] = value
        self.client.hset(key, mapping=payload)
        self.client.expire(key, ttl)

    def hash_get(self, cache_id: str) -> dict[str, Any] | None:
        raw = self.client.hgetall(f"{self.prefix}{cache_id}")
        if not raw:
            return None
        result: dict[str, Any] = {}
        for key, value in raw.items():
            name = key.decode("utf-8") if isinstance(key, bytes) else str(key)
            if isinstance(value, bytes):
                try:
                    result[name] = value.decode("utf-8")
                except UnicodeDecodeError:
                    result[name] = value
            else:
                result[name] = value
        return result

    def hash_delete(self, cache_id: str) -> None:
        self.client.delete(f"{self.prefix}{cache_id}")

    def knn(
        self, embedding: Sequence[float], filters: dict[str, str], top_k: int = 1
    ) -> tuple[str, float] | None:
        self._ensure_index()
        vector = struct.pack(f"<{len(embedding)}f", *[float(x) for x in embedding])
        expressions = [
            f"@{name}:{{{escape_tag(value)}}}" for name, value in filters.items() if value
        ]
        filter_expr = " ".join(expressions)
        query = (
            f"({filter_expr})=>[KNN {top_k} @embedding $vec AS score]"
            if filter_expr
            else f"*=>[KNN {top_k} @embedding $vec AS score]"
        )
        result = self.client.execute_command(
            "FT.SEARCH",
            self.index,
            query,
            "PARAMS",
            "2",
            "vec",
            vector,
            "SORTBY",
            "score",
            "RETURN",
            "1",
            "score",
            "DIALECT",
            "2",
            "LIMIT",
            "0",
            str(top_k),
        )
        if isinstance(result, dict):
            rows = result.get("results") or result.get(b"results") or []
            if not rows:
                return None
            row = rows[0]
            key = row.get("id") or row.get(b"id")
            attributes = row.get("extra_attributes") or row.get(b"extra_attributes") or {}
        else:
            if not result or int(result[0]) == 0 or len(result) < 3:
                return None
            key = result[1]
            fields = result[2] or []
            attributes = dict(zip(fields[0::2], fields[1::2]))

        if key is None:
            return None
        cache_id = key.decode("utf-8") if isinstance(key, bytes) else str(key)
        cache_id = cache_id.removeprefix(self.prefix)
        raw_distance = attributes.get("score") or attributes.get(b"score")
        distance = float(raw_distance) if raw_distance is not None else 0.0
        return cache_id, distance

    def count(self) -> int:
        try:
            self._ensure_index()
            info = self.client.execute_command("FT.INFO", self.index)
            data = dict(zip(info[0::2], info[1::2]))
            value = data.get(b"num_docs") or data.get("num_docs") or 0
            return int(value)
        except Exception:
            return sum(1 for _ in self.client.scan_iter(match=f"{self.prefix}*"))

    def clear(self) -> int:
        removed = 0
        for pattern in (f"{self.prefix}*", f"{self.exact_prefix}*"):
            batch: list[Any] = []
            for key in self.client.scan_iter(match=pattern, count=200):
                batch.append(key)
                if len(batch) >= 200:
                    removed += int(self.client.delete(*batch))
                    batch = []
            if batch:
                removed += int(self.client.delete(*batch))
        return removed


class InMemoryCacheBackend:
    """测试/无 Redis 场景的内存后端（余弦检索，确定性）。"""

    def __init__(self):
        self.hashes: dict[str, dict[str, Any]] = {}
        self.exacts: dict[str, str] = {}
        self.available = True

    def ping(self) -> bool:
        if not self.available:
            raise ConnectionError("backend unavailable")
        return True

    def exact_get(self, key: str) -> str | None:
        return self.exacts.get(key)

    def exact_set(self, key: str, value: str, ttl: int) -> None:
        self.exacts[key] = value

    def exact_delete(self, key: str) -> None:
        self.exacts.pop(key, None)

    def hash_set(self, cache_id: str, mapping: dict[str, Any], ttl: int) -> None:
        self.hashes[cache_id] = {
            name: value for name, value in mapping.items() if value is not None
        }

    def hash_get(self, cache_id: str) -> dict[str, Any] | None:
        return self.hashes.get(cache_id)

    def hash_delete(self, cache_id: str) -> None:
        self.hashes.pop(cache_id, None)

    def knn(
        self, embedding: Sequence[float], filters: dict[str, str], top_k: int = 1
    ) -> tuple[str, float] | None:
        import math

        best: tuple[str, float] | None = None
        for cache_id, payload in self.hashes.items():
            if any(
                value and str(payload.get(name)) != value
                for name, value in filters.items()
            ):
                continue
            vector = payload.get("embedding")
            if not vector:
                continue
            dot = sum(a * b for a, b in zip(embedding, vector))
            norm_a = math.sqrt(sum(a * a for a in embedding)) or 1.0
            norm_b = math.sqrt(sum(b * b for b in vector)) or 1.0
            similarity = dot / (norm_a * norm_b)
            distance = 1.0 - similarity
            if best is None or distance < best[1]:
                best = (cache_id, distance)
        return best

    def count(self) -> int:
        return len(self.hashes)

    def clear(self) -> int:
        removed = len(self.hashes) + len(self.exacts)
        self.hashes.clear()
        self.exacts.clear()
        return removed


class SemanticCache:
    def __init__(
        self,
        config: CacheConfig,
        embedder: Any,
        backend: CacheBackend | None = None,
    ):
        self.config = config
        self.embedder = embedder
        self._backend = backend
        self._disabled = False
        self.counters: dict[str, int] = {
            "hit_exact": 0,
            "hit_semantic": 0,
            "miss": 0,
            "protected_skip": 0,
            "store": 0,
            "delete": 0,
            "clear": 0,
            "errors": 0,
        }

    @property
    def enabled(self) -> bool:
        return self.config.enabled and not self._disabled

    def _get_backend(self) -> CacheBackend | None:
        if not self.config.enabled:
            return None
        if self._backend is None:
            try:
                self._backend = RedisBackend(self.config)
            except Exception as exc:
                self._disable(f"Redis 初始化失败: {exc!r}")
                return None
        try:
            self._backend.ping()
        except Exception as exc:
            self._disable(f"Redis 连接失败: {exc!r}")
            return None
        return self._backend

    def _disable(self, reason: str) -> None:
        if not self._disabled:
            self._disabled = True
            self.counters["errors"] += 1
            get_logger().warning("语义缓存已降级（本次进程内不再重试）: %s", reason)

    def _filters(
        self, model_fp: str, memory_fp: str, context_fp: str, prompt_fp: str
    ) -> dict[str, str]:
        return {
            "model_fp": model_fp,
            "memory_fp": memory_fp,
            "context_fp": context_fp or "root",
            "prompt_fp": prompt_fp,
        }

    def _exact_key(self, question: str, filters: dict[str, str]) -> str:
        digest = sha1_text(normalize_question(question) + "|" + "|".join(
            f"{name}={value}" for name, value in sorted(filters.items())
        ))
        return f"{self.config.namespace}:exact:{digest}"

    def lookup(
        self,
        question: str,
        *,
        model_fp: str,
        memory_fp: str,
        context_fp: str = "",
        prompt_fp: str = "",
    ) -> CacheHit | None:
        if not self.enabled:
            return None
        backend = self._get_backend()
        if backend is None:
            return None
        filters = self._filters(model_fp, memory_fp, context_fp, prompt_fp)
        try:
            cache_id = backend.exact_get(self._exact_key(question, filters))
            if cache_id:
                payload = backend.hash_get(cache_id)
                if payload:
                    self.counters["hit_exact"] += 1
                    return self._to_hit(payload, 1.0, "exact")
                backend.exact_delete(self._exact_key(question, filters))
        except Exception as exc:
            self._disable(f"精确缓存查询失败: {exc!r}")
            return None

        if self.config.protect_numeric and is_numeric_sensitive(question):
            self.counters["protected_skip"] += 1
            return None

        try:
            embedding = self.embedder.embed_query(question)
            nearest = backend.knn(embedding, filters, top_k=1)
        except Exception as exc:
            self._disable(f"语义缓存查询失败: {exc!r}")
            return None

        if nearest is not None:
            cache_id, distance = nearest
            similarity = 1.0 - float(distance)
            if similarity >= self.config.threshold:
                try:
                    payload = backend.hash_get(cache_id)
                except Exception as exc:
                    self._disable(f"缓存读取失败: {exc!r}")
                    return None
                if payload:
                    self.counters["hit_semantic"] += 1
                    return self._to_hit(payload, similarity, "semantic")
        self.counters["miss"] += 1
        return None

    def store(
        self,
        question: str,
        answer: str,
        sources: list[dict] | None = None,
        *,
        model_fp: str,
        memory_fp: str,
        context_fp: str = "",
        prompt_fp: str = "",
    ) -> str | None:
        if not self.enabled or not str(answer or "").strip():
            return None
        backend = self._get_backend()
        if backend is None:
            return None
        filters = self._filters(model_fp, memory_fp, context_fp, prompt_fp)
        cache_id = uuid.uuid4().hex
        exact_key = self._exact_key(question, filters)
        mapping: dict[str, Any] = {
            "cache_id": cache_id,
            "question": str(question),
            "answer": str(answer),
            "sources": json.dumps(sources or [], ensure_ascii=False),
            "created_at": int(time.time()),
            "exact_key": exact_key,
            **filters,
        }
        if not (self.config.protect_numeric and is_numeric_sensitive(question)):
            try:
                mapping["embedding"] = [
                    float(x) for x in self.embedder.embed_query(question)
                ]
            except Exception as exc:
                get_logger().warning("缓存向量化失败，仅存精确缓存: %r", exc)
        try:
            backend.hash_set(cache_id, mapping, self.config.ttl_seconds)
            backend.exact_set(exact_key, cache_id, self.config.ttl_seconds)
        except Exception as exc:
            self._disable(f"缓存写入失败: {exc!r}")
            return None
        self.counters["store"] += 1
        return cache_id

    def delete(self, cache_id: str) -> bool:
        backend = self._get_backend()
        if backend is None or not cache_id:
            return False
        try:
            payload = backend.hash_get(cache_id)
            if payload and payload.get("exact_key"):
                backend.exact_delete(str(payload["exact_key"]))
            backend.hash_delete(cache_id)
        except Exception as exc:
            self._disable(f"缓存删除失败: {exc!r}")
            return False
        self.counters["delete"] += 1
        return bool(payload)

    def clear(self) -> int:
        backend = self._get_backend()
        if backend is None:
            return 0
        try:
            removed = backend.clear()
        except Exception as exc:
            self._disable(f"缓存清空失败: {exc!r}")
            return 0
        self.counters["clear"] += 1
        return removed

    def stats_payload(self) -> dict[str, Any]:
        connected = self._disabled is False and self.config.enabled
        entries = 0
        if self.config.enabled and not self._disabled:
            try:
                backend = self._get_backend()
                entries = backend.count() if backend is not None else 0
            except Exception:
                connected = False
        hits = self.counters["hit_exact"] + self.counters["hit_semantic"]
        lookups = hits + self.counters["miss"] + self.counters["protected_skip"]
        return {
            "enabled": self.config.enabled,
            "connected": connected,
            "namespace": self.config.namespace,
            "threshold": self.config.threshold,
            "protect_numeric": self.config.protect_numeric,
            "entries": entries,
            "hit_rate": round(hits / lookups, 4) if lookups else None,
            "counters": dict(self.counters),
        }

    @staticmethod
    def _to_hit(payload: dict[str, Any], score: float, hit_type: str) -> CacheHit:
        sources: list[dict] = []
        raw_sources = payload.get("sources")
        if isinstance(raw_sources, str) and raw_sources.strip():
            try:
                parsed = json.loads(raw_sources)
                if isinstance(parsed, list):
                    sources = parsed
            except json.JSONDecodeError:
                sources = []
        return CacheHit(
            cache_id=str(payload.get("cache_id") or payload.get("id") or ""),
            answer=str(payload.get("answer") or ""),
            sources=sources,
            score=float(score),
            hit_type=hit_type,
        )
