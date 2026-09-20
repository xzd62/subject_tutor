"""全局配置：路径、学科枚举、切块参数、嵌入模型与 Chroma 设置。

所有参数都可以通过环境变量或项目根目录的 .env 覆盖。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None and (PROJECT_ROOT / ".env").exists():
    load_dotenv(PROJECT_ROOT / ".env")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


SUBJECTS: tuple[str, ...] = ("语文", "数学", "英语", "物理", "化学", "生物", "政治", "历史", "地理")

GRADES: tuple[str, ...] = ("七上", "七下", "八上", "八下", "九上", "九下")

GRADE_ALIASES: dict[str, str] = {
    "七年级上": "七上",
    "七年级下": "七下",
    "八年级上": "八上",
    "八年级下": "八下",
    "九年级上": "九上",
    "九年级下": "九下",
    "七年级上册": "七上",
    "七年级下册": "七下",
    "八年级上册": "八上",
    "八年级下册": "八下",
    "九年级上册": "九上",
    "九年级下册": "九下",
}

COMMON_VERSIONS: tuple[str, ...] = (
    "人教版", "部编版", "统编版", "苏教版", "北师大版", "沪教版", "外研版",
    "译林版", "湘教版", "鲁教版", "浙教版", "沪科版", "岳麓版", "科普版",
    "冀教版", "鄂教版", "川教版", "鲁科版", "粤教版", "苏科版",
)


@dataclass(frozen=True)
class PathConfig:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    md_dir: Path = PROJECT_ROOT / "data" / "md"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    reports_dir: Path = PROJECT_ROOT / "outputs" / "reports"
    logs_dir: Path = PROJECT_ROOT / "outputs" / "logs"
    chunks_dir: Path = PROJECT_ROOT / "outputs" / "chunks"
    retrieval_dir: Path = PROJECT_ROOT / "outputs" / "retrieval"
    evaluation_dir: Path = PROJECT_ROOT / "outputs" / "evaluation"
    memories_dir: Path = PROJECT_ROOT / "data" / "memories"
    manifest_path: Path = PROJECT_ROOT / "outputs" / "manifest.json"
    chunks_jsonl: Path = PROJECT_ROOT / "outputs" / "chunks.jsonl"

    def ensure_dirs(self) -> None:
        for path in (
            self.raw_dir,
            self.md_dir,
            self.chroma_dir,
            self.reports_dir,
            self.logs_dir,
            self.chunks_dir,
            self.retrieval_dir,
            self.evaluation_dir,
            self.memories_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ChunkConfig:
    max_chars: int = _env_int("TUTOR_RAG_CHUNK_MAX_CHARS", 450)
    overlap_chars: int = _env_int("TUTOR_RAG_CHUNK_OVERLAP", 80)
    min_chars: int = _env_int("TUTOR_RAG_CHUNK_MIN", 100)
    separators: tuple[str, ...] = (
        "\n\n", "\n", "。", "；", "！", "？", "…", "：", ". ", "; ", "! ", "? ", ": ",
        "，", ", ", "、", " ",
    )


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = _env_str("TUTOR_RAG_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    max_length: int = _env_int("TUTOR_RAG_EMBEDDING_MAX_LENGTH", 512)
    batch_size: int = _env_int("TUTOR_RAG_EMBEDDING_BATCH", 32)
    device: str = _env_str("TUTOR_RAG_DEVICE", "auto")
    normalize: bool = True
    query_instruction: str = "为这个句子生成表示以用于检索相关文章："


@dataclass(frozen=True)
class ChromaConfig:
    collection_name: str = _env_str("TUTOR_RAG_COLLECTION", "textbook_chunks")
    distance: str = "cosine"
    upsert_batch_size: int = 128


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = _env_int("TUTOR_RAG_RETRIEVAL_TOP_K", 3)
    rrf_k: int = _env_int("TUTOR_RAG_RETRIEVAL_RRF_K", 60)
    enabled: bool = _env_bool("TUTOR_RAG_RETRIEVAL_ENABLED", True)
    trace: bool = _env_bool("TUTOR_RAG_RETRIEVAL_TRACE", True)


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = _env_bool("TUTOR_RAG_CACHE_ENABLED", True)
    redis_url: str = _env_str("TUTOR_RAG_CACHE_REDIS_URL", "redis://127.0.0.1:6379/0")
    namespace: str = _env_str("TUTOR_RAG_CACHE_NAMESPACE", "tutor")
    threshold: float = _env_float("TUTOR_RAG_CACHE_THRESHOLD", 0.95)
    ttl_seconds: int = _env_int("TUTOR_RAG_CACHE_TTL", 604800)
    vector_dim: int = _env_int("TUTOR_RAG_CACHE_VECTOR_DIM", 1024)
    protect_numeric: bool = _env_bool("TUTOR_RAG_CACHE_PROTECT_NUMERIC", True)


@dataclass(frozen=True)
class SubAgentConfig:
    enabled: bool = _env_bool("TUTOR_RAG_SUBAGENTS_ENABLED", True)
    top_k: int = _env_int("TUTOR_RAG_SUBAGENT_TOP_K", 5)


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool = _env_bool("TUTOR_RAG_MEMORY_ENABLED", True)
    index_limit: int = _env_int("TUTOR_RAG_MEMORY_INDEX_LIMIT", 50)
    description_limit: int = _env_int("TUTOR_RAG_MEMORY_DESCRIPTION_LIMIT", 30)
    read_max_chars: int = _env_int("TUTOR_RAG_MEMORY_READ_MAX_CHARS", 6000)


@dataclass(frozen=True)
class ChatConfig:
    model: str = _env_str("TUTOR_RAG_LLM_MODEL", "deepseek:deepseek-flash")
    api_key: str = ""
    fallback_models: tuple[str, ...] = tuple(
        item.strip()
        for item in _env_str("TUTOR_RAG_LLM_FALLBACKS", "").split(",")
        if item.strip()
    )
    temperature: float = _env_float("TUTOR_RAG_LLM_TEMPERATURE", 0.3)
    max_tokens: int = _env_int("TUTOR_RAG_LLM_MAX_TOKENS", 8192)
    timeout_s: int = _env_int("TUTOR_RAG_LLM_TIMEOUT", 60)
    max_retries: int = _env_int("TUTOR_RAG_LLM_MAX_RETRIES", 2)


@dataclass(frozen=True)
class Settings:
    paths: PathConfig = field(default_factory=PathConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    subagents: SubAgentConfig = field(default_factory=SubAgentConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)


_cached: Settings | None = None


def get_settings() -> Settings:
    global _cached
    if _cached is None:
        _cached = Settings()
    return _cached
