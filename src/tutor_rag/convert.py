"""教材转换：md/txt/pdf -> 统一 UTF-8 Markdown。

- md：直通，仅做编码校验与换行归一化；
- txt：优先 MarkItDown，编码异常时回退本地解码（gb18030/big5）；
- pdf：MarkItDown（pdfminer），扫描版/图片版会产出空文本并给出告警。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .trace import get_logger

SOURCE_EXTS: tuple[str, ...] = (".md", ".markdown", ".txt", ".pdf")
TEXT_EXTS: tuple[str, ...] = (".md", ".markdown", ".txt")

_TEXT_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "gb18030", "big5")

_markitdown = None


def _get_markitdown():
    global _markitdown
    if _markitdown is None:
        from markitdown import MarkItDown

        _markitdown = MarkItDown()
    return _markitdown


def normalize_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    stripped = text.strip()
    return stripped + "\n" if stripped else ""


def decode_text(data: bytes) -> tuple[str, str | None]:
    for encoding in _TEXT_ENCODINGS:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        fallback = None if encoding.startswith("utf-8") else encoding
        return text, fallback
    raise ValueError("无法识别的文本编码（已尝试 utf-8/gb18030/big5）")


def _convert_with_markitdown(raw_path: Path) -> str:
    result = _get_markitdown().convert(str(raw_path))
    return result.text_content or ""


@dataclass
class ConversionResult:
    raw_path: Path
    md_path: Path
    source_format: str
    content_chars: int
    duration_s: float
    converter: str
    cached: bool = False
    fallback: str | None = None
    warnings: list[str] = field(default_factory=list)


def discover_sources(raw_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SOURCE_EXTS
    )


def md_output_path(raw_path: Path, md_dir: Path) -> Path:
    return md_dir / f"{raw_path.stem}.md"


def convert_file(raw_path: Path, md_dir: Path, use_cache: bool = True) -> ConversionResult:
    start = time.perf_counter()
    md_dir.mkdir(parents=True, exist_ok=True)
    out_path = md_output_path(raw_path, md_dir)
    suffix = raw_path.suffix.lower()
    warnings: list[str] = []
    fallback: str | None = None
    converter = ""

    if use_cache and out_path.exists() and out_path.stat().st_mtime >= raw_path.stat().st_mtime:
        text = out_path.read_text(encoding="utf-8")
        return ConversionResult(
            raw_path=raw_path,
            md_path=out_path,
            source_format=suffix.lstrip("."),
            content_chars=len(text),
            duration_s=time.perf_counter() - start,
            converter="cache",
            cached=True,
        )

    if suffix in (".md", ".markdown"):
        text, fallback = decode_text(raw_path.read_bytes())
        converter = "passthrough"
    elif suffix == ".txt":
        data = raw_path.read_bytes()
        decoded, fallback = decode_text(data)
        if fallback is None:
            try:
                text = _convert_with_markitdown(raw_path)
                converter = "markitdown"
            except Exception as exc:
                get_logger().warning("MarkItDown 处理 txt 失败，回退本地解码: %s (%r)", raw_path.name, exc)
                text, converter = decoded, "decode-fallback"
        else:
            text, converter = decoded, "decode-fallback"
    elif suffix == ".pdf":
        text = _convert_with_markitdown(raw_path)
        converter = "markitdown"
        if not text.strip():
            warnings.append("PDF 未提取到文本，可能是扫描版/图片版，本期不处理图片")
    else:
        raise ValueError(f"不支持的源格式: {raw_path.suffix}")

    normalized = normalize_markdown(text)
    out_path.write_text(normalized, encoding="utf-8")
    return ConversionResult(
        raw_path=raw_path,
        md_path=out_path,
        source_format=suffix.lstrip("."),
        content_chars=len(normalized),
        duration_s=time.perf_counter() - start,
        converter=converter,
        fallback=fallback,
        warnings=warnings,
    )


def convert_all(raw_dir: Path, md_dir: Path, use_cache: bool = True) -> list[ConversionResult]:
    results: list[ConversionResult] = []
    for raw_path in discover_sources(raw_dir):
        results.append(convert_file(raw_path, md_dir, use_cache=use_cache))
    return results
