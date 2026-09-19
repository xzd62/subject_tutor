"""把转换后的 md 读成带元数据的 TextbookDocument。

命名约定：{学科}-{版本}-{年级}[-{补充说明}].{md|txt|pdf}
例如：数学-人教版-七上.md、英语-外研版-八下-词汇表.pdf
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import GRADE_ALIASES, GRADES, SUBJECTS

_STEM_RE = re.compile(
    r"^(?P<subject>[^-]+)-(?P<version>[^-]+)-(?P<grade>[^-]+)(?:-(?P<variant>.+))?$"
)

_DOC_NAMESPACE = uuid.UUID("a3f1c2d4-5b6e-4f70-8a91-2c3d4e5f6071")


class DocumentParseError(ValueError):
    pass


@dataclass
class TextbookDocument:
    doc_id: str
    text: str
    metadata: dict[str, Any]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    stripped = cleaned.strip()
    return stripped + "\n" if stripped else ""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_source_name(stem: str) -> dict[str, str]:
    match = _STEM_RE.match(stem)
    if match is None:
        raise DocumentParseError(
            f"文件名不符合 `学科-版本-年级[-补充说明]` 约定: {stem!r}"
        )
    subject = match.group("subject").strip()
    version = match.group("version").strip()
    grade_raw = match.group("grade").strip()
    grade = GRADE_ALIASES.get(grade_raw, grade_raw)
    if subject not in SUBJECTS:
        raise DocumentParseError(
            f"未知学科 {subject!r}，可选：{'/'.join(SUBJECTS)}"
        )
    if grade not in GRADES:
        raise DocumentParseError(
            f"未知年级 {grade_raw!r}，可选：{'/'.join(GRADES)}（或写 七年级上册 之类）"
        )
    if not version:
        raise DocumentParseError(f"缺少教材版本: {stem!r}")
    return {
        "subject": subject,
        "version": version,
        "grade": grade,
        "variant": (match.group("variant") or "").strip(),
    }


def make_doc_id(stem: str) -> str:
    canonical = re.sub(r"\s+", "", stem).lower()
    return uuid.uuid5(_DOC_NAMESPACE, f"tutor-rag:{canonical}").hex


def load_document(
    md_path: Path,
    source_file: str,
    source_format: str,
) -> TextbookDocument:
    text = normalize_text(md_path.read_text(encoding="utf-8"))
    if not text:
        raise DocumentParseError(f"文档内容为空: {md_path}")

    parsed = parse_source_name(Path(source_file).stem)
    doc_id = make_doc_id(Path(source_file).stem)
    metadata: dict[str, Any] = {
        "doc_id": doc_id,
        "subject": parsed["subject"],
        "grade": parsed["grade"],
        "textbook_version": parsed["version"],
        "source_file": source_file,
        "source_format": source_format,
        "content_hash": sha256_text(text),
    }
    return TextbookDocument(doc_id=doc_id, text=text, metadata=metadata)
