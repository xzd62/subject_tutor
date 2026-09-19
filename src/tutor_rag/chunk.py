"""标题层级切块。

层次化策略：
1. 解析 #~###### 标题树，按叶子小节划定切分范围（降级切分是局部临时的，
   遇到下一个标题自动回到标题切分，不存在“回不去”的问题）；
2. 小节正文先切出原子块（表格 / 围栏代码 / $$ 公式），原子块不拦腰截断；
3. 其余文本按分隔符逐级降级（\\n\\n -> \\n -> 句末 -> 逗号 -> 空格 -> 硬切），
   贪心装箱到 max_chars，相邻块从前块尾部取 overlap_chars 作为衔接；
4. 过小的小节块先与同父级相邻块合并，避免碎片。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .config import ChunkConfig
from .load import TextbookDocument, sha256_text

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

_OVERLAP_BOUNDARIES: tuple[str, ...] = ("\n\n", "\n", "。", "；", "！", "？", "，", "、", " ")


@dataclass
class Section:
    level: int
    title: str
    path: tuple[str, ...]
    body_lines: list[str] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)


@dataclass
class Block:
    kind: str
    text: str


@dataclass
class PackedPiece:
    text: str
    overlap_prefix: str = ""


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    embed_text: str
    metadata: dict[str, Any]
    char_count: int


@dataclass
class ChunkStats:
    total: int = 0
    size_ok: int = 0
    min_ok: int = 0
    sections_with_body: int = 0
    multi_piece_sections: int = 0
    continuation_pieces: int = 0
    overlap_pieces: int = 0
    atomic_blocks: int = 0
    hard_cut_atoms: int = 0

    def merge(self, other: "ChunkStats") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass
class ChunkResult:
    chunks: list[ChunkRecord]
    stats: ChunkStats


def parse_sections(text: str) -> Section:
    root = Section(level=0, title="", path=())
    stack = [root]
    for line in text.split("\n"):
        match = HEADING_RE.match(line)
        if match is None:
            stack[-1].body_lines.append(line)
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        while len(stack) > 1 and stack[-1].level >= level:
            stack.pop()
        parent = stack[-1]
        section = Section(level=level, title=title, path=parent.path + (title,))
        parent.children.append(section)
        stack.append(section)
    return root


def extract_blocks(text: str) -> list[Block]:
    lines = text.split("\n")
    blocks: list[Block] = []
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        joined = "\n".join(buffer)
        if joined.strip():
            blocks.append(Block(kind="text", text=joined))
        buffer.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush()
            fence = stripped[:3]
            body = [line]
            index += 1
            while index < len(lines):
                body.append(lines[index])
                closing = lines[index].strip().startswith(fence)
                index += 1
                if closing:
                    break
            blocks.append(Block(kind="code", text="\n".join(body)))
            continue

        if stripped.startswith("$$"):
            flush()
            body = [line]
            index += 1
            if stripped.count("$$") < 2:
                while index < len(lines):
                    body.append(lines[index])
                    closing = "$$" in lines[index]
                    index += 1
                    if closing:
                        break
            blocks.append(Block(kind="math", text="\n".join(body)))
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            flush()
            body = [line]
            index += 1
            while index < len(lines) and lines[index].strip().startswith("|"):
                body.append(lines[index])
                index += 1
            blocks.append(Block(kind="table", text="\n".join(body)))
            continue

        buffer.append(line)
        index += 1

    flush()
    return blocks


def _split_keep_sep(text: str, separator: str) -> list[str]:
    parts = text.split(separator)
    with_sep = [part + separator for part in parts[:-1]] + [parts[-1]]
    return [part for part in with_sep if part]


def _split_recursive(
    text: str, separators: tuple[str, ...], max_chars: int
) -> tuple[list[str], int]:
    if len(text) <= max_chars:
        return [text], 0
    if not separators:
        pieces = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
        return pieces, 1
    separator = separators[0]
    if separator not in text:
        return _split_recursive(text, separators[1:], max_chars)

    atoms: list[str] = []
    hard_cuts = 0
    for part in _split_keep_sep(text, separator):
        if len(part) > max_chars:
            sub_atoms, sub_cuts = _split_recursive(part, separators[1:], max_chars)
            atoms.extend(sub_atoms)
            hard_cuts += sub_cuts
        else:
            atoms.append(part)
    return atoms, hard_cuts


def split_to_atoms(
    text: str, separators: tuple[str, ...], max_chars: int
) -> tuple[list[str], int]:
    return _split_recursive(text, separators, max_chars)


def tail_overlap(previous: str, overlap_chars: int, limit: int) -> str:
    take = min(overlap_chars, limit, len(previous))
    if take <= 0:
        return ""
    raw = previous[-take:]
    for boundary in _OVERLAP_BOUNDARIES:
        position = raw.find(boundary)
        if position < 0:
            continue
        candidate = raw[position + len(boundary) :]
        if len(candidate) >= max(20, take // 2):
            return candidate
    return raw


def pack_atoms(
    atoms: list[str], max_chars: int, overlap_chars: int
) -> list[PackedPiece]:
    pieces: list[PackedPiece] = []
    current = ""
    current_prefix = ""
    for atom in atoms:
        if not current:
            current = atom
            continue
        if len(current) + len(atom) <= max_chars:
            current += atom
            continue
        pieces.append(PackedPiece(text=current, overlap_prefix=current_prefix))
        limit = max(0, max_chars - len(atom))
        current_prefix = tail_overlap(current, overlap_chars, limit)
        current = current_prefix + atom if current_prefix else atom
    if current:
        pieces.append(PackedPiece(text=current, overlap_prefix=current_prefix))
    return pieces


def chunk_section_body(
    body: str, config: ChunkConfig
) -> tuple[list[PackedPiece], int, int]:
    atoms: list[str] = []
    atomic_blocks = 0
    hard_cuts = 0
    for block in extract_blocks(body):
        if block.kind == "text":
            text_atoms, cuts = split_to_atoms(block.text, config.separators, config.max_chars)
            atoms.extend(text_atoms)
            hard_cuts += cuts
        else:
            atomic_blocks += 1
            if len(block.text) <= config.max_chars:
                atoms.append(block.text)
            else:
                atoms.extend(
                    block.text[i : i + config.max_chars]
                    for i in range(0, len(block.text), config.max_chars)
                )
                hard_cuts += 1
    if not atoms:
        return [], atomic_blocks, hard_cuts
    return pack_atoms(atoms, config.max_chars, config.overlap_chars), atomic_blocks, hard_cuts


@dataclass
class _Draft:
    text: str
    path: tuple[str, ...]
    overlap_prefix: str = ""


def _relation(previous: _Draft, draft: _Draft) -> str | None:
    if previous.path[:-1] == draft.path[:-1]:
        return "sibling"
    if len(previous.path) < len(draft.path) and draft.path[: len(previous.path)] == previous.path:
        return "ancestor"
    return None


def merge_small_drafts(drafts: list[_Draft], config: ChunkConfig) -> list[_Draft]:
    merged: list[_Draft] = []
    for draft in drafts:
        if merged:
            previous = merged[-1]
            relation = _relation(previous, draft)
            fits = len(previous.text) + len(draft.text) + 2 <= config.max_chars
            if relation == "ancestor" and fits and len(previous.text) < config.min_chars:
                draft.text = previous.text.rstrip() + "\n\n" + draft.text.lstrip()
                merged[-1] = draft
                continue
            too_small = (
                len(previous.text) < config.min_chars or len(draft.text) < config.min_chars
            )
            if relation is not None and too_small and fits:
                previous.text = previous.text.rstrip() + "\n\n" + draft.text.lstrip()
                continue
        merged.append(draft)
    return merged


def chunk_document(doc: TextbookDocument, config: ChunkConfig) -> ChunkResult:
    root = parse_sections(doc.text)
    drafts: list[_Draft] = []
    stats = ChunkStats()

    def walk(section: Section) -> None:
        body = "\n".join(section.body_lines).strip()
        if body:
            stats.sections_with_body += 1
            pieces, atomic_blocks, hard_cuts = chunk_section_body(body, config)
            stats.atomic_blocks += atomic_blocks
            stats.hard_cut_atoms += hard_cuts
            if len(pieces) > 1:
                stats.multi_piece_sections += 1
            for index, piece in enumerate(pieces):
                if index > 0:
                    stats.continuation_pieces += 1
                    if piece.overlap_prefix:
                        stats.overlap_pieces += 1
                drafts.append(
                    _Draft(
                        text=piece.text,
                        path=section.path,
                        overlap_prefix=piece.overlap_prefix,
                    )
                )
        for child in section.children:
            walk(child)

    walk(root)
    merged = merge_small_drafts(drafts, config)

    chunks: list[ChunkRecord] = []
    for index, draft in enumerate(merged):
        text = draft.text.strip()
        heading_path = " > ".join(draft.path)
        embed_text = f"{heading_path}\n{text}" if heading_path else text
        chunk_id = f"{doc.doc_id}:{index:04d}"
        metadata: dict[str, Any] = {
            "doc_id": doc.doc_id,
            "chunk_id": chunk_id,
            "chunk_index": index,
            "subject": doc.metadata["subject"],
            "grade": doc.metadata["grade"],
            "textbook_version": doc.metadata["textbook_version"],
            "source_file": doc.metadata["source_file"],
            "heading_path": heading_path,
            "content_hash": doc.metadata["content_hash"],
            "chunk_hash": sha256_text(text),
        }
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                text=text,
                embed_text=embed_text,
                metadata=metadata,
                char_count=len(text),
            )
        )

    stats.total = len(chunks)
    stats.size_ok = sum(1 for chunk in chunks if chunk.char_count <= config.max_chars)
    stats.min_ok = sum(1 for chunk in chunks if chunk.char_count >= config.min_chars)
    return ChunkResult(chunks=chunks, stats=stats)
