"""入库流水线：扫描 -> 转换 -> 加载 -> 变更比对 -> 切块 -> 向量化入库 -> 落盘。

增量策略：
- 以 content_hash（规范化文本的 sha256）判断文档是否变化；
- 嵌入模型变化时自动全量重嵌入；
- 变化文档先按 doc_id 删除旧块再写入，保证无脏数据；
- manifest.json 记录每篇文档的状态，chunks/<doc_id>.jsonl 保留块级明细供二期 BM25 使用。
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .chunk import ChunkResult, ChunkStats, chunk_document
from .config import Settings, get_settings
from .convert import ConversionResult, convert_file, discover_sources
from .index import ChromaStore, Embedder, format_embedding_summary
from .load import DocumentParseError, TextbookDocument, load_document, parse_source_name
from .trace import Progress, RunReport, get_logger, setup_logging

MANIFEST_VERSION = 1


@dataclass
class IngestResult:
    report: RunReport
    ingested: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)
    chunk_results: dict[str, ChunkResult] = field(default_factory=dict)


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": MANIFEST_VERSION, "documents": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("documents", {})
    return data


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def chunks_file(paths, doc_id: str) -> Path:
    return paths.chunks_dir / f"{doc_id}.jsonl"


def write_doc_chunks(path: Path, chunk_result: ChunkResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in chunk_result.chunks:
            handle.write(
                json.dumps(
                    {
                        "chunk_id": record.chunk_id,
                        "text": record.text,
                        "embed_text": record.embed_text,
                        "metadata": record.metadata,
                        "char_count": record.char_count,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def rebuild_chunks_jsonl(paths, manifest: dict[str, Any]) -> int:
    total = 0
    with paths.chunks_jsonl.open("w", encoding="utf-8") as out:
        for doc_id in sorted(manifest.get("documents", {})):
            per_doc = chunks_file(paths, doc_id)
            if not per_doc.exists():
                continue
            for line in per_doc.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.write(line + "\n")
                    total += 1
    return total


def delete_document(settings: Settings, doc_id: str) -> dict[str, Any]:
    paths = settings.paths
    manifest = load_manifest(paths.manifest_path)
    entry = manifest.get("documents", {}).pop(doc_id, None)

    store = ChromaStore(paths.chroma_dir, settings.chroma)
    removed_chunks = store.count_document(doc_id)
    store.delete_document(doc_id)

    per_doc = chunks_file(paths, doc_id)
    if per_doc.exists():
        per_doc.unlink()

    source_file = str(entry.get("source_file", "")) if entry else ""
    if source_file:
        raw_path = paths.raw_dir / source_file
        if raw_path.exists():
            raw_path.unlink()
        md_path = paths.md_dir / f"{Path(source_file).stem}.md"
        if md_path.exists():
            md_path.unlink()

    save_manifest(paths.manifest_path, manifest)
    rebuild_chunks_jsonl(paths, manifest)
    get_logger().info(
        "已删除教材: %s（%d 个块）", source_file or doc_id, removed_chunks
    )
    return {
        "doc_id": doc_id,
        "source_file": source_file,
        "removed_chunks": removed_chunks,
    }


def _scan_sources(paths, subject: str | None, report: RunReport) -> tuple[list[Path], list[Path]]:
    all_sources = discover_sources(paths.raw_dir)
    selected: list[Path] = []
    for source in all_sources:
        try:
            parsed = parse_source_name(source.stem)
        except DocumentParseError as exc:
            report.event("filename_parse_error", file=source.name, error=str(exc))
            continue
        if subject and parsed["subject"] != subject:
            continue
        selected.append(source)

    groups: dict[str, list[Path]] = defaultdict(list)
    for source in selected:
        groups[source.stem].append(source)
    duplicates = {stem: files for stem, files in groups.items() if len(files) > 1}
    for stem, files in duplicates.items():
        report.event(
            "duplicate_source_stem",
            level="error",
            stem=stem,
            files=[file.name for file in files],
        )
    if duplicates:
        selected = [source for source in selected if source.stem not in duplicates]
    return all_sources, selected


def _convert_selected(
    settings: Settings,
    selected: list[Path],
    report: RunReport,
    failed: list[dict[str, Any]],
    use_cache: bool,
) -> dict[Path, ConversionResult]:
    conversions: dict[Path, ConversionResult] = {}
    progress = Progress(len(selected), "转换")
    for index, source in enumerate(selected, 1):
        try:
            conversion = convert_file(source, settings.paths.md_dir, use_cache=use_cache)
        except Exception as exc:
            report.event("convert_error", level="error", file=source.name, error=repr(exc))
            failed.append({"file": source.name, "stage": "convert", "error": repr(exc)})
            continue
        conversions[source] = conversion
        if conversion.fallback:
            report.event(
                "encoding_fallback",
                level="info",
                file=source.name,
                encoding=conversion.fallback,
            )
        for warning in conversion.warnings:
            report.event("convert_warning", file=source.name, warning=warning)
        progress.tick(index, file=source.name)
    return conversions


def _load_documents(
    conversions: dict[Path, ConversionResult],
    report: RunReport,
    failed: list[dict[str, Any]],
) -> list[TextbookDocument]:
    documents: list[TextbookDocument] = []
    for source, conversion in conversions.items():
        try:
            document = load_document(
                conversion.md_path,
                source_file=source.name,
                source_format=conversion.source_format,
            )
        except Exception as exc:
            report.event("load_error", level="error", file=source.name, error=repr(exc))
            failed.append({"file": source.name, "stage": "load", "error": repr(exc)})
            continue
        documents.append(document)
    return documents


def run_convert(
    settings: Settings | None = None,
    subject: str | None = None,
    use_cache: bool = True,
) -> RunReport:
    settings = settings or get_settings()
    settings.paths.ensure_dirs()
    report = RunReport(command="convert")
    setup_logging(settings.paths.logs_dir / f"convert-{report.run_id}.log")

    with report.stage("scan") as stage:
        all_sources, selected = _scan_sources(settings.paths, subject, report)
        stage.set("raw_files", len(all_sources))
        stage.set("selected", len(selected))
        stage.ratio("select_success", len(selected), len(all_sources))

    failed: list[dict[str, Any]] = []
    with report.stage("convert") as stage:
        conversions = _convert_selected(settings, selected, report, failed, use_cache)
        cache_hits = sum(1 for item in conversions.values() if item.cached)
        fallbacks = sum(1 for item in conversions.values() if item.fallback)
        chars = sum(item.content_chars for item in conversions.values())
        stage.set("converted", len(conversions))
        stage.set("cache_hits", cache_hits)
        stage.set("chars_total", chars)
        stage.ratio("convert_success", len(conversions), len(conversions) + len(failed))
        stage.ratio("cache_hit_rate", cache_hits, len(conversions))

    report.finish("success" if not failed else "partial")
    report.save(settings.paths.reports_dir / f"run-{report.run_id}.json")
    report.log_summary()
    return report


def run_ingest(
    settings: Settings | None = None,
    subject: str | None = None,
    force: bool = False,
    prune: bool = False,
    use_cache: bool = True,
    embedder: Embedder | None = None,
    force_doc_ids: Collection[str] | None = None,
) -> IngestResult:
    settings = settings or get_settings()
    paths = settings.paths
    paths.ensure_dirs()
    report = RunReport(command="ingest")
    setup_logging(paths.logs_dir / f"ingest-{report.run_id}.log")
    result = IngestResult(report=report)
    store = ChromaStore(paths.chroma_dir, settings.chroma)
    manifest = load_manifest(paths.manifest_path)
    known: dict[str, Any] = manifest["documents"]
    embed_model_name = settings.embedding.model_name

    with report.stage("scan") as stage:
        all_sources, selected = _scan_sources(paths, subject, report)
        stage.set("raw_files", len(all_sources))
        stage.set("selected", len(selected))
        stage.ratio("select_success", len(selected), len(all_sources))

    with report.stage("convert") as stage:
        conversions = _convert_selected(settings, selected, report, result.failed, use_cache)
        cache_hits = sum(1 for item in conversions.values() if item.cached)
        stage.set("converted", len(conversions))
        stage.set("cache_hits", cache_hits)
        stage.ratio("convert_success", len(conversions), len(conversions) + len(result.failed))
        stage.ratio("cache_hit_rate", cache_hits, len(conversions))

    with report.stage("load") as stage:
        documents = _load_documents(conversions, report, result.failed)
        stage.set("documents", len(documents))
        stage.ratio("load_success", len(documents), len(conversions))

    to_ingest: list[TextbookDocument] = []
    with report.stage("compare") as stage:
        content_changed = 0
        model_changed = 0
        forced = set(force_doc_ids or ())
        force_rebuilt = 0
        for document in documents:
            previous = known.get(document.doc_id)
            if force or document.doc_id in forced or previous is None:
                if document.doc_id in forced and previous is not None:
                    force_rebuilt += 1
                to_ingest.append(document)
                continue
            if previous.get("content_hash") != document.metadata["content_hash"]:
                content_changed += 1
                to_ingest.append(document)
                continue
            if previous.get("embedding_model") != embed_model_name:
                model_changed += 1
                to_ingest.append(document)
                continue
            result.skipped.append(document.doc_id)
        stage.set("new_or_changed", len(to_ingest))
        stage.set("unchanged", len(result.skipped))
        stage.set("content_changed", content_changed)
        stage.set("model_changed", model_changed)
        stage.set("force_rebuilt", force_rebuilt)
        stage.ratio("unchanged_skip_rate", len(result.skipped), len(documents))
        if model_changed:
            stage.note(f"嵌入模型变化，重嵌入 {model_changed} 篇")

    with report.stage("chunk") as stage:
        aggregate = ChunkStats()
        total_chars = 0
        for document in to_ingest:
            chunk_result = chunk_document(document, settings.chunk)
            result.chunk_results[document.doc_id] = chunk_result
            aggregate.merge(chunk_result.stats)
            total_chars += sum(record.char_count for record in chunk_result.chunks)
            if not chunk_result.chunks:
                report.event(
                    "empty_chunks",
                    doc_id=document.doc_id,
                    file=document.metadata["source_file"],
                )
        stage.set("documents", len(to_ingest))
        stage.set("chunks", aggregate.total)
        stage.set(
            "avg_chars",
            round(total_chars / aggregate.total, 1) if aggregate.total else 0.0,
        )
        stage.set("multi_piece_sections", aggregate.multi_piece_sections)
        stage.set("atomic_blocks", aggregate.atomic_blocks)
        stage.ratio("size_compliance", aggregate.size_ok, aggregate.total)
        stage.ratio("min_compliance", aggregate.min_ok, aggregate.total)
        stage.ratio(
            "overlap_coverage", aggregate.overlap_pieces, aggregate.continuation_pieces
        )
        if aggregate.hard_cut_atoms:
            stage.note(f"存在 {aggregate.hard_cut_atoms} 处无分隔符硬切")
        if aggregate.total and aggregate.size_ok != aggregate.total:
            stage.note("存在超过 max_chars 的块，请检查！")

    with report.stage("embed_store") as stage:
        active = [
            document
            for document in to_ingest
            if result.chunk_results.get(document.doc_id)
            and result.chunk_results[document.doc_id].chunks
        ]
        total_chunks = sum(
            len(result.chunk_results[document.doc_id].chunks) for document in active
        )
        engine = embedder or Embedder(settings.embedding)
        progress = Progress(total_chunks, "向量化")
        done = 0
        verified = 0
        dimension = 0

        def make_callback(offset: int):
            return lambda count: progress.tick(offset + count)

        for document in active:
            records = result.chunk_results[document.doc_id].chunks
            vectors = engine.embed_documents(
                [record.embed_text for record in records],
                on_progress=make_callback(done),
            )
            done += len(records)
            if len(vectors) != len(records):
                report.event(
                    "embedding_mismatch",
                    level="error",
                    doc_id=document.doc_id,
                    expected=len(records),
                    got=len(vectors),
                )
                continue
            if vectors and not dimension:
                dimension = len(vectors[0])
            store.delete_document(document.doc_id)
            store.upsert_chunks(records, vectors)
            stored = store.count_document(document.doc_id)
            if stored == len(records):
                verified += 1
            else:
                report.event(
                    "upsert_mismatch",
                    level="error",
                    doc_id=document.doc_id,
                    expected=len(records),
                    stored=stored,
                )
            write_doc_chunks(chunks_file(paths, document.doc_id), result.chunk_results[document.doc_id])
            known[document.doc_id] = {
                "source_file": document.metadata["source_file"],
                "source_format": document.metadata["source_format"],
                "subject": document.metadata["subject"],
                "grade": document.metadata["grade"],
                "textbook_version": document.metadata["textbook_version"],
                "content_hash": document.metadata["content_hash"],
                "chunk_count": len(records),
                "embedding_model": embed_model_name,
                "ingested_at": datetime.now().isoformat(timespec="seconds"),
            }
            result.ingested.append(document.doc_id)

        stage.set("documents", len(active))
        stage.set("vectors", done)
        if dimension:
            stage.set(
                "embedding",
                format_embedding_summary(embed_model_name, getattr(engine, "device", None), dimension),
            )
        stage.ratio("upsert_verification", verified, len(active))

    with report.stage("persist") as stage:
        if prune:
            current_names = {source.name for source in all_sources}
            for doc_id in list(known):
                if known[doc_id].get("source_file") not in current_names:
                    store.delete_document(doc_id)
                    stale = chunks_file(paths, doc_id)
                    if stale.exists():
                        stale.unlink()
                    known.pop(doc_id)
                    result.pruned.append(doc_id)
        manifest["version"] = MANIFEST_VERSION
        manifest["documents"] = known
        manifest["embedding_model"] = embed_model_name
        manifest["collection"] = settings.chroma.collection_name
        save_manifest(paths.manifest_path, manifest)
        total_lines = rebuild_chunks_jsonl(paths, manifest)
        counts = store.count_by_doc()
        consistent = sum(
            1
            for doc_id, entry in known.items()
            if counts.get(doc_id, 0) == int(entry.get("chunk_count", -1))
        )
        stage.set("manifest_documents", len(known))
        stage.set("collection_chunks", store.count())
        stage.set("chunks_jsonl_lines", total_lines)
        stage.set("pruned", len(result.pruned))
        stage.ratio("manifest_consistency", consistent, len(known))

    status = "success" if not result.failed else "partial"
    report.finish(status)
    report.save(paths.reports_dir / f"run-{report.run_id}.json")
    report.log_summary()
    get_logger().info("运行报告: %s", paths.reports_dir / f"run-{report.run_id}.json")
    return result
