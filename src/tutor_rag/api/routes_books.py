"""教材上传与列表：上传后立即入库并刷新检索（当次会话即可命中）。"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from ..config import GRADES, SUBJECTS
from ..convert import SOURCE_EXTS, decode_text
from ..load import make_doc_id
from ..pipeline import delete_document, load_manifest, run_ingest, save_manifest
from ..trace import get_logger
from .schemas import TextbookContentPayload
from .scheduler import run_sweep
from .state import RuntimeState

EDITABLE_FORMATS = {"md", "markdown", "txt"}

router = APIRouter()

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_READ_CHUNK = 1024 * 1024


def _runtime(request: Request) -> RuntimeState:
    return request.app.state.runtime


@router.post("/api/upload/textbook")
def upload_textbook(
    request: Request,
    file: UploadFile = File(...),
    subject: str = Form(...),
    grade: str = Form(...),
    version: str = Form(...),
) -> dict:
    runtime = _runtime(request)
    if subject not in SUBJECTS:
        raise HTTPException(400, f"未知学科: {subject}")
    if grade not in GRADES:
        raise HTTPException(400, f"未知年级: {grade}")
    version = version.strip()
    if not version or "-" in version:
        raise HTTPException(400, "版本名不能为空、不能包含短横线")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SOURCE_EXTS:
        raise HTTPException(400, f"仅支持 {'/'.join(SOURCE_EXTS)} 格式")

    stem = f"{subject}-{version}-{grade}"
    raw_dir = runtime.settings.paths.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    for old in raw_dir.glob(f"{stem}.*"):
        old.unlink()
    target = raw_dir / f"{stem}{suffix}"

    size = 0
    with target.open("wb") as handle:
        while True:
            chunk = file.file.read(_READ_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                handle.close()
                target.unlink(missing_ok=True)
                raise HTTPException(413, "文件超过 50MB 限制")
            handle.write(chunk)

    started = time.perf_counter()
    with runtime.ingest_lock:
        result = run_ingest(runtime.settings, embedder=runtime.retrieval.embedder)
        runtime.retrieval.refresh()
    elapsed = time.perf_counter() - started

    manifest = load_manifest(runtime.settings.paths.manifest_path)
    doc_id = make_doc_id(stem)
    entry = manifest["documents"].get(doc_id, {})
    get_logger().info(
        "教材上传完成: %s 入库=%s 跳过=%s 失败=%s 块数=%s 耗时=%.1fs",
        target.name,
        result.ingested,
        result.skipped,
        result.failed,
        entry.get("chunk_count", 0),
        elapsed,
    )
    return {
        "ok": not result.failed,
        "doc_id": doc_id,
        "source_file": target.name,
        "chunks": int(entry.get("chunk_count", 0)),
        "collection_chunks": runtime.retrieval.store.count(),
        "ingested": result.ingested,
        "skipped": result.skipped,
        "failed": result.failed,
        "elapsed_s": round(elapsed, 1),
    }


def _manifest_entry(runtime: RuntimeState, doc_id: str) -> dict:
    manifest = load_manifest(runtime.settings.paths.manifest_path)
    entry = manifest.get("documents", {}).get(doc_id)
    if entry is None:
        raise HTTPException(404, "教材不存在")
    return entry


def _editable_raw_path(runtime: RuntimeState, entry: dict) -> Path:
    source_format = str(entry.get("source_format", ""))
    if source_format not in EDITABLE_FORMATS:
        raise HTTPException(400, "仅 md/txt 支持在线编辑，PDF 请重新上传替换")
    path = runtime.settings.paths.raw_dir / str(entry.get("source_file", ""))
    if not path.is_file():
        raise HTTPException(404, "原文件不存在")
    return path


@router.get("/api/textbooks/{doc_id}/content")
def textbook_content(doc_id: str, request: Request) -> dict:
    runtime = _runtime(request)
    entry = _manifest_entry(runtime, doc_id)
    path = _editable_raw_path(runtime, entry)
    text, encoding = decode_text(path.read_bytes())
    return {
        "doc_id": doc_id,
        "source_file": entry.get("source_file", ""),
        "source_format": entry.get("source_format", ""),
        "encoding": encoding or "utf-8",
        "content": text,
    }


@router.put("/api/textbooks/{doc_id}/content")
def update_textbook_content(
    doc_id: str, payload: TextbookContentPayload, request: Request
) -> dict:
    runtime = _runtime(request)
    entry = _manifest_entry(runtime, doc_id)
    path = _editable_raw_path(runtime, entry)
    path.write_text(payload.content, encoding="utf-8")

    md_path = runtime.settings.paths.md_dir / f"{Path(str(entry.get('source_file'))).stem}.md"
    if md_path.exists():
        md_path.unlink()

    manifest = load_manifest(runtime.settings.paths.manifest_path)
    manifest["documents"][doc_id]["pending_rebuild"] = True
    save_manifest(runtime.settings.paths.manifest_path, manifest)
    get_logger().info("教材内容已保存: %s（等待重建）", entry.get("source_file"))
    return {
        "ok": True,
        "pending_rebuild": True,
        "message": "内容已保存，点击「重建向量」后生效",
    }


@router.post("/api/textbooks/{doc_id}/rebuild")
def rebuild_textbook(doc_id: str, request: Request) -> dict:
    runtime = _runtime(request)
    _manifest_entry(runtime, doc_id)
    removed_before = runtime.retrieval.store.count_document(doc_id)
    started = time.perf_counter()
    try:
        with runtime.ingest_lock:
            result = run_ingest(
                runtime.settings,
                embedder=runtime.retrieval.embedder,
                force_doc_ids={doc_id},
            )
            runtime.retrieval.refresh()
    except Exception as exc:
        get_logger().warning("教材重建失败，原索引保持不变: %r", exc)
        raise HTTPException(500, f"重建失败: {exc}") from exc
    if result.failed:
        raise HTTPException(500, f"重建失败: {result.failed}")
    if doc_id not in result.ingested:
        raise HTTPException(400, "重建未产生知识块，请检查教材内容")
    entry = load_manifest(runtime.settings.paths.manifest_path)["documents"].get(doc_id, {})
    get_logger().info("教材重建完成: %s", entry.get("source_file", doc_id))
    return {
        "ok": True,
        "doc_id": doc_id,
        "removed_chunks": removed_before,
        "chunk_count": int(entry.get("chunk_count", 0)),
        "collection_chunks": runtime.retrieval.store.count(),
        "elapsed_s": round(time.perf_counter() - started, 1),
    }


@router.post("/api/textbooks/recheck")
def recheck_textbooks(request: Request) -> dict:
    runtime = _runtime(request)
    return run_sweep(runtime, trigger="manual")


@router.get("/api/textbooks/{doc_id}/chunks")
def textbook_chunks(doc_id: str, request: Request) -> dict:
    runtime = _runtime(request)
    entry = _manifest_entry(runtime, doc_id)

    records = []
    for record in runtime.retrieval.store.export_all():
        metadata = record.get("metadata") or {}
        if metadata.get("doc_id") != doc_id:
            continue
        heading = str(metadata.get("heading_path", ""))
        text = record.get("text") or ""
        if heading and text.startswith(f"{heading}\n"):
            text = text[len(heading) + 1 :]
        records.append(
            {
                "chunk_id": record["chunk_id"],
                "chunk_index": int(metadata.get("chunk_index", 0)),
                "heading_path": heading,
                "char_count": len(text),
                "text": text,
            }
        )
    records.sort(key=lambda item: item["chunk_index"])
    return {
        "doc_id": doc_id,
        "source_file": entry.get("source_file", ""),
        "source_format": entry.get("source_format", ""),
        "subject": entry.get("subject", ""),
        "grade": entry.get("grade", ""),
        "textbook_version": entry.get("textbook_version", ""),
        "pending_rebuild": bool(entry.get("pending_rebuild", False)),
        "chunks": records,
    }


@router.get("/api/textbooks/{doc_id}/file")
def textbook_file(doc_id: str, request: Request) -> FileResponse:
    runtime = _runtime(request)
    entry = _manifest_entry(runtime, doc_id)
    path = runtime.settings.paths.raw_dir / str(entry.get("source_file", ""))
    if not path.is_file():
        raise HTTPException(404, "原文件不存在")
    return FileResponse(path, filename=path.name)


@router.delete("/api/textbooks/{doc_id}")
def delete_textbook(doc_id: str, request: Request) -> dict:
    runtime = _runtime(request)
    _manifest_entry(runtime, doc_id)
    with runtime.ingest_lock:
        result = delete_document(runtime.settings, doc_id)
        runtime.retrieval.refresh()
    return {
        "ok": True,
        **result,
        "collection_chunks": runtime.retrieval.store.count(),
    }


@router.get("/api/textbooks")
def list_textbooks(request: Request) -> dict:
    runtime = _runtime(request)
    manifest = load_manifest(runtime.settings.paths.manifest_path)
    counts = runtime.retrieval.store.count_by_doc()
    items = []
    for doc_id, entry in sorted(
        manifest.get("documents", {}).items(),
        key=lambda item: item[1].get("source_file", ""),
    ):
        items.append(
            {
                "doc_id": doc_id,
                "source_file": entry.get("source_file", ""),
                "subject": entry.get("subject", ""),
                "grade": entry.get("grade", ""),
                "textbook_version": entry.get("textbook_version", ""),
                "chunk_count": int(entry.get("chunk_count", 0)),
                "actual_chunks": counts.get(doc_id, 0),
                "ingested_at": entry.get("ingested_at", ""),
                "pending_rebuild": bool(entry.get("pending_rebuild", False)),
            }
        )
    return {"items": items, "collection_chunks": runtime.retrieval.store.count()}
