"""命令行入口：

    tutor-rag convert [--subject 数学] [--no-cache]
    tutor-rag ingest  [--subject 数学] [--force] [--prune] [--no-cache]
    tutor-rag stats
    tutor-rag query "有理数的定义" [--top-k 5] [--subject 数学]
    tutor-rag reset --yes
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from collections import defaultdict

from .config import get_settings
from .index import ChromaStore, Embedder
from .pipeline import load_manifest, run_convert, run_ingest
from .trace import force_utf8_console, setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tutor-rag", description="初中家教 RAG 入库工具")
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", help="原始教材 -> 统一 Markdown")
    convert.add_argument("--subject", default=None, help="只处理某学科")
    convert.add_argument("--no-cache", action="store_true", help="忽略转换缓存")

    ingest = sub.add_parser("ingest", help="转换 + 切块 + 向量化 + Chroma 入库")
    ingest.add_argument("--subject", default=None, help="只处理某学科")
    ingest.add_argument("--force", action="store_true", help="忽略哈希，全量重嵌入")
    ingest.add_argument("--prune", action="store_true", help="清理 raw 中已删除文档的块")
    ingest.add_argument("--no-cache", action="store_true", help="忽略转换缓存")

    sub.add_parser("stats", help="查看 manifest 与 Chroma 统计")

    query = sub.add_parser("query", help="检索冒烟测试（正式检索在二期）")
    query.add_argument("text", help="查询文本")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--subject", default=None)

    chat = sub.add_parser("chat", help="终端对话（create_agent + 内存 checkpointer）")
    chat.add_argument("--once", default=None, help="只问一次后退出")
    chat.add_argument("--model", default=None, help="覆盖模型（provider:model）")
    chat.add_argument("--thread", default="cli", help="会话线程 id")
    chat.add_argument("--no-stream", action="store_true", help="关闭流式输出")
    chat.add_argument("--no-retrieval", action="store_true", help="关闭教材检索")

    retrieve = sub.add_parser("retrieve", help="检索调试（vector / bm25 / rrf）")
    retrieve.add_argument("text", help="查询文本")
    retrieve.add_argument("--top-k", type=int, default=None, help="召回数量（默认取配置）")
    retrieve.add_argument(
        "--mode", choices=("vector", "bm25", "rrf"), default="rrf", help="查看哪一路结果"
    )
    retrieve.add_argument("--subject", default=None, help="限定学科")

    evaluate = sub.add_parser("eval", help="离线检索评测（三种模式消融）")
    evaluate.add_argument("--k", type=int, default=3, help="评测截断 k")
    evaluate.add_argument("--dataset", default=None, help="标注集路径")

    web = sub.add_parser("web", help="启动 Web 服务（FastAPI + 前端静态页）")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=6076)
    web.add_argument("--reload", action="store_true", help="开发模式热重载")

    reset = sub.add_parser("reset", help="清空 collection 与 manifest")
    reset.add_argument("--yes", action="store_true", help="确认执行")
    return parser


def cmd_convert(args: argparse.Namespace) -> int:
    report = run_convert(subject=args.subject, use_cache=not args.no_cache)
    return 0 if report.status == "success" else 1


def cmd_ingest(args: argparse.Namespace) -> int:
    result = run_ingest(
        subject=args.subject,
        force=args.force,
        prune=args.prune,
        use_cache=not args.no_cache,
    )
    return 0 if not result.failed else 1


def _display_width(text: str) -> int:
    width = 0
    for char in str(text):
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def _pad(text: str, width: int, align: str = "left") -> str:
    text = str(text)
    padding = " " * max(0, width - _display_width(text))
    return padding + text if align == "right" else text + padding


def cmd_stats(args: argparse.Namespace) -> int:
    settings = get_settings()
    setup_logging(settings.paths.logs_dir / "stats.log")
    manifest = load_manifest(settings.paths.manifest_path)
    documents = manifest.get("documents", {})
    store = ChromaStore(settings.paths.chroma_dir, settings.chroma)
    counts = store.count_by_doc()

    print(f"collection={settings.chroma.collection_name} 块总数={store.count()}")
    print(f"文档数={len(documents)} 嵌入模型={manifest.get('embedding_model', '?')}")

    columns = [("来源文件", 32), ("学科", 6), ("年级", 6), ("版本", 10), ("块数", 6), ("入库时间", 21)]
    gap = "  "
    line_width = sum(width for _, width in columns) + len(gap) * (len(columns) - 1)
    print("-" * line_width)
    print(gap.join(_pad(name, width, "right" if name == "块数" else "left") for name, width in columns))
    subject_chunks: dict[str, int] = defaultdict(int)
    mismatches = 0
    for doc_id, entry in sorted(documents.items(), key=lambda item: item[1].get("source_file", "")):
        actual = counts.get(doc_id, 0)
        expected = int(entry.get("chunk_count", 0))
        if actual != expected:
            mismatches += 1
        subject_chunks[entry.get("subject", "?")] += expected
        values = [
            entry.get("source_file", "?"),
            entry.get("subject", "?"),
            entry.get("grade", "?"),
            entry.get("textbook_version", "?"),
            expected,
            entry.get("ingested_at", ""),
        ]
        print(
            gap.join(
                _pad(value, width, "right" if name == "块数" else "left")
                for value, (name, width) in zip(values, columns)
            )
        )
    print("-" * line_width)
    for subject, chunks in sorted(subject_chunks.items()):
        print(f"  {subject}: {chunks} 块")
    if mismatches:
        print(f"警告: {mismatches} 篇文档的 manifest 块数与 collection 不一致")
    return 1 if mismatches else 0


def cmd_query(args: argparse.Namespace) -> int:
    settings = get_settings()
    setup_logging(settings.paths.logs_dir / "query.log")
    store = ChromaStore(settings.paths.chroma_dir, settings.chroma)
    embedder = Embedder(settings.embedding)
    vector = embedder.embed_query(args.text)
    hits = store.query(vector, top_k=args.top_k, subject=args.subject)
    for rank, hit in enumerate(hits, 1):
        metadata = hit["metadata"] or {}
        preview = (hit["document"] or "").replace("\n", " ")[:80]
        distance = hit["distance"]
        distance_text = f"{distance:.4f}" if isinstance(distance, float) else "n/a"
        print(
            f"#{rank} distance={distance_text} subject={metadata.get('subject', '?')} "
            f"path={metadata.get('heading_path', '')}\n    {preview}"
        )
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from .agent.chat import run_chat

    return run_chat(
        model_name=args.model,
        thread_id=args.thread,
        once=args.once,
        stream=not args.no_stream,
        retrieval=not args.no_retrieval,
    )


def cmd_retrieve(args: argparse.Namespace) -> int:
    from .retrieval.service import RetrievalService

    settings = get_settings()
    setup_logging(settings.paths.logs_dir / "retrieve.log")
    service = RetrievalService(settings)
    result = service.search(args.text, top_k=args.top_k, subject=args.subject)
    hits = result.by_mode(args.mode)

    print(
        f"query={result.query!r} mode={args.mode} top_k={result.top_k} "
        f"corpus={service.corpus_size} 块"
    )
    if result.degraded:
        print(f"降级: {result.degraded}")
    if not hits:
        print("无命中")
        return 0
    for rank, hit in enumerate(hits, 1):
        scores: list[str] = []
        if hit.vector_rank is not None and hit.vector_score is not None:
            scores.append(f"vector#{hit.vector_rank}({hit.vector_score:.3f})")
        if hit.bm25_rank is not None and hit.bm25_score is not None:
            scores.append(f"bm25#{hit.bm25_rank}({hit.bm25_score:.2f})")
        print(f"#{rank} rrf={hit.rrf_score:.5f} {' '.join(scores)}")
        print(f"    [{hit.subject}] {hit.heading_path} · {hit.source_file}")
        print(f"    {hit.text.replace(chr(10), ' ')[:110]}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .evaluation import report_text, run_evaluation, save_report

    settings = get_settings()
    setup_logging(settings.paths.logs_dir / "eval.log")
    dataset = Path(args.dataset) if args.dataset else None
    report = run_evaluation(settings=settings, dataset_path=dataset, k=args.k)
    print(report_text(report))
    path = save_report(
        report,
        settings.paths.evaluation_dir / f"retrieval-report-{report['run_id']}.json",
    )
    print(f"报告已保存: {path}")
    return 0 if not report["unmatched_labels"] else 1


def cmd_web(args: argparse.Namespace) -> int:
    import uvicorn

    print(f"Web 服务启动: http://{args.host}:{args.port}（Ctrl+C 退出）")
    uvicorn.run(
        "tutor_rag.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    if not args.yes:
        print("拒绝执行：reset 会清空向量库与 manifest，请加 --yes 确认")
        return 2
    settings = get_settings()
    setup_logging(settings.paths.logs_dir / "reset.log")
    store = ChromaStore(settings.paths.chroma_dir, settings.chroma)
    store.reset()
    manifest_path = settings.paths.manifest_path
    if manifest_path.exists():
        manifest_path.unlink()
    if settings.paths.chunks_jsonl.exists():
        settings.paths.chunks_jsonl.unlink()
    for per_doc in settings.paths.chunks_dir.glob("*.jsonl"):
        per_doc.unlink()
    print("已清空 collection、manifest 与 chunks 明细")
    return 0


def main(argv: list[str] | None = None) -> int:
    force_utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "convert": cmd_convert,
        "ingest": cmd_ingest,
        "stats": cmd_stats,
        "query": cmd_query,
        "chat": cmd_chat,
        "retrieve": cmd_retrieve,
        "eval": cmd_eval,
        "web": cmd_web,
        "reset": cmd_reset,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
