"""离线检索评测：vector / bm25 / rrf 三种模式消融，输出 hit@1、hit@k、Recall@k、MRR 与延迟。

标注格式（evaluation/retrieval_eval.jsonl，每行一条）：
{"query": "...", "expected_heading": "章节名片段", "expected_source": "文件名.md", "subject": "数学"}
命中判定：检索结果的 heading_path 含 expected_heading（且 source 匹配）即视为相关块。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, Settings, get_settings
from .retrieval.service import RetrievalService
from .trace import Progress, get_logger

DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "retrieval_eval.jsonl"
MODES = ("vector", "bm25", "rrf")


@dataclass
class EvalQuery:
    query: str
    expected_heading: str
    expected_source: str | None = None
    subject: str | None = None
    note: str = ""


def load_eval_set(path: Path | None = None) -> list[EvalQuery]:
    dataset = path or DEFAULT_DATASET
    queries: list[EvalQuery] = []
    for raw_line in dataset.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        payload = json.loads(line)
        queries.append(
            EvalQuery(
                query=payload["query"],
                expected_heading=payload["expected_heading"],
                expected_source=payload.get("expected_source"),
                subject=payload.get("subject"),
                note=payload.get("note", ""),
            )
        )
    return queries


def relevant_chunk_ids(corpus: list[dict[str, Any]], item: EvalQuery) -> set[str]:
    ids: set[str] = set()
    for chunk in corpus:
        metadata = chunk.get("metadata") or {}
        if item.expected_heading not in str(metadata.get("heading_path", "")):
            continue
        if item.expected_source and metadata.get("source_file") != item.expected_source:
            continue
        ids.add(chunk["chunk_id"])
    return ids


def _latency_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {
        "avg": round(sum(ordered) / len(ordered), 1),
        "p50": round(ordered[len(ordered) // 2], 1),
        "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 1),
    }


def run_evaluation(
    settings: Settings | None = None,
    dataset_path: Path | None = None,
    k: int = 3,
    service: RetrievalService | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    service = service or RetrievalService(settings)
    queries = load_eval_set(dataset_path)
    corpus = service.store.export_all()

    buckets = {
        mode: {"hit1": 0, "hitk": 0, "recall": 0.0, "mrr": 0.0, "count": 0}
        for mode in MODES
    }
    per_query: list[dict[str, Any]] = []
    latencies: list[float] = []
    unmatched_labels: list[str] = []
    progress = Progress(len(queries), "评测")

    for index, item in enumerate(queries, 1):
        relevant = relevant_chunk_ids(corpus, item)
        if not relevant:
            unmatched_labels.append(item.query)
            progress.tick(index, query=item.query[:20])
            continue
        result = service.search(item.query, top_k=k, subject=item.subject)
        latencies.append(result.elapsed_ms.get("total", 0.0))
        row: dict[str, Any] = {
            "query": item.query,
            "expected_heading": item.expected_heading,
            "relevant_count": len(relevant),
            "modes": {},
        }
        for mode in MODES:
            ranked = [chunk.chunk_id for chunk in result.by_mode(mode)]
            top_list = ranked[:k]
            hit1 = int(bool(top_list) and top_list[0] in relevant)
            hitk = int(any(chunk_id in relevant for chunk_id in top_list))
            recall = len(set(top_list) & relevant) / len(relevant)
            mrr = 0.0
            for rank, chunk_id in enumerate(ranked, 1):
                if chunk_id in relevant:
                    mrr = 1.0 / rank
                    break
            bucket = buckets[mode]
            bucket["hit1"] += hit1
            bucket["hitk"] += hitk
            bucket["recall"] += recall
            bucket["mrr"] += mrr
            bucket["count"] += 1
            row["modes"][mode] = {
                "hit@1": hit1,
                "hit@k": hitk,
                "recall@k": round(recall, 4),
                "mrr": round(mrr, 4),
                "top_ids": top_list,
            }
        per_query.append(row)
        progress.tick(index, query=item.query[:20])

    summary: dict[str, dict[str, float]] = {}
    for mode, bucket in buckets.items():
        count = bucket["count"] or 1
        summary[mode] = {
            "hit@1": round(bucket["hit1"] / count, 4),
            "hit@k": round(bucket["hitk"] / count, 4),
            "recall@k": round(bucket["recall"] / count, 4),
            "mrr": round(bucket["mrr"] / count, 4),
            "queries": bucket["count"],
        }

    report = {
        "run_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "top_k": k,
        "dataset_queries": len(queries),
        "evaluated_queries": len(per_query),
        "unmatched_labels": unmatched_labels,
        "corpus_size": len(corpus),
        "summary": summary,
        "latency_ms": _latency_stats(latencies),
        "per_query": per_query,
    }
    for mode, values in summary.items():
        get_logger().info(
            "评测[%s] hit@1=%.1f%% hit@%d=%.1f%% recall@%d=%.1f%% MRR=%.4f (n=%d)",
            mode,
            values["hit@1"] * 100,
            k,
            values["hit@k"] * 100,
            k,
            values["recall@k"] * 100,
            values["mrr"],
            values["queries"],
        )
    if unmatched_labels:
        get_logger().warning("有 %d 条标注在语料中找不到对应块", len(unmatched_labels))
    return report


def report_text(report: dict[str, Any]) -> str:
    k = report["top_k"]
    lines = [
        f"检索评测 top_k={k} 语料={report['corpus_size']} 块 "
        f"查询={report['evaluated_queries']}/{report['dataset_queries']}",
        f"{'模式':<8}{'hit@1':>8}{'hit@k':>8}{'recall@k':>10}{'MRR':>8}",
    ]
    for mode, values in report["summary"].items():
        lines.append(
            f"{mode:<8}{values['hit@1'] * 100:>7.1f}%{values['hit@k'] * 100:>7.1f}%"
            f"{values['recall@k'] * 100:>9.1f}%{values['mrr']:>8.4f}"
        )
    latency = report["latency_ms"]
    lines.append(
        f"延迟(ms): avg={latency['avg']} p50={latency['p50']} p95={latency['p95']}"
    )
    if report["unmatched_labels"]:
        lines.append(f"未匹配标注 {len(report['unmatched_labels'])} 条:")
        lines.extend(f"  - {query}" for query in report["unmatched_labels"])
    return "\n".join(lines)


def save_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
