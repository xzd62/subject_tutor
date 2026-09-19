"""运行离线检索评测并保存报告到 outputs/evaluation。

用法：python scripts/evaluate_retrieval.py [--k 3] [--dataset evaluation/retrieval_eval.jsonl]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tutor_rag.config import get_settings
from tutor_rag.evaluation import report_text, run_evaluation, save_report


def main() -> int:
    parser = argparse.ArgumentParser(description="检索三模式消融评测")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    settings = get_settings()
    dataset = Path(args.dataset) if args.dataset else None
    report = run_evaluation(settings=settings, dataset_path=dataset, k=args.k)
    print(report_text(report))
    path = save_report(
        report,
        settings.paths.evaluation_dir / f"retrieval-report-{report['run_id']}.json",
    )
    print(f"报告已保存: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
