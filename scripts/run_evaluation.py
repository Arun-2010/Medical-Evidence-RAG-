"""Run the evaluation and save results.jsonl / summary.json / summary.csv.

Examples:
    python scripts/run_evaluation.py --split dev --limit 20
    python scripts/run_evaluation.py --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import setup_logging
from app.evaluation.runner import EvalConfig, run_evaluation


def _print_section(title: str, metrics: dict) -> None:
    print(f"\n{title}")
    for name, m in metrics.items():
        if m["status"] == "computed":
            print(f"  {name:<28} {m['value']:.4f}  (n={m['n']})")
        else:
            print(f"  {name:<28} not computed: {m['reason']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["dev", "validation", "test"], default="dev")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N questions")
    parser.add_argument(
        "--context-mode",
        choices=["dataset", "corpus"],
        default="dataset",
        help="dataset: index PubMedQA's own abstracts (gold PMIDs known); "
        "corpus: use the locally ingested PubMed index",
    )
    parser.add_argument(
        "--distractors", type=int, default=80, help="Extra same-split abstracts to index"
    )
    parser.add_argument("--no-stress", action="store_true", help="Skip the unanswerable set")
    parser.add_argument("--no-ragas", action="store_true", help="Skip RAGAS metrics")
    parser.add_argument("--min-evidence-count", type=int, help="Override MIN_EVIDENCE_COUNT")
    parser.add_argument("--output-dir", type=Path, help="Report directory")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    try:
        settings = get_settings()
        setup_logging(settings.log_level)
        if not settings.llm_configured:
            print(
                "Note: GEMINI_API_KEY is not set. Retrieval metrics will be computed; "
                "generation, judge and RAGAS metrics will be reported as not computed."
            )
        summary = run_evaluation(
            settings,
            EvalConfig(
                split=args.split,
                limit=args.limit,
                context_mode=args.context_mode,
                distractors=args.distractors,
                include_stress=not args.no_stress,
                use_ragas=not args.no_ragas,
                min_evidence_count=args.min_evidence_count,
                output_dir=args.output_dir,
            ),
        )
    except AppError as exc:
        print(f"Error: {exc.message}", file=sys.stderr)
        return 1

    print(f"\nCounts: {summary['counts']}")
    for section, metrics in summary["metrics"].items():
        _print_section(section, metrics)
    print(f"\nReport saved to {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
