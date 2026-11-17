"""Rebuild the Chroma and BM25 indexes from a processed JSONL file.

Example:
    python scripts/rebuild_index.py
    python scripts/rebuild_index.py --input tests/fixtures/sample_articles.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import setup_logging
from app.ingestion.pipeline import load_articles
from app.retrieval.service import RetrievalService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input", type=Path, help="Articles JSONL (default: data/processed/articles.jsonl)"
    )
    args = parser.parse_args()

    try:
        settings = get_settings()
        setup_logging(settings.log_level)
        path = args.input or settings.processed_articles_path
        if not path.exists():
            print(f"Error: {path} not found. Run scripts/ingest_pubmed.py first.", file=sys.stderr)
            return 1
        articles, bad_rows = load_articles(path)
        print(f"Loaded {len(articles)} articles from {path} ({bad_rows} unreadable rows skipped)")
        if not articles:
            print("Error: no usable articles; the index was left unchanged.", file=sys.stderr)
            return 1
        status = RetrievalService(settings).index_articles(articles)
        print(
            f"Index: {status.dense_count} dense / {status.bm25_count} BM25 chunks. {status.message}"
        )
    except AppError as exc:
        print(f"Error: {exc.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
