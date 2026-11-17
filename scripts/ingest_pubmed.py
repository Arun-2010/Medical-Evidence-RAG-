"""Ingest PubMed abstracts for a query and update the search indexes.

Example:
    python scripts/ingest_pubmed.py --query "asthma treatment" --max-records 50
"""

from __future__ import annotations

import argparse
import sys

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import setup_logging
from app.ingestion.pipeline import ingest_pubmed, load_articles
from app.retrieval.service import RetrievalService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query", required=True, help="PubMed search query")
    parser.add_argument("--max-records", type=int, default=50, help="Maximum PMIDs to fetch")
    parser.add_argument("--min-date", help="Earliest publication date (YYYY[/MM[/DD]])")
    parser.add_argument("--max-date", help="Latest publication date (YYYY[/MM[/DD]])")
    parser.add_argument(
        "--no-index", action="store_true", help="Only update the JSONL store; skip indexing"
    )
    args = parser.parse_args()
    if not 1 <= args.max_records <= 10000:
        parser.error("--max-records must be between 1 and 10000")

    try:
        settings = get_settings()
        setup_logging(settings.log_level)
        report = ingest_pubmed(settings, args.query, args.max_records, args.min_date, args.max_date)
        print(
            f"PMIDs found: {report.pmids_found} | already stored: {report.already_stored} | "
            f"fetched: {report.fetched} | added: {report.added} | "
            f"skipped (no abstract): {report.skipped_no_abstract} | malformed: {report.malformed}"
        )
        print(f"Articles in {settings.processed_articles_path}: {report.total_stored}")
        if args.no_index:
            return 0
        articles, _ = load_articles(settings.processed_articles_path)
        if not articles:
            print("No articles stored; nothing to index.")
            return 0
        print("Indexing (the first run downloads the embedding model)...")
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
