"""Ingestion pipeline: PubMed search -> fetch -> parse -> merge into processed JSONL."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.ingestion.parser import Article, parse_pubmed_xml
from app.ingestion.pubmed import PubMedClient

logger = logging.getLogger(__name__)


class IngestReport(BaseModel):
    """Counts describing one ingestion run."""

    query: str
    pmids_found: int = 0
    already_stored: int = 0
    fetched: int = 0
    added: int = 0
    skipped_no_abstract: int = 0
    malformed: int = 0
    total_stored: int = 0


def load_articles(path: Path) -> tuple[list[Article], int]:
    """Load articles from JSONL, de-duplicated by PMID (last wins).

    Returns the articles and the number of unreadable rows that were skipped.
    """
    if not path.exists():
        return [], 0
    by_pmid: dict[str, Article] = {}
    bad_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                article = Article.model_validate_json(line)
            except ValidationError:
                bad_rows += 1
                continue
            by_pmid[article.pmid] = article
    if bad_rows:
        logger.warning("Skipped %d unreadable rows in %s", bad_rows, path)
    return list(by_pmid.values()), bad_rows


def save_articles(path: Path, articles: list[Article]) -> None:
    """Atomically write articles as JSONL (write to a temp file, then replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for article in articles:
            handle.write(article.model_dump_json() + "\n")
    os.replace(tmp, path)


def _save_raw(raw_dir: Path, query: str, xml_documents: list[str]) -> None:
    """Keep the raw EFetch payloads for provenance/debugging."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:8]
    for i, xml_text in enumerate(xml_documents):
        (raw_dir / f"efetch_{stamp}_{digest}_{i:03d}.xml").write_text(xml_text, encoding="utf-8")
    manifest = {"query": query, "fetched_at": stamp, "batches": len(xml_documents)}
    (raw_dir / f"efetch_{stamp}_{digest}.json").write_text(json.dumps(manifest), encoding="utf-8")


def ingest_pubmed(
    settings: Settings,
    query: str,
    max_records: int,
    min_date: str | None = None,
    max_date: str | None = None,
    client: PubMedClient | None = None,
    save_raw: bool = True,
) -> IngestReport:
    """Search PubMed and merge new abstracts into the processed JSONL store.

    Idempotent and resumable: PMIDs already stored are not fetched again, and the
    store is keyed by PMID, so re-running a query never creates duplicates.
    """
    report = IngestReport(query=query)
    existing, _ = load_articles(settings.processed_articles_path)
    by_pmid = {a.pmid: a for a in existing}

    owns_client = client is None
    client = client or PubMedClient(settings)
    try:
        pmids = client.search(query, max_records, min_date, max_date)
        report.pmids_found = len(pmids)
        new_pmids = [p for p in pmids if p not in by_pmid]
        report.already_stored = len(pmids) - len(new_pmids)
        xml_documents = client.fetch_xml(new_pmids) if new_pmids else []
    finally:
        if owns_client:
            client.close()

    report.fetched = len(new_pmids)
    for xml_text in xml_documents:
        parsed = parse_pubmed_xml(xml_text)
        report.skipped_no_abstract += parsed.skipped_no_abstract
        report.malformed += parsed.malformed
        for article in parsed.articles:
            if article.pmid not in by_pmid:
                by_pmid[article.pmid] = article
                report.added += 1

    if xml_documents and save_raw:
        _save_raw(settings.raw_dir, query, xml_documents)
    if report.added:
        save_articles(settings.processed_articles_path, list(by_pmid.values()))
    report.total_stored = len(by_pmid)
    logger.info("Ingestion finished: %s", report.model_dump())
    return report
