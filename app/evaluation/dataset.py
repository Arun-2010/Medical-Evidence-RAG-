"""Evaluation datasets: PubMedQA (expert-labelled) and a synthetic unanswerable stress set."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.core.exceptions import DatasetError
from app.ingestion.parser import PUBMED_URL_TEMPLATE, Article

logger = logging.getLogger(__name__)

PUBMEDQA_REPO = "https://github.com/pubmedqa/pubmedqa"
PUBMEDQA_FILES = {
    "ori_pqal.json": "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json",
    "test_ground_truth.json": (
        "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json"
    ),
}
MANIFEST_NAME = "manifest.json"
DEV_SIZE = 100
Split = Literal["dev", "validation", "test"]


class EvalExample(BaseModel):
    """One normalized evaluation question."""

    id: str
    kind: Literal["pubmedqa", "synthetic_unanswerable"]
    split: str
    question: str = Field(min_length=3)
    expected_abstain: bool
    gold_pmid: str | None = None
    contexts: list[str] = Field(default_factory=list)
    """Dataset-provided context (PubMedQA abstract sections). Not retrieved text."""
    reference_answer: str | None = None
    label: str | None = None
    year: int | None = None


def download_pubmedqa(target_dir: Path, timeout: float = 60.0, force: bool = False) -> dict:
    """Download PubMedQA PQA-L from the official repository and write a manifest.

    The manifest records source URLs, SHA-256 digests and the download time; that
    digest is the dataset version reported with every evaluation run.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / MANIFEST_NAME
    if (
        manifest_path.exists()
        and not force
        and all((target_dir / name).exists() for name in PUBMEDQA_FILES)
    ):
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    files = {}
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for name, url in PUBMEDQA_FILES.items():
            try:
                response = client.get(url)
                response.raise_for_status()
                json.loads(response.content)
            except (httpx.HTTPError, ValueError) as exc:
                raise DatasetError(f"Could not download {url}: {type(exc).__name__}") from exc
            (target_dir / name).write_bytes(response.content)
            files[name] = {
                "url": url,
                "sha256": hashlib.sha256(response.content).hexdigest(),
                "bytes": len(response.content),
            }
    manifest = {
        "dataset": "PubMedQA PQA-L (expert-labelled)",
        "repository": PUBMEDQA_REPO,
        "license": "MIT (see repository)",
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def dataset_version(dataset_dir: Path) -> dict | None:
    """Return the download manifest, or ``None`` if the dataset is not present."""
    path = dataset_dir / MANIFEST_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def split_pmids(all_pmids: list[str], test_pmids: set[str], seed: int = 42) -> dict[str, list[str]]:
    """Deterministic dev/validation/test split.

    ``test`` is the official 500-question PubMedQA test set. The remaining labelled
    questions are shuffled with ``seed``; the first ``DEV_SIZE`` form ``dev`` (smoke
    tests) and the rest ``validation`` (threshold tuning). Splits never overlap.
    """
    rest = sorted(p for p in all_pmids if p not in test_pmids)
    random.Random(seed).shuffle(rest)
    return {
        "dev": rest[:DEV_SIZE],
        "validation": rest[DEV_SIZE:],
        "test": sorted(p for p in all_pmids if p in test_pmids),
    }


def _normalize(pmid: str, row: object, split: str) -> EvalExample:
    """Convert one raw PubMedQA record; raises on malformed rows."""
    if not isinstance(row, dict) or not str(pmid).isdigit():
        raise ValueError("row is not a PMID-keyed object")
    contexts = [c.strip() for c in row["CONTEXTS"] if isinstance(c, str) and c.strip()]
    label = row["final_decision"]
    if not contexts or label not in {"yes", "no", "maybe"}:
        raise ValueError("missing contexts or label")
    year = row.get("YEAR")
    return EvalExample(
        id=f"pubmedqa-{pmid}",
        kind="pubmedqa",
        split=split,
        question=str(row["QUESTION"]).strip(),
        expected_abstain=False,
        gold_pmid=str(pmid),
        contexts=contexts,
        reference_answer=(row.get("LONG_ANSWER") or "").strip() or None,
        label=label,
        year=int(year) if str(year or "").isdigit() else None,
    )


def load_pubmedqa(
    dataset_dir: Path, split: Split, limit: int | None = None, seed: int = 42
) -> tuple[list[EvalExample], int]:
    """Load one split. Returns ``(examples, skipped_malformed_rows)``."""
    data_path = dataset_dir / "ori_pqal.json"
    truth_path = dataset_dir / "test_ground_truth.json"
    if not data_path.exists() or not truth_path.exists():
        raise DatasetError(
            f"PubMedQA not found in {dataset_dir}. Run: python scripts/download_pubmedqa.py"
        )
    try:
        data = json.loads(data_path.read_text(encoding="utf-8"))
        test_pmids = set(json.loads(truth_path.read_text(encoding="utf-8")))
        if not isinstance(data, dict):
            raise ValueError("top-level JSON is not an object")
    except ValueError as exc:
        raise DatasetError(f"PubMedQA files in {dataset_dir} are corrupt: {exc}") from exc

    examples: list[EvalExample] = []
    skipped = 0
    for pmid in split_pmids(list(data), test_pmids, seed)[split]:
        try:
            examples.append(_normalize(pmid, data[pmid], split))
        except (KeyError, TypeError, ValueError, ValidationError):
            skipped += 1
    if skipped:
        logger.warning("Skipped %d malformed PubMedQA rows in split %s", skipped, split)
    return (examples[:limit] if limit is not None else examples), skipped


def load_unanswerable(path: Path) -> tuple[list[EvalExample], int]:
    """Load the synthetic unanswerable stress set (JSONL with ``id`` and ``question``).

    These questions are hand-written stress tests, not expert-labelled ground truth,
    and are always reported separately from PubMedQA.
    """
    if not path.exists():
        return [], 0
    examples: list[EvalExample] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            examples.append(
                EvalExample(
                    id=str(row["id"]),
                    kind="synthetic_unanswerable",
                    split="stress",
                    question=str(row["question"]).strip(),
                    expected_abstain=True,
                )
            )
        except (ValueError, KeyError, TypeError):
            skipped += 1
    return examples, skipped


def example_to_article(example: EvalExample) -> Article:
    """Turn a PubMedQA example's own context into an indexable article.

    PubMedQA does not ship article titles, so the question (which is derived from
    the title) is *not* used; a neutral placeholder keeps retrieval honest.
    """
    if example.gold_pmid is None or not example.contexts:
        raise ValueError(f"example {example.id} has no dataset context")
    return Article(
        pmid=example.gold_pmid,
        title=f"PubMed abstract {example.gold_pmid}",
        abstract=" ".join(example.contexts),
        publication_year=example.year,
        journal=None,
        pubmed_url=PUBMED_URL_TEMPLATE.format(pmid=example.gold_pmid),
        source="pubmedqa",
    )
