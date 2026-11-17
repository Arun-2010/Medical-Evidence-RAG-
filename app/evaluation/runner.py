"""Evaluation runner: retrieval, grounding and abstention metrics with saved reports."""

from __future__ import annotations

import csv
import json
import logging
import platform
from collections import Counter
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.exceptions import AppError, LLMNotConfiguredError
from app.evaluation import deterministic as det
from app.evaluation.dataset import (
    EvalExample,
    Split,
    dataset_version,
    example_to_article,
    load_pubmedqa,
    load_unanswerable,
)
from app.evaluation.judge import ClaimJudge
from app.evaluation.ragas_adapter import METRIC_NAMES, RagasAdapter
from app.generation.answer import AnswerService
from app.generation.citations import strip_citations
from app.generation.providers import LLMProvider, get_provider
from app.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

_PACKAGES = (
    "ragas", "chromadb", "sentence-transformers", "transformers", "torch", "rank-bm25",
    "fastapi", "pydantic", "httpx", "openai",
)  # fmt: skip


class EvalConfig(BaseModel):
    """Options for one evaluation run (saved verbatim in the report)."""

    split: Split = "dev"
    limit: int | None = Field(default=None, ge=1)
    context_mode: Literal["dataset", "corpus"] = "dataset"
    """``dataset``: index PubMedQA's own abstracts in a separate eval collection, so the
    gold PMID is known. ``corpus``: query the locally ingested PubMed index as-is."""
    distractors: int = Field(default=80, ge=0)
    include_stress: bool = True
    use_ragas: bool = True
    min_evidence_count: int | None = Field(default=None, ge=1)
    output_dir: Path | None = None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _eval_settings(settings: Settings, config: EvalConfig) -> Settings:
    """Settings used for the run. Dataset mode gets its own Chroma collection."""
    update: dict = {}
    if config.context_mode == "dataset":
        update.update(
            chroma_path=settings.data_dir / "eval" / "chroma",
            chroma_collection=f"pubmedqa_{config.split}",
            # Each PubMedQA question has exactly one gold abstract, so requiring two
            # relevant sources would force an abstention on every question.
            min_evidence_count=1,
        )
    if config.min_evidence_count is not None:
        update["min_evidence_count"] = config.min_evidence_count
    return settings.model_copy(update=update)


def evaluate_example(
    example: EvalExample,
    retrieval: RetrievalService,
    answerer: AnswerService,
    indexed: set[str],
    judge: ClaimJudge | None,
    min_rerank_score: float,
) -> dict:
    """Run one question through retrieval and answering; never raises for app errors."""
    row: dict = {
        "id": example.id,
        "kind": example.kind,
        "split": example.split,
        "question": example.question,
        "expected_abstain": example.expected_abstain,
        "label": example.label,
        "gold_pmid": example.gold_pmid,
        "gold_indexed": example.gold_pmid in indexed if example.gold_pmid else False,
        "retrieved_pmids": [],
        "gold_rank": None,
        "generation_status": "not_run",
        "abstained": None,
        "abstention_reason": None,
        "answer": None,
        "citations": None,
        "claim_check": None,
        "ragas": None,
        "error": None,
    }
    try:
        evidence = retrieval.retrieve(example.question)
    except AppError as exc:
        row.update(generation_status="error", error=f"retrieval: {exc.code}: {exc.message}")
        return row
    row["retrieved_pmids"] = [e.pmid for e in evidence]
    row["rerank_scores"] = [round(e.rerank_score, 4) for e in evidence]
    row["gold_rank"] = det.gold_rank(row["retrieved_pmids"], example.gold_pmid)
    relevant = [e for e in evidence if e.rerank_score >= min_rerank_score]
    row["_contexts"] = [e.abstract for e in relevant]

    try:
        result = answerer.answer_from_evidence(example.question, evidence)
    except LLMNotConfiguredError:
        row["generation_status"] = "skipped_llm_not_configured"
        return row
    except AppError as exc:
        row.update(generation_status="error", error=f"generation: {exc.code}: {exc.message}")
        return row

    row.update(
        generation_status="completed",
        abstained=result.abstained,
        abstention_reason=result.abstention_reason,
        answer=result.answer,
        citations=result.diagnostics.get("citations"),
        claim_check=result.diagnostics.get("claim_check"),
        llm_called=result.diagnostics.get("llm_called", False),
    )
    if not result.abstained and row["claim_check"] is None and judge is not None:
        row["claim_check"] = judge.judge(example.question, result.answer, relevant).to_dict()
    return row


def summarize(rows: list[dict], k: int) -> dict:
    """Aggregate result rows into metric sections. Malformed rows are counted and skipped."""
    valid = [r for r in rows if det.is_valid_row(r)]
    pubmedqa = [r for r in valid if r["kind"] == "pubmedqa"]
    stress = [r for r in valid if r["kind"] == "synthetic_unanswerable"]

    ragas_section = {}
    for name in METRIC_NAMES:
        cells = [(r.get("ragas") or {}).get(name) for r in valid]
        cells = [c for c in cells if isinstance(c, dict)]
        reasons = Counter(c.get("reason") for c in cells if c.get("value") is None)
        reason = reasons.most_common(1)[0][0] if reasons else "RAGAS did not run on any answer"
        ragas_section[name] = det.mean_metric([c.get("value") for c in cells], reason)

    delivered = det.claim_metrics(valid, delivered_only=True)
    drafts = det.claim_metrics(valid, delivered_only=False)
    return {
        "counts": {
            "rows": len(rows),
            "malformed_rows": len(rows) - len(valid),
            "pubmedqa_rows": len(pubmedqa),
            "stress_rows": len(stress),
            "generation_completed": sum(r.get("generation_status") == "completed" for r in valid),
            "generation_skipped_llm_not_configured": sum(
                r.get("generation_status") == "skipped_llm_not_configured" for r in valid
            ),
            "errors": sum(r.get("generation_status") == "error" for r in valid),
            "answered": sum(r.get("abstained") is False for r in valid),
            "abstained": sum(r.get("abstained") is True for r in valid),
            "abstention_reasons": dict(
                Counter(r["abstention_reason"] for r in valid if r.get("abstention_reason"))
            ),
        },
        "metrics": {
            "retrieval": det.retrieval_metrics(pubmedqa, k),
            "grounding": {
                "claim_support_rate": delivered["claim_support_rate"],
                "citation_coverage": delivered["citation_coverage"],
                "draft_claim_support_rate": drafts["claim_support_rate"],
                "draft_citation_coverage": drafts["citation_coverage"],
                "citation_validity_rate": det.citation_validity_rate(valid),
            },
            "abstention": det.abstention_metrics(valid),
            "abstention_pubmedqa_only": det.abstention_metrics(pubmedqa),
            "abstention_stress_only": det.abstention_metrics(stress),
            "ragas": ragas_section,
        },
    }


def write_report(output_dir: Path, rows: list[dict], summary: dict) -> None:
    """Write ``results.jsonl``, ``summary.json`` and ``summary.csv``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            public = {k: v for k, v in row.items() if not k.startswith("_")}
            handle.write(json.dumps(public, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "metric", "value", "status", "n", "reason"])
        for section, metrics in summary["metrics"].items():
            for name, m in metrics.items():
                writer.writerow([section, name, m["value"], m["status"], m["n"], m["reason"] or ""])


def run_evaluation(
    settings: Settings,
    config: EvalConfig,
    *,
    retrieval: RetrievalService | None = None,
    provider: LLMProvider | None = None,
    ragas: RagasAdapter | None = None,
    examples: list[EvalExample] | None = None,
    distractors: list[EvalExample] | None = None,
    stress: list[EvalExample] | None = None,
) -> dict:
    """Run an evaluation and save the report. Returns the summary dict.

    The keyword-only arguments exist for tests (inject fakes / fixture data); the
    CLI leaves them unset so data is loaded from disk and real models are used.
    """
    eval_settings = _eval_settings(settings, config)
    eval_dir = settings.data_dir / "eval"
    skipped = {"pubmedqa": 0, "stress": 0}

    if examples is None:
        pool, skipped["pubmedqa"] = load_pubmedqa(
            eval_dir / "pubmedqa", config.split, seed=settings.seed
        )
        cut = config.limit if config.limit is not None else len(pool)
        examples = pool[:cut]
        distractors = pool[cut : cut + config.distractors]
    distractors = distractors or []
    if stress is None:
        stress = []
        if config.include_stress:
            stress, skipped["stress"] = load_unanswerable(eval_dir / "unanswerable_stress.jsonl")

    retrieval = retrieval or RetrievalService(eval_settings)
    if config.context_mode == "dataset":
        logger.info(
            "Indexing %d dataset abstracts for evaluation", len(examples) + len(distractors)
        )
        retrieval.index_articles([example_to_article(e) for e in [*examples, *distractors]])
    indexed = retrieval.indexed_pmids()

    if provider is None and eval_settings.llm_configured:
        provider = get_provider(eval_settings)
    answerer = AnswerService(eval_settings, retrieval, provider=provider)
    judge = ClaimJudge(provider) if provider and not eval_settings.enable_claim_check else None

    rows = []
    for i, example in enumerate([*examples, *stress], start=1):
        logger.info("Evaluating %d/%d (%s)", i, len(examples) + len(stress), example.id)
        rows.append(
            evaluate_example(
                example, retrieval, answerer, indexed, judge, eval_settings.min_rerank_score
            )
        )

    references = {e.id: e.reference_answer for e in examples}
    answered = [r for r in rows if r["abstained"] is False]
    if config.use_ragas and answered:
        adapter = ragas or RagasAdapter(eval_settings)
        samples = [
            {
                "question": r["question"],
                "response": strip_citations(r["answer"]),
                "contexts": r.get("_contexts") or [],
                "reference": references.get(r["id"]),
            }
            for r in answered
        ]
        for row, scores in zip(answered, adapter.score(samples), strict=True):
            row["ragas"] = scores

    summary = summarize(rows, k=eval_settings.top_k_rerank)
    summary["counts"].update(
        pubmedqa_skipped_malformed=skipped["pubmedqa"],
        stress_skipped_malformed=skipped["stress"],
        indexed_documents=len(indexed),
        distractor_documents=len(distractors),
        pubmedqa_label_distribution=dict(Counter(e.label for e in examples)),
    )
    stamp = datetime.now(UTC)
    summary = {
        "run": {
            "timestamp": stamp.isoformat(timespec="seconds"),
            "config": config.model_dump(mode="json"),
            "llm_configured": provider is not None,
            "llm_model": provider.model if provider else None,
            "judge_model": provider.model if provider else None,
            "embedding_model": eval_settings.embedding_model,
            "reranker_model": eval_settings.reranker_model,
            "settings": {
                "top_k_bm25": eval_settings.top_k_bm25,
                "top_k_dense": eval_settings.top_k_dense,
                "top_k_fusion": eval_settings.top_k_fusion,
                "top_k_rerank": eval_settings.top_k_rerank,
                "rrf_k": eval_settings.rrf_k,
                "min_rerank_score": eval_settings.min_rerank_score,
                "min_evidence_count": eval_settings.min_evidence_count,
                "enable_claim_check": eval_settings.enable_claim_check,
                "min_claim_support_rate": eval_settings.min_claim_support_rate,
                "min_citation_coverage": eval_settings.min_citation_coverage,
                "seed": eval_settings.seed,
            },
            "dataset": dataset_version(eval_dir / "pubmedqa"),
            "python": platform.python_version(),
            "packages": _package_versions(),
        },
        **summary,
        "notes": [
            "PubMedQA questions are treated as answerable from their own abstract; the "
            "synthetic stress set is hand-written, not expert-labelled ground truth.",
            "Claim metrics come from an LLM judge, which is imperfect and not ground truth. "
            "Delivered answers already passed the same judge, so their support rate is "
            "biased upward; the draft_* metrics include rejected drafts.",
            "Metrics with status 'not_computed' were not measured; they are not zeros.",
        ],
    }
    output_dir = config.output_dir or (
        eval_dir / "runs" / f"{stamp.strftime('%Y%m%dT%H%M%SZ')}_{config.split}"
    )
    write_report(output_dir, rows, summary)
    summary["output_dir"] = str(output_dir)
    return summary
