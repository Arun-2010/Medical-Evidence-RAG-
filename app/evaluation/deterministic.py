"""Deterministic evaluation metrics. None of these need an LLM.

Every metric is returned as a dict with ``value``, ``status`` (``computed`` or
``not_computed``), ``reason`` and ``n`` so that a missing metric is never mistaken
for a zero.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def metric(value: float | None, n: int = 0, reason: str | None = None) -> dict:
    """Build a metric record; ``value=None`` (or NaN) means it could not be computed."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {
            "value": None,
            "status": "not_computed",
            "reason": reason or "no eligible rows",
            "n": n,
        }
    return {"value": round(float(value), 4), "status": "computed", "reason": reason, "n": n}


def _ratio(numerator: float, denominator: float, reason: str) -> dict:
    if denominator <= 0:
        return metric(None, 0, reason)
    return metric(numerator / denominator, int(denominator))


def gold_rank(retrieved_pmids: list[str], gold_pmid: str | None) -> int | None:
    """1-based rank of the gold PMID in the retrieved list, or ``None`` if absent."""
    if gold_pmid is None:
        return None
    for rank, pmid in enumerate(retrieved_pmids, start=1):
        if pmid == gold_pmid:
            return rank
    return None


def is_valid_row(row: object) -> bool:
    """A usable result row has an id, a kind, a boolean label and a PMID list."""
    return (
        isinstance(row, dict)
        and isinstance(row.get("id"), str)
        and isinstance(row.get("kind"), str)
        and isinstance(row.get("expected_abstain"), bool)
        and isinstance(row.get("retrieved_pmids"), list)
    )


def retrieval_metrics(rows: Iterable[dict], k: int) -> dict[str, dict]:
    """Recall@k, MRR and nDCG@k over rows whose gold PMID is present in the index.

    Each question has a single gold document, so nDCG@k reduces to
    ``1 / log2(1 + rank)`` when the gold document is in the top ``k``.
    """
    eligible = [r for r in rows if r.get("gold_pmid") and r.get("gold_indexed")]
    reason = "no rows with a gold PMID present in the index"
    if not eligible:
        return {name: metric(None, 0, reason) for name in (f"recall_at_{k}", "mrr", f"ndcg_at_{k}")}
    ranks = [gold_rank(r["retrieved_pmids"], r["gold_pmid"]) for r in eligible]
    n = len(ranks)
    hits = [rank for rank in ranks if rank is not None and rank <= k]
    return {
        f"recall_at_{k}": metric(len(hits) / n, n),
        "mrr": metric(sum(1 / rank for rank in ranks if rank is not None) / n, n),
        f"ndcg_at_{k}": metric(sum(1 / math.log2(1 + rank) for rank in hits) / n, n),
    }


def abstention_metrics(rows: Iterable[dict]) -> dict[str, dict]:
    """Abstention classification metrics (positive class = abstain).

    ``abstained`` must be a bool for every row; if any row is undetermined (for
    example because generation could not run without an LLM key) the metrics are
    reported as not computed instead of being computed on a biased subset.
    """
    rows = list(rows)
    names = (
        "refusal_correctness", "false_abstention_rate", "unsafe_answer_rate",
        "abstention_precision", "abstention_recall", "abstention_f1",
    )  # fmt: skip
    undetermined = sum(not isinstance(r.get("abstained"), bool) for r in rows)
    if not rows or undetermined:
        reason = (
            f"{undetermined} of {len(rows)} rows have no abstention outcome "
            "(generation did not run)"
            if rows
            else "no eligible rows"
        )
        return {name: metric(None, 0, reason) for name in names}

    tp = sum(r["expected_abstain"] and r["abstained"] for r in rows)
    fp = sum(not r["expected_abstain"] and r["abstained"] for r in rows)
    fn = sum(r["expected_abstain"] and not r["abstained"] for r in rows)
    unanswerable = sum(r["expected_abstain"] for r in rows)
    answerable = len(rows) - unanswerable
    no_unanswerable = "no unanswerable rows in this run"
    no_answerable = "no answerable rows in this run"
    precision = _ratio(tp, tp + fp, "the system never abstained")
    recall = _ratio(tp, unanswerable, no_unanswerable)
    if precision["value"] is None or recall["value"] is None:
        f1 = metric(None, 0, "precision or recall not computable")
    elif tp == 0:
        f1 = metric(0.0, len(rows))
    else:
        f1 = metric(2 * tp / (2 * tp + fp + fn), len(rows))
    return {
        "refusal_correctness": _ratio(tp, unanswerable, no_unanswerable),
        "false_abstention_rate": _ratio(fp, answerable, no_answerable),
        "unsafe_answer_rate": _ratio(fn, unanswerable, no_unanswerable),
        "abstention_precision": precision,
        "abstention_recall": recall,
        "abstention_f1": f1,
    }


def citation_validity_rate(rows: Iterable[dict]) -> dict:
    """Citations resolving to retrieved PMIDs / citations emitted, over generated drafts.

    Counts the final draft of every question that reached generation, including
    drafts later rejected for invalid citations (delivered answers alone would
    score 1.0 by construction).
    """
    emitted = valid = 0
    for row in rows:
        citations = row.get("citations")
        if isinstance(citations, dict):
            emitted += len(citations.get("cited_pmids") or [])
            valid += len(citations.get("valid_pmids") or [])
    return _ratio(valid, emitted, "no citations were emitted (no drafts generated)")


def claim_metrics(rows: Iterable[dict], delivered_only: bool) -> dict[str, dict]:
    """Claim support rate and citation coverage from the LLM judge's verdicts.

    Support rate = supported / checkable claims; coverage = checkable claims with
    at least one valid citation / checkable claims. Pooled over claims, not rows.
    """
    supported = cited = checkable = judged = 0
    for row in rows:
        check = row.get("claim_check")
        if not isinstance(check, dict) or check.get("error"):
            continue
        if delivered_only and row.get("abstained") is not False:
            continue
        n = check.get("checkable_claims") or 0
        if not n:
            continue
        judged += 1
        checkable += n
        supported += (check.get("verdict_counts") or {}).get("supported", 0)
        cited += round((check.get("citation_coverage") or 0.0) * n)
    reason = "no judged answers (LLM judge did not run)"
    if not checkable:
        return {
            "claim_support_rate": metric(None, 0, reason),
            "citation_coverage": metric(None, 0, reason),
        }
    return {
        "claim_support_rate": {**metric(supported / checkable, checkable), "answers": judged},
        "citation_coverage": {**metric(cited / checkable, checkable), "answers": judged},
    }


def mean_metric(values: Iterable[float | None], reason: str) -> dict:
    """Mean of the non-missing values; not computed if there are none."""
    usable = [v for v in values if isinstance(v, int | float) and not math.isnan(v)]
    if not usable:
        return metric(None, 0, reason)
    return metric(sum(usable) / len(usable), len(usable))
