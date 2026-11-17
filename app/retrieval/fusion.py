"""Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FusedHit:
    """A fused result with the 1-based rank it had in each component ranking."""

    doc_id: str
    rrf_score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]], k: int = 60, top_k: int | None = None
) -> list[FusedHit]:
    """Fuse ranked ID lists with ``RRF(d) = sum_i 1 / (k + rank_i(d))``.

    ``rankings`` maps a component name (e.g. ``"bm25"``) to document IDs ordered
    best-first. Ranks are 1-based. An ID repeated inside one list only counts at its
    best (first) position. Ties are broken by document ID for determinism.
    """
    if k < 1:
        raise ValueError("RRF k must be >= 1")
    fused: dict[str, FusedHit] = {}
    for name, doc_ids in rankings.items():
        rank = 0
        seen: set[str] = set()
        for doc_id in doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank += 1
            hit = fused.setdefault(doc_id, FusedHit(doc_id=doc_id))
            hit.rrf_score += 1.0 / (k + rank)
            hit.ranks[name] = rank
    ordered = sorted(fused.values(), key=lambda h: (-h.rrf_score, h.doc_id))
    return ordered[:top_k] if top_k is not None else ordered
