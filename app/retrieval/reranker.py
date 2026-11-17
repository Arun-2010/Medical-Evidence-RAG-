"""Cross-encoder reranking."""

from __future__ import annotations

import logging
import threading
from typing import Protocol

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    """Scores ``(query, text)`` pairs; higher means more relevant."""

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Lazy wrapper around a sentence-transformers ``CrossEncoder``.

    Scores are raw model outputs (logits for the default MS MARCO model). They are
    useful for ordering and as a heuristic threshold, but they are not calibrated
    probabilities and must not be presented as confidence.
    """

    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        self.model_name = model_name
        self._batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                logger.info("Loading cross-encoder %s on CPU", self.model_name)
                self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Score every text against the query in batches."""
        if not texts:
            return []
        scores = self._load().predict(
            [(query, text) for text in texts],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]


def rerank(
    reranker: Reranker, query: str, texts: list[str], top_k: int | None = None
) -> list[tuple[int, float]]:
    """Return ``(original_index, score)`` pairs sorted by descending score.

    Ties keep the original (fused) order. Raises if the reranker returns the wrong
    number of scores rather than silently mis-assigning them.
    """
    if not texts:
        return []
    scores = reranker.score(query, texts)
    if len(scores) != len(texts):
        raise ValueError(f"Reranker returned {len(scores)} scores for {len(texts)} candidates")
    ranked = sorted(enumerate(scores), key=lambda pair: (-pair[1], pair[0]))
    return ranked[:top_k] if top_k is not None else ranked
