"""Version-tolerant wrapper around RAGAS.

Written against ragas 0.4.x, which exposes per-metric classes in
``ragas.metrics.collections`` with an async ``ascore(...)`` method and LLMs built
by ``ragas.llms.llm_factory``. Everything is imported lazily; if the installed
version does not offer these interfaces, or no LLM key is configured, every metric
is reported as ``not_computed`` with a reason instead of a silent zero.
"""

from __future__ import annotations

import asyncio
import logging
import math
from importlib import metadata
from typing import Any

from app.core.config import LLM_KEY_HINT, Settings

logger = logging.getLogger(__name__)

METRIC_NAMES = ("faithfulness", "context_precision", "context_recall")


def ragas_version() -> str | None:
    """Installed ragas version, or ``None`` if it is not installed."""
    try:
        return metadata.version("ragas")
    except metadata.PackageNotFoundError:
        return None


def _not_computed(reason: str) -> dict:
    return {"value": None, "reason": reason}


class RagasAdapter:
    """Computes faithfulness, context precision and context recall per sample."""

    def __init__(self, settings: Settings, metrics: dict[str, Any] | None = None) -> None:
        """``metrics`` may be injected (name -> object with ``ascore``) for testing."""
        self._settings = settings
        self._metrics = metrics
        self._unavailable_reason: str | None = None

    def _build_metrics(self) -> dict[str, Any] | None:
        """Create ragas metric objects, or record why that is impossible."""
        if self._metrics is not None:
            return self._metrics
        if self._unavailable_reason:
            return None
        if self._settings.llm_api_key is None:
            self._unavailable_reason = f"LLM not configured ({LLM_KEY_HINT} missing)"
            return None
        try:
            from openai import AsyncOpenAI
            from ragas.llms import llm_factory
            from ragas.metrics.collections import (
                ContextPrecisionWithoutReference,
                ContextPrecisionWithReference,
                ContextRecall,
                Faithfulness,
            )

            client = AsyncOpenAI(
                api_key=self._settings.llm_api_key.get_secret_value(),
                base_url=self._settings.llm_base_url,
                timeout=self._settings.request_timeout_seconds,
                max_retries=self._settings.llm_max_retries,
            )
            llm = llm_factory(self._settings.llm_model, client=client)
            self._metrics = {
                "faithfulness": Faithfulness(llm=llm),
                "context_precision_with_reference": ContextPrecisionWithReference(llm=llm),
                "context_precision_without_reference": ContextPrecisionWithoutReference(llm=llm),
                "context_recall": ContextRecall(llm=llm),
            }
        except Exception as exc:  # noqa: BLE001 - any import/API mismatch must degrade gracefully
            self._unavailable_reason = (
                f"ragas {ragas_version()} interface unavailable: {type(exc).__name__}: {exc}"
            )[:300]
            logger.warning("RAGAS disabled: %s", self._unavailable_reason)
            return None
        return self._metrics

    async def _score_one(self, metrics: dict[str, Any], sample: dict) -> dict[str, dict]:
        question = sample.get("question")
        response = sample.get("response")
        contexts = sample.get("contexts") or []
        reference = sample.get("reference")
        if not question or not response or not contexts:
            reason = "sample is missing question, response or contexts"
            return {name: _not_computed(reason) for name in METRIC_NAMES}

        async def run(key: str, **kwargs: Any) -> dict:
            if key not in metrics:
                return _not_computed(f"metric '{key}' not available in ragas {ragas_version()}")
            try:
                result = await metrics[key].ascore(**kwargs)
                value = float(getattr(result, "value", result))
            except Exception as exc:  # noqa: BLE001 - one failed metric must not abort the run
                return _not_computed(f"{type(exc).__name__}: {exc}"[:200])
            if math.isnan(value):
                return _not_computed("ragas returned NaN (no statements could be extracted)")
            return {"value": value, "reason": None}

        scores = {
            "faithfulness": await run(
                "faithfulness", user_input=question, response=response, retrieved_contexts=contexts
            )
        }
        if reference:
            scores["context_precision"] = await run(
                "context_precision_with_reference",
                user_input=question,
                reference=reference,
                retrieved_contexts=contexts,
            )
            scores["context_recall"] = await run(
                "context_recall",
                user_input=question,
                retrieved_contexts=contexts,
                reference=reference,
            )
        else:
            scores["context_precision"] = await run(
                "context_precision_without_reference",
                user_input=question,
                response=response,
                retrieved_contexts=contexts,
            )
            scores["context_recall"] = _not_computed("no reference answer for this sample")
        return scores

    def score(self, samples: list[dict]) -> list[dict[str, dict]]:
        """Score samples (``question``, ``response``, ``contexts``, optional ``reference``).

        Returns one dict per sample mapping each metric name to
        ``{"value": float | None, "reason": str | None}``.
        """
        metrics = self._build_metrics()
        if metrics is None:
            reason = self._unavailable_reason or "ragas unavailable"
            return [{name: _not_computed(reason) for name in METRIC_NAMES} for _ in samples]

        async def run_all() -> list[dict[str, dict]]:
            return [await self._score_one(metrics, sample) for sample in samples]

        return asyncio.run(run_all())
