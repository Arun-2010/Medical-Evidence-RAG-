"""Evidence-grounded answering with citation enforcement and abstention."""

from __future__ import annotations

import logging
from typing import Protocol

from pydantic import BaseModel, Field

from app.core.config import ABSTENTION_PHRASE, Settings
from app.evaluation.judge import ClaimJudge, JudgeResult
from app.generation.citations import is_abstention, validate_citations
from app.generation.prompts import (
    ANSWER_SYSTEM_PROMPT,
    build_answer_prompt,
    build_correction_prompt,
)
from app.generation.providers import LLMProvider, get_provider
from app.retrieval.service import RetrievalService
from app.schemas.api import ABSTENTION_MESSAGES, AbstentionReason, Citation, EvidenceItem

logger = logging.getLogger(__name__)


class ClaimChecker(Protocol):
    """Anything that can judge an answer's claims against evidence."""

    def judge(self, question: str, answer: str, evidence: list[EvidenceItem]) -> JudgeResult: ...


class AnswerResult(BaseModel):
    """Outcome of answering one question."""

    answer: str
    abstained: bool
    abstention_reason: AbstentionReason | None = None
    abstention_message: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    diagnostics: dict = Field(default_factory=dict)


class AnswerService:
    """Retrieves evidence, gates on relevance, generates, then verifies the answer.

    An answer is returned only if it survives every check; otherwise the result is
    the standard abstention phrase with a structured reason. Thresholds are never
    relaxed automatically.
    """

    def __init__(
        self,
        settings: Settings,
        retrieval: RetrievalService,
        provider: LLMProvider | None = None,
        claim_checker: ClaimChecker | None = None,
    ) -> None:
        self._settings = settings
        self._retrieval = retrieval
        self._provider = provider
        self._claim_checker = claim_checker

    def _get_provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = get_provider(self._settings)
        return self._provider

    def _get_claim_checker(self) -> ClaimChecker:
        if self._claim_checker is None:
            self._claim_checker = ClaimJudge(self._get_provider())
        return self._claim_checker

    def ask(self, question: str, top_k: int | None = None) -> AnswerResult:
        """Retrieve evidence for ``question`` and answer from it."""
        return self.answer_from_evidence(question, self._retrieval.retrieve(question, top_k))

    def answer_from_evidence(self, question: str, evidence: list[EvidenceItem]) -> AnswerResult:
        """Answer from already-retrieved evidence (also used by the evaluation runner)."""
        settings = self._settings
        relevant = [e for e in evidence if e.rerank_score >= settings.min_rerank_score]
        diagnostics: dict = {
            "retrieved_count": len(evidence),
            "relevant_count": len(relevant),
            "top_rerank_score": max((e.rerank_score for e in evidence), default=None),
            "min_rerank_score": settings.min_rerank_score,
            "min_evidence_count": settings.min_evidence_count,
            "llm_called": False,
        }

        def abstain(reason: AbstentionReason) -> AnswerResult:
            logger.info("Abstaining: %s", reason)
            return AnswerResult(
                answer=ABSTENTION_PHRASE,
                abstained=True,
                abstention_reason=reason,
                abstention_message=ABSTENTION_MESSAGES[reason],
                evidence=evidence,
                diagnostics=diagnostics,
            )

        if not evidence:
            return abstain("no_evidence")
        if not relevant:
            return abstain("low_relevance")
        if len(relevant) < settings.min_evidence_count:
            return abstain("insufficient_evidence_count")

        provider = self._get_provider()
        diagnostics.update(llm_called=True, llm_model=provider.model, citation_retry=False)
        draft = provider.complete(ANSWER_SYSTEM_PROMPT, build_answer_prompt(question, relevant))
        if is_abstention(draft):
            return abstain("model_reported_insufficient_evidence")

        report = validate_citations(draft, relevant)
        if not report.ok:
            diagnostics.update(citation_retry=True, first_attempt_citations=report.to_dict())
            draft = provider.complete(
                ANSWER_SYSTEM_PROMPT,
                build_correction_prompt(question, relevant, draft, report.problems()),
            )
            if is_abstention(draft):
                return abstain("model_reported_insufficient_evidence")
            report = validate_citations(draft, relevant)
        diagnostics["citations"] = report.to_dict()
        if report.invalid_pmids:
            return abstain("invalid_citations")
        if not report.valid_pmids:
            return abstain("no_valid_citations")

        if settings.enable_claim_check:
            verdict = self._get_claim_checker().judge(question, draft, relevant)
            diagnostics["claim_check"] = verdict.to_dict()
            support, coverage = verdict.support_rate, verdict.citation_coverage
            if verdict.error or support is None or coverage is None:
                return abstain("claim_check_failed")
            if support < settings.min_claim_support_rate:
                return abstain("unsupported_claims")
            if coverage < settings.min_citation_coverage:
                return abstain("uncited_claims")

        return AnswerResult(
            answer=draft,
            abstained=False,
            citations=report.citations,
            evidence=evidence,
            diagnostics=diagnostics,
        )
