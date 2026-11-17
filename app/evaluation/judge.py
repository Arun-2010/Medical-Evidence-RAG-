"""Claim-level LLM-as-judge.

The judge is itself an LLM: it is imperfect and is *not* ground truth. Its output
is validated against a strict schema and cross-checked deterministically where
possible (cited PMIDs must really appear in the answer and in the evidence).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.core.exceptions import LLMProviderError
from app.generation.citations import extract_cited_pmids
from app.generation.prompts import JUDGE_SYSTEM_PROMPT, build_judge_prompt
from app.generation.providers import LLMProvider
from app.schemas.api import EvidenceItem

logger = logging.getLogger(__name__)

Verdict = Literal["supported", "partially_supported", "unsupported", "not_verifiable"]
CHECKABLE: tuple[str, ...] = ("supported", "partially_supported", "unsupported")


class ClaimVerdict(BaseModel):
    """The judge's decision for one claim."""

    claim: str = Field(min_length=1)
    cited_pmids: list[str] = Field(default_factory=list)
    verdict: Verdict

    @field_validator("cited_pmids", mode="before")
    @classmethod
    def _coerce_pmids(cls, value: object) -> object:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip().isdigit()]
        return value


class JudgeOutput(BaseModel):
    """Strict schema for the judge's JSON response."""

    claims: list[ClaimVerdict]


@dataclass
class JudgeResult:
    """Validated judge output, or an error if the judge could not be used."""

    claims: list[ClaimVerdict] = field(default_factory=list)
    error: str | None = None
    valid_pmids: set[str] = field(default_factory=set)

    @property
    def checkable(self) -> list[ClaimVerdict]:
        return [c for c in self.claims if c.verdict in CHECKABLE]

    @property
    def support_rate(self) -> float | None:
        """Fully supported claims / checkable claims (``None`` if nothing checkable)."""
        checkable = self.checkable
        if self.error or not checkable:
            return None
        return sum(c.verdict == "supported" for c in checkable) / len(checkable)

    @property
    def citation_coverage(self) -> float | None:
        """Checkable claims with >= 1 valid citation / checkable claims."""
        checkable = self.checkable
        if self.error or not checkable:
            return None
        cited = sum(any(p in self.valid_pmids for p in c.cited_pmids) for c in checkable)
        return cited / len(checkable)

    def to_dict(self) -> dict:
        counts = {
            v: sum(c.verdict == v for c in self.claims) for v in (*CHECKABLE, "not_verifiable")
        }
        return {
            "error": self.error,
            "claim_count": len(self.claims),
            "checkable_claims": len(self.checkable),
            "verdict_counts": counts,
            "support_rate": self.support_rate,
            "citation_coverage": self.citation_coverage,
            "claims": [c.model_dump() for c in self.claims],
        }


def parse_judge_output(raw: str) -> JudgeOutput:
    """Parse and validate the judge's JSON (tolerating a Markdown code fence)."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    try:
        return JudgeOutput.model_validate(json.loads(raw[start : end + 1]))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid judge JSON: {type(exc).__name__}") from exc


class ClaimJudge:
    """Judges each factual claim of an answer against the retrieved evidence."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def judge(self, question: str, answer: str, evidence: list[EvidenceItem]) -> JudgeResult:
        """Return per-claim verdicts. Retries once on malformed output, then reports an error."""
        prompt = build_judge_prompt(question, answer, evidence)
        evidence_pmids = {item.pmid for item in evidence}
        in_answer = set(extract_cited_pmids(answer))
        last_error = "unknown error"
        for attempt in range(2):
            try:
                raw = self._provider.complete(JUDGE_SYSTEM_PROMPT, prompt, json_mode=True)
                output = parse_judge_output(raw)
            except LLMProviderError as exc:
                return JudgeResult(error=f"judge provider error: {exc.message}")
            except ValueError as exc:
                last_error = str(exc)
                logger.warning("Judge output rejected (attempt %d): %s", attempt + 1, last_error)
                continue
            # Trust a claimed citation only if the PMID is really in the answer text.
            for claim in output.claims:
                claim.cited_pmids = [p for p in claim.cited_pmids if p in in_answer]
            return JudgeResult(claims=output.claims, valid_pmids=evidence_pmids & in_answer)
        return JudgeResult(error=f"judge returned malformed output twice ({last_error})")
