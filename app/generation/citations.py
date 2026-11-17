"""Deterministic citation parsing and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.config import ABSTENTION_PHRASE
from app.schemas.api import Citation, EvidenceItem

# Matches [PMID: 123], [PMID 123], [PMID: 123, 456] and [PMID: 123; PMID: 456].
_CITATION_RE = re.compile(
    r"\[\s*PMID\s*:?\s*(\d+(?:\s*[,;]\s*(?:PMID\s*:?\s*)?\d+)*)\s*\]", re.IGNORECASE
)
# Any PMID-looking mention, bracketed or not, so unbracketed fabricated IDs are also caught.
_LOOSE_PMID_RE = re.compile(r"PMID\s*:?\s*(\d+)", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")
_LEADING_CITATIONS_RE = re.compile(r"^(?:\[\s*PMID[^\]]*\]\s*[.;,]?\s*)+", re.IGNORECASE)
_NORMALISE_RE = re.compile(r"[^a-z ]")


@dataclass
class CitationReport:
    """Result of checking an answer's citations against the retrieved evidence."""

    cited_pmids: list[str] = field(default_factory=list)
    valid_pmids: list[str] = field(default_factory=list)
    invalid_pmids: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    sentence_count: int = 0
    uncited_sentences: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when at least one citation exists and all of them resolve."""
        return bool(self.valid_pmids) and not self.invalid_pmids

    def problems(self) -> str:
        """Human-readable description used in the correction prompt."""
        issues = []
        if self.invalid_pmids:
            issues.append(
                f"these cited PMIDs are not in the evidence: {', '.join(self.invalid_pmids)}"
            )
        if not self.valid_pmids:
            issues.append("the answer has no valid [PMID: <id>] citation")
        return "; ".join(issues) or "none"

    def to_dict(self) -> dict:
        return {
            "cited_pmids": self.cited_pmids,
            "valid_pmids": self.valid_pmids,
            "invalid_pmids": self.invalid_pmids,
            "sentence_count": self.sentence_count,
            "uncited_sentence_count": len(self.uncited_sentences),
        }


def extract_cited_pmids(answer: str) -> list[str]:
    """Return every PMID mentioned in ``answer``, in first-appearance order, de-duplicated."""
    return list(dict.fromkeys(_LOOSE_PMID_RE.findall(answer)))


def split_sentences(text: str) -> list[str]:
    """Rough sentence splitter; trailing citations stay attached to their sentence."""
    sentences: list[str] = []
    for part in _SENTENCE_RE.split(text.strip()):
        part = part.strip()
        # Citations placed after the full stop belong to the sentence before them.
        leading = _LEADING_CITATIONS_RE.match(part)
        if sentences and leading:
            sentences[-1] = f"{sentences[-1]} {leading.group(0).strip()}"
            part = part[leading.end() :].strip()
        if part:
            sentences.append(part)
    return sentences


def strip_citations(text: str) -> str:
    """Remove bracketed PMID citations (used before sending text to metrics/judges)."""
    return re.sub(r"\s+([.,;])", r"\1", _CITATION_RE.sub("", text)).strip()


def is_abstention(answer: str) -> bool:
    """True when the answer is (or contains) the standard abstention phrase."""

    def normalise(text: str) -> str:
        return " ".join(_NORMALISE_RE.sub(" ", text.lower()).split())

    return normalise(ABSTENTION_PHRASE) in normalise(answer)


def validate_citations(answer: str, evidence: list[EvidenceItem]) -> CitationReport:
    """Check that every cited PMID belongs to ``evidence``.

    ``uncited_sentences`` is a *heuristic* (sentences with no bracketed citation). It
    cannot tell whether a sentence is a factual claim; the LLM claim checker covers
    that, imperfectly.
    """
    by_pmid = {item.pmid: item for item in evidence}
    report = CitationReport(cited_pmids=extract_cited_pmids(answer))
    for pmid in report.cited_pmids:
        if pmid in by_pmid:
            item = by_pmid[pmid]
            report.valid_pmids.append(pmid)
            report.citations.append(Citation(pmid=pmid, title=item.title, url=item.pubmed_url))
        else:
            report.invalid_pmids.append(pmid)
    sentences = split_sentences(answer)
    report.sentence_count = len(sentences)
    report.uncited_sentences = [s for s in sentences if not _CITATION_RE.search(s)]
    return report
