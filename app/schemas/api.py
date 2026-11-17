"""Pydantic request/response schemas shared by the service layer and the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_QUESTION_CHARS = 500

AbstentionReason = Literal[
    "no_evidence",
    "low_relevance",
    "insufficient_evidence_count",
    "model_reported_insufficient_evidence",
    "invalid_citations",
    "no_valid_citations",
    "claim_check_failed",
    "unsupported_claims",
    "uncited_claims",
]

ABSTENTION_MESSAGES: dict[str, str] = {
    "no_evidence": "No indexed abstracts matched the question.",
    "low_relevance": "The best-matching abstracts were below the relevance threshold.",
    "insufficient_evidence_count": "Too few sufficiently relevant abstracts were found.",
    "model_reported_insufficient_evidence": (
        "The retrieved abstracts do not support an answer, or they conflict."
    ),
    "invalid_citations": "The draft answer cited sources that were not retrieved.",
    "no_valid_citations": "The draft answer was not backed by valid citations.",
    "claim_check_failed": "The claim-level support check could not be completed.",
    "unsupported_claims": "The draft answer contained claims not supported by the abstracts.",
    "uncited_claims": "The draft answer contained factual claims without citations.",
}


class _Query(BaseModel):
    @field_validator("query", "question", mode="before", check_fields=False)
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class RetrieveRequest(_Query):
    """Evidence search request."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Do inhaled corticosteroids reduce asthma exacerbations?",
                "top_k": 5,
            }
        }
    )

    query: str = Field(
        min_length=3, max_length=MAX_QUESTION_CHARS, description="Research question."
    )
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Results to return (default: TOP_K_RERANK)."
    )


class AskRequest(_Query):
    """Question answering request."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"question": "Do inhaled corticosteroids reduce asthma exacerbations?"}
        }
    )

    question: str = Field(
        min_length=3, max_length=MAX_QUESTION_CHARS, description="Research question."
    )
    top_k: int | None = Field(default=None, ge=1, le=20)


class EvidenceItem(BaseModel):
    """One retrieved abstract (or abstract chunk) with full score provenance."""

    chunk_id: str
    pmid: str
    pubmed_url: str
    title: str
    publication_year: int | None = None
    journal: str | None = None
    abstract: str
    bm25_rank: int | None = Field(default=None, description="1-based BM25 rank, if retrieved.")
    bm25_score: float | None = None
    dense_rank: int | None = Field(default=None, description="1-based dense rank, if retrieved.")
    dense_distance: float | None = Field(
        default=None, description="Cosine distance (1 - cosine similarity); lower is closer."
    )
    rrf_score: float = Field(description="Reciprocal Rank Fusion score.")
    rerank_score: float = Field(
        description="Raw cross-encoder relevance score. Not a probability or confidence."
    )
    final_rank: int = Field(description="1-based rank after reranking.")


class RetrieveResponse(BaseModel):
    query: str
    results: list[EvidenceItem]
    request_id: str


class Citation(BaseModel):
    """A citation that resolved to a retrieved abstract."""

    pmid: str
    title: str
    url: str


class AskResponse(BaseModel):
    """Answer with citations, or the standard abstention phrase."""

    answer: str
    abstained: bool
    abstention_reason: AbstentionReason | None = None
    abstention_message: str | None = Field(
        default=None, description="Short user-facing explanation of the abstention."
    )
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    diagnostics: dict = Field(
        default_factory=dict, description="Citation/claim-check metadata for debugging."
    )
    disclaimer: str
    request_id: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    app_env: str
    llm_configured: bool
    llm_model: str
    embedding_model: str
    reranker_model: str
    disclaimer: str


class IndexStatusResponse(BaseModel):
    ready: bool
    dense_count: int
    bm25_count: int
    in_sync: bool = Field(description="True when both indexes hold the same number of chunks.")
    collection: str
    message: str


class IngestRequest(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    max_records: int = Field(default=50, ge=1, le=200)
    min_date: str | None = Field(default=None, pattern=r"^\d{4}(/\d{2}){0,2}$")
    max_date: str | None = Field(default=None, pattern=r"^\d{4}(/\d{2}){0,2}$")


class IngestResponse(BaseModel):
    report: dict
    index: IndexStatusResponse
    request_id: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: str
