"""HTTP routes. Handlers only translate between HTTP and the service layer."""

from __future__ import annotations

import secrets
from functools import lru_cache

from fastapi import APIRouter, Depends, Header, Request

from app.core.config import DISCLAIMER, Settings, get_settings
from app.core.exceptions import AppError
from app.generation.answer import AnswerService
from app.ingestion.pipeline import ingest_pubmed, load_articles
from app.retrieval.service import IndexStatus, RetrievalService
from app.schemas.api import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    HealthResponse,
    IndexStatusResponse,
    IngestRequest,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
)

router = APIRouter()

_ERRORS = {
    409: {"model": ErrorResponse, "description": "Index is empty; ingest documents first."},
    422: {"model": ErrorResponse, "description": "Invalid request."},
}
_LLM_ERRORS = {
    502: {"model": ErrorResponse, "description": "The LLM provider call failed."},
    503: {"model": ErrorResponse, "description": "No LLM API key is configured."},
}


class AdminForbiddenError(AppError):
    code = "forbidden"
    status_code = 403


@lru_cache(maxsize=1)
def _retrieval_singleton() -> RetrievalService:
    return RetrievalService(get_settings())


@lru_cache(maxsize=1)
def _answer_singleton() -> AnswerService:
    return AnswerService(get_settings(), _retrieval_singleton())


def get_app_settings() -> Settings:
    return get_settings()


def get_retrieval_service() -> RetrievalService:
    """Process-wide retrieval service (models load lazily on first query)."""
    return _retrieval_singleton()


def get_answer_service() -> AnswerService:
    return _answer_singleton()


def _status_response(status: IndexStatus) -> IndexStatusResponse:
    return IndexStatusResponse(
        ready=status.ready,
        dense_count=status.dense_count,
        bm25_count=status.bm25_count,
        in_sync=status.in_sync,
        collection=status.collection,
        message=status.message,
    )


@router.get("/health", response_model=HealthResponse, summary="Service health")
def health(settings: Settings = Depends(get_app_settings)) -> HealthResponse:
    """Liveness plus non-secret configuration. Never returns API keys."""
    return HealthResponse(
        app_env=settings.app_env,
        llm_configured=settings.llm_configured,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
        disclaimer=DISCLAIMER,
    )


@router.get("/index/status", response_model=IndexStatusResponse, summary="Index readiness")
def index_status(
    service: RetrievalService = Depends(get_retrieval_service),
) -> IndexStatusResponse:
    """Document counts for the dense and BM25 indexes."""
    return _status_response(service.status())


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    responses=_ERRORS,
    summary="Hybrid evidence retrieval (no generation)",
)
def retrieve(
    body: RetrieveRequest,
    request: Request,
    service: RetrievalService = Depends(get_retrieval_service),
) -> RetrieveResponse:
    """BM25 + dense retrieval, RRF fusion and cross-encoder reranking.

    `rerank_score` is a raw model score used for ordering, not a probability.
    """
    return RetrieveResponse(
        query=body.query,
        results=service.retrieve(body.query, body.top_k),
        request_id=request.state.request_id,
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={**_ERRORS, **_LLM_ERRORS},
    summary="Answer a research question from indexed abstracts",
)
def ask(
    body: AskRequest,
    request: Request,
    service: AnswerService = Depends(get_answer_service),
) -> AskResponse:
    """Retrieve evidence and answer only from it, with `[PMID: ...]` citations.

    Returns `Insufficient evidence in the indexed sources.` with `abstained=true`
    when evidence is weak or the draft answer fails citation/claim checks.
    Research/demo tool only - not medical advice.
    """
    result = service.ask(body.question, body.top_k)
    return AskResponse(
        **result.model_dump(), disclaimer=DISCLAIMER, request_id=request.state.request_id
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={403: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
    summary="Ingest PubMed abstracts and rebuild the index (admin)",
)
def ingest(
    body: IngestRequest,
    request: Request,
    x_admin_token: str | None = Header(default=None),
    settings: Settings = Depends(get_app_settings),
    service: RetrievalService = Depends(get_retrieval_service),
) -> IngestResponse:
    """Disabled unless `ADMIN_TOKEN` is set; requires it in the `X-Admin-Token` header."""
    if settings.admin_token is None:
        raise AdminForbiddenError(
            "Ingestion over HTTP is disabled. Set ADMIN_TOKEN or use scripts/ingest_pubmed.py."
        )
    expected = settings.admin_token.get_secret_value()
    if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise AdminForbiddenError("Missing or invalid X-Admin-Token header.")
    report = ingest_pubmed(settings, body.query, body.max_records, body.min_date, body.max_date)
    articles, _ = load_articles(settings.processed_articles_path)
    status = service.index_articles(articles)
    return IngestResponse(
        report=report.model_dump(),
        index=_status_response(status),
        request_id=request.state.request_id,
    )
