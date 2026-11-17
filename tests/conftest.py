"""Shared fixtures. Nothing here touches the network or downloads a model."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from app.core.config import Settings, load_settings
from app.ingestion.parser import Article
from app.retrieval.lexical import tokenize
from app.retrieval.service import RetrievalService
from app.schemas.api import EvidenceItem

FIXTURES = Path(__file__).parent / "fixtures"


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedding (no model download)."""

    dim = 64

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in tokenize(text):
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                vector[bucket] += 1.0
            norm = math.sqrt(sum(x * x for x in vector)) or 1.0
            vectors.append([x / norm for x in vector])
        return vectors


class FakeReranker:
    """Scores by the number of distinct query tokens found in the text."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        query_tokens = set(tokenize(query))
        return [float(len(query_tokens & set(tokenize(text)))) for text in texts]


class FakeLLM:
    """Scripted LLM provider: returns queued responses and records the prompts."""

    model = "fake-model"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        self.calls.append({"system": system, "user": user, "json_mode": json_mode})
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self.responses.pop(0)


def make_article(pmid: str, title: str, abstract: str, year: int | None = 2020) -> Article:
    return Article(
        pmid=pmid,
        title=title,
        abstract=abstract,
        publication_year=year,
        journal="Test Journal",
        pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    )


def make_evidence(
    pmid: str, rerank_score: float, rank: int = 1, abstract: str = ""
) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=f"PMID_{pmid}_0",
        pmid=pmid,
        pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        title=f"Title {pmid}",
        publication_year=2020,
        abstract=abstract or f"Abstract text for {pmid}.",
        rrf_score=0.03,
        rerank_score=rerank_score,
        final_rank=rank,
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated settings: temp data dirs, no ``.env``, no LLM key."""
    return load_settings(
        _env_file=None,
        app_env="test",
        gemini_api_key=None,
        openai_api_key=None,
        pubmed_api_key=None,
        pubmed_email="tester@example.org",
        admin_token=None,
        data_dir=tmp_path / "data",
        chroma_path=tmp_path / "data" / "chroma",
        chroma_collection="test_collection",
        top_k_rerank=3,
        top_k_fusion=10,
        min_rerank_score=1.0,
        min_evidence_count=2,
        pubmed_max_retries=2,
    )


@pytest.fixture
def articles() -> list[Article]:
    return [
        make_article(
            "111",
            "Inhaled corticosteroids in asthma",
            "Inhaled corticosteroids reduced asthma exacerbations in adults in a randomized trial.",
        ),
        make_article(
            "222",
            "Budesonide for childhood asthma",
            "Budesonide, an inhaled corticosteroid, improved asthma control in children.",
        ),
        make_article(
            "333",
            "Statins and cardiovascular risk",
            "Statin therapy lowered LDL cholesterol and reduced cardiovascular events.",
        ),
        make_article(
            "444",
            "Metformin in type 2 diabetes",
            "Metformin improved glycaemic control in patients with type 2 diabetes.",
        ),
    ]


@pytest.fixture
def retrieval_service(settings: Settings) -> RetrievalService:
    return RetrievalService(settings, embedder=FakeEmbedder(), reranker=FakeReranker())


@pytest.fixture
def indexed_service(
    retrieval_service: RetrievalService, articles: list[Article]
) -> RetrievalService:
    retrieval_service.index_articles(articles)
    return retrieval_service
