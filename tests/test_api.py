from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api import routes
from app.core.config import ABSTENTION_PHRASE
from app.generation.answer import AnswerService
from app.ingestion.pipeline import IngestReport, save_articles
from app.main import create_app
from tests.conftest import FakeLLM

ANSWER = "Steroids reduced exacerbations [PMID: 111]. Control improved in children [PMID: 222]."
JUDGE_OK = json.dumps(
    {
        "claims": [
            {
                "claim": "Steroids reduced exacerbations",
                "cited_pmids": ["111"],
                "verdict": "supported",
            },
            {
                "claim": "Control improved in children",
                "cited_pmids": ["222"],
                "verdict": "supported",
            },
        ]
    }
)
QUESTION = "inhaled corticosteroids asthma exacerbations children"


def make_client(settings, retrieval, llm=None) -> TestClient:
    app = create_app()
    app.dependency_overrides[routes.get_app_settings] = lambda: settings
    app.dependency_overrides[routes.get_retrieval_service] = lambda: retrieval
    answer_service = AnswerService(settings, retrieval, provider=llm)
    app.dependency_overrides[routes.get_answer_service] = lambda: answer_service
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client(settings, indexed_service) -> TestClient:
    return make_client(settings, indexed_service, FakeLLM([ANSWER, JUDGE_OK]))


def test_health_exposes_no_secrets(settings, indexed_service):
    secret_settings = settings.model_copy(update={"gemini_api_key": SecretStr("sk-super-secret")})
    response = make_client(secret_settings, indexed_service).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["llm_configured"] is True
    assert "not a medical device" in body["disclaimer"]
    assert "sk-super-secret" not in response.text


def test_request_id_generated_and_echoed(client):
    generated = client.get("/health").headers["X-Request-ID"]
    assert len(generated) == 32
    assert (
        client.get("/health", headers={"X-Request-ID": "abc-123"}).headers["X-Request-ID"]
        == "abc-123"
    )
    unsafe = client.get("/health", headers={"X-Request-ID": "bad id <script>"})
    assert unsafe.headers["X-Request-ID"] != "bad id <script>"


def test_index_status(client):
    body = client.get("/index/status").json()
    assert body["ready"] is True and body["dense_count"] == 4 and body["bm25_count"] == 4


def test_retrieve(client):
    response = client.post("/retrieve", json={"query": QUESTION, "top_k": 2})
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2
    assert body["request_id"] == response.headers["X-Request-ID"]
    top = body["results"][0]
    assert top["pmid"] in {"111", "222"}
    for key in ("pubmed_url", "title", "abstract", "bm25_rank", "bm25_score", "dense_rank",
                "dense_distance", "rrf_score", "rerank_score", "final_rank"):  # fmt: skip
        assert key in top


def test_ask_returns_cited_answer(client):
    response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is False and body["answer"] == ANSWER
    assert {c["pmid"] for c in body["citations"]} == {"111", "222"}
    assert body["citations"][0]["url"].startswith("https://pubmed.ncbi.nlm.nih.gov/")
    assert body["evidence"] and body["request_id"] and body["disclaimer"]


def test_ask_abstains_on_weak_evidence(settings, indexed_service):
    client = make_client(settings, indexed_service, FakeLLM([]))
    body = client.post("/ask", json={"question": "zebrafish fin regeneration"}).json()
    assert body["abstained"] is True
    assert body["answer"] == ABSTENTION_PHRASE
    assert body["abstention_reason"] == "low_relevance"
    assert body["citations"] == []


def test_ask_rejects_fabricated_citations(settings, indexed_service):
    fabricated = "Steroids cure asthma [PMID: 99999999]."
    client = make_client(settings, indexed_service, FakeLLM([fabricated, fabricated]))
    body = client.post("/ask", json={"question": QUESTION}).json()
    assert body["abstained"] is True and body["abstention_reason"] == "invalid_citations"
    assert "99999999" not in body["answer"]


def test_ask_without_llm_key_returns_configuration_error(settings, indexed_service):
    client = make_client(settings, indexed_service, llm=None)
    response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "llm_not_configured"
    assert "GEMINI_API_KEY" in body["error"]["message"]
    # Retrieval keeps working without a key.
    assert client.post("/retrieve", json={"query": QUESTION}).status_code == 200


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/retrieve", {"query": "ab"}),
        ("/retrieve", {"query": "   "}),
        ("/retrieve", {"query": "x" * 501}),
        ("/retrieve", {"query": "valid question", "top_k": 0}),
        ("/retrieve", {}),
        ("/ask", {"question": ""}),
        ("/ask", {"question": "x" * 501}),
    ],
)
def test_validation_errors(client, path, payload):
    response = client.post(path, json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error" and body["request_id"]


def test_empty_index_behaviour(settings, retrieval_service):
    client = make_client(settings, retrieval_service, FakeLLM([]))
    status = client.get("/index/status").json()
    assert status["ready"] is False and "ingest documents first" in status["message"]
    for path, payload in (("/retrieve", {"query": QUESTION}), ("/ask", {"question": QUESTION})):
        response = client.post(path, json=payload)
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "index_empty"
        assert "ingest documents first" in response.json()["error"]["message"]


def test_unexpected_errors_hide_details(settings, indexed_service):
    class Exploding:
        def retrieve(self, *args, **kwargs):
            raise RuntimeError("secret internal detail")

    response = make_client(settings, Exploding()).post("/retrieve", json={"query": QUESTION})
    assert response.status_code == 500
    assert response.json()["error"] == {
        "code": "internal_error",
        "message": "Internal server error.",
    }
    assert "secret internal detail" not in response.text


def test_ingest_disabled_without_admin_token(client):
    response = client.post("/ingest", json={"query": "asthma"})
    assert response.status_code == 403 and response.json()["error"]["code"] == "forbidden"


def test_ingest_requires_matching_token(settings, indexed_service):
    protected = settings.model_copy(update={"admin_token": SecretStr("letmein")})
    client = make_client(protected, indexed_service)
    response = client.post("/ingest", json={"query": "asthma"}, headers={"X-Admin-Token": "nope"})
    assert response.status_code == 403
    assert client.post("/ingest", json={"query": "asthma"}).status_code == 403


def test_ingest_with_valid_token_runs_pipeline_and_reindexes(
    settings, indexed_service, articles, monkeypatch
):
    protected = settings.model_copy(update={"admin_token": SecretStr("letmein")})
    save_articles(protected.processed_articles_path, articles[:3])
    calls = []

    def fake_ingest(cfg, query, max_records, min_date, max_date):
        calls.append((query, max_records))
        return IngestReport(query=query, pmids_found=3, total_stored=3)

    monkeypatch.setattr(routes, "ingest_pubmed", fake_ingest)
    response = make_client(protected, indexed_service).post(
        "/ingest",
        json={"query": "asthma", "max_records": 3},
        headers={"X-Admin-Token": "letmein"},
    )
    assert response.status_code == 200
    assert calls == [("asthma", 3)]
    assert response.json()["index"]["dense_count"] == 3


def test_openapi_documents_endpoints(client):
    spec = client.get("/openapi.json").json()
    assert {"/health", "/index/status", "/retrieve", "/ask", "/ingest"} <= set(spec["paths"])
    assert "not a medical device" in spec["info"]["description"]
