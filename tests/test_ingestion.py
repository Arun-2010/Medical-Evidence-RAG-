from __future__ import annotations

import json
from urllib.parse import parse_qs

import httpx
import pytest

from app.core.exceptions import PubMedError
from app.ingestion.pipeline import ingest_pubmed, load_articles
from app.ingestion.pubmed import PubMedClient
from tests.conftest import FIXTURES

SAMPLE_XML = (FIXTURES / "efetch_sample.xml").read_text(encoding="utf-8")


class FakePubMed:
    """httpx.MockTransport handler emulating ESearch/EFetch."""

    def __init__(self, idlist: list[str], failures: int = 0, fail_status: int = 503) -> None:
        self.idlist = idlist
        self.failures = failures
        self.fail_status = fail_status
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        params = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        self.requests.append((request.url.path, params))
        if self.failures > 0:
            self.failures -= 1
            return httpx.Response(self.fail_status)
        if request.url.path.endswith("esearch.fcgi"):
            start = int(params["retstart"])
            page = self.idlist[start : start + int(params["retmax"])]
            body = {"esearchresult": {"count": str(len(self.idlist)), "idlist": page}}
            return httpx.Response(200, json=body)
        return httpx.Response(200, text=SAMPLE_XML)

    def count(self, endpoint: str) -> int:
        return sum(path.endswith(endpoint) for path, _ in self.requests)


def make_client(settings, handler) -> tuple[PubMedClient, list[float]]:
    sleeps: list[float] = []
    client = PubMedClient(
        settings, client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=sleeps.append
    )
    return client, sleeps


def test_search_deduplicates_and_sends_identity(settings):
    handler = FakePubMed(["1", "2", "2", "3"])
    client, _ = make_client(settings, handler)
    assert client.search("asthma", 10, min_date="2015") == ["1", "2", "3"]
    params = handler.requests[0][1]
    assert params["email"] == "tester@example.org"
    assert params["term"] == "asthma"
    assert params["mindate"] == "2015" and params["datetype"] == "pdat"
    assert "api_key" not in params


def test_search_respects_max_records(settings):
    client, _ = make_client(settings, FakePubMed([str(i) for i in range(1, 50)]))
    assert len(client.search("asthma", 5)) == 5


def test_retries_transient_errors_with_backoff(settings):
    handler = FakePubMed(["1"], failures=2)
    client, sleeps = make_client(settings, handler)
    assert client.search("asthma", 5) == ["1"]
    assert handler.count("esearch.fcgi") == 3
    assert [s for s in sleeps if s >= 1.0] == [1.0, 2.0]


def test_gives_up_after_max_retries(settings):
    handler = FakePubMed(["1"], failures=99)
    client, _ = make_client(settings, handler)
    with pytest.raises(PubMedError, match="HTTP 503"):
        client.search("asthma", 5)
    assert handler.count("esearch.fcgi") == settings.pubmed_max_retries + 1


def test_does_not_retry_client_errors(settings):
    handler = FakePubMed(["1"], failures=99, fail_status=400)
    client, _ = make_client(settings, handler)
    with pytest.raises(PubMedError, match="HTTP 400"):
        client.search("asthma", 5)
    assert handler.count("esearch.fcgi") == 1


def test_network_error_becomes_pubmed_error(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    client, _ = make_client(settings, handler)
    with pytest.raises(PubMedError, match="ConnectError"):
        client.search("asthma", 5)


def test_empty_query_rejected(settings):
    client, _ = make_client(settings, FakePubMed([]))
    with pytest.raises(PubMedError):
        client.search("   ", 5)


def test_ingest_empty_search_results(settings):
    handler = FakePubMed([])
    client, _ = make_client(settings, handler)
    report = ingest_pubmed(settings, "nothing matches", 10, client=client)
    assert report.pmids_found == 0 and report.added == 0 and report.total_stored == 0
    assert handler.count("efetch.fcgi") == 0
    assert not settings.processed_articles_path.exists()


def test_ingest_stores_normalized_articles_and_reports_skips(settings):
    client, _ = make_client(settings, FakePubMed(["1001", "1002", "1003", "1004"]))
    report = ingest_pubmed(settings, "asthma", 10, client=client)
    assert report.added == 3
    assert report.skipped_no_abstract == 2
    assert report.malformed == 2
    rows = [
        json.loads(line)
        for line in settings.processed_articles_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [r["pmid"] for r in rows] == ["1001", "1002", "1004"]
    assert set(rows[0]) == {
        "pmid", "title", "abstract", "publication_year", "journal", "pubmed_url", "source"
    }  # fmt: skip
    assert list(settings.raw_dir.glob("efetch_*.xml"))


def test_ingest_is_idempotent_and_skips_stored_pmids(settings):
    handler = FakePubMed(["1001", "1002", "1004"])
    client, _ = make_client(settings, handler)
    ingest_pubmed(settings, "asthma", 10, client=client)
    report = ingest_pubmed(settings, "asthma", 10, client=client)
    assert report.already_stored == 3 and report.fetched == 0 and report.added == 0
    assert handler.count("efetch.fcgi") == 1
    articles, bad = load_articles(settings.processed_articles_path)
    assert len(articles) == 3 and bad == 0


def test_load_articles_skips_bad_rows_and_duplicates(tmp_path):
    path = tmp_path / "articles.jsonl"
    good = {
        "pmid": "1", "title": "T", "abstract": "A", "publication_year": None,
        "journal": None, "pubmed_url": "u", "source": "pubmed",
    }  # fmt: skip
    path.write_text(
        "\n".join([json.dumps(good), "not json", json.dumps({"pmid": "x"}), json.dumps(good)]),
        encoding="utf-8",
    )
    articles, bad = load_articles(path)
    assert len(articles) == 1 and bad == 2
