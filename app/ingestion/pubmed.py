"""Thin NCBI E-utilities client (ESearch + EFetch) with pacing and retries."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import httpx

from app.core.config import Settings
from app.core.exceptions import PubMedError

logger = logging.getLogger(__name__)

EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
USER_AGENT = "medical-evidence-rag/0.1 (research demo; httpx)"
EFETCH_BATCH_SIZE = 100
ESEARCH_PAGE_SIZE = 500
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class PubMedClient:
    """Client for PubMed search and abstract retrieval.

    NCBI allows roughly 3 requests/second without an API key and 10 with one; this
    client stays below both by spacing requests, and retries transient failures
    with exponential backoff.
    """

    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._sleep = sleep
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(settings.request_timeout_seconds),
            headers={"User-Agent": USER_AGENT},
        )
        self._min_interval = 0.12 if settings.pubmed_api_key else 0.4
        self._last_request = 0.0

    def close(self) -> None:
        """Close the underlying HTTP client if this instance created it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> PubMedClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _common_params(self) -> dict[str, str]:
        params = {"db": "pubmed", "tool": "medical-evidence-rag"}
        if self._settings.pubmed_email:
            params["email"] = self._settings.pubmed_email
        if self._settings.pubmed_api_key:
            params["api_key"] = self._settings.pubmed_api_key.get_secret_value()
        return params

    def _request(self, endpoint: str, params: dict[str, str]) -> httpx.Response:
        """POST to an E-utilities endpoint with pacing, retries and backoff."""
        attempts = self._settings.pubmed_max_retries + 1
        last_error = "unknown error"
        for attempt in range(attempts):
            wait = self._min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                self._sleep(wait)
            self._last_request = time.monotonic()
            try:
                response = self._client.post(
                    f"{EUTILS_BASE_URL}/{endpoint}", data={**self._common_params(), **params}
                )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}"
            else:
                if response.status_code == 200:
                    return response
                last_error = f"HTTP {response.status_code}"
                if response.status_code not in _RETRYABLE_STATUS:
                    break
            if attempt < attempts - 1:
                delay = min(2.0**attempt, 30.0)
                logger.warning(
                    "PubMed %s failed (%s); retry %d/%d in %.0fs",
                    endpoint,
                    last_error,
                    attempt + 1,
                    attempts - 1,
                    delay,
                )
                self._sleep(delay)
        raise PubMedError(f"PubMed {endpoint} request failed: {last_error}")

    def search(
        self,
        query: str,
        max_records: int,
        min_date: str | None = None,
        max_date: str | None = None,
    ) -> list[str]:
        """Return up to ``max_records`` de-duplicated PMIDs for ``query``.

        ``min_date``/``max_date`` use NCBI's ``YYYY``, ``YYYY/MM`` or ``YYYY/MM/DD``
        formats and filter on publication date. An open end defaults to 1800 / 3000.
        """
        if not query.strip():
            raise PubMedError("PubMed search query must not be empty.")
        if max_records < 1:
            raise PubMedError("max_records must be at least 1.")
        pmids: list[str] = []
        seen: set[str] = set()
        start = 0
        while len(pmids) < max_records:
            params = {
                "term": query,
                "retmode": "json",
                "retstart": str(start),
                "retmax": str(min(ESEARCH_PAGE_SIZE, max_records - len(pmids))),
                "sort": "relevance",
            }
            if min_date or max_date:
                params.update(
                    datetype="pdat", mindate=min_date or "1800", maxdate=max_date or "3000"
                )
            response = self._request("esearch.fcgi", params)
            try:
                result = response.json()["esearchresult"]
                page = [str(p) for p in result.get("idlist", [])]
                total = int(result.get("count", 0))
            except (ValueError, KeyError, TypeError) as exc:
                raise PubMedError("PubMed ESearch returned an unexpected response.") from exc
            if "ERROR" in result:
                raise PubMedError(f"PubMed ESearch rejected the query: {result['ERROR']}")
            for pmid in page:
                if pmid.isdigit() and pmid not in seen:
                    seen.add(pmid)
                    pmids.append(pmid)
            start += len(page)
            if not page or start >= total:
                break
        logger.info("PubMed search returned %d PMIDs", len(pmids))
        return pmids[:max_records]

    def fetch_xml(self, pmids: list[str]) -> list[str]:
        """Fetch article XML for ``pmids`` in batches; one XML document per batch."""
        documents: list[str] = []
        for i in range(0, len(pmids), EFETCH_BATCH_SIZE):
            batch = pmids[i : i + EFETCH_BATCH_SIZE]
            response = self._request(
                "efetch.fcgi", {"id": ",".join(batch), "retmode": "xml", "rettype": "abstract"}
            )
            documents.append(response.text)
        return documents
