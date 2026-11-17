"""Streamlit UI. A thin client: all RAG logic lives behind the FastAPI backend.

Run with:  streamlit run app/ui.py
The backend URL is read from the API_BASE_URL environment variable.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
DISCLAIMER = (
    "Research/demo tool only. This is not a medical device and not a substitute for "
    "professional medical advice, diagnosis, or treatment. It summarises indexed PubMed "
    "abstracts and has not reviewed any patient's medical record."
)
ASK_TIMEOUT_SECONDS = 180.0


def call_api(method: str, path: str, payload: dict | None = None, timeout: float = 10.0):
    """Return ``(data, error_message)``; exactly one of them is ``None``."""
    try:
        response = httpx.request(method, f"{API_BASE_URL}{path}", json=payload, timeout=timeout)
    except httpx.TimeoutException:
        return None, "The API took too long to respond. Try again in a moment."
    except httpx.HTTPError:
        return None, (
            f"Cannot reach the API at {API_BASE_URL}. Start it with: uvicorn app.main:app --reload"
        )
    try:
        data = response.json()
    except ValueError:
        return None, f"The API returned an unexpected response (HTTP {response.status_code})."
    if response.status_code >= 400:
        error = data.get("error", {}) if isinstance(data, dict) else {}
        return None, error.get("message") or f"Request failed (HTTP {response.status_code})."
    return data, None


def fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_evidence(evidence: list[dict]) -> None:
    st.subheader(f"Retrieved evidence ({len(evidence)})")
    for item in evidence:
        year = item.get("publication_year") or "n.d."
        with st.expander(f"{item['final_rank']}. {item['title']} ({year})"):
            st.markdown(f"[PMID {item['pmid']}]({item['pubmed_url']})")
            if item.get("journal"):
                st.caption(item["journal"])
            abstract = item["abstract"]
            st.write(abstract if len(abstract) <= 1200 else abstract[:1200] + " ...")
            cols = st.columns(4)
            cols[0].metric("Reranker score (raw)", fmt(item["rerank_score"], 2))
            cols[1].metric("RRF score", fmt(item["rrf_score"], 4))
            cols[2].metric(
                "BM25 rank / score",
                f"{item.get('bm25_rank') or 'n/a'} / {fmt(item.get('bm25_score'), 2)}",
            )
            cols[3].metric(
                "Dense rank / distance",
                f"{item.get('dense_rank') or 'n/a'} / {fmt(item.get('dense_distance'))}",
            )
    if evidence:
        st.caption(
            "Reranker scores are raw model outputs used for ordering. "
            "They are not probabilities or confidence levels."
        )


def main() -> None:
    st.set_page_config(page_title="Medical Evidence RAG", page_icon=":microscope:")
    st.title("Medical Evidence RAG")
    st.warning(DISCLAIMER)

    health, health_error = call_api("GET", "/health")
    if health_error:
        st.error(health_error)
        st.stop()
    status, status_error = call_api("GET", "/index/status", timeout=60.0)

    with st.sidebar:
        st.header("Status")
        if status_error:
            st.error(status_error)
        elif status["ready"]:
            st.success(f"{status['dense_count']} abstracts indexed")
        else:
            st.error(status["message"])
        if health["llm_configured"]:
            st.success(f"LLM: {health['llm_model']}")
        else:
            st.error("LLM key missing: set GEMINI_API_KEY in .env. Evidence search still works.")
        st.caption(f"Embedding: {health['embedding_model']}")
        st.caption(f"Reranker: {health['reranker_model']}")

    with st.form("ask"):
        question = st.text_input(
            "Research question",
            max_chars=500,
            placeholder="Do inhaled corticosteroids reduce asthma exacerbations?",
        )
        mode = st.radio("Mode", ["Answer with citations", "Evidence search only"], horizontal=True)
        submitted = st.form_submit_button("Ask")

    if not submitted:
        return
    if len(question.strip()) < 3:
        st.error("Please enter a question (at least 3 characters).")
        return

    if mode == "Evidence search only":
        with st.spinner("Searching indexed abstracts..."):
            data, error = call_api(
                "POST", "/retrieve", {"query": question}, timeout=ASK_TIMEOUT_SECONDS
            )
        if error:
            st.error(error)
            return
        render_evidence(data["results"])
        return

    with st.spinner("Retrieving evidence and generating an answer..."):
        data, error = call_api("POST", "/ask", {"question": question}, timeout=ASK_TIMEOUT_SECONDS)
    if error:
        st.error(error)
        return

    st.subheader("Answer")
    if data["abstained"]:
        st.info(data["answer"])
        if data.get("abstention_message"):
            st.caption(f"Why: {data['abstention_message']}")
    else:
        st.write(data["answer"])
        st.subheader("Sources")
        for citation in data["citations"]:
            st.markdown(f"- [PMID {citation['pmid']}]({citation['url']}) - {citation['title']}")
    render_evidence(data["evidence"])
    st.caption(f"Request ID: {data['request_id']}")


main()
