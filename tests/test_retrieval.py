from __future__ import annotations

import pytest

from app.core.exceptions import IndexEmptyError
from app.retrieval.chunking import chunk_articles
from app.retrieval.dense import DenseIndex
from app.retrieval.lexical import BM25Index, tokenize
from app.retrieval.reranker import rerank
from app.retrieval.service import RetrievalService
from tests.conftest import FakeEmbedder, FakeReranker, make_article


# --- BM25 -----------------------------------------------------------------
def test_tokenize_handles_case_and_punctuation():
    assert tokenize("Asthma, COPD & IL-5 (2020)!") == ["asthma", "copd", "il", "5", "2020"]


def test_bm25_empty_corpus_and_empty_query(articles):
    assert BM25Index().search("asthma", 5) == []
    assert BM25Index([]).search("asthma", 5) == []
    index = BM25Index(chunk_articles(articles, 4000))
    assert index.search("", 5) == []
    assert index.search("?!", 5) == []


def test_bm25_ranking(articles):
    index = BM25Index(chunk_articles(articles, 4000))
    hits = index.search("statin cholesterol", 5)
    assert [h.chunk.pmid for h in hits] == ["333"]
    assert hits[0].score > 0
    hits = index.search("ASTHMA children budesonide", 5)
    assert [h.chunk.pmid for h in hits] == ["222", "111"]
    assert hits[0].score > hits[1].score
    assert hits[0].chunk.title == "Budesonide for childhood asthma"


def test_bm25_no_match_returns_empty(articles):
    assert BM25Index(chunk_articles(articles, 4000)).search("zebrafish", 5) == []


def test_bm25_persistence_round_trip(articles, tmp_path):
    index = BM25Index(chunk_articles(articles, 4000))
    path = tmp_path / "bm25.json"
    index.save(path)
    loaded = BM25Index.load(path)
    assert loaded is not None and len(loaded) == len(index)
    assert [h.chunk.chunk_id for h in loaded.search("asthma", 5)] == [
        h.chunk.chunk_id for h in index.search("asthma", 5)
    ]


def test_bm25_load_missing_or_incompatible(tmp_path):
    assert BM25Index.load(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert BM25Index.load(bad) is None
    bad.write_text('{"version": 999, "chunks": [], "tokens": []}', encoding="utf-8")
    assert BM25Index.load(bad) is None


# --- Dense ----------------------------------------------------------------
def test_dense_empty_collection(tmp_path):
    index = DenseIndex(tmp_path / "chroma", "empty_collection", FakeEmbedder())
    assert index.count() == 0
    assert index.search("asthma", 5) == []


def test_dense_search_and_upsert_idempotent(articles, tmp_path):
    index = DenseIndex(tmp_path / "chroma", "dense_collection", FakeEmbedder())
    chunks = chunk_articles(articles, 4000)
    index.upsert(chunks)
    index.upsert(chunks)
    assert index.count() == len(chunks)
    hits = index.search("statin therapy cholesterol cardiovascular", 2)
    assert hits[0].chunk.pmid == "333"
    assert hits[0].chunk.title == "Statins and cardiovascular risk"
    assert hits[0].chunk.publication_year == 2020
    assert 0.0 <= hits[0].distance <= hits[1].distance <= 2.0
    assert index.search("   ", 2) == []


def test_dense_delete_missing(articles, tmp_path):
    index = DenseIndex(tmp_path / "chroma", "prune_collection", FakeEmbedder())
    chunks = chunk_articles(articles, 4000)
    index.upsert(chunks)
    assert index.delete_missing({chunks[0].chunk_id}) == len(chunks) - 1
    assert [c.chunk_id for c in index.all_chunks()] == [chunks[0].chunk_id]


# --- Reranker -------------------------------------------------------------
def test_rerank_orders_descending_and_keeps_indices():
    texts = ["nothing relevant", "asthma only", "asthma and corticosteroids"]
    ranked = rerank(FakeReranker(), "asthma corticosteroids", texts)
    assert [i for i, _ in ranked] == [2, 1, 0]
    assert [s for _, s in ranked] == [2.0, 1.0, 0.0]
    assert rerank(FakeReranker(), "asthma corticosteroids", texts, top_k=1) == [(2, 2.0)]


def test_rerank_empty_candidates():
    assert rerank(FakeReranker(), "asthma", []) == []


def test_rerank_rejects_wrong_score_count():
    class Broken:
        def score(self, query, texts):
            return [1.0]

    with pytest.raises(ValueError):
        rerank(Broken(), "q", ["a", "b"])


# --- Service --------------------------------------------------------------
def test_retrieve_on_empty_index_raises(retrieval_service):
    status = retrieval_service.status()
    assert not status.ready and "empty" in status.message.lower()
    with pytest.raises(IndexEmptyError):
        retrieval_service.retrieve("asthma")


def test_hybrid_retrieval_returns_traceable_results(indexed_service):
    results = indexed_service.retrieve("inhaled corticosteroids asthma exacerbations adults")
    assert results[0].pmid == "111"
    assert [r.final_rank for r in results] == list(range(1, len(results) + 1))
    assert [r.rerank_score for r in results] == sorted(
        (r.rerank_score for r in results), reverse=True
    )
    top = results[0]
    assert top.pubmed_url == "https://pubmed.ncbi.nlm.nih.gov/111/"
    assert top.bm25_rank == 1 and top.bm25_score > 0
    assert top.dense_rank is not None and top.dense_distance is not None
    assert top.rrf_score > 0
    assert len({r.pmid for r in results}) == len(results)
    assert len(results) <= 3  # TOP_K_RERANK in test settings


def test_dense_only_hit_has_no_bm25_fields(indexed_service):
    results = indexed_service.retrieve("zebrafish", top_k=4)
    assert results, "dense retrieval should still return nearest neighbours"
    assert all(r.bm25_rank is None and r.bm25_score is None for r in results)
    assert all(r.dense_rank is not None for r in results)


def test_reindexing_is_idempotent_and_prunes(indexed_service, articles):
    status = indexed_service.index_articles(articles)
    assert status.dense_count == status.bm25_count == 4 and status.ready
    status = indexed_service.index_articles(articles[:2])
    assert status.dense_count == status.bm25_count == 2


def test_multi_chunk_article_returned_once(retrieval_service, settings):
    long_abstract = " ".join(f"Asthma finding number {i} was reported." for i in range(300))
    assert len(long_abstract) > settings.max_abstract_chars
    retrieval_service.index_articles(
        [
            make_article("900", "Long asthma review", long_abstract),
            make_article("901", "Other", "Asthma."),
        ]
    )
    results = retrieval_service.retrieve("asthma finding", top_k=3)
    assert sorted(r.pmid for r in results) == ["900", "901"]


def test_bm25_rebuilt_from_chroma_when_artifact_missing(indexed_service, settings):
    settings.bm25_path.unlink()
    fresh = RetrievalService(settings, embedder=FakeEmbedder(), reranker=FakeReranker())
    assert fresh.status().ready
    assert settings.bm25_path.exists()
    assert fresh.retrieve("statin cholesterol")[0].pmid == "333"
