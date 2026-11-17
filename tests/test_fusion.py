from __future__ import annotations

import pytest

from app.retrieval.fusion import reciprocal_rank_fusion


def test_one_based_ranks_and_scores():
    fused = reciprocal_rank_fusion({"bm25": ["a", "b"], "dense": ["b", "c"]}, k=60)
    by_id = {h.doc_id: h for h in fused}
    assert by_id["a"].ranks == {"bm25": 1}
    assert by_id["a"].rrf_score == pytest.approx(1 / 61)
    assert by_id["b"].ranks == {"bm25": 2, "dense": 1}
    assert by_id["b"].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert by_id["c"].rrf_score == pytest.approx(1 / 62)


def test_ordering_descending_with_deterministic_ties():
    fused = reciprocal_rank_fusion({"bm25": ["a", "b", "c"], "dense": ["c", "b", "a"]}, k=1)
    # a and c tie (1/2 + 1/4); b scores 1/3 + 1/3.
    assert [h.doc_id for h in fused] == ["a", "c", "b"]
    assert fused[0].rrf_score == pytest.approx(fused[1].rrf_score)
    assert fused[1].rrf_score > fused[2].rrf_score


def test_document_in_both_lists_beats_single_list():
    fused = reciprocal_rank_fusion({"bm25": ["x", "both"], "dense": ["y", "both"]})
    assert fused[0].doc_id == "both"
    assert len(fused) == 3


def test_duplicates_within_a_list_count_once():
    fused = reciprocal_rank_fusion({"bm25": ["a", "a", "b"]}, k=60)
    by_id = {h.doc_id: h for h in fused}
    assert len(fused) == 2
    assert by_id["a"].rrf_score == pytest.approx(1 / 61)
    assert by_id["b"].ranks == {"bm25": 2}


def test_empty_inputs():
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"bm25": [], "dense": []}) == []
    assert [h.doc_id for h in reciprocal_rank_fusion({"bm25": [], "dense": ["a"]})] == ["a"]


def test_top_k_and_invalid_k():
    assert len(reciprocal_rank_fusion({"bm25": ["a", "b", "c"]}, top_k=2)) == 2
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"bm25": ["a"]}, k=0)
