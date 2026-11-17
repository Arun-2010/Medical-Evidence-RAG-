from __future__ import annotations

from app.retrieval.chunking import Chunk, chunk_article, chunk_articles, split_text
from tests.conftest import make_article


def test_short_abstract_is_one_chunk():
    article = make_article("123", "Title", "One sentence. Two sentences.")
    chunks = chunk_article(article, max_chars=4000)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "PMID_123_0"
    assert chunks[0].text == article.abstract


def test_long_abstract_splits_on_sentences_with_stable_ids():
    abstract = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = chunk_article(make_article("77", "Long", abstract), max_chars=200)
    assert len(chunks) > 1
    assert [c.chunk_id for c in chunks] == [f"PMID_77_{i}" for i in range(len(chunks))]
    assert all(len(c.text) <= 200 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)
    assert " ".join(c.text for c in chunks) == abstract


def test_metadata_preserved_on_every_chunk():
    abstract = " ".join(f"Sentence number {i} is here." for i in range(40))
    for chunk in chunk_article(make_article("77", "Long title", abstract, year=2018), 200):
        meta = chunk.metadata()
        assert meta["pmid"] == "77"
        assert meta["title"] == "Long title"
        assert meta["publication_year"] == 2018
        assert meta["journal"] == "Test Journal"
        assert meta["pubmed_url"].endswith("/77/")
        assert all(isinstance(v, str | int) for v in meta.values())


def test_metadata_omits_missing_values():
    article = make_article("5", "T", "Text.", year=None).model_copy(update={"journal": None})
    meta = chunk_article(article, 4000)[0].metadata()
    assert "publication_year" not in meta and "journal" not in meta


def test_record_round_trip():
    chunk = chunk_article(make_article("9", "T", "Some text."), 4000)[0]
    assert Chunk.from_record(chunk.to_record()) == chunk


def test_oversized_sentence_is_hard_split():
    pieces = split_text("x" * 450, 200)
    assert [len(p) for p in pieces] == [200, 200, 50]


def test_empty_text():
    assert split_text("   ", 100) == []


def test_chunk_articles_deduplicates_by_pmid():
    a = make_article("1", "First", "Text one.")
    b = make_article("1", "Second", "Text two.")
    chunks = chunk_articles([a, b], 4000)
    assert len(chunks) == 1 and chunks[0].title == "First"
