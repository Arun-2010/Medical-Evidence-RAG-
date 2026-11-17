from __future__ import annotations

import pytest

from app.core.exceptions import PubMedError
from app.ingestion.parser import parse_pubmed_xml
from tests.conftest import FIXTURES


@pytest.fixture(scope="module")
def parsed():
    return parse_pubmed_xml((FIXTURES / "efetch_sample.xml").read_text(encoding="utf-8"))


def test_counts(parsed):
    assert [a.pmid for a in parsed.articles] == ["1001", "1002", "1004"]
    assert parsed.skipped_no_abstract == 2  # 1003 and the book record
    assert parsed.malformed == 2  # no PMID, and no Article element


def test_structured_abstract_and_inline_markup(parsed):
    article = parsed.articles[0]
    assert article.title == "Inhaled corticosteroids and exacerbation risk."
    assert article.abstract == (
        "BACKGROUND: Asthma exacerbations are common. "
        "RESULTS: Exacerbations fell by 30% with H2O-based delivery."
    )
    assert article.publication_year == 2019
    assert article.journal == "Journal of Asthma Research"
    assert article.pubmed_url == "https://pubmed.ncbi.nlm.nih.gov/1001/"
    assert article.source == "pubmed"


def test_medline_date_and_whitespace(parsed):
    article = parsed.articles[1]
    assert article.publication_year == 2015
    assert article.abstract == "Single paragraph abstract text."


def test_missing_date_and_journal(parsed):
    article = parsed.articles[2]
    assert article.publication_year is None
    assert article.journal is None


def test_duplicate_pmid_keeps_first(parsed):
    assert sum(a.pmid == "1001" for a in parsed.articles) == 1
    assert "Duplicate" not in parsed.articles[0].title


def test_empty_set():
    result = parse_pubmed_xml("<PubmedArticleSet></PubmedArticleSet>")
    assert result.articles == [] and result.malformed == 0 and result.skipped_no_abstract == 0


def test_invalid_xml_raises():
    with pytest.raises(PubMedError):
        parse_pubmed_xml("<PubmedArticleSet><PubmedArticle>")
