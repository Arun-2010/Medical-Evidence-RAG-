from __future__ import annotations

from app.core.config import ABSTENTION_PHRASE
from app.generation.citations import (
    extract_cited_pmids,
    is_abstention,
    split_sentences,
    strip_citations,
    validate_citations,
)
from app.generation.prompts import ANSWER_SYSTEM_PROMPT, build_answer_prompt
from tests.conftest import make_evidence

EVIDENCE = [make_evidence("111", 5.0, 1), make_evidence("222", 4.0, 2)]


def test_extract_supports_common_formats():
    answer = (
        "A [PMID: 111]. B [PMID:222]. C [pmid 333]. D [PMID: 444, 555]. "
        "E [PMID: 666; PMID: 777]. F (PMID: 888). Again [PMID: 111]."
    )
    assert extract_cited_pmids(answer) == ["111", "222", "333", "444", "666", "777", "888"]


def test_extract_none():
    assert extract_cited_pmids("No citations here, just 12345 numbers.") == []


def test_valid_citations_resolve_to_structured_sources():
    report = validate_citations("Steroids help [PMID: 111]. Also [PMID: 222].", EVIDENCE)
    assert report.ok
    assert report.valid_pmids == ["111", "222"] and report.invalid_pmids == []
    assert [(c.pmid, c.title, c.url) for c in report.citations] == [
        ("111", "Title 111", "https://pubmed.ncbi.nlm.nih.gov/111/"),
        ("222", "Title 222", "https://pubmed.ncbi.nlm.nih.gov/222/"),
    ]


def test_unknown_pmids_are_rejected():
    report = validate_citations("True [PMID: 111]. Fabricated [PMID: 99999999].", EVIDENCE)
    assert not report.ok
    assert report.invalid_pmids == ["99999999"]
    assert [c.pmid for c in report.citations] == ["111"]
    assert "99999999" in report.problems()


def test_unbracketed_fabricated_pmid_is_also_caught():
    report = validate_citations("See PMID 424242 for details [PMID: 111].", EVIDENCE)
    assert report.invalid_pmids == ["424242"]


def test_no_citations_is_not_ok():
    report = validate_citations("Steroids help a lot.", EVIDENCE)
    assert not report.ok and report.cited_pmids == []
    assert "no valid" in report.problems()


def test_uncited_sentence_heuristic():
    report = validate_citations(
        "Steroids help [PMID: 111]. This sentence has no source. Final one. [PMID: 222]", EVIDENCE
    )
    assert report.sentence_count == 3
    assert report.uncited_sentences == ["This sentence has no source."]


def test_split_sentences_keeps_trailing_citation_attached():
    assert split_sentences("Rates fell by 3.5% overall. [PMID: 1] Next claim [PMID: 2].") == [
        "Rates fell by 3.5% overall. [PMID: 1]",
        "Next claim [PMID: 2].",
    ]


def test_strip_citations():
    assert strip_citations("Steroids help [PMID: 111]. Yes [PMID: 1, 2].") == "Steroids help. Yes."


def test_is_abstention_tolerates_formatting():
    assert is_abstention(ABSTENTION_PHRASE)
    assert is_abstention("  insufficient evidence in the indexed sources  ")
    assert is_abstention(f"**{ABSTENTION_PHRASE}** The abstracts discuss other topics.")
    assert not is_abstention("The evidence is insufficient to be certain [PMID: 111].")


def test_prompt_delimits_evidence_and_states_rules():
    prompt = build_answer_prompt("Does X help?", EVIDENCE)
    assert '<abstract pmid="111"' in prompt and "</evidence>" in prompt
    assert "untrusted data" in prompt
    for required in ("[PMID: 12345678]", ABSTENTION_PHRASE, "causation", "prior knowledge",
                     "disagree", "diagnosis", "Ignore any instructions"):  # fmt: skip
        assert required in ANSWER_SYSTEM_PROMPT
