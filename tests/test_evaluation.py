from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app.core.exceptions import DatasetError
from app.evaluation import deterministic as det
from app.evaluation.dataset import (
    EvalExample,
    example_to_article,
    load_pubmedqa,
    load_unanswerable,
    split_pmids,
)
from app.evaluation.judge import parse_judge_output
from app.evaluation.ragas_adapter import RagasAdapter
from app.evaluation.runner import EvalConfig, run_evaluation, summarize
from tests.conftest import FakeLLM


# --- Dataset ----------------------------------------------------------------
def raw_row(question: str = "Does X improve Y?", decision: str = "yes") -> dict:
    return {
        "QUESTION": question,
        "CONTEXTS": ["Background text.", "Results text."],
        "YEAR": "2012",
        "final_decision": decision,
        "LONG_ANSWER": "X improved Y.",
    }


@pytest.fixture
def pubmedqa_dir(tmp_path):
    data = {str(1000 + i): raw_row(f"Does treatment {i} work?") for i in range(130)}
    data["2001"] = {"QUESTION": "No contexts?", "CONTEXTS": [], "final_decision": "yes"}
    data["2002"] = {"QUESTION": "Bad label?", "CONTEXTS": ["c"], "final_decision": "perhaps"}
    data["2003"] = "not an object"
    data["2004"] = {"CONTEXTS": ["c"], "final_decision": "no"}
    test_ids = {str(1000 + i): "yes" for i in range(20)}
    (tmp_path / "ori_pqal.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "test_ground_truth.json").write_text(json.dumps(test_ids), encoding="utf-8")
    return tmp_path


def test_load_pubmedqa_normalizes_and_counts_malformed(pubmedqa_dir):
    test, test_skipped = load_pubmedqa(pubmedqa_dir, "test")
    dev, dev_skipped = load_pubmedqa(pubmedqa_dir, "dev")
    validation, val_skipped = load_pubmedqa(pubmedqa_dir, "validation")
    assert len(test) == 20 and test_skipped == 0
    assert len(dev) + len(validation) == 110
    assert dev_skipped + val_skipped == 4
    example = test[0]
    assert example.kind == "pubmedqa" and example.expected_abstain is False
    assert example.gold_pmid == "1000" and example.id == "pubmedqa-1000"
    assert example.contexts == ["Background text.", "Results text."]
    assert example.reference_answer == "X improved Y." and example.label == "yes"
    assert example.year == 2012


def test_splits_are_disjoint_and_deterministic(pubmedqa_dir):
    ids = {
        split: {e.id for e in load_pubmedqa(pubmedqa_dir, split)[0]}
        for split in ("dev", "validation", "test")
    }
    assert not ids["dev"] & ids["validation"]
    assert not ids["dev"] & ids["test"]
    assert not ids["validation"] & ids["test"]
    again = {e.id for e in load_pubmedqa(pubmedqa_dir, "dev")[0]}
    assert again == ids["dev"]
    pmids = [str(i) for i in range(300)]
    assert split_pmids(pmids, {"1", "2"}, seed=1) == split_pmids(pmids[::-1], {"1", "2"}, seed=1)
    assert (
        split_pmids(pmids, {"1", "2"}, seed=1)["dev"]
        != split_pmids(pmids, {"1", "2"}, seed=2)["dev"]
    )


def test_limit_and_missing_dataset(pubmedqa_dir, tmp_path):
    assert len(load_pubmedqa(pubmedqa_dir, "test", limit=5)[0]) == 5
    with pytest.raises(DatasetError, match="download_pubmedqa"):
        load_pubmedqa(tmp_path / "missing", "dev")
    (pubmedqa_dir / "ori_pqal.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(DatasetError, match="corrupt"):
        load_pubmedqa(pubmedqa_dir, "dev")


def test_load_unanswerable_marks_synthetic_and_skips_bad_lines(tmp_path):
    path = tmp_path / "stress.jsonl"
    path.write_text(
        '{"id": "s1", "question": "What is the capital of France?"}\n'
        "not json\n"
        '{"id": "s2"}\n'
        '{"id": "s3", "question": ""}\n\n',
        encoding="utf-8",
    )
    examples, skipped = load_unanswerable(path)
    assert [e.id for e in examples] == ["s1"] and skipped == 3
    assert examples[0].kind == "synthetic_unanswerable" and examples[0].expected_abstain is True
    assert load_unanswerable(tmp_path / "missing.jsonl") == ([], 0)


def test_bundled_stress_set_is_well_formed():
    from pathlib import Path

    examples, skipped = load_unanswerable(Path("data/eval/unanswerable_stress.jsonl"))
    assert len(examples) >= 10 and skipped == 0


def test_example_to_article_does_not_leak_question(pubmedqa_dir):
    example = load_pubmedqa(pubmedqa_dir, "test")[0][0]
    article = example_to_article(example)
    assert article.pmid == "1000" and article.source == "pubmedqa"
    assert article.abstract == "Background text. Results text."
    assert example.question not in article.title
    with pytest.raises(ValueError):
        example_to_article(example.model_copy(update={"contexts": []}))


# --- Deterministic metrics ---------------------------------------------------
def row(**overrides) -> dict:
    base = {
        "id": "r", "kind": "pubmedqa", "expected_abstain": False, "retrieved_pmids": ["1", "2"],
        "gold_pmid": "1", "gold_indexed": True, "abstained": False,
    }  # fmt: skip
    return {**base, **overrides}


def test_metric_marks_missing_values_as_not_computed():
    assert det.metric(None, reason="why") == {
        "value": None, "status": "not_computed", "reason": "why", "n": 0,
    }  # fmt: skip
    assert det.metric(float("nan"))["status"] == "not_computed"
    assert det.metric(0.0, 3) == {"value": 0.0, "status": "computed", "reason": None, "n": 3}


def test_gold_rank():
    assert det.gold_rank(["5", "7", "9"], "7") == 2
    assert det.gold_rank(["5"], "7") is None
    assert det.gold_rank([], "7") is None
    assert det.gold_rank(["5"], None) is None


def test_retrieval_metrics():
    rows = [
        row(retrieved_pmids=["1", "2", "3"], gold_pmid="1"),  # rank 1
        row(retrieved_pmids=["2", "1", "3"], gold_pmid="1"),  # rank 2
        row(retrieved_pmids=["2", "3", "4"], gold_pmid="1"),  # miss
        row(retrieved_pmids=["2", "3", "1"], gold_pmid="1"),  # rank 3, outside k=2
        row(gold_pmid="9", gold_indexed=False),  # not eligible
        row(gold_pmid=None, gold_indexed=False),  # not eligible
    ]
    metrics = det.retrieval_metrics(rows, k=2)
    assert metrics["recall_at_2"] == {"value": 0.5, "status": "computed", "reason": None, "n": 4}
    assert metrics["mrr"]["value"] == pytest.approx((1 + 0.5 + 0 + 1 / 3) / 4, abs=1e-4)
    assert metrics["ndcg_at_2"]["value"] == pytest.approx((1 + 1 / 1.58496) / 4, abs=1e-4)


def test_retrieval_metrics_without_gold_are_not_computed():
    for rows in ([], [row(gold_pmid=None, gold_indexed=False)]):
        metrics = det.retrieval_metrics(rows, k=5)
        assert all(m["status"] == "not_computed" and m["value"] is None for m in metrics.values())
        assert "gold" in metrics["mrr"]["reason"]


def test_abstention_metrics():
    rows = [
        row(expected_abstain=True, abstained=True),
        row(expected_abstain=True, abstained=True),
        row(expected_abstain=True, abstained=False),  # unsafe answer
        row(expected_abstain=False, abstained=False),
        row(expected_abstain=False, abstained=False),
        row(expected_abstain=False, abstained=False),
        row(expected_abstain=False, abstained=True),  # false abstention
    ]
    metrics = det.abstention_metrics(rows)
    assert metrics["refusal_correctness"]["value"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["unsafe_answer_rate"]["value"] == pytest.approx(1 / 3, abs=1e-4)
    assert metrics["false_abstention_rate"]["value"] == 0.25
    assert metrics["abstention_precision"]["value"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["abstention_recall"]["value"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["abstention_f1"]["value"] == pytest.approx(2 / 3, abs=1e-4)


def test_abstention_metrics_empty_undetermined_and_single_class():
    assert all(m["status"] == "not_computed" for m in det.abstention_metrics([]).values())
    undetermined = det.abstention_metrics([row(abstained=True), row(abstained=None)])
    assert all(m["status"] == "not_computed" for m in undetermined.values())
    assert "1 of 2" in undetermined["abstention_f1"]["reason"]
    answerable_only = det.abstention_metrics([row(abstained=False), row(abstained=False)])
    assert answerable_only["false_abstention_rate"]["value"] == 0.0
    assert answerable_only["refusal_correctness"]["status"] == "not_computed"
    assert answerable_only["abstention_precision"]["status"] == "not_computed"
    assert answerable_only["abstention_f1"]["status"] == "not_computed"


def test_citation_validity_rate():
    rows = [
        row(citations={"cited_pmids": ["1", "2"], "valid_pmids": ["1", "2"]}),
        row(citations={"cited_pmids": ["1", "99"], "valid_pmids": ["1"]}),
        row(citations=None),
        row(),
    ]
    assert det.citation_validity_rate(rows)["value"] == 0.75
    assert det.citation_validity_rate([row()])["status"] == "not_computed"


def test_claim_metrics_pool_claims_and_skip_judge_errors():
    def check(supported: int, checkable: int, coverage: float, error=None) -> dict:
        return {
            "error": error, "checkable_claims": checkable, "citation_coverage": coverage,
            "verdict_counts": {"supported": supported},
        }  # fmt: skip

    rows = [
        row(claim_check=check(3, 4, 1.0)),
        row(claim_check=check(1, 4, 0.5), abstained=True),  # rejected draft
        row(claim_check=check(0, 0, 0.0, error="judge failed")),
        row(claim_check=None),
    ]
    delivered = det.claim_metrics(rows, delivered_only=True)
    assert delivered["claim_support_rate"]["value"] == 0.75
    assert delivered["citation_coverage"]["value"] == 1.0
    drafts = det.claim_metrics(rows, delivered_only=False)
    assert drafts["claim_support_rate"]["value"] == 0.5 and drafts["claim_support_rate"]["n"] == 8
    assert drafts["citation_coverage"]["value"] == 0.75
    empty = det.claim_metrics([row()], delivered_only=False)
    assert empty["claim_support_rate"]["status"] == "not_computed"


def test_summarize_handles_empty_and_malformed_rows():
    empty = summarize([], k=5)
    assert empty["counts"]["rows"] == 0
    for section in empty["metrics"].values():
        assert all(m["status"] == "not_computed" for m in section.values())

    rows = [row(), "garbage", {"id": 3}, {"id": "x", "kind": "pubmedqa"}, None]
    summary = summarize(rows, k=5)
    assert summary["counts"]["rows"] == 5 and summary["counts"]["malformed_rows"] == 4
    assert summary["metrics"]["retrieval"]["recall_at_5"]["value"] == 1.0
    assert summary["metrics"]["ragas"]["faithfulness"]["status"] == "not_computed"


def test_parse_judge_output_validation():
    good = '{"claims": [{"claim": "c", "cited_pmids": [123, "x"], "verdict": "supported"}]}'
    assert parse_judge_output(good).claims[0].cited_pmids == ["123"]
    for bad in ("", "[]", '{"claims": "none"}',
                '{"claims": [{"claim": "", "verdict": "supported"}]}',
                '{"claims": [{"claim": "c", "verdict": "true"}]}'):  # fmt: skip
        with pytest.raises(ValueError):
            parse_judge_output(bad)


# --- RAGAS adapter -----------------------------------------------------------
class FakeMetric:
    def __init__(self, value=None, error: Exception | None = None) -> None:
        self.value, self.error, self.calls = value, error, []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(value=self.value)


SAMPLE = {"question": "q?", "response": "answer", "contexts": ["ctx"], "reference": "ref"}


def test_ragas_not_computed_without_llm_key(settings):
    scores = RagasAdapter(settings).score([SAMPLE])
    assert set(scores[0]) == {"faithfulness", "context_precision", "context_recall"}
    assert all(s["value"] is None and "GEMINI_API_KEY" in s["reason"] for s in scores[0].values())


def test_ragas_scores_with_reference(settings):
    metrics = {
        "faithfulness": FakeMetric(0.9),
        "context_precision_with_reference": FakeMetric(0.8),
        "context_precision_without_reference": FakeMetric(0.1),
        "context_recall": FakeMetric(0.7),
    }
    scores = RagasAdapter(settings, metrics=metrics).score([SAMPLE])[0]
    assert {k: v["value"] for k, v in scores.items()} == {
        "faithfulness": 0.9, "context_precision": 0.8, "context_recall": 0.7,
    }  # fmt: skip
    assert metrics["faithfulness"].calls == [
        {"user_input": "q?", "response": "answer", "retrieved_contexts": ["ctx"]}
    ]
    assert metrics["context_precision_without_reference"].calls == []


def test_ragas_degrades_per_metric(settings):
    metrics = {
        "faithfulness": FakeMetric(float("nan")),
        "context_precision_without_reference": FakeMetric(error=RuntimeError("provider down")),
    }
    no_reference = {**SAMPLE, "reference": None}
    scores = RagasAdapter(settings, metrics=metrics).score([no_reference, {"question": "q"}])
    assert (
        scores[0]["faithfulness"]["value"] is None and "NaN" in scores[0]["faithfulness"]["reason"]
    )
    assert "provider down" in scores[0]["context_precision"]["reason"]
    assert "no reference" in scores[0]["context_recall"]["reason"]
    assert all("missing" in s["reason"] for s in scores[1].values())


def test_ragas_real_interface_builds_without_network(settings):
    """The adapter targets the installed ragas API; building metrics must not fail."""
    configured = settings.model_copy(update={"gemini_api_key": SecretStr("not-a-real-key")})
    adapter = RagasAdapter(configured)
    assert adapter._build_metrics() is not None, adapter._unavailable_reason


# --- Runner (end to end with fakes) ------------------------------------------
EXAMPLES = [
    EvalExample(
        id="pubmedqa-111", kind="pubmedqa", split="dev", expected_abstain=False, gold_pmid="111",
        question="Do inhaled corticosteroids reduce asthma exacerbations in adults?",
        contexts=["Inhaled corticosteroids reduced asthma exacerbations in adults."],
        reference_answer="Yes, they reduced exacerbations.", label="yes",
    ),
    EvalExample(
        id="pubmedqa-333", kind="pubmedqa", split="dev", expected_abstain=False, gold_pmid="333",
        question="Does statin therapy lower cholesterol and cardiovascular events?",
        contexts=["Statin therapy lowered LDL cholesterol and reduced cardiovascular events."],
        reference_answer="Yes.", label="yes",
    ),
]  # fmt: skip
DISTRACTOR = EvalExample(
    id="pubmedqa-444", kind="pubmedqa", split="dev", expected_abstain=False, gold_pmid="444",
    question="Does metformin help?", contexts=["Metformin improved glycaemic control."],
    label="yes",
)  # fmt: skip
STRESS = [
    EvalExample(
        id="stress-1", kind="synthetic_unanswerable", split="stress", expected_abstain=True,
        question="Which chess opening is best?",
    )
]  # fmt: skip
JUDGE_OK = json.dumps(
    {
        "claims": [
            {
                "claim": "Steroids reduced exacerbations",
                "cited_pmids": ["111"],
                "verdict": "supported",
            }
        ]
    }
)


def test_run_evaluation_end_to_end(settings, retrieval_service, tmp_path):
    fabricated = "Statins cut events by 90% [PMID: 99999999]."
    llm = FakeLLM(["Steroids reduced exacerbations [PMID: 111].", JUDGE_OK, fabricated, fabricated])
    ragas = RagasAdapter(
        settings,
        metrics={
            "faithfulness": FakeMetric(1.0),
            "context_precision_with_reference": FakeMetric(0.5),
            "context_recall": FakeMetric(1.0),
        },
    )
    out = tmp_path / "report"
    summary = run_evaluation(
        settings,
        EvalConfig(split="dev", output_dir=out),
        retrieval=retrieval_service,
        provider=llm,
        ragas=ragas,
        examples=EXAMPLES,
        distractors=[DISTRACTOR],
        stress=STRESS,
    )
    counts, metrics = summary["counts"], summary["metrics"]
    assert counts["rows"] == 3 and counts["indexed_documents"] == 3
    assert counts["answered"] == 1 and counts["abstained"] == 2
    assert counts["abstention_reasons"] == {"invalid_citations": 1, "low_relevance": 1}
    assert metrics["retrieval"]["recall_at_3"]["value"] == 1.0
    assert metrics["retrieval"]["mrr"]["value"] == 1.0
    assert metrics["abstention"]["refusal_correctness"]["value"] == 1.0
    assert metrics["abstention"]["unsafe_answer_rate"]["value"] == 0.0
    assert metrics["abstention"]["false_abstention_rate"]["value"] == 0.5
    assert metrics["abstention"]["abstention_f1"]["value"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["grounding"]["citation_validity_rate"]["value"] == 0.5
    assert metrics["grounding"]["claim_support_rate"]["value"] == 1.0
    assert metrics["ragas"]["faithfulness"] == {
        "value": 1.0,
        "status": "computed",
        "reason": None,
        "n": 1,
    }
    assert metrics["ragas"]["context_precision"]["value"] == 0.5

    run = summary["run"]
    assert run["llm_model"] == "fake-model" and run["settings"]["min_evidence_count"] == 1
    assert run["packages"]["ragas"] and run["timestamp"] and run["config"]["split"] == "dev"

    rows = [json.loads(line) for line in (out / "results.jsonl").read_text("utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["pubmedqa-111", "pubmedqa-333", "stress-1"]
    assert rows[0]["gold_rank"] == 1 and rows[0]["ragas"]["faithfulness"]["value"] == 1.0
    assert "99999999" not in rows[1]["answer"]
    assert not any(key.startswith("_") for key in rows[0])
    saved = json.loads((out / "summary.json").read_text("utf-8"))
    assert saved["metrics"]["retrieval"]["mrr"]["value"] == 1.0
    with (out / "summary.csv").open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    assert {"section", "metric", "value", "status", "n", "reason"} == set(table[0])
    assert any(r["metric"] == "mrr" and r["status"] == "computed" for r in table)


def test_run_evaluation_without_llm_reports_not_computed(settings, retrieval_service, tmp_path):
    summary = run_evaluation(
        settings,
        EvalConfig(split="dev", output_dir=tmp_path / "report"),
        retrieval=retrieval_service,
        examples=EXAMPLES,
        distractors=[],
        stress=STRESS,
    )
    counts, metrics = summary["counts"], summary["metrics"]
    assert counts["generation_skipped_llm_not_configured"] == 2
    assert counts["abstained"] == 1  # the stress question is stopped by the relevance gate
    assert metrics["retrieval"]["recall_at_3"]["status"] == "computed"
    assert metrics["abstention"]["refusal_correctness"]["status"] == "not_computed"
    assert "generation did not run" in metrics["abstention"]["refusal_correctness"]["reason"]
    assert metrics["abstention_stress_only"]["refusal_correctness"]["value"] == 1.0
    assert metrics["grounding"]["claim_support_rate"]["status"] == "not_computed"
    assert metrics["ragas"]["faithfulness"]["status"] == "not_computed"
    assert summary["run"]["llm_configured"] is False
