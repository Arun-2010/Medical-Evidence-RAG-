from __future__ import annotations

import json

import httpx
import pytest

from app.core.config import ABSTENTION_PHRASE
from app.core.exceptions import LLMNotConfiguredError, LLMProviderError
from app.evaluation.judge import ClaimJudge
from app.generation.answer import AnswerService
from app.generation.providers import OpenAICompatibleProvider, get_provider
from tests.conftest import FakeLLM, make_evidence

GOOD_EVIDENCE = [
    make_evidence("111", 5.0, 1),
    make_evidence("222", 4.0, 2),
    make_evidence("333", -3.0, 3),
]
GOOD_ANSWER = (
    "Steroids reduced exacerbations [PMID: 111]. Control improved in children [PMID: 222]."
)


def judge_json(*verdicts: tuple[str, list[str], str]) -> str:
    return json.dumps(
        {"claims": [{"claim": c, "cited_pmids": p, "verdict": v} for c, p, v in verdicts]}
    )


ALL_SUPPORTED = judge_json(
    ("Steroids reduced exacerbations", ["111"], "supported"),
    ("Control improved in children", ["222"], "supported"),
)


def make_service(settings, responses, **overrides) -> tuple[AnswerService, FakeLLM]:
    llm = FakeLLM(responses)
    service = AnswerService(settings.model_copy(update=overrides), retrieval=None, provider=llm)
    return service, llm


def assert_abstained(result, reason: str) -> None:
    assert result.abstained
    assert result.answer == ABSTENTION_PHRASE
    assert result.abstention_reason == reason
    assert result.abstention_message
    assert result.citations == []


# --- Pre-generation gates (no LLM call) ------------------------------------
def test_abstains_with_no_evidence(settings):
    service, llm = make_service(settings, [])
    assert_abstained(service.answer_from_evidence("q?", []), "no_evidence")
    assert llm.calls == []


def test_abstains_when_top_score_below_threshold(settings):
    service, llm = make_service(settings, [])
    evidence = [make_evidence("111", 0.5, 1), make_evidence("222", -2.0, 2)]
    result = service.answer_from_evidence("q?", evidence)
    assert_abstained(result, "low_relevance")
    assert result.evidence == evidence  # still shown for transparency
    assert llm.calls == []


def test_abstains_with_too_few_relevant_sources(settings):
    service, llm = make_service(settings, [])
    evidence = [make_evidence("111", 5.0, 1), make_evidence("222", 0.2, 2)]
    assert_abstained(service.answer_from_evidence("q?", evidence), "insufficient_evidence_count")
    assert llm.calls == []


def test_gates_abstain_even_without_llm_key(settings):
    service = AnswerService(settings, retrieval=None)
    assert_abstained(service.answer_from_evidence("q?", []), "no_evidence")


# --- Generation fallback when the API key is absent ------------------------
def test_generation_without_api_key_raises_clear_error(settings):
    service = AnswerService(settings, retrieval=None)
    with pytest.raises(LLMNotConfiguredError, match="GEMINI_API_KEY"):
        service.answer_from_evidence("q?", GOOD_EVIDENCE)
    with pytest.raises(LLMNotConfiguredError):
        get_provider(settings)


# --- Happy path ------------------------------------------------------------
def test_supported_answer_is_returned_with_citations(settings):
    service, llm = make_service(settings, [GOOD_ANSWER, ALL_SUPPORTED])
    result = service.answer_from_evidence("Do steroids help?", GOOD_EVIDENCE)
    assert not result.abstained and result.abstention_reason is None
    assert result.answer == GOOD_ANSWER
    assert [c.pmid for c in result.citations] == ["111", "222"]
    assert result.diagnostics["claim_check"]["support_rate"] == 1.0
    # Only evidence above the relevance threshold is shown to the model.
    assert 'pmid="333"' not in llm.calls[0]["user"]
    assert llm.calls[1]["json_mode"] is True


def test_claim_check_can_be_disabled(settings):
    service, llm = make_service(settings, [GOOD_ANSWER], enable_claim_check=False)
    assert not service.answer_from_evidence("q?", GOOD_EVIDENCE).abstained
    assert len(llm.calls) == 1


# --- Post-generation safety paths ------------------------------------------
def test_model_reported_insufficient_evidence(settings):
    service, _ = make_service(settings, [f"{ABSTENTION_PHRASE} The abstracts are off-topic."])
    result = service.answer_from_evidence("q?", GOOD_EVIDENCE)
    assert_abstained(result, "model_reported_insufficient_evidence")


def test_fabricated_citation_retried_then_accepted(settings):
    bad = "Steroids help [PMID: 111]. Mortality fell 50% [PMID: 99999999]."
    service, llm = make_service(settings, [bad, GOOD_ANSWER, ALL_SUPPORTED])
    result = service.answer_from_evidence("q?", GOOD_EVIDENCE)
    assert not result.abstained
    assert "99999999" not in result.answer
    assert result.diagnostics["citation_retry"] is True
    assert "99999999" in llm.calls[1]["user"]  # correction prompt names the bad PMID


def test_fabricated_citation_twice_abstains(settings):
    bad = "Steroids help [PMID: 111]. Mortality fell 50% [PMID: 99999999]."
    service, llm = make_service(settings, [bad, bad])
    result = service.answer_from_evidence("q?", GOOD_EVIDENCE)
    assert_abstained(result, "invalid_citations")
    assert len(llm.calls) == 2  # exactly one retry
    assert "99999999" not in result.answer


def test_citing_filtered_out_low_relevance_source_is_invalid(settings):
    bad = "Statins help [PMID: 333]."
    service, _ = make_service(settings, [bad, bad])
    assert_abstained(service.answer_from_evidence("q?", GOOD_EVIDENCE), "invalid_citations")


def test_answer_without_citations_abstains(settings):
    service, _ = make_service(settings, ["Steroids help.", "Steroids really help."])
    assert_abstained(service.answer_from_evidence("q?", GOOD_EVIDENCE), "no_valid_citations")


def test_abstention_on_retry_is_respected(settings):
    service, _ = make_service(settings, ["Steroids help.", ABSTENTION_PHRASE])
    result = service.answer_from_evidence("q?", GOOD_EVIDENCE)
    assert_abstained(result, "model_reported_insufficient_evidence")


def test_unsupported_claims_abstain(settings):
    verdict = judge_json(
        ("Steroids reduced exacerbations", ["111"], "supported"),
        ("Control improved in children", ["222"], "unsupported"),
    )
    service, _ = make_service(settings, [GOOD_ANSWER, verdict])
    result = service.answer_from_evidence("q?", GOOD_EVIDENCE)
    assert_abstained(result, "unsupported_claims")
    assert result.diagnostics["claim_check"]["support_rate"] == 0.5


def test_uncited_claims_abstain(settings):
    answer = "Steroids reduced exacerbations [PMID: 111]. Control improved in children."
    verdict = judge_json(
        ("Steroids reduced exacerbations", ["111"], "supported"),
        ("Control improved in children", [], "supported"),
    )
    service, _ = make_service(settings, [answer, verdict])
    assert_abstained(service.answer_from_evidence("q?", GOOD_EVIDENCE), "uncited_claims")


def test_judge_cannot_invent_citations_absent_from_answer(settings):
    answer = "Steroids reduced exacerbations [PMID: 111]. Control improved in children."
    verdict = judge_json(
        ("Steroids reduced exacerbations", ["111"], "supported"),
        ("Control improved in children", ["222"], "supported"),  # 222 is not in the answer
    )
    service, _ = make_service(settings, [answer, verdict])
    assert_abstained(service.answer_from_evidence("q?", GOOD_EVIDENCE), "uncited_claims")


def test_malformed_judge_output_retried_once_then_abstains(settings):
    service, llm = make_service(
        settings, [GOOD_ANSWER, "not json", '{"claims": [{"verdict": "maybe"}]}']
    )
    result = service.answer_from_evidence("q?", GOOD_EVIDENCE)
    assert_abstained(result, "claim_check_failed")
    assert "malformed" in result.diagnostics["claim_check"]["error"]
    assert len(llm.calls) == 3


def test_malformed_judge_output_recovers_on_retry(settings):
    fenced = f"```json\n{ALL_SUPPORTED}\n```"
    service, _ = make_service(settings, [GOOD_ANSWER, "oops", fenced])
    assert not service.answer_from_evidence("q?", GOOD_EVIDENCE).abstained


def test_no_checkable_claims_abstains(settings):
    verdict = judge_json(("More research is needed", [], "not_verifiable"))
    service, _ = make_service(settings, [GOOD_ANSWER, verdict])
    assert_abstained(service.answer_from_evidence("q?", GOOD_EVIDENCE), "claim_check_failed")


def test_judge_provider_failure_is_reported_not_raised():
    class Failing:
        model = "x"

        def complete(self, system, user, *, json_mode=False):
            raise LLMProviderError("boom")

    result = ClaimJudge(Failing()).judge("q?", GOOD_ANSWER, GOOD_EVIDENCE)
    assert result.error and result.support_rate is None


def test_generation_provider_failure_propagates(settings):
    class Failing:
        model = "x"

        def complete(self, system, user, *, json_mode=False):
            raise LLMProviderError("boom")

    service = AnswerService(settings, retrieval=None, provider=Failing())
    with pytest.raises(LLMProviderError):
        service.answer_from_evidence("q?", GOOD_EVIDENCE)


def test_ask_uses_retrieval(settings, indexed_service):
    answer = "Steroids reduced exacerbations [PMID: 111]. Control improved in children [PMID: 222]."
    llm = FakeLLM([answer, ALL_SUPPORTED])
    service = AnswerService(settings, indexed_service, provider=llm)
    result = service.ask("inhaled corticosteroids asthma exacerbations children")
    assert not result.abstained
    assert {c.pmid for c in result.citations} == {"111", "222"}


# --- OpenAI-compatible provider (mocked HTTP) -------------------------------
def make_provider(handler, **kwargs) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        api_key="sk-secret",
        base_url="https://llm.example/v1",
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
        **kwargs,
    )


def test_provider_sends_expected_request():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": " hello "}}]})

    assert make_provider(handler).complete("sys", "usr", json_mode=True) == "hello"
    assert seen["url"] == "https://llm.example/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-secret"
    assert seen["body"]["model"] == "test-model"
    assert seen["body"]["temperature"] == 0 and seen["body"]["seed"] == 42
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]


def test_provider_retries_then_fails_without_leaking_key():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    with pytest.raises(LLMProviderError) as excinfo:
        make_provider(handler, max_retries=2).complete("s", "u")
    assert len(calls) == 3
    assert "rate limited" in excinfo.value.message
    assert "sk-secret" not in excinfo.value.message


def test_provider_does_not_retry_auth_errors_and_rejects_empty_output():
    calls = []

    def unauthorized(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with pytest.raises(LLMProviderError, match="HTTP 401"):
        make_provider(unauthorized).complete("s", "u")
    assert len(calls) == 1

    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    with pytest.raises(LLMProviderError, match="empty"):
        make_provider(empty).complete("s", "u")
