from __future__ import annotations

import pytest

from app.core.config import GEMINI_DEFAULTS, OPENAI_DEFAULTS, load_settings
from app.core.exceptions import ConfigurationError
from app.generation.providers import get_provider


@pytest.fixture(autouse=True)
def _no_llm_env(monkeypatch):
    for name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_gemini_key_uses_gemini_defaults(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test-value")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ignored")
    settings = load_settings(_env_file=None)
    assert settings.llm_configured
    assert settings.llm_api_key.get_secret_value() == "gm-test-value"
    assert (settings.llm_base_url, settings.llm_model) == GEMINI_DEFAULTS
    assert settings.llm_seed is None
    assert "gm-test-value" not in repr(settings)
    provider = get_provider(settings)
    assert provider.model == GEMINI_DEFAULTS[1]
    assert provider._url == f"{GEMINI_DEFAULTS[0]}/chat/completions"
    assert provider._headers == {"Authorization": "Bearer gm-test-value"}


def test_openai_only_key_uses_openai_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-value")
    settings = load_settings(_env_file=None)
    assert (settings.llm_base_url, settings.llm_model) == OPENAI_DEFAULTS
    assert settings.llm_seed == settings.seed


def test_explicit_model_and_blank_values(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test-value")
    monkeypatch.setenv("LLM_MODEL", "gemini-custom")
    monkeypatch.setenv("LLM_BASE_URL", "")
    settings = load_settings(_env_file=None)
    assert settings.llm_model == "gemini-custom"
    assert settings.llm_base_url == GEMINI_DEFAULTS[0]


def test_defaults_without_env_file():
    settings = load_settings(_env_file=None)
    assert not settings.llm_configured
    assert (settings.llm_base_url, settings.llm_model) == GEMINI_DEFAULTS
    assert settings.top_k_rerank <= settings.top_k_fusion
    assert settings.bm25_path.parent == settings.chroma_path


def test_environment_variables_are_read(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-value")
    monkeypatch.setenv("TOP_K_RERANK", "4")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1/")
    settings = load_settings(_env_file=None)
    assert settings.llm_configured and settings.top_k_rerank == 4
    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert "sk-test-value" not in repr(settings)


def test_blank_key_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    assert not load_settings(_env_file=None).llm_configured


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"top_k_rerank": 0}, "TOP_K_RERANK"),
        ({"rrf_k": 0}, "RRF_K"),
        ({"request_timeout_seconds": -1}, "REQUEST_TIMEOUT_SECONDS"),
        ({"top_k_rerank": 10, "top_k_fusion": 5}, "TOP_K_FUSION"),
        ({"min_evidence_count": 9, "top_k_rerank": 5}, "MIN_EVIDENCE_COUNT"),
        ({"llm_base_url": "ftp://example.org"}, "LLM_BASE_URL"),
        ({"log_level": "LOUD"}, "LOG_LEVEL"),
    ],
)
def test_invalid_settings_fail_with_actionable_message(overrides, expected):
    with pytest.raises(ConfigurationError, match=expected):
        load_settings(_env_file=None, **overrides)
