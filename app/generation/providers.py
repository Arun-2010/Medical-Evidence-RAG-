"""LLM provider abstraction. All provider-specific code lives in this module."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Protocol

import httpx

from app.core.config import LLM_KEY_HINT, Settings
from app.core.exceptions import LLMNotConfiguredError, LLMProviderError

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class LLMProvider(Protocol):
    """Minimal chat interface used for both generation and judging."""

    model: str

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """Return the assistant message text for one system + user exchange."""
        ...


class OpenAICompatibleProvider:
    """Calls ``POST {base_url}/chat/completions`` on any OpenAI-compatible server."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        max_retries: int = 2,
        seed: int | None = 42,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._max_retries = max_retries
        self._seed = seed
        self._sleep = sleep
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout))
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        }
        if self._seed is not None:
            payload["seed"] = self._seed
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error = "unknown error"
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(self._url, json=payload, headers=self._headers)
            except httpx.HTTPError as exc:
                last_error = type(exc).__name__
            else:
                if response.status_code == 200:
                    return self._extract_text(response)
                last_error = f"HTTP {response.status_code}: {self._error_message(response)}"
                if response.status_code not in _RETRYABLE_STATUS:
                    break
            if attempt < self._max_retries:
                delay = min(2.0**attempt, 20.0)
                logger.warning("LLM call failed (%s); retrying in %.0fs", last_error, delay)
                self._sleep(delay)
        raise LLMProviderError(
            f"LLM request to model '{self.model}' failed: {last_error}. "
            f"Check {LLM_KEY_HINT}, LLM_BASE_URL and LLM_MODEL."
        )

    @staticmethod
    def _extract_text(response: httpx.Response) -> str:
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("LLM returned a response in an unexpected format.") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError("LLM returned an empty response.")
        return content.strip()

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        """Short provider error text (never includes request headers or the key)."""
        try:
            message = response.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = response.text
        return str(message)[:300]


def get_provider(settings: Settings) -> LLMProvider:
    """Build the configured provider, or raise a clear error if no key is set."""
    if settings.llm_api_key is None:
        raise LLMNotConfiguredError(
            f"No LLM API key configured. Set {LLM_KEY_HINT} (and optionally LLM_BASE_URL / "
            "LLM_MODEL) in your .env file to enable answer generation. Retrieval still works."
        )
    return OpenAICompatibleProvider(
        api_key=settings.llm_api_key.get_secret_value(),
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout=settings.request_timeout_seconds,
        max_retries=settings.llm_max_retries,
        seed=settings.llm_seed,
    )
