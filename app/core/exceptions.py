"""Application exception hierarchy.

Each error carries a stable machine-readable ``code`` and an HTTP status so the
API layer can translate failures without leaking stack traces.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all expected application errors."""

    code = "app_error"
    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigurationError(AppError):
    """Invalid or missing configuration."""

    code = "configuration_error"
    status_code = 500


class LLMNotConfiguredError(AppError):
    """Generation was requested but no LLM API key is configured."""

    code = "llm_not_configured"
    status_code = 503


class LLMProviderError(AppError):
    """The LLM provider call failed after retries."""

    code = "llm_provider_error"
    status_code = 502


class PubMedError(AppError):
    """A PubMed E-utilities call failed after retries or returned bad data."""

    code = "pubmed_error"
    status_code = 502


class IndexEmptyError(AppError):
    """Retrieval was requested but no documents are indexed."""

    code = "index_empty"
    status_code = 409


class RetrievalError(AppError):
    """A retrieval stage failed."""

    code = "retrieval_error"
    status_code = 500


class DatasetError(AppError):
    """An evaluation dataset is missing or unusable."""

    code = "dataset_error"
    status_code = 500
