"""Typed application settings loaded from environment variables / ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.exceptions import ConfigurationError

ABSTENTION_PHRASE = "Insufficient evidence in the indexed sources."
DISCLAIMER = (
    "Research/demo tool only. This is not a medical device and not a substitute for "
    "professional medical advice, diagnosis, or treatment. It summarises indexed PubMed "
    "abstracts and has not reviewed any patient's medical record."
)


GEMINI_DEFAULTS = ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash")
OPENAI_DEFAULTS = ("https://api.openai.com/v1", "gpt-4o-mini")
LLM_KEY_HINT = "GEMINI_API_KEY (or OPENAI_API_KEY)"


class Settings(BaseSettings):
    """All runtime configuration. Field names map to upper-case environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    gemini_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    llm_base_url: str = ""
    llm_model: str = ""
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_batch_size: int = Field(default=32, ge=1, le=512)

    data_dir: Path = Path("data")
    chroma_path: Path = Path("data/chroma")
    chroma_collection: str = Field(default="pubmed_abstracts", min_length=3, max_length=63)

    pubmed_email: str | None = None
    pubmed_api_key: SecretStr | None = None
    pubmed_max_retries: int = Field(default=3, ge=0, le=8)

    top_k_bm25: int = Field(default=20, ge=1, le=200)
    top_k_dense: int = Field(default=20, ge=1, le=200)
    top_k_fusion: int = Field(default=15, ge=1, le=200)
    top_k_rerank: int = Field(default=5, ge=1, le=50)
    rrf_k: int = Field(default=60, ge=1, le=1000)

    min_rerank_score: float = Field(default=0.0, ge=-50.0, le=50.0)
    min_evidence_count: int = Field(default=2, ge=1, le=50)
    enable_claim_check: bool = True
    min_claim_support_rate: float = Field(default=0.8, ge=0.0, le=1.0)
    min_citation_coverage: float = Field(default=0.8, ge=0.0, le=1.0)

    max_abstract_chars: int = Field(default=4000, ge=200, le=20000)
    request_timeout_seconds: float = Field(default=30.0, gt=0, le=300)

    admin_token: SecretStr | None = None
    seed: int = 42

    @field_validator(
        "gemini_api_key",
        "openai_api_key",
        "pubmed_api_key",
        "admin_token",
        "pubmed_email",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        """Treat empty strings (as left in a copied ``.env.example``) as unset."""
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("llm_base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        value = value.strip()
        if value and not value.startswith(("http://", "https://")):
            raise ValueError("LLM_BASE_URL must start with http:// or https://")
        return value.rstrip("/")

    @model_validator(mode="after")
    def _resolve_llm_defaults(self) -> Settings:
        """Fill an unset LLM_BASE_URL / LLM_MODEL from the provider whose key is present.

        Gemini is the default provider; OpenAI defaults apply only when just
        OPENAI_API_KEY is set.
        """
        openai_only = self.gemini_api_key is None and self.openai_api_key is not None
        base_url, model = OPENAI_DEFAULTS if openai_only else GEMINI_DEFAULTS
        self.llm_base_url = self.llm_base_url or base_url
        self.llm_model = self.llm_model.strip() or model
        return self

    @model_validator(mode="after")
    def _check_top_k(self) -> Settings:
        if self.top_k_rerank > self.top_k_fusion:
            raise ValueError(
                f"TOP_K_RERANK ({self.top_k_rerank}) must be <= TOP_K_FUSION ({self.top_k_fusion})"
            )
        if self.min_evidence_count > self.top_k_rerank:
            raise ValueError(
                f"MIN_EVIDENCE_COUNT ({self.min_evidence_count}) must be <= "
                f"TOP_K_RERANK ({self.top_k_rerank}); otherwise every question would abstain"
            )
        return self

    @property
    def llm_configured(self) -> bool:
        """True when an LLM API key is available."""
        return self.llm_api_key is not None

    @property
    def llm_api_key(self) -> SecretStr | None:
        """Key sent to the LLM endpoint: GEMINI_API_KEY if set, otherwise OPENAI_API_KEY."""
        return self.gemini_api_key or self.openai_api_key

    @property
    def llm_seed(self) -> int | None:
        """Seed for the LLM request; omitted for Gemini's endpoint, which may reject it."""
        return None if "googleapis.com" in self.llm_base_url else self.seed

    @property
    def processed_articles_path(self) -> Path:
        """Normalized article store (JSONL), the source of truth for index rebuilds."""
        return self.data_dir / "processed" / "articles.jsonl"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def bm25_path(self) -> Path:
        """Versioned BM25 artifact, stored next to the Chroma data."""
        return self.chroma_path / f"bm25_{self.chroma_collection}.json"


def load_settings(**overrides: object) -> Settings:
    """Build settings, converting validation failures into an actionable error."""
    try:
        return Settings(**overrides)  # type: ignore[arg-type]
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']).upper() or 'SETTINGS'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigurationError(
            f"Invalid configuration ({problems}). Fix the value in your .env file or environment."
        ) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached process-wide settings."""
    return load_settings()
