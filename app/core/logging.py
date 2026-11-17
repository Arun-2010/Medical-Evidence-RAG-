"""Logging setup with a per-request ID attached to every record."""

from __future__ import annotations

import logging
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once. Safe to call repeatedly."""
    root = logging.getLogger()
    root.setLevel(level)
    if any(getattr(h, "_medrag", False) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler._medrag = True  # type: ignore[attr-defined]
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s")
    )
    root.addHandler(handler)
    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
