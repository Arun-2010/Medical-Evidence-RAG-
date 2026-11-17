"""BM25 lexical index with a versioned on-disk artifact."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.retrieval.chunking import Chunk

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 1
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case and split on anything that is not a letter or digit."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class LexicalHit:
    chunk: Chunk
    score: float


class BM25Index:
    """BM25 over chunks. Position ``i`` in the corpus always maps to ``chunks[i]``."""

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self._chunks: list[Chunk] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        if chunks:
            self.build(chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def build(self, chunks: list[Chunk]) -> None:
        """Rebuild the index from ``chunks`` (de-duplicated by chunk ID, sorted by ID)."""
        unique = {c.chunk_id: c for c in chunks}
        self._chunks = [unique[key] for key in sorted(unique)]
        self._tokens = [tokenize(c.index_text) for c in self._chunks]
        self._fit()

    def _fit(self) -> None:
        # BM25Okapi divides by the corpus size, so it cannot be built on an empty corpus.
        usable = any(self._tokens)
        self._bm25 = BM25Okapi(self._tokens) if usable else None

    def search(self, query: str, top_k: int) -> list[LexicalHit]:
        """Return up to ``top_k`` chunks sharing at least one term with the query."""
        query_tokens = tokenize(query)
        if self._bm25 is None or not query_tokens or top_k < 1:
            return []
        scores = self._bm25.get_scores(query_tokens)
        query_terms = set(query_tokens)
        candidates = [
            (float(scores[i]), i)
            for i, tokens in enumerate(self._tokens)
            if query_terms.intersection(tokens)
        ]
        candidates.sort(key=lambda pair: (-pair[0], self._chunks[pair[1]].chunk_id))
        return [LexicalHit(self._chunks[i], score) for score, i in candidates[:top_k]]

    def save(self, path: Path) -> None:
        """Persist the tokenized corpus and chunk records atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": ARTIFACT_VERSION,
            "chunks": [c.to_record() for c in self._chunks],
            "tokens": self._tokens,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> BM25Index | None:
        """Load an artifact; return ``None`` if it is missing, corrupt or incompatible."""
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("version") != ARTIFACT_VERSION:
                raise ValueError(f"unsupported artifact version {payload.get('version')!r}")
            chunks = [Chunk.from_record(r) for r in payload["chunks"]]
            tokens = payload["tokens"]
            if len(chunks) != len(tokens):
                raise ValueError("chunk/token count mismatch")
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Ignoring unusable BM25 artifact %s (%s); it will be rebuilt", path, exc)
            return None
        index = cls()
        index._chunks, index._tokens = chunks, tokens
        index._fit()
        return index
