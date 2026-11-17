"""Dense retrieval: sentence-transformers embeddings stored in ChromaDB."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.retrieval.chunking import Chunk

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Turns texts into fixed-size vectors."""

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Lazy wrapper around a ``SentenceTransformer`` model (loaded on first use, CPU)."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self._batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %s on CPU", self.model_name)
                self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches; vectors are L2-normalised."""
        if not texts:
            return []
        vectors = self._load().encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in vector] for vector in vectors]


@dataclass(frozen=True)
class DenseHit:
    chunk: Chunk
    distance: float
    """Cosine distance (``1 - cosine similarity``); lower means closer."""


class DenseIndex:
    """Persistent ChromaDB collection using cosine distance.

    Chroma returns *distances*, not similarities: for the cosine space the value is
    ``1 - cosine_similarity`` (0 = identical direction, 2 = opposite). Results are
    ordered by ascending distance.
    """

    def __init__(
        self, path: Path, collection: str, embedder: Embedder, batch_size: int = 32
    ) -> None:
        self._path = path
        self._collection_name = collection
        self._embedder = embedder
        self._batch_size = batch_size
        self._client = None
        self._collection = None
        self._lock = threading.Lock()

    def _get_collection(self):
        with self._lock:
            if self._collection is None:
                import chromadb

                self._path.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=str(self._path))
                self._collection = self._client.get_or_create_collection(
                    name=self._collection_name,
                    configuration={"hnsw": {"space": "cosine"}},
                    embedding_function=None,
                )
        return self._collection

    def count(self) -> int:
        """Number of stored chunks."""
        return int(self._get_collection().count())

    def upsert(self, chunks: list[Chunk]) -> None:
        """Embed and upsert chunks by stable chunk ID (no duplicates on re-ingestion)."""
        collection = self._get_collection()
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=self._embedder.encode([c.index_text for c in batch]),
                documents=[c.text for c in batch],
                metadatas=[c.metadata() for c in batch],
            )

    def delete_missing(self, keep_ids: set[str]) -> int:
        """Remove stored chunks whose IDs are not in ``keep_ids``; return how many."""
        collection = self._get_collection()
        stale = [i for i in collection.get(include=[])["ids"] if i not in keep_ids]
        for start in range(0, len(stale), 500):
            collection.delete(ids=stale[start : start + 500])
        return len(stale)

    def all_chunks(self) -> list[Chunk]:
        """Return every stored chunk (used to rebuild BM25 if its artifact is lost)."""
        data = self._get_collection().get(include=["documents", "metadatas"])
        return [
            Chunk.from_record({**(meta or {}), "chunk_id": chunk_id, "text": text or ""})
            for chunk_id, text, meta in zip(
                data["ids"], data["documents"], data["metadatas"], strict=True
            )
        ]

    def search(self, query: str, top_k: int) -> list[DenseHit]:
        """Return the ``top_k`` nearest chunks, closest first."""
        if not query.strip() or top_k < 1:
            return []
        collection = self._get_collection()
        total = collection.count()
        if total == 0:
            return []
        result = collection.query(
            query_embeddings=self._embedder.encode([query]),
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )
        return [
            DenseHit(
                chunk=Chunk.from_record({**(meta or {}), "chunk_id": chunk_id, "text": text or ""}),
                distance=float(distance),
            )
            for chunk_id, text, meta, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]
