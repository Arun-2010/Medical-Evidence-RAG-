"""Hybrid retrieval orchestration: BM25 + dense -> RRF -> cross-encoder rerank."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from app.core.config import Settings
from app.core.exceptions import IndexEmptyError
from app.ingestion.parser import Article
from app.retrieval.chunking import Chunk, chunk_articles
from app.retrieval.dense import DenseIndex, Embedder, SentenceTransformerEmbedder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.lexical import BM25Index
from app.retrieval.reranker import CrossEncoderReranker, Reranker, rerank
from app.schemas.api import EvidenceItem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexStatus:
    dense_count: int
    bm25_count: int
    collection: str

    @property
    def in_sync(self) -> bool:
        return self.dense_count == self.bm25_count

    @property
    def ready(self) -> bool:
        return self.dense_count > 0 and self.in_sync

    @property
    def message(self) -> str:
        if self.dense_count == 0 and self.bm25_count == 0:
            return "Index is empty; ingest documents first."
        if not self.in_sync:
            return "Dense and BM25 indexes are out of sync; rebuild the index."
        return "Index is ready."


class RetrievalService:
    """Owns both indexes and runs the retrieval pipeline."""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._settings = settings
        self._reranker = reranker or CrossEncoderReranker(settings.reranker_model)
        self._dense = DenseIndex(
            settings.chroma_path,
            settings.chroma_collection,
            embedder
            or SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_batch_size),
            batch_size=settings.embedding_batch_size,
        )
        self._bm25: BM25Index | None = None
        self._lock = threading.Lock()

    def _get_bm25(self) -> BM25Index:
        """Load the BM25 artifact, rebuilding it from Chroma if missing or incompatible."""
        with self._lock:
            if self._bm25 is None:
                index = BM25Index.load(self._settings.bm25_path)
                if index is None:
                    index = BM25Index(self._dense.all_chunks())
                    if len(index):
                        logger.info("Rebuilt BM25 index from Chroma (%d chunks)", len(index))
                        index.save(self._settings.bm25_path)
                self._bm25 = index
            return self._bm25

    def index_articles(self, articles: list[Article]) -> IndexStatus:
        """Make both indexes reflect exactly ``articles`` (upsert + prune stale chunks)."""
        chunks = chunk_articles(articles, self._settings.max_abstract_chars)
        self._dense.upsert(chunks)
        removed = self._dense.delete_missing({c.chunk_id for c in chunks})
        index = BM25Index(chunks)
        index.save(self._settings.bm25_path)
        with self._lock:
            self._bm25 = index
        logger.info("Indexed %d chunks (%d stale removed)", len(chunks), removed)
        return self.status()

    def status(self) -> IndexStatus:
        """Report index sizes and readiness."""
        return IndexStatus(
            dense_count=self._dense.count(),
            bm25_count=len(self._get_bm25()),
            collection=self._settings.chroma_collection,
        )

    def indexed_pmids(self) -> set[str]:
        """PMIDs currently present in the index."""
        return {chunk.pmid for chunk in self._get_bm25().chunks}

    def retrieve(self, query: str, top_k: int | None = None) -> list[EvidenceItem]:
        """Run hybrid retrieval and return reranked evidence, best first.

        Raises :class:`IndexEmptyError` when nothing is indexed. Returns an empty
        list when the index has documents but none match.
        """
        settings = self._settings
        top_k = top_k or settings.top_k_rerank
        bm25 = self._get_bm25()
        if self._dense.count() == 0 and len(bm25) == 0:
            raise IndexEmptyError("Index is empty; ingest documents first.")

        lexical_hits = bm25.search(query, settings.top_k_bm25)
        dense_hits = self._dense.search(query, settings.top_k_dense)
        chunks: dict[str, Chunk] = {h.chunk.chunk_id: h.chunk for h in dense_hits}
        chunks.update({h.chunk.chunk_id: h.chunk for h in lexical_hits})
        bm25_scores = {h.chunk.chunk_id: h.score for h in lexical_hits}
        dense_distances = {h.chunk.chunk_id: h.distance for h in dense_hits}

        fused = reciprocal_rank_fusion(
            {
                "bm25": [h.chunk.chunk_id for h in lexical_hits],
                "dense": [h.chunk.chunk_id for h in dense_hits],
            },
            k=settings.rrf_k,
            top_k=max(settings.top_k_fusion, top_k),
        )
        ranked = rerank(self._reranker, query, [chunks[h.doc_id].index_text for h in fused])

        results: list[EvidenceItem] = []
        seen_pmids: set[str] = set()
        for index, score in ranked:
            hit = fused[index]
            chunk = chunks[hit.doc_id]
            if chunk.pmid in seen_pmids:  # keep only the best chunk per article
                continue
            seen_pmids.add(chunk.pmid)
            results.append(
                EvidenceItem(
                    chunk_id=chunk.chunk_id,
                    pmid=chunk.pmid,
                    pubmed_url=chunk.pubmed_url,
                    title=chunk.title,
                    publication_year=chunk.publication_year,
                    journal=chunk.journal,
                    abstract=chunk.text,
                    bm25_rank=hit.ranks.get("bm25"),
                    bm25_score=bm25_scores.get(hit.doc_id),
                    dense_rank=hit.ranks.get("dense"),
                    dense_distance=dense_distances.get(hit.doc_id),
                    rrf_score=hit.rrf_score,
                    rerank_score=score,
                    final_rank=len(results) + 1,
                )
            )
            if len(results) >= top_k:
                break
        logger.info(
            "Retrieved %d results (bm25=%d, dense=%d, fused=%d)",
            len(results),
            len(lexical_hits),
            len(dense_hits),
            len(fused),
        )
        return results
