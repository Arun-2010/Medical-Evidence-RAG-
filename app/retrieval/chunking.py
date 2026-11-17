"""Turn articles into indexable chunks with stable IDs."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.parser import Article

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """An indexable unit. Short abstracts map to exactly one chunk."""

    chunk_id: str
    pmid: str
    chunk_number: int
    title: str
    text: str
    publication_year: int | None
    journal: str | None
    pubmed_url: str
    source: str = "pubmed"

    @property
    def index_text(self) -> str:
        """Text used for embedding, BM25 and reranking (title + abstract text)."""
        return f"{self.title}\n{self.text}"

    def metadata(self) -> dict[str, str | int]:
        """Flat metadata for ChromaDB (no ``None`` values or nested objects)."""
        meta: dict[str, str | int] = {
            "pmid": self.pmid,
            "chunk_number": self.chunk_number,
            "title": self.title,
            "pubmed_url": self.pubmed_url,
            "source": self.source,
        }
        if self.publication_year is not None:
            meta["publication_year"] = self.publication_year
        if self.journal:
            meta["journal"] = self.journal
        return meta

    def to_record(self) -> dict:
        """JSON-serialisable form used by the BM25 artifact."""
        return {**self.metadata(), "chunk_id": self.chunk_id, "text": self.text}

    @classmethod
    def from_record(cls, record: dict) -> Chunk:
        """Rebuild a chunk from :meth:`to_record` output or Chroma metadata + text."""
        return cls(
            chunk_id=record["chunk_id"],
            pmid=str(record["pmid"]),
            chunk_number=int(record.get("chunk_number", 0)),
            title=record.get("title", ""),
            text=record["text"],
            publication_year=record.get("publication_year"),
            journal=record.get("journal"),
            pubmed_url=record.get("pubmed_url", ""),
            source=record.get("source", "pubmed"),
        )


def chunk_id(pmid: str, chunk_number: int) -> str:
    """Stable chunk identifier."""
    return f"PMID_{pmid}_{chunk_number}"


def split_text(text: str, max_chars: int) -> list[str]:
    """Split ``text`` on sentence boundaries into pieces of at most ``max_chars``.

    Text that already fits is returned unchanged as a single piece. A single
    sentence longer than ``max_chars`` is hard-split.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        while len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_article(article: Article, max_chars: int) -> list[Chunk]:
    """Chunk one article; every chunk keeps the PMID and article metadata."""
    return [
        Chunk(
            chunk_id=chunk_id(article.pmid, number),
            pmid=article.pmid,
            chunk_number=number,
            title=article.title,
            text=piece,
            publication_year=article.publication_year,
            journal=article.journal,
            pubmed_url=article.pubmed_url,
            source=article.source,
        )
        for number, piece in enumerate(split_text(article.abstract, max_chars))
    ]


def chunk_articles(articles: list[Article], max_chars: int) -> list[Chunk]:
    """Chunk many articles, de-duplicating by PMID (first occurrence wins)."""
    seen: set[str] = set()
    chunks: list[Chunk] = []
    for article in articles:
        if article.pmid in seen:
            continue
        seen.add(article.pmid)
        chunks.extend(chunk_article(article, max_chars))
    return chunks
