"""Defensive parser for PubMed EFetch XML."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.core.exceptions import PubMedError

logger = logging.getLogger(__name__)

PUBMED_URL_TEMPLATE = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_WS_RE = re.compile(r"\s+")


class Article(BaseModel):
    """A normalized PubMed article with a usable abstract."""

    pmid: str = Field(pattern=r"^\d+$")
    title: str
    abstract: str = Field(min_length=1)
    publication_year: int | None = None
    journal: str | None = None
    pubmed_url: str
    source: str = "pubmed"


@dataclass
class ParseResult:
    """Outcome of parsing one EFetch payload."""

    articles: list[Article] = field(default_factory=list)
    skipped_no_abstract: int = 0
    malformed: int = 0


def _text(element: ET.Element | None) -> str:
    """Flatten an element (including inline markup such as <i>/<sub>) to clean text."""
    if element is None:
        return ""
    return _WS_RE.sub(" ", "".join(element.itertext())).strip()


def _abstract(article: ET.Element) -> str:
    """Join all abstract sections, keeping structured labels (e.g. ``METHODS:``)."""
    parts: list[str] = []
    for node in article.findall("./Abstract/AbstractText"):
        text = _text(node)
        if not text:
            continue
        label = (node.get("Label") or "").strip()
        parts.append(f"{label}: {text}" if label else text)
    return " ".join(parts)


def _year(article: ET.Element) -> int | None:
    """Best-effort publication year from PubDate/Year, MedlineDate, or ArticleDate."""
    for path in (
        "./Journal/JournalIssue/PubDate/Year",
        "./Journal/JournalIssue/PubDate/MedlineDate",
        "./ArticleDate/Year",
    ):
        match = _YEAR_RE.search(_text(article.find(path)))
        if match:
            return int(match.group(1))
    return None


def parse_pubmed_xml(xml_text: str) -> ParseResult:
    """Parse EFetch XML into normalized articles.

    Records without a PMID or title are counted as malformed; records without
    abstract text are counted as skipped. Duplicated PMIDs within the payload are
    collapsed. Raises :class:`PubMedError` only when the document itself is not XML.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise PubMedError(f"PubMed returned XML that could not be parsed: {exc}") from exc

    result = ParseResult()
    seen: set[str] = set()
    for record in root.iter("PubmedArticle"):
        citation = record.find("MedlineCitation")
        article = citation.find("Article") if citation is not None else None
        pmid = _text(citation.find("PMID")) if citation is not None else ""
        if article is None or not pmid.isdigit():
            result.malformed += 1
            continue
        if pmid in seen:
            continue
        seen.add(pmid)
        title = _text(article.find("ArticleTitle"))
        if not title:
            result.malformed += 1
            continue
        abstract = _abstract(article)
        if not abstract:
            result.skipped_no_abstract += 1
            continue
        result.articles.append(
            Article(
                pmid=pmid,
                title=title,
                abstract=abstract,
                publication_year=_year(article),
                journal=_text(article.find("./Journal/Title")) or None,
                pubmed_url=PUBMED_URL_TEMPLATE.format(pmid=pmid),
            )
        )
    # Book records carry no journal abstract in the format we index.
    result.skipped_no_abstract += sum(1 for _ in root.iter("PubmedBookArticle"))
    logger.info(
        "Parsed PubMed XML: %d usable, %d without abstract, %d malformed",
        len(result.articles),
        result.skipped_no_abstract,
        result.malformed,
    )
    return result
