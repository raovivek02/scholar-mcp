"""arXiv Atom API client."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date

from .errors import InvalidInputError, ScholarError
from .http import HttpClient
from .models import Paper

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
TTL_SEARCH = 3600
CATEGORY_RE = re.compile(r"^[A-Za-z\-]+(\.[A-Za-z\-]+)?$")
_ID_RE = re.compile(r"/abs/(.+?)(?:v\d+)?$")

SORT_OPTIONS = {"relevance": "relevance", "newest": "submittedDate", "updated": "lastUpdatedDate"}


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def build_query(
    query: str, category: str | None, year_from: int | None, year_to: int | None
) -> str:
    q = query.replace('"', " ").strip()
    if not q:
        raise InvalidInputError("query is empty.")
    parts = [q if re.match(r"^(all|ti|abs|au|cat):", q) else f'all:"{q}"']
    if category:
        if not CATEGORY_RE.match(category):
            raise InvalidInputError(f"Invalid arXiv category '{category}' (example: cs.CV).")
        parts.append(f"cat:{category}")
    if year_from or year_to:
        start = f"{year_from or 1991}01010000"
        end = f"{year_to or date.today().year}12312359"
        parts.append(f"submittedDate:[{start} TO {end}]")
    return " AND ".join(parts)


def parse_feed(xml_text: str) -> list[Paper]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ScholarError("arXiv returned an unreadable response.") from exc
    papers: list[Paper] = []
    for entry in root.findall("a:entry", NS):
        raw_id = _clean(entry.findtext("a:id", namespaces=NS))
        if "api/errors" in raw_id:
            raise ScholarError(f"arXiv rejected the query: {_clean(entry.findtext('a:summary', namespaces=NS))}")
        m = _ID_RE.search(raw_id)
        arxiv_id = m.group(1) if m else None
        published = entry.findtext("a:published", namespaces=NS) or ""
        pdf = next(
            (
                lk.get("href")
                for lk in entry.findall("a:link", NS)
                if lk.get("title") == "pdf" and lk.get("href")
            ),
            f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None,
        )
        primary = entry.find("x:primary_category", NS)
        cats = [c.get("term") for c in entry.findall("a:category", NS) if c.get("term")]
        if primary is not None and primary.get("term") in cats:
            cats.remove(primary.get("term"))
            cats.insert(0, primary.get("term"))
        papers.append(
            Paper(
                source="arxiv",
                arxiv_id=arxiv_id,
                title=_clean(entry.findtext("a:title", namespaces=NS)),
                authors=[
                    _clean(a.findtext("a:name", namespaces=NS))
                    for a in entry.findall("a:author", NS)
                ],
                year=int(published[:4]) if published[:4].isdigit() else None,
                venue=_clean(entry.findtext("x:journal_ref", namespaces=NS)) or "arXiv",
                abstract=_clean(entry.findtext("a:summary", namespaces=NS)),
                doi=_clean(entry.findtext("x:doi", namespaces=NS)) or None,
                url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else raw_id,
                pdf_url=pdf,
                categories=cats,
            )
        )
    return papers


class ArxivClient:
    def __init__(self, http: HttpClient):
        self._http = http

    async def search(
        self,
        query: str,
        max_results: int = 10,
        category: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        sort_by: str = "relevance",
    ) -> list[Paper]:
        if sort_by not in SORT_OPTIONS:
            raise InvalidInputError(f"sort_by must be one of {sorted(SORT_OPTIONS)}.")
        params = {
            "search_query": build_query(query, category, year_from, year_to),
            "start": 0,
            "max_results": max_results,
            "sortBy": SORT_OPTIONS[sort_by],
            "sortOrder": "descending",
        }
        return parse_feed(await self._http.get_text(API, params, TTL_SEARCH))
