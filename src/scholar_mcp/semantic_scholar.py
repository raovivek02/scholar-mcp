"""Semantic Scholar Graph API client (search, details, citations, references, recommendations)."""

from __future__ import annotations

from typing import Any

from .http import HttpClient
from .ids import normalize_paper_id
from .models import Paper

GRAPH = "https://api.semanticscholar.org/graph/v1"
RECS = "https://api.semanticscholar.org/recommendations/v1"
FULL_FIELDS = (
    "paperId,title,abstract,year,venue,citationCount,influentialCitationCount,"
    "authors,externalIds,url,openAccessPdf"
)
LIST_FIELDS = (
    "paperId,title,year,venue,citationCount,influentialCitationCount,authors,externalIds,url"
)
# Citation lists are not sortable server-side, so we fetch a window and rank locally.
FETCH_WINDOW = 200

TTL_SEARCH = 3600
TTL_PAPER = 24 * 3600
TTL_EDGES = 6 * 3600


def to_paper(d: dict[str, Any]) -> Paper:
    ext = d.get("externalIds") or {}
    arxiv = ext.get("ArXiv")
    pdf = (d.get("openAccessPdf") or {}).get("url") or (
        f"https://arxiv.org/pdf/{arxiv}" if arxiv else None
    )
    url = d.get("url") or (f"https://arxiv.org/abs/{arxiv}" if arxiv else None)
    return Paper(
        source="semantic_scholar",
        paper_id=d.get("paperId"),
        arxiv_id=arxiv,
        doi=ext.get("DOI"),
        title=d.get("title") or "(untitled)",
        authors=[a["name"] for a in d.get("authors") or [] if a.get("name")],
        year=d.get("year"),
        venue=d.get("venue") or None,
        abstract=d.get("abstract"),
        citation_count=d.get("citationCount"),
        influential_citation_count=d.get("influentialCitationCount"),
        url=url,
        pdf_url=pdf,
    )


class SemanticScholarClient:
    def __init__(self, http: HttpClient):
        self._http = http

    async def search(
        self,
        query: str,
        limit: int = 10,
        year_from: int | None = None,
        year_to: int | None = None,
        min_citations: int | None = None,
    ) -> list[Paper]:
        params: dict[str, Any] = {"query": query, "limit": limit, "fields": FULL_FIELDS}
        if year_from or year_to:
            if year_from and year_to and year_from == year_to:
                params["year"] = str(year_from)
            else:
                params["year"] = f"{year_from or ''}-{year_to or ''}"
        if min_citations:
            params["minCitationCount"] = min_citations
        data = await self._http.get_json(f"{GRAPH}/paper/search", params, TTL_SEARCH)
        return [to_paper(d) for d in data.get("data") or []]

    async def get_paper(self, paper_id: str) -> Paper:
        pid = normalize_paper_id(paper_id)
        data = await self._http.get_json(
            f"{GRAPH}/paper/{pid}", {"fields": FULL_FIELDS}, TTL_PAPER
        )
        return to_paper(data)

    async def _edges(self, paper_id: str, kind: str) -> list[tuple[Paper, bool | None]]:
        pid = normalize_paper_id(paper_id)
        key = "citingPaper" if kind == "citations" else "citedPaper"
        data = await self._http.get_json(
            f"{GRAPH}/paper/{pid}/{kind}",
            {"fields": LIST_FIELDS + ",isInfluential", "limit": FETCH_WINDOW},
            TTL_EDGES,
        )
        out: list[tuple[Paper, bool | None]] = []
        for row in data.get("data") or []:
            inner = row.get(key) or {}
            if inner.get("paperId") and inner.get("title"):
                out.append((to_paper(inner), row.get("isInfluential")))
        return out

    async def get_citations(self, paper_id: str) -> list[tuple[Paper, bool | None]]:
        """Papers that cite ``paper_id`` (up to FETCH_WINDOW)."""
        return await self._edges(paper_id, "citations")

    async def get_references(self, paper_id: str) -> list[tuple[Paper, bool | None]]:
        """Papers that ``paper_id`` cites (up to FETCH_WINDOW)."""
        return await self._edges(paper_id, "references")

    async def recommend(self, paper_id: str, limit: int = 10) -> list[Paper]:
        pid = normalize_paper_id(paper_id)
        data = await self._http.get_json(
            f"{RECS}/papers/forpaper/{pid}", {"fields": FULL_FIELDS, "limit": limit}, TTL_EDGES
        )
        return [to_paper(d) for d in data.get("recommendedPapers") or []]
