"""Data models returned by the tools. Every paper carries a URL so claims can be verified."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Paper(BaseModel):
    title: str
    source: str  # "semantic_scholar" | "arxiv"
    paper_id: str | None = None  # Semantic Scholar id
    arxiv_id: str | None = None
    doi: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    citation_count: int | None = None
    influential_citation_count: int | None = None
    url: str | None = None
    pdf_url: str | None = None
    categories: list[str] = Field(default_factory=list)

    def brief(self, abstract_chars: int | None = 600, max_authors: int = 6) -> dict[str, Any]:
        """Compact dict for LLM consumption: drops empty fields, trims long text."""
        data = self.model_dump(exclude_none=True, exclude_defaults=False)
        data = {k: v for k, v in data.items() if v not in (None, [], "")}
        if len(self.authors) > max_authors:
            data["authors"] = [*self.authors[:max_authors], f"... (+{len(self.authors) - max_authors} more)"]
        abstract = data.get("abstract")
        if abstract is not None:
            if abstract_chars == 0:
                del data["abstract"]
            elif abstract_chars and len(abstract) > abstract_chars:
                data["abstract"] = abstract[:abstract_chars].rstrip() + "..."
        return data
