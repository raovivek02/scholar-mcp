"""MCP server definition: tools, resources and prompts.

Design rule: tools *fetch and compute* (search, graphs, ranking, BibTeX); the LLM client does
the *reasoning and writing*. Summaries of related work are produced by the client from tool
output, guided by the prompt templates below, so every claim is traceable to a returned paper.
"""

import json
from typing import Annotated, Any, Literal

from mcp.types import ToolAnnotations
from pydantic import Field

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from .bibtex import to_bibtex
from .errors import InvalidInputError
from .graph import build_citation_graph as _build_graph
from .models import Paper
from .services import get_services

INSTRUCTIONS = (
    "Research-paper assistant backed by arXiv and Semantic Scholar. Typical flow: "
    "search_papers (or search_arxiv for the newest preprints) -> get_paper_details / "
    "get_citations / find_related_papers -> build_citation_graph for structure -> "
    "export_bibtex. Only cite papers that tool results returned; include their url or arXiv id."
)

mcp = _Server("scholar-mcp", instructions=INSTRUCTIONS)
READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=True, idempotentHint=True)

PaperId = Annotated[
    str,
    Field(description="arXiv id (1706.03762), arXiv URL, DOI (10.xxxx/...) or Semantic Scholar id"),
]
Limit = Annotated[int, Field(ge=1, le=50, description="Maximum number of results")]
Year = Annotated[int | None, Field(ge=1900, le=2100, description="Publication year bound")]


@mcp.tool(annotations=READ_ONLY)
async def search_papers(
    query: Annotated[str, Field(min_length=2, description="Keywords, e.g. 'LiDAR moving object segmentation'")],
    limit: Limit = 10,
    year_from: Year = None,
    year_to: Year = None,
    min_citations: Annotated[int | None, Field(ge=0, description="Minimum citation count")] = None,
    sort_by: Literal["relevance", "citations"] = "relevance",
) -> dict[str, Any]:
    """Search papers on Semantic Scholar (all venues, with citation counts and abstracts).

    Best for finding established or influential work. Use search_arxiv for brand-new preprints.
    """
    papers = await get_services().s2.search(query, limit, year_from, year_to, min_citations)
    if sort_by == "citations":
        papers.sort(key=lambda p: p.citation_count or 0, reverse=True)
    return {"query": query, "count": len(papers), "papers": [p.brief() for p in papers]}


@mcp.tool(annotations=READ_ONLY)
async def search_arxiv(
    query: Annotated[str, Field(min_length=2, description="Keywords or an arXiv field query like 'ti:transformer'")],
    limit: Limit = 10,
    category: Annotated[str | None, Field(description="arXiv category such as cs.CV, cs.RO, cs.LG")] = None,
    year_from: Year = None,
    year_to: Year = None,
    sort_by: Literal["relevance", "newest", "updated"] = "relevance",
) -> dict[str, Any]:
    """Search arXiv directly. Use sort_by='newest' to find the latest preprints in a category.

    Results have no citation counts; call get_paper_details for those.
    """
    papers = await get_services().arxiv.search(query, limit, category, year_from, year_to, sort_by)
    return {"query": query, "count": len(papers), "papers": [p.brief() for p in papers]}


@mcp.tool(annotations=READ_ONLY)
async def get_paper_details(paper_id: PaperId) -> dict[str, Any]:
    """Full metadata for one paper: complete abstract, authors, venue, citation counts, links."""
    paper = await get_services().s2.get_paper(paper_id)
    return paper.brief(abstract_chars=None, max_authors=30)


async def _edges(paper_id: str, kind: str, limit: int, influential_only: bool, sort_by: str) -> dict[str, Any]:
    s2 = get_services().s2
    rows = await (s2.get_citations(paper_id) if kind == "citations" else s2.get_references(paper_id))
    fetched = len(rows)
    if influential_only:
        rows = [r for r in rows if r[1]]
    key = (lambda r: r[0].year or 0) if sort_by == "year" else (lambda r: r[0].citation_count or 0)
    rows.sort(key=key, reverse=True)
    items = []
    for paper, influential in rows[:limit]:
        item = paper.brief(abstract_chars=0)
        if influential is not None:
            item["influential"] = influential
        items.append(item)
    return {"paper_id": paper_id, "fetched": fetched, "returned": len(items), kind: items}


@mcp.tool(annotations=READ_ONLY)
async def get_citations(
    paper_id: PaperId,
    limit: Limit = 20,
    influential_only: Annotated[bool, Field(description="Only citations marked influential")] = False,
    sort_by: Literal["citations", "year"] = "citations",
) -> dict[str, Any]:
    """Papers that cite the given paper (forward citations), from a window of up to 200."""
    return await _edges(paper_id, "citations", limit, influential_only, sort_by)


@mcp.tool(annotations=READ_ONLY)
async def get_references(
    paper_id: PaperId,
    limit: Limit = 20,
    sort_by: Literal["citations", "year"] = "citations",
) -> dict[str, Any]:
    """Papers the given paper cites (its bibliography), from a window of up to 200."""
    return await _edges(paper_id, "references", limit, False, sort_by)


@mcp.tool(annotations=READ_ONLY)
async def find_related_papers(paper_id: PaperId, limit: Limit = 10) -> dict[str, Any]:
    """Papers similar to a seed paper, using Semantic Scholar's recommendation model."""
    papers = await get_services().s2.recommend(paper_id, limit)
    return {"seed": paper_id, "count": len(papers), "papers": [p.brief() for p in papers]}


@mcp.tool(annotations=READ_ONLY)
async def build_citation_graph(
    paper_id: PaperId,
    depth: Annotated[int, Field(ge=1, le=2, description="1 = direct neighbours, 2 = neighbours of neighbours")] = 1,
    direction: Literal["citations", "references", "both"] = "both",
    max_nodes: Annotated[int, Field(ge=5, le=100)] = 40,
) -> dict[str, Any]:
    """Build a bounded citation graph around a seed paper.

    Returns nodes, edges (source cites target), and key_papers ranked by how often they are
    cited *within* the graph - a quick way to find foundational work in a subfield. The graph
    is a top-cited sample, not the full network (see the 'truncated' flag).
    """
    return await _build_graph(
        get_services().s2, paper_id, depth=depth, direction=direction, max_nodes=max_nodes
    )


@mcp.tool(annotations=READ_ONLY)
async def export_bibtex(
    paper_ids: Annotated[list[str], Field(min_length=1, max_length=20, description="Paper ids to export")],
) -> dict[str, Any]:
    """Generate BibTeX entries for papers. Done in code, so entries are exact and reproducible."""
    s2 = get_services().s2
    papers: list[Paper] = []
    missing: list[dict[str, str]] = []
    for pid in paper_ids:
        try:
            papers.append(await s2.get_paper(pid))
        except Exception as exc:  # report per-paper failures instead of failing the batch
            missing.append({"paper_id": pid, "error": str(exc)})
    if not papers:
        raise InvalidInputError("None of the papers could be resolved: " + json.dumps(missing))
    return {"count": len(papers), "bibtex": to_bibtex(papers), "missing": missing}


@mcp.resource("paper://{paper_id}", mime_type="text/markdown")
async def paper_resource(paper_id: str) -> str:
    """A paper's metadata and abstract as a Markdown document (arXiv id or Semantic Scholar id)."""
    p = await get_services().s2.get_paper(paper_id)
    lines = [f"# {p.title}", ""]
    meta = [
        ", ".join(p.authors),
        f"{p.venue or 'n/a'} ({p.year or 'n/a'})",
        f"Citations: {p.citation_count if p.citation_count is not None else 'n/a'}",
        p.url or "",
    ]
    lines += [m for m in meta if m] + ["", "## Abstract", "", p.abstract or "No abstract available."]
    return "\n".join(lines)


@mcp.resource("scholar://cache/stats", mime_type="application/json")
def cache_stats() -> str:
    """Cache size and hit/miss counters for this server process."""
    return json.dumps(get_services().cache.stats())


@mcp.prompt()
def related_work_review(
    topic: str, num_papers: int = 10, year_from: int | None = None
) -> str:
    """Write a grounded related-work section on a topic using the tools."""
    since = f" published since {year_from}" if year_from else ""
    return f"""You are writing the related-work section of a research paper on: {topic}

Work in this order, using the scholar-mcp tools:
1. search_papers for the topic{since}; run 2-3 differently-worded queries and, if recency
   matters, search_arxiv with sort_by='newest'. Select the {num_papers} most relevant papers,
   favouring highly cited foundational work plus a few recent ones.
2. For the 2-3 most central papers, call build_citation_graph (depth 1) or get_references to
   see what the field builds on, and find_related_papers to catch anything you missed.
3. Write 3-5 paragraphs grouped by THEME or METHOD (not paper-by-paper). For each paper say
   what it contributes and how it relates to the others.

Rules:
- Use ONLY information found in tool results (titles, abstracts, venues, years, counts).
  If an abstract does not support a claim, do not make it.
- Cite every claim as (FirstAuthor et al., Year) and list arXiv ids or URLs at the end.
- Never cite a paper that no tool returned.
- State limitations plainly: coverage is limited to what the APIs returned.
- Finish by calling export_bibtex for the papers you cited and include the result."""


@mcp.prompt()
def literature_map(seed_paper: str) -> str:
    """Map the intellectual neighbourhood of one paper: foundations, followers, open directions."""
    return f"""Create a literature map around this paper: {seed_paper}

1. get_paper_details for the seed, then build_citation_graph (depth 2, direction 'both').
2. Report: (a) foundations - key_papers it builds on, (b) direct follow-ups (get_citations
   sorted by year), (c) parallel work (find_related_papers), (d) apparent gaps or open
   directions, labelled clearly as your inference.
3. Ground every statement in returned data and cite arXiv ids or URLs. Note the graph is a
   sample (check the 'truncated' flag)."""
