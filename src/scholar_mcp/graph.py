"""Bounded citation-graph construction on top of the Semantic Scholar client."""

from __future__ import annotations

from typing import Any

from .models import Paper
from .semantic_scholar import SemanticScholarClient


def _rank(papers: list[Paper]) -> list[Paper]:
    return sorted(papers, key=lambda p: p.citation_count or 0, reverse=True)


async def build_citation_graph(
    s2: SemanticScholarClient,
    seed_id: str,
    *,
    depth: int = 1,
    direction: str = "both",
    max_nodes: int = 40,
    per_node: int = 10,
    expand_per_level: int = 3,
) -> dict[str, Any]:
    """Breadth-first expansion around a seed paper.

    Edges point from the citing paper to the cited paper (``source`` cites ``target``).
    Growth is bounded three ways: ``per_node`` neighbours per expanded paper (highest cited
    first), ``expand_per_level`` papers expanded per level, and ``max_nodes`` overall.
    """
    seed = await s2.get_paper(seed_id)
    if not seed.paper_id:
        raise ValueError("Seed paper has no Semantic Scholar id.")
    nodes: dict[str, tuple[Paper, int]] = {seed.paper_id: (seed, 0)}
    edges: dict[tuple[str, str], bool | None] = {}
    truncated = False
    frontier = [seed]

    for level in range(1, depth + 1):
        discovered: list[Paper] = []
        for node in frontier[: max(expand_per_level, 1) if level > 1 else 1]:
            fetches: list[tuple[str, list[tuple[Paper, bool | None]]]] = []
            if direction in ("citations", "both"):
                fetches.append(("citations", await s2.get_citations(node.paper_id or "")))
            if direction in ("references", "both"):
                fetches.append(("references", await s2.get_references(node.paper_id or "")))
            for kind, rows in fetches:
                ranked = sorted(rows, key=lambda r: r[0].citation_count or 0, reverse=True)
                for paper, influential in ranked[:per_node]:
                    pid = paper.paper_id
                    if not pid:
                        continue
                    if pid not in nodes:
                        if len(nodes) >= max_nodes:
                            truncated = True
                            continue
                        nodes[pid] = (paper, level)
                        discovered.append(paper)
                    edge = (pid, node.paper_id) if kind == "citations" else (node.paper_id, pid)
                    edges[edge] = influential  # type: ignore[index]
        frontier = _rank(discovered)
        if not frontier:
            break

    edge_list = [
        {"source": s, "target": t, **({"influential": inf} if inf is not None else {})}
        for (s, t), inf in edges.items()
        if s in nodes and t in nodes
    ]
    in_degree: dict[str, int] = {}
    for e in edge_list:
        in_degree[e["target"]] = in_degree.get(e["target"], 0) + 1

    node_list = [
        {**p.brief(abstract_chars=0), "depth": d, "cited_within_graph": in_degree.get(pid, 0)}
        for pid, (p, d) in nodes.items()
    ]
    key_papers = sorted(
        node_list,
        key=lambda n: (n["cited_within_graph"], n.get("citation_count") or 0),
        reverse=True,
    )[:5]
    return {
        "seed": seed.paper_id,
        "direction": direction,
        "depth": depth,
        "nodes": node_list,
        "edges": edge_list,
        "key_papers": [
            {"paper_id": n["paper_id"], "title": n["title"], "cited_within_graph": n["cited_within_graph"]}
            for n in key_papers
        ],
        "truncated": truncated,
        "note": "edges: source cites target. Graph is a bounded sample (top-cited neighbours), "
        "not the complete citation network.",
    }
