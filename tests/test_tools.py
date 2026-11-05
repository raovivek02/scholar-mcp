import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from scholar_mcp import server
from scholar_mcp.errors import InvalidInputError, NotFoundError

EXPECTED_TOOLS = {
    "search_papers", "search_arxiv", "get_paper_details", "get_citations",
    "get_references", "find_related_papers", "build_citation_graph", "export_bibtex",
}


async def test_search_papers_sorting(services):
    out = await server.search_papers("lidar", limit=5, sort_by="citations")
    assert [p["title"] for p in out["papers"]] == ["Foundation Paper", "Seed Paper"]
    assert "url" in out["papers"][0]


async def test_search_arxiv(services):
    out = await server.search_arxiv("lidar mos", limit=2, category="cs.RO")
    assert out["count"] == 2 and out["papers"][0]["arxiv_id"] == "2105.08971"


async def test_citations_filters(services):
    out = await server.get_citations("1706.03762", limit=5, influential_only=True)
    assert out["fetched"] == 2 and out["returned"] == 1
    assert out["citations"][0]["title"] == "Follow-up A" and "abstract" not in out["citations"][0]


async def test_references_and_related(services):
    refs = await server.get_references("1706.03762")
    assert refs["references"][0]["title"] == "Foundation Paper"
    rel = await server.find_related_papers("1706.03762", limit=2)
    assert rel["count"] == 2


async def test_details_full_abstract(services):
    out = await server.get_paper_details("1706.03762")
    assert out["abstract"] == "Abstract of Seed Paper."


async def test_bibtex_reports_missing(services):
    out = await server.export_bibtex(["1706.03762", "9999.99999"])
    assert out["count"] == 1 and "@article{lovelace2018seed" in out["bibtex"]
    assert out["missing"][0]["paper_id"] == "9999.99999"


async def test_not_found_is_actionable(services):
    with pytest.raises(NotFoundError):
        await server.get_paper_details("9999.99999")
    with pytest.raises(InvalidInputError, match="Unrecognised"):
        await server.get_paper_details("garbage")


async def test_prompt_and_resource(services):
    text = server.related_work_review("lidar mos", 8, 2020)
    assert "lidar mos" in text and "Never cite a paper that no tool returned" in text
    assert "# Seed Paper" in await server.paper_resource("1706.03762")


async def test_stdio_protocol_end_to_end():
    """Spawn the real server over stdio: catches stdout pollution and schema problems."""
    params = StdioServerParameters(command=sys.executable, args=["-m", "scholar_mcp"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name for t in (await session.list_tools()).tools}
            assert tools == EXPECTED_TOOLS
            bad = await session.call_tool("get_paper_details", {"paper_id": "garbage"})
            assert bad.is_error if hasattr(bad, 'is_error') else bad.isError
            assert "Unrecognised paper id" in bad.content[0].text
            invalid = await session.call_tool("search_papers", {"query": "x", "limit": 999})
            assert getattr(invalid, 'is_error', getattr(invalid, 'isError', False))  # schema validation (limit <= 50) rejects it
