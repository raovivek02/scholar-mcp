import json

import httpx

from scholar_mcp.arxiv_client import build_query, parse_feed
from scholar_mcp.graph import build_citation_graph
from tests.conftest import ATOM, pid


def test_parse_arxiv_feed():
    papers = parse_feed(ATOM)
    first, legacy = papers
    assert first.arxiv_id == "2105.08971"  # version stripped
    assert first.title == "Moving Object Segmentation in 3D LiDAR Data"  # whitespace collapsed
    assert first.abstract == "We propose a method for MOS."
    assert first.year == 2021 and first.venue == "IEEE RA-L 2021"
    assert first.categories[0] == "cs.RO"
    assert first.pdf_url == "http://arxiv.org/pdf/2105.08971v2"
    assert legacy.arxiv_id == "hep-th/9901001" and legacy.venue == "arXiv"
    assert legacy.pdf_url == "https://arxiv.org/pdf/hep-th/9901001"


def test_build_query():
    q = build_query("lidar segmentation", "cs.CV", 2020, 2024)
    assert q == 'all:"lidar segmentation" AND cat:cs.CV AND submittedDate:[202001010000 TO 202412312359]'
    assert build_query("ti:transformer", None, None, None) == "ti:transformer"


async def test_s2_search_params_and_auth(services):
    seen = []
    inner = services.s2._http._client._transport

    def spy(request):
        seen.append(request)
        return inner.handler(request)

    services.s2._http._client._transport = httpx.MockTransport(spy)
    papers = await services.s2.search("lidar", limit=5, year_from=2020, min_citations=10)
    req = seen[0]
    assert req.url.params["year"] == "2020-" and req.url.params["minCitationCount"] == "10"
    assert req.headers["x-api-key"] == "test-key"
    assert papers[0].arxiv_id == "1706.03762"
    assert papers[0].pdf_url == "https://arxiv.org/pdf/1706.03762"


async def test_get_paper_by_arxiv_id(services):
    paper = await services.s2.get_paper("arXiv:1706.03762v3")
    assert paper.title == "Seed Paper"


async def test_graph_depth1(services):
    g = await build_citation_graph(services.s2, "1706.03762", depth=1)
    ids = {n["paper_id"] for n in g["nodes"]}
    assert ids == {pid(1), pid(2), pid(3), pid(5)}
    edges = {(e["source"], e["target"]) for e in g["edges"]}
    assert edges == {(pid(2), pid(1)), (pid(3), pid(1)), (pid(1), pid(5))}
    assert g["key_papers"][0]["paper_id"] in {pid(1), pid(5)}
    assert not g["truncated"]


async def test_graph_depth2_and_cap(services):
    g = await build_citation_graph(services.s2, "1706.03762", depth=2)
    assert pid(4) in {n["paper_id"] for n in g["nodes"]}
    assert (pid(4), pid(2)) in {(e["source"], e["target"]) for e in g["edges"]}
    capped = await build_citation_graph(services.s2, "1706.03762", depth=2, max_nodes=3)
    assert len(capped["nodes"]) == 3 and capped["truncated"]
    assert all(
        e["source"] in {n["paper_id"] for n in capped["nodes"]} for e in capped["edges"]
    )
    json.dumps(g)  # must be JSON serialisable
