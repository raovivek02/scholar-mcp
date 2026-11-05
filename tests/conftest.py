from urllib.parse import unquote

import httpx
import pytest

from scholar_mcp.services import Services, build_services, set_services


def pid(n: int) -> str:
    return f"{n:040x}"


def mk(n: int, title: str, cites: int, year: int = 2020, arxiv: str | None = None, **extra):
    d = {
        "paperId": pid(n),
        "title": title,
        "year": year,
        "venue": extra.pop("venue", "NeurIPS"),
        "citationCount": cites,
        "influentialCitationCount": cites // 10,
        "authors": [{"name": "Ada Lovelace"}, {"name": "Alan Turing"}],
        "externalIds": {"ArXiv": arxiv} if arxiv else {},
        "abstract": f"Abstract of {title}.",
        "url": f"https://www.semanticscholar.org/paper/{pid(n)}",
    }
    d.update(extra)
    return d


P1 = mk(1, "Seed Paper", 100, 2018, arxiv="1706.03762")
P2 = mk(2, "Follow-up A", 50, 2020)
P3 = mk(3, "Follow-up B", 30, 2021)
P4 = mk(4, "Second-hop Paper", 5, 2022)
P5 = mk(5, "Foundation Paper", 900, 2012)
PAPERS = {p["paperId"]: p for p in (P1, P2, P3, P4, P5)}
CITATIONS = {pid(1): [P2, P3], pid(2): [P4], pid(3): [], pid(4): [], pid(5): []}
REFERENCES = {pid(1): [P5], pid(2): [], pid(3): [], pid(4): [], pid(5): []}


def s2_handler(request: httpx.Request) -> httpx.Response:
    path = unquote(request.url.path)
    if path.endswith("/paper/search"):
        return httpx.Response(200, json={"total": 2, "data": [P1, P5]})
    if "/recommendations/v1/" in path:
        return httpx.Response(200, json={"recommendedPapers": [P2, P3]})
    parts = path.split("/paper/", 1)[1].split("/")
    ident = parts[0]
    key = pid(1) if ident == "ARXIV:1706.03762" else ident
    if key not in PAPERS:
        return httpx.Response(404, json={"error": "Paper not found"})
    if len(parts) == 1:
        return httpx.Response(200, json=PAPERS[key])
    if parts[1] == "citations":
        data = [{"isInfluential": i == 0, "citingPaper": c} for i, c in enumerate(CITATIONS[key])]
        return httpx.Response(200, json={"data": data})
    data = [{"isInfluential": False, "citedPaper": c} for c in REFERENCES[key]]
    return httpx.Response(200, json={"data": data})


@pytest.fixture
def services() -> Services:
    svc = build_services(
        cache_path=":memory:",
        api_key="test-key",
        s2_transport=httpx.MockTransport(s2_handler),
        arxiv_transport=httpx.MockTransport(lambda r: httpx.Response(200, text=ATOM)),
        min_interval_s2=0,
        min_interval_arxiv=0,
    )
    set_services(svc)
    yield svc
    set_services(None)


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2105.08971v2</id>
    <published>2021-05-19T10:00:00Z</published>
    <title>Moving Object Segmentation
      in 3D LiDAR Data</title>
    <summary>  We propose a   method for MOS. </summary>
    <author><name>Xieyuanli Chen</name></author>
    <author><name>Cyrill Stachniss</name></author>
    <arxiv:primary_category term="cs.RO"/>
    <category term="cs.CV"/>
    <category term="cs.RO"/>
    <arxiv:journal_ref>IEEE RA-L 2021</arxiv:journal_ref>
    <link href="http://arxiv.org/abs/2105.08971v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2105.08971v2" rel="related" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/hep-th/9901001v1</id>
    <published>1999-01-01T00:00:00Z</published>
    <title>Old Style Id</title>
    <summary>Legacy.</summary>
    <author><name>A. Person</name></author>
    <category term="hep-th"/>
  </entry>
</feed>"""
