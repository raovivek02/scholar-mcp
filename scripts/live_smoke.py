"""One-off check against the REAL arXiv / Semantic Scholar APIs (the unit tests use mocks).

    python scripts/live_smoke.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scholar_mcp import server  # noqa: E402


async def main() -> None:
    steps = [
        ("search_papers", server.search_papers("LiDAR moving object segmentation", limit=3)),
        ("search_arxiv", server.search_arxiv("moving object segmentation", limit=3, category="cs.CV")),
        ("get_paper_details", server.get_paper_details("1706.03762")),
        ("get_citations", server.get_citations("1706.03762", limit=3)),
        ("build_citation_graph", server.build_citation_graph("1706.03762", depth=1, max_nodes=15)),
        ("export_bibtex", server.export_bibtex(["1706.03762"])),
    ]
    for name, coro in steps:
        try:
            out = await coro
            print(f"OK    {name}: keys={list(out)[:5]}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {name}: {exc}")


asyncio.run(main())
