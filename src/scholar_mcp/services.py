"""Wiring: build the HTTP clients, cache and API wrappers from environment configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import httpx

from .arxiv_client import ArxivClient
from .cache import Cache
from .http import HttpClient
from .semantic_scholar import SemanticScholarClient

DEFAULT_CACHE = "~/.cache/scholar-mcp/cache.sqlite"


@dataclass
class Services:
    s2: SemanticScholarClient
    arxiv: ArxivClient
    cache: Cache


def build_services(
    *,
    cache_path: str | Path | None = None,
    api_key: str | None = None,
    s2_transport: httpx.AsyncBaseTransport | None = None,
    arxiv_transport: httpx.AsyncBaseTransport | None = None,
    min_interval_s2: float | None = None,
    min_interval_arxiv: float | None = None,
) -> Services:
    cache = Cache(cache_path or os.environ.get("SCHOLAR_MCP_CACHE", DEFAULT_CACHE))
    key = api_key if api_key is not None else os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers = {"x-api-key": key} if key else {}
    s2_http = HttpClient(
        "Semantic Scholar",
        min_interval=1.6 if min_interval_s2 is None else min_interval_s2,
        headers=headers,
        cache=cache,
        max_retries=5,
        transport=s2_transport,
    )
    arxiv_http = HttpClient(
        "arXiv",
        min_interval=3.0 if min_interval_arxiv is None else min_interval_arxiv,
        cache=cache,
        transport=arxiv_transport,
    )
    return Services(SemanticScholarClient(s2_http), ArxivClient(arxiv_http), cache)


_services: Services | None = None


def get_services() -> Services:
    global _services
    if _services is None:
        _services = build_services()
    return _services


def set_services(services: Services | None) -> None:
    global _services
    _services = services
