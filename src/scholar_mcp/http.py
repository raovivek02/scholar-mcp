"""HTTP client with per-host rate limiting, retry/backoff, and optional response caching."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from .cache import Cache
from .errors import NotFoundError, RateLimitedError, ScholarError, UpstreamError

log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
USER_AGENT = "scholar-mcp/0.1 (research assistant MCP server)"


class HttpClient:
    def __init__(
        self,
        name: str,
        *,
        min_interval: float = 1.0,
        headers: dict[str, str] | None = None,
        cache: Cache | None = None,
        max_retries: int = 3,
        timeout: float = 20.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.name = name
        self._min_interval = min_interval
        self._cache = cache
        self._max_retries = max_retries
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._last = 0.0
        self._client = httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT, **(headers or {})},
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )

    async def _throttle(self) -> None:
        async with self._lock:
            wait = self._last + self._min_interval - time.monotonic()
            if wait > 0:
                await self._sleep(wait)
            self._last = time.monotonic()

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> float:
        if retry_after and retry_after.isdigit():
            return min(float(retry_after), 60.0)
        return float(min(2**attempt, 30))

    async def _fetch(self, url: str, params: dict[str, Any] | None) -> str:
        for attempt in range(self._max_retries + 1):
            await self._throttle()
            last = attempt == self._max_retries
            try:
                resp = await self._client.get(url, params=params)
            except httpx.TransportError as exc:
                if last:
                    raise UpstreamError(
                        f"{self.name} is unreachable ({exc.__class__.__name__}). Try again later."
                    ) from exc
                await self._sleep(self._backoff(attempt, None))
                continue
            code = resp.status_code
            if code == 200:
                return resp.text
            if code == 404:
                raise NotFoundError(f"{self.name}: resource not found ({url}).")
            if code == 429 or code >= 500:
                if last:
                    if code == 429:
                        raise RateLimitedError(
                            f"{self.name} is rate limiting requests. Wait a minute, or set "
                            "SEMANTIC_SCHOLAR_API_KEY for higher limits."
                        )
                    raise UpstreamError(f"{self.name} returned HTTP {code}. Try again later.")
                log.warning("%s returned %s; retrying (attempt %s)", self.name, code, attempt + 1)
                await self._sleep(self._backoff(attempt, resp.headers.get("Retry-After")))
                continue
            raise ScholarError(f"{self.name} rejected the request (HTTP {code}): {resp.text[:200]}")
        raise UpstreamError(f"{self.name}: request failed")  # pragma: no cover

    async def get_text(
        self, url: str, params: dict[str, Any] | None = None, ttl: float | None = None
    ) -> str:
        key = None
        if self._cache is not None and ttl:
            key = Cache.make_key(self.name, url, params)
            hit = self._cache.get(key)
            if hit is not None:
                return hit
        text = await self._fetch(url, params)
        if key is not None and self._cache is not None:
            self._cache.set(key, text, ttl or 0)
        return text

    async def get_json(
        self, url: str, params: dict[str, Any] | None = None, ttl: float | None = None
    ) -> Any:
        text = await self.get_text(url, params, ttl)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise UpstreamError(f"{self.name} returned invalid JSON.") from exc

    async def aclose(self) -> None:
        await self._client.aclose()
