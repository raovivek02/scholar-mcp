import httpx
import pytest

from scholar_mcp.cache import Cache
from scholar_mcp.errors import NotFoundError, RateLimitedError, ScholarError
from scholar_mcp.http import HttpClient


def make(handler, **kw):
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    client = HttpClient("test", min_interval=0, transport=httpx.MockTransport(handler),
                        sleep=fake_sleep, **kw)
    return client, sleeps


async def test_retry_honours_retry_after():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, json={"ok": True})

    client, sleeps = make(handler)
    assert await client.get_json("https://x.test/a") == {"ok": True}
    assert calls["n"] == 2 and 7.0 in sleeps


async def test_gives_up_with_actionable_error():
    client, _ = make(lambda r: httpx.Response(429), max_retries=2)
    with pytest.raises(RateLimitedError, match="SEMANTIC_SCHOLAR_API_KEY"):
        await client.get_text("https://x.test/a")


async def test_404_and_4xx():
    client, _ = make(lambda r: httpx.Response(404))
    with pytest.raises(NotFoundError):
        await client.get_text("https://x.test/a")
    client, _ = make(lambda r: httpx.Response(400, text="bad"))
    with pytest.raises(ScholarError, match="400"):
        await client.get_text("https://x.test/a")


async def test_response_cache_avoids_second_request():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json={"n": calls["n"]})

    client, _ = make(handler, cache=Cache(":memory:"))
    first = await client.get_json("https://x.test/a", {"q": 1}, ttl=60)
    second = await client.get_json("https://x.test/a", {"q": 1}, ttl=60)
    third = await client.get_json("https://x.test/a", {"q": 2}, ttl=60)
    assert first == second and third != first and calls["n"] == 2


def test_cache_ttl_expiry():
    now = [1000.0]
    cache = Cache(":memory:", clock=lambda: now[0])
    cache.set("k", "v", ttl=10)
    assert cache.get("k") == "v"
    now[0] += 11
    assert cache.get("k") is None
    assert cache.stats()["entries"] == 0
