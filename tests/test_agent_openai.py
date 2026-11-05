"""Run the real stdio server with a scripted fake LLM (no network, no API key needed)."""

import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "client"))

from agent_openai import run_agent_openai  # noqa: E402


async def test_full_loop_with_scripted_llm():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if len(seen) == 1:
            assert {t["function"]["name"] for t in body["tools"]} >= {"search_papers", "build_citation_graph"}
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "get_paper_details", "arguments": json.dumps({"paper_id": "garbage"})},
            }]}
        else:
            last = body["messages"][-1]
            assert last["role"] == "tool" and "Unrecognised paper id" in last["content"]
            msg = {"role": "assistant", "content": "That paper id is invalid, so I could not find it."}
        return httpx.Response(200, json={"choices": [{"message": msg}]})

    result = await run_agent_openai("Summarise garbage", transport=httpx.MockTransport(handler))
    assert result.turns == 2 and "could not find" in result.answer
    assert result.tool_calls[0].name == "get_paper_details" and result.tool_calls[0].is_error


def test_simplify_schema_for_strict_providers():
    from agent_openai import simplify_schema

    raw = {
        "title": "x", "type": "object", "additionalProperties": False,
        "properties": {
            "year_from": {"anyOf": [{"type": "integer", "minimum": 1900}, {"type": "null"}], "default": None, "title": "Y"},
            "limit": {"type": "integer", "default": 10, "maximum": 50},
        },
    }
    out = simplify_schema(raw)
    assert out == {"type": "object", "properties": {
        "year_from": {"type": "integer", "minimum": 1900}, "limit": {"type": "integer", "maximum": 50}}}


async def test_retries_on_429():
    from agent_openai import _post_with_retry

    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"ok": True})

    async def no_sleep(_):
        return None

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        resp = await _post_with_retry(http, "https://x.test/chat", {}, sleep=no_sleep)
    assert resp.status_code == 200 and calls["n"] == 3
