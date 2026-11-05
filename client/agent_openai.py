"""Agent loop for any OpenAI-compatible chat API: Ollama (local, free), Groq, Gemini, OpenRouter...

    # local, free (after `ollama pull qwen2.5:32b`):
    python client/agent_openai.py "Related work on LiDAR moving object segmentation"

    # Gemini free tier (key from aistudio.google.com):
    export LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
    export LLM_MODEL=gemini-2.5-flash      # any free model listed in AI Studio

Environment (all optional):
    LLM_BASE_URL  default http://localhost:11434/v1   (Ollama)
    LLM_MODEL     default qwen2.5:32b
    LLM_API_KEY   not needed for Ollama; required by hosted providers
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any

import httpx
from agent import SYSTEM, AgentResult, ToolCall, _get, server_params
from mcp import ClientSession
from mcp.client.stdio import stdio_client

DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:32b")


_DROP_KEYS = {"title", "default", "additionalProperties", "$schema"}


def simplify_schema(node: Any) -> Any:
    """Make a Pydantic JSON schema acceptable to stricter providers such as Gemini.

    Drops keys they reject (title/default/additionalProperties) and turns the ``X | None`` form
    (``anyOf: [X, {"type": "null"}]``) into plain ``X``.
    """
    if isinstance(node, list):
        return [simplify_schema(n) for n in node]
    if not isinstance(node, dict):
        return node
    out = {k: simplify_schema(v) for k, v in node.items() if k not in _DROP_KEYS}
    variants = out.get("anyOf")
    if variants:
        real = [v for v in variants if v.get("type") != "null"]
        if len(real) == 1:
            out.pop("anyOf")
            out = {**real[0], **out}
    return out


MAX_TOOL_CHARS = 12000  # keep tool results small: faster and cheaper on free tiers
TIMEOUT = httpx.Timeout(connect=15, read=120, write=30, pool=30)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


async def _post_with_retry(
    http: httpx.AsyncClient, url: str, payload: dict[str, Any], sleep: Any = asyncio.sleep
) -> httpx.Response:
    """Free tiers rate limit aggressively: back off and retry on 429 / 5xx / timeouts."""
    for attempt in range(6):
        try:
            resp = await http.post(url, json=payload)
        except httpx.TransportError as exc:  # includes read timeouts
            if attempt == 5:
                raise RuntimeError(
                    f"LLM endpoint did not respond ({exc.__class__.__name__}) after 6 attempts. "
                    "Try a different LLM_MODEL (a Flash-Lite model is faster) or retry later."
                ) from exc
            _log(f"[llm] {exc.__class__.__name__}; retrying (attempt {attempt + 2}/6)")
            await sleep(min(2 ** (attempt + 1), 30))
            continue
        if resp.status_code not in (429, 500, 502, 503) or attempt == 5:
            return resp
        _log(f"[llm] HTTP {resp.status_code}; retrying (attempt {attempt + 2}/6)")
        retry_after = resp.headers.get("Retry-After", "")
        await sleep(float(retry_after) if retry_after.replace(".", "", 1).isdigit() else min(2 ** (attempt + 1), 60))
    return resp  # pragma: no cover


async def run_agent_openai(
    question: str,
    model: str = DEFAULT_MODEL,
    max_turns: int = 12,
    base_url: str = DEFAULT_BASE_URL,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AgentResult:
    headers = {"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"} if os.environ.get("LLM_API_KEY") else {}
    async with httpx.AsyncClient(headers=headers, timeout=TIMEOUT, transport=transport) as http:
        async with stdio_client(server_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = (await session.list_tools()).tools
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description or "",
                            "parameters": simplify_schema(_get(t, "inputSchema", "input_schema")),
                        },
                    }
                    for t in listed
                ]
                messages: list[dict[str, Any]] = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": question},
                ]
                result = AgentResult(answer="")
                for turn in range(1, max_turns + 1):
                    result.turns = turn
                    started = time.monotonic()
                    _log(f"[llm] turn {turn}: asking {model} ...")
                    resp = await _post_with_retry(
                        http,
                        f"{base_url.rstrip('/')}/chat/completions",
                        {"model": model, "messages": messages, "tools": tools, "temperature": 0.2},
                    )
                    if resp.status_code != 200:
                        raise RuntimeError(f"LLM endpoint returned HTTP {resp.status_code}: {resp.text[:300]}")
                    _log(f"[llm] turn {turn}: answered in {time.monotonic() - started:.1f}s")
                    msg = resp.json()["choices"][0]["message"]
                    calls = msg.get("tool_calls") or []
                    assistant: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
                    if calls:
                        assistant["tool_calls"] = calls
                    messages.append(assistant)
                    if not calls:
                        result.answer = msg.get("content") or ""
                        return result
                    for call in calls:
                        fn = call["function"]
                        raw = fn.get("arguments") or "{}"
                        args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
                        out = await session.call_tool(fn["name"], args)
                        text = "\n".join(c.text for c in out.content if c.type == "text")
                        is_error = bool(_get(out, "isError", "is_error", default=False))
                        result.tool_calls.append(ToolCall(fn["name"], args, text, is_error))
                        _log(f"[tool] {fn['name']}({args}){' [error]' if is_error else ''}")
                        if len(text) > MAX_TOOL_CHARS:
                            text = text[:MAX_TOOL_CHARS] + "\n...[truncated]"
                        tool_msg = {"role": "tool", "tool_call_id": call.get("id", fn["name"])}
                        messages.append({**tool_msg, "name": fn["name"], "content": text})
                result.answer = "(stopped: reached max_turns without a final answer)"
                return result


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-turns", type=int, default=12)
    args = parser.parse_args()
    try:
        res = await run_agent_openai(args.question, args.model, args.max_turns, args.base_url)
    except BaseException as exc:  # noqa: BLE001 - show the root cause, not an ExceptionGroup wall
        while getattr(exc, "exceptions", None):
            exc = exc.exceptions[0]
        sys.exit(f"Agent failed: {exc.__class__.__name__}: {exc}")
    print(res.answer)


if __name__ == "__main__":
    asyncio.run(_main())
