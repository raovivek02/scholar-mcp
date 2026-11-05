"""Tool-calling agent loop: an LLM (Anthropic API) driving the scholar-mcp server over stdio.

    export ANTHROPIC_API_KEY=...
    python client/agent.py "Summarise the most influential LiDAR moving-object-segmentation papers"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
SYSTEM = """You are a careful research assistant with access to arXiv and Semantic Scholar tools.
- Ground every factual claim about a paper in tool results. Never cite a paper no tool returned.
- Cite papers as (FirstAuthor et al., Year) and give the arXiv id or URL.
- If a tool errors or finds nothing, say so plainly instead of guessing."""


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Read an attribute across mcp 1.x (camelCase) and 2.x (snake_case) result types."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    output: str
    is_error: bool


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    turns: int = 0


def server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable, args=["-m", "scholar_mcp"], env=dict(os.environ)
    )


async def run_agent(question: str, model: str = DEFAULT_MODEL, max_turns: int = 12) -> AgentResult:
    from anthropic import AsyncAnthropic  # imported lazily so other providers don't need it

    llm = AsyncAnthropic()
    async with stdio_client(server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = (await session.list_tools()).tools
            tools = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": _get(t, "inputSchema", "input_schema"),
                }
                for t in listed
            ]
            messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
            result = AgentResult(answer="")
            for turn in range(1, max_turns + 1):
                result.turns = turn
                resp = await llm.messages.create(
                    model=model, max_tokens=4096, system=SYSTEM, tools=tools, messages=messages
                )
                messages.append({"role": "assistant", "content": resp.content})
                if resp.stop_reason != "tool_use":
                    result.answer = "".join(b.text for b in resp.content if b.type == "text")
                    return result
                tool_results = []
                for block in resp.content:
                    if block.type != "tool_use":
                        continue
                    out = await session.call_tool(block.name, block.input)
                    text = "\n".join(c.text for c in out.content if c.type == "text")
                    is_error = bool(_get(out, "isError", "is_error", default=False))
                    result.tool_calls.append(ToolCall(block.name, dict(block.input), text, is_error))
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": text,
                            "is_error": is_error,
                        }
                    )
                messages.append({"role": "user", "content": tool_results})
            result.answer = "(stopped: reached max_turns without a final answer)"
            return result


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-turns", type=int, default=12)
    args = parser.parse_args()
    res = await run_agent(args.question, args.model, args.max_turns)
    for i, call in enumerate(res.tool_calls, 1):
        flag = " [error]" if call.is_error else ""
        print(f"[{i}] {call.name}({call.arguments}){flag}", file=sys.stderr)
    print(res.answer)


if __name__ == "__main__":
    asyncio.run(_main())
