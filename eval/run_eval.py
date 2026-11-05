"""Evaluate the agent against eval/cases.json and write eval/results.json + results.md.

Metrics (per case, then averaged):
  citation_support every '(Author, Year)' citation matches a paper that a tool returned
  tool_selection   required tools were called (all-of / any-of)
  retrieval_recall fraction of expected arXiv ids that appeared in tool output
  grounded         every arXiv id in the final answer appeared in tool output or the prompt
  graceful_failure (missing-paper case) the answer admits the paper could not be found

    python eval/run_eval.py --model claude-sonnet-5-5
    python eval/run_eval.py --provider openai --model qwen2.5:32b   # local Ollama, free
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "client"))

from agent import DEFAULT_MODEL, run_agent  # noqa: E402
from agent_openai import DEFAULT_BASE_URL, run_agent_openai  # noqa: E402

from scholar_mcp.ids import ARXIV_IN_TEXT_RE  # noqa: E402

# Phrases that admit a paper could not be found (about NOT finding it, not about summarising it).
FAILURE_RE = re.compile(
    r"not found|could(?: not|n't)|can(?:not|'t) (?:find|locate)|unable|"
    r"(?:does not|doesn't) (?:exist|correspond|appear|match)|not (?:a )?valid|invalid|"
    r"no (?:such |matching )?(?:paper|record|result|entry)|nonexistent|non-existent",
    re.IGNORECASE,
)


def ids_in(text: str) -> set[str]:
    return set(ARXIV_IN_TEXT_RE.findall(text))


_NAME = r"[^\W\d_]+(?:[-'][^\W\d_]+)*"
CITE_RE = re.compile(
    rf"({_NAME}[^\W\d_]*)(?: et al\.?| (?:and|&) {_NAME})(?:,|\s)\s*\(?(\d{{4}})\)?"
)
BIBTEX_RE = re.compile(r"author = \{(.*?)\},\s*year = \{(\d{4})\}", re.DOTALL)


def _fold(name: str) -> str:
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return plain.strip().split()[-1].lower() if plain.strip() else ""


def known_author_years(outputs: list[str]) -> set[tuple[str, int]]:
    """(surname, year) pairs of every paper that appeared in tool output (JSON or BibTeX)."""
    pairs: set[tuple[str, int]] = set()

    def add_bibtex(text: str) -> None:
        for authors, year in BIBTEX_RE.findall(text):
            pairs.update((_fold(a.split(",")[0]), int(year)) for a in authors.split(" and "))

    def walk(node) -> None:
        if isinstance(node, str):
            add_bibtex(node)  # decoded JSON strings (export_bibtex output) contain real newlines
        elif isinstance(node, dict):
            authors, year = node.get("authors"), node.get("year")
            if isinstance(authors, list) and isinstance(year, int):
                pairs.update((_fold(a), year) for a in authors if isinstance(a, str) and not a.startswith("..."))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for text in outputs:
        try:
            walk(json.loads(text))
        except json.JSONDecodeError:
            add_bibtex(text)
    return pairs


def unsupported_citations(answer: str, outputs: list[str]) -> list[str] | None:
    """Author-year citations in the answer that no tool result supports; None if none are cited."""
    cited = {(_fold(m.group(1)), int(m.group(2))) for m in CITE_RE.finditer(answer)}
    if not cited:
        return None
    known = known_author_years(outputs)
    return sorted(f"{name} {year}" for name, year in cited if (name, year) not in known)


def score(case: dict, result) -> dict:
    called = {c.name for c in result.tool_calls}
    all_of = set(case.get("expected_tools_all", []))
    any_of = set(case.get("expected_tools_any", []))
    tool_ok = all_of <= called and (not any_of or bool(any_of & called))
    tool_text = "\n".join(c.output for c in result.tool_calls)
    seen = ids_in(tool_text)
    expected = case.get("expected_paper_ids", [])
    recall = (sum(i in seen for i in expected) / len(expected)) if expected else None
    allowed = seen | ids_in(case["prompt"])
    unsupported = sorted(ids_in(result.answer) - allowed)
    bad_cites = unsupported_citations(result.answer, [c.output for c in result.tool_calls])
    row = {
        "id": case["id"],
        "tool_selection": tool_ok,
        "retrieval_recall": recall,
        "grounded": not unsupported,
        "unsupported_ids": unsupported,
        "citation_support": None if bad_cites is None else not bad_cites,
        "unsupported_citations": bad_cites or [],
        "answer": result.answer,
        "tools_called": [c.name for c in result.tool_calls],
        "tool_errors": sum(c.is_error for c in result.tool_calls),
        "turns": result.turns,
    }
    if case.get("expect_failure_acknowledged"):
        row["graceful_failure"] = bool(FAILURE_RE.search(result.answer))
    return row


def summarise(rows: list[dict]) -> dict:
    def rate(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    keys = ("tool_selection", "retrieval_recall", "grounded", "citation_support", "graceful_failure")
    return {k: rate(k) for k in keys}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic",
                        help="'openai' = any OpenAI-compatible API, e.g. local Ollama (free)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--cases", default=str(ROOT / "eval" / "cases.json"))
    parser.add_argument("--only", default="", help="comma-separated case ids to run (writes results.only.*)")
    args = parser.parse_args()
    local_default = os.environ.get("LLM_MODEL", "qwen2.5:32b")
    model = args.model or (DEFAULT_MODEL if args.provider == "anthropic" else local_default)
    cases = json.loads(Path(args.cases).read_text())["cases"]
    only = {c.strip() for c in args.only.split(",") if c.strip()}
    if only:
        cases = [c for c in cases if c["id"] in only]
    stem = "results.only" if only else "results"
    rows = []
    for case in cases:
        print(f"running {case['id']} ...", file=sys.stderr)
        if args.provider == "anthropic":
            result = await run_agent(case["prompt"], model)
        else:
            result = await run_agent_openai(case["prompt"], model, base_url=args.base_url)
        rows.append(score(case, result))
    summary = summarise(rows)
    out = {"model": model, "n_cases": len(rows), "summary": summary, "cases": rows}
    (ROOT / "eval" / f"{stem}.json").write_text(json.dumps(out, indent=2))
    lines = [f"Model: `{model}` - {len(rows)} cases", "", "| metric | score |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary.items() if v is not None]
    lines += ["", "| case | tools ok | recall | grounded | turns |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['tool_selection']} | {r['retrieval_recall']} | {r['grounded']} | {r['turns']} |"
        )
    (ROOT / "eval" / f"{stem}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if only:
        for r in rows:
            print(f"\n--- final answer for {r['id']} ---\n{r['answer']}")


if __name__ == "__main__":
    asyncio.run(main())
