"""The scorer itself is tested offline so eval numbers can be trusted."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "client"))

import run_eval  # noqa: E402


def call(name, output="", err=False):
    return SimpleNamespace(name=name, output=output, is_error=err)


def test_scoring():
    case = {"id": "x", "prompt": "about 1706.03762", "expected_tools_all": ["export_bibtex"],
            "expected_paper_ids": ["1706.03762", "1810.04805"]}
    result = SimpleNamespace(
        answer="See 1706.03762 and also 2001.00001.", turns=3,
        tool_calls=[call("export_bibtex", "eprint = {1706.03762}")],
    )
    row = run_eval.score(case, result)
    assert row["tool_selection"] and row["retrieval_recall"] == 0.5
    assert not row["grounded"] and row["unsupported_ids"] == ["2001.00001"]


def test_graceful_failure_and_summary():
    case = {"id": "m", "prompt": "p", "expected_tools_any": ["search_papers"],
            "expect_failure_acknowledged": True}
    result = SimpleNamespace(answer="I could not find that paper.", turns=2,
                             tool_calls=[call("search_papers", "{}")])
    row = run_eval.score(case, result)
    assert row["graceful_failure"] and row["grounded"]
    assert run_eval.summarise([row])["tool_selection"] == 1.0
