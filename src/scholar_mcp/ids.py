"""Normalise the many ways people write paper identifiers into Semantic Scholar's format."""

from __future__ import annotations

import re

from .errors import InvalidInputError

_NEW_ARXIV = r"\d{4}\.\d{4,5}"
_OLD_ARXIV = r"[a-z\-]+(?:\.[A-Za-z\-]+)?/\d{7}"
ARXIV_RE = re.compile(rf"^(?:arxiv:)?({_NEW_ARXIV}|{_OLD_ARXIV})(?:v\d+)?$", re.IGNORECASE)
ARXIV_URL_RE = re.compile(
    rf"arxiv\.org/(?:abs|pdf)/({_NEW_ARXIV}|{_OLD_ARXIV})(?:v\d+)?(?:\.pdf)?", re.IGNORECASE
)
ARXIV_IN_TEXT_RE = re.compile(rf"(?<![\w.])({_NEW_ARXIV})(?:v\d+)?(?![\w.]*\d)")
S2_ID_RE = re.compile(r"^[0-9a-f]{40}$")


def normalize_paper_id(raw: str) -> str:
    """Return an id usable with the Semantic Scholar API (``ARXIV:``, ``DOI:`` or raw S2 id)."""
    value = (raw or "").strip()
    if not value:
        raise InvalidInputError("paper_id is empty. Provide an arXiv id, DOI, or Semantic Scholar id.")
    if (m := ARXIV_URL_RE.search(value)) or (m := ARXIV_RE.match(value)):
        return f"ARXIV:{m.group(1)}"
    lowered = value.lower()
    if lowered.startswith("doi:"):
        return "DOI:" + value[4:].strip()
    if value.startswith("10.") and "/" in value:
        return f"DOI:{value}"
    if "doi.org/" in lowered:
        return "DOI:" + value.split("doi.org/", 1)[1]
    if S2_ID_RE.match(value) or value.upper().startswith(("CORPUSID:", "ARXIV:", "DOI:")):
        return value
    raise InvalidInputError(
        f"Unrecognised paper id '{raw}'. Use an arXiv id (e.g. 1706.03762), a DOI "
        "(10.xxxx/...), or a 40-character Semantic Scholar id. Use search_papers to find ids."
    )


def extract_arxiv_id(text: str) -> str | None:
    m = ARXIV_URL_RE.search(text) or ARXIV_RE.match(text.strip())
    return m.group(1) if m else None
