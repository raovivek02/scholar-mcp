"""BibTeX generation (done in code, never by the LLM, so entries are reproducible)."""

from __future__ import annotations

import re
import unicodedata

from .models import Paper

_ESCAPES = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_"}


def _escape(text: str) -> str:
    return "".join(_ESCAPES.get(ch, ch) for ch in text)


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _surname(name: str) -> str:
    return name.strip().split()[-1] if name.strip() else "unknown"


def _author_field(authors: list[str]) -> str:
    out = []
    for name in authors:
        parts = name.strip().split()
        out.append(f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else name)
    return " and ".join(out)


def cite_key(paper: Paper) -> str:
    surname = re.sub(r"[^a-z]", "", _ascii(_surname(paper.authors[0] if paper.authors else "")).lower())
    first = next(
        (w for w in re.findall(r"[a-z]+", _ascii(paper.title).lower()) if len(w) > 3), "paper"
    )
    return f"{surname or 'unknown'}{paper.year or ''}{first}"


def to_bibtex(papers: list[Paper]) -> str:
    seen: dict[str, int] = {}
    entries = []
    for p in papers:
        key = cite_key(p)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            key += chr(ord("a") + seen[key] - 2)
        has_venue = p.venue and p.venue.lower() != "arxiv"
        kind = "article" if has_venue else "misc"
        fields = [f"title = {{{{{_escape(p.title)}}}}}"]
        if p.authors:
            fields.append(f"author = {{{_escape(_author_field(p.authors))}}}")
        if p.year:
            fields.append(f"year = {{{p.year}}}")
        if has_venue:
            fields.append(f"journal = {{{_escape(p.venue or '')}}}")
        if p.arxiv_id:
            fields += [f"eprint = {{{p.arxiv_id}}}", "archivePrefix = {arXiv}"]
        if p.doi:
            fields.append(f"doi = {{{p.doi}}}")
        if p.url:
            fields.append(f"url = {{{p.url}}}")
        entries.append(f"@{kind}{{{key},\n  " + ",\n  ".join(fields) + "\n}")
    return "\n\n".join(entries)
