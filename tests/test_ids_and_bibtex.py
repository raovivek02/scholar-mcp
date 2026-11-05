import pytest

from scholar_mcp.bibtex import cite_key, to_bibtex
from scholar_mcp.errors import InvalidInputError
from scholar_mcp.ids import extract_arxiv_id, normalize_paper_id
from scholar_mcp.models import Paper


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1706.03762", "ARXIV:1706.03762"),
        ("arXiv:1706.03762v5", "ARXIV:1706.03762"),
        ("https://arxiv.org/abs/2105.08971v2", "ARXIV:2105.08971"),
        ("https://arxiv.org/pdf/2105.08971.pdf", "ARXIV:2105.08971"),
        ("hep-th/9901001", "ARXIV:hep-th/9901001"),
        ("10.1145/3292500.3330701", "DOI:10.1145/3292500.3330701"),
        ("https://doi.org/10.1000/xyz123", "DOI:10.1000/xyz123"),
        ("a" * 40, "a" * 40),
    ],
)
def test_normalize(raw, expected):
    assert normalize_paper_id(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "not an id", "12345"])
def test_normalize_rejects(raw):
    with pytest.raises(InvalidInputError):
        normalize_paper_id(raw)


def test_extract_arxiv_id():
    assert extract_arxiv_id("see https://arxiv.org/abs/2105.08971v2") == "2105.08971"
    assert extract_arxiv_id("nothing here") is None


def test_bibtex_entry_and_dedup():
    a = Paper(title="Attention & Memory", source="arxiv", arxiv_id="1706.03762", year=2017,
              authors=["Ashish Vaswani", "Noam Shazeer"], url="https://arxiv.org/abs/1706.03762")
    b = a.model_copy()
    out = to_bibtex([a, b])
    assert cite_key(a) == "vaswani2017attention"
    assert "@misc{vaswani2017attention," in out
    assert "@misc{vaswani2017attentiona," in out  # duplicate key gets a suffix
    assert r"Attention \& Memory" in out
    assert "author = {Vaswani, Ashish and Shazeer, Noam}" in out
    assert "eprint = {1706.03762}" in out


def test_bibtex_article_when_venue():
    p = Paper(title="Deep Nets", source="semantic_scholar", year=2015, venue="Nature",
              authors=["Yann LeCun"])
    assert to_bibtex([p]).startswith("@article{lecun2015deep,")
