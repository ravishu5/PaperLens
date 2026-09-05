"""Resource resolution and its failure modes."""
import pytest

from paperlens.graph.store import Store
from paperlens.resources import ResourceNotFound, resolve


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "g.db")


def test_unknown_uri_scheme_is_rejected(store):
    with pytest.raises(ResourceNotFound, match="Unrecognised"):
        resolve(store, "paperlens://nonsense/thing")


def test_uningested_paper_says_so_rather_than_inventing(store):
    with pytest.raises(ResourceNotFound, match="has not been ingested"):
        resolve(store, "paperlens://paper/2103.00020")


def test_non_arxiv_identifier_is_rejected(store):
    with pytest.raises(ResourceNotFound, match="not an arXiv identifier"):
        resolve(store, "paperlens://paper/the-clip-paper")


# ── identifier resolution ─────────────────────────────────────────────────
@pytest.fixture
def two_papers(tmp_path):
    from paperlens.graph.store import Store as _S

    s = _S(tmp_path / "g.db")
    for aid, title, doi in [("2103.00020", "CLIP", "10.1234/clip"),
                            ("1706.03762", "Transformer", None)]:
        s.upsert_paper(aid, title, doi=doi, latest_version=f"{aid}v1")
        s.upsert_paper_version(f"{aid}v1", aid, 1, source_kind="LATEX",
                               fidelity="LATEX_EXACT", source_sha256=None,
                               flattened_tex="x")
    s.commit()
    return s


def test_exact_arxiv_id_resolves(two_papers):
    from paperlens.resources import _pv
    assert _pv(two_papers, "2103.00020") == "2103.00020v1"
    assert _pv(two_papers, "arXiv:1706.03762") == "1706.03762v1"


def test_versioned_id_resolves_verbatim(two_papers):
    from paperlens.resources import _pv
    assert _pv(two_papers, "2103.00020v1") == "2103.00020v1"


def test_non_arxiv_paper_resolves_by_exact_doi(two_papers):
    """Supports papers ingested from a DOI or PDF rather than arXiv."""
    from paperlens.resources import _pv
    assert _pv(two_papers, "10.1234/clip") == "2103.00020v1"


@pytest.mark.parametrize("probe", ["2103", "0002", "1", ".", "00020"])
def test_partial_identifiers_do_not_silently_resolve(two_papers, probe):
    """A substring fallback made _pv('.') resolve to an arbitrary paper, which
    then flowed into mappings and reports as though it had been asked for."""
    from paperlens.resources import _pv
    with pytest.raises(ResourceNotFound):
        _pv(two_papers, probe)


def test_empty_identifier_is_rejected(two_papers):
    from paperlens.resources import _pv
    with pytest.raises(ResourceNotFound, match="empty"):
        _pv(two_papers, "   ")


def test_ambiguous_identifier_raises_rather_than_picking(tmp_path):
    from paperlens.graph.store import Store as _S
    from paperlens.resources import _pv

    s = _S(tmp_path / "g.db")
    # Two papers sharing a DOI is malformed data, but the resolver must refuse
    # to choose rather than return whichever row comes first.
    for aid in ("2103.00020", "1706.03762"):
        s.upsert_paper(aid, aid, doi="10.1234/dup", latest_version=f"{aid}v1")
        s.upsert_paper_version(f"{aid}v1", aid, 1, source_kind="LATEX",
                               fidelity="LATEX_EXACT", source_sha256=None,
                               flattened_tex="x")
    s.commit()
    with pytest.raises(ResourceNotFound, match="matches 2 ingested papers"):
        _pv(s, "10.1234/dup")
