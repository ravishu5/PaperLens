"""Code intelligence against a real clone of openai/CLIP.

Parametrised over every available backend, which is the point of the port: both
must answer the same questions the same way. Run with `pytest -m network`.
"""
import pytest

from paperlens.code import indexer
from paperlens.code.python_ast import PythonAstProvider
from paperlens.graph.store import Store
from paperlens.resources import resolve, symbol_uri

pytestmark = pytest.mark.network

REPO = "openai/CLIP"


def _providers():
    out = [("python-ast", PythonAstProvider())]
    try:
        from paperlens.code.jcodemunch_adapter import JCodeMunchProvider

        p = JCodeMunchProvider()
        if p.available():
            out.append(("jcodemunch", p))
    except Exception:
        pass
    return out


@pytest.fixture(scope="module", params=[n for n, _ in _providers()])
def wired(request, tmp_path_factory, monkeypatch_module):
    # This fixture swaps the process-wide provider, so it must put it back:
    # leaking a backend into later modules made an unrelated gap-detection test
    # fail only when the full suite ran.
    saved_provider = indexer._provider
    saved_seen = set(indexer._backend_seen)  # (backend, repo) pairs

    prov = dict(_providers())[request.param]
    indexer._provider = prov
    indexer._backend_seen.clear()
    store = Store(tmp_path_factory.mktemp(request.param) / "g.db")
    summary = indexer.index_repository(store, REPO)
    yield store, prov, summary

    indexer._provider = saved_provider
    indexer._backend_seen.clear()
    indexer._backend_seen.update(saved_seen)


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_index_reports_a_commit_sha_and_symbols(wired):
    _, _, s = wired
    assert len(s.commit_sha) == 40
    assert s.symbol_count > 0 and s.file_count > 0


def test_reindexing_the_same_sha_is_cached(wired):
    store, _, _ = wired
    again = indexer.index_repository(store, REPO)
    assert again.cached is True


def test_the_phase_zero_anchor_resolves_to_its_symbol(wired):
    """tau = 0.07 lives in CLIP.__init__ at model.py:295. This is the code half
    of the CONFIRMED chain from RESEARCH.md section 7."""
    store, prov, _ = wired
    hits = prov.search_text(REPO, "logit_scale")
    line_295 = [h for h in hits if h.line == 295]
    assert line_295, f"expected a hit at model.py:295, got {[(h.file_path, h.line) for h in hits]}"
    assert "0.07" in line_295[0].text


def test_symbol_resource_resolves_and_carries_source(wired):
    store, _, _ = wired
    uri = symbol_uri(REPO, "clip/model.py::CLIP.encode_image#method")
    got = resolve(store, uri)
    assert got["qualified_name"] == "clip/model.py::CLIP.encode_image#method"
    assert got["file_path"] == "clip/model.py"
    assert "def encode_image" in (got["source"] or "")


def test_file_outline_lists_the_model_classes(wired):
    store, _, _ = wired
    got = resolve(store, f"paperlens://repo/{REPO}/file/clip/model.py")
    names = {s["qualified_name"] for s in got["symbols"]}
    assert "clip/model.py::CLIP#class" in names
    assert "clip/model.py::VisionTransformer#class" in names


def test_the_papers_central_method_is_absent_and_the_absence_is_citable(wired):
    """openai/CLIP ships inference weights, not the contrastive objective the
    paper is about. The backend must say so with a citable reference, not return
    an empty list that reads like an error."""
    _, prov, _ = wired
    res = prov.search_symbols(REPO, "contrastive loss cross entropy")
    assert res.found is False
    assert res.absence_ref and res.absence_ref.startswith("absent:")


def test_referenced_symbols_are_persisted_on_first_read(wired):
    store, _, snap = wired
    resolve(store, symbol_uri(REPO, "clip/model.py::CLIP.forward#method"))
    row = store.one(
        "SELECT * FROM symbols WHERE snapshot_id = ? AND qualified_name = ?",
        (snap.snapshot_id, "clip/model.py::CLIP.forward#method"))
    assert row is not None and row["file_path"] == "clip/model.py"


def test_unknown_symbol_is_reported_not_invented(wired):
    from paperlens.resources import ResourceNotFound

    store, _, _ = wired
    with pytest.raises(ResourceNotFound):
        resolve(store, symbol_uri(REPO, "clip/model.py::NoSuchThing#class"))


def test_unindexed_repo_says_so(tmp_path):
    store = Store(tmp_path / "g.db")
    with pytest.raises(LookupError, match="has not been indexed"):
        indexer.require_snapshot(store, "some/other")
