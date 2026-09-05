"""Research lineage against the real CLIP paper.

The backward half needs no network beyond ingestion; the forward half degrades
when citation indexes are unavailable, and the tests assert that it degrades
honestly rather than inventing edges. Run with `pytest -m network`.
"""
import pytest

from paperlens.correlate.lineage import (find_sota_successors, trace_method,
                                         trace_research_lineage)
from paperlens.graph.store import Store
from paperlens.paper.ingest import ingest_paper
from paperlens.resources import resolve

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = Store(tmp_path_factory.mktemp("lineage") / "g.db")
    ingest_paper(s, "2103.00020")
    return s


def test_the_bibliography_is_parsed_and_partly_resolved(store):
    refs = resolve(store, "paperlens://paper/2103.00020/references")
    assert refs["count"] > 150
    assert refs["resolved_to_arxiv"] > 50
    assert all(r["title"] for r in refs["references"][:20])


def test_the_temperature_citation_resolves_to_its_source_paper(store):
    """The Phase 0 chain closes here: the paper's tau = 0.07 maps to
    clip/model.py:295 in code, and traces back to arXiv:1805.01978 in the
    literature -- all three established, none inferred."""
    m = trace_method(store, "2103.00020", "temperature")
    origins = {o["paper"]: o for o in m["introduced_by"]}
    assert "1805.01978" in origins
    o = origins["1805.01978"]
    assert o["confidence"] == "CONFIRMED"
    assert "0.07" in o["context"] and "temperature" in o["context"]
    assert "Unsupervised feature learning" in o["title"]


def test_backward_edges_are_confirmed_and_quotable(store):
    lin = trace_research_lineage(store, "2103.00020", direction="back", limit=8)
    assert lin["counts"]["predecessors"] == 8
    for n in lin["predecessors"]:
        assert n["confidence"] == "CONFIRMED"
        assert n["contexts"], f"{n['paper']} has no quoted context"


def test_each_context_quotes_its_own_citation(store):
    """Attaching a neighbouring sentence to a reference produced plausible,
    wrong provenance."""
    lin = trace_research_lineage(store, "2103.00020", direction="back", limit=15)
    checked = 0
    for n in lin["predecessors"]:
        rows = store.all(
            "SELECT bib_key FROM bib_entries WHERE paper_version='2103.00020v1' "
            "AND (arxiv_id = ? OR bib_key = ?)",
            (n["paper"], n["paper"].removeprefix("bib:")))
        if not rows:
            continue
        key = rows[0]["bib_key"]
        assert any(key in c for c in n["contexts"]), \
            f"{key}: contexts quote a different citation: {n['contexts'][:1]}"
        checked += 1
    assert checked >= 5


def test_forward_edges_are_never_confirmed_without_a_quotation(store):
    lin = trace_research_lineage(store, "2103.00020", direction="forward", limit=10)
    for n in lin["successors"]:
        if n["confidence"] == "CONFIRMED":
            assert n["contexts"], "CONFIRMED with nothing quoted"


def test_successors_are_returned_without_claiming_what_they_changed(store):
    out = find_sota_successors(store, "2103.00020", limit=8)
    assert out["successors"]
    assert any("not asserted" in n for n in out["notes"])


def test_the_direction_asymmetry_is_stated(store):
    lin = trace_research_lineage(store, "2103.00020", limit=5)
    assert any("not verified against" in n for n in lin["notes"])


def test_lineage_is_readable_as_a_resource(store):
    trace_research_lineage(store, "2103.00020", direction="back", limit=5)
    got = resolve(store, "paperlens://lineage/2103.00020")
    assert got["edges"]
    assert any(e["to"] == "1805.01978" for e in got["edges"])


def test_tracing_an_absent_method_says_so(store):
    m = trace_method(store, "2103.00020", "quantum annealing")
    assert m["introduced_by"] == []
    assert any("No sentence" in n for n in m["notes"])
