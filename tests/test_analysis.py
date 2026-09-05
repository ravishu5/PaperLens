"""Agent write-back validation: the evidence invariant at the API boundary."""
import pytest

from paperlens.correlate.analysis import get_analysis, record_paper_analysis
from paperlens.graph.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "g.db")
    s.upsert_paper("2103.00020", "CLIP")
    s.upsert_paper_version("2103.00020v1", "2103.00020", 1, source_kind="LATEX",
                           fidelity="LATEX_EXACT", source_sha256=None,
                           flattened_tex=r"\section{Approach} text")
    s.execute(
        "INSERT INTO sections (id, paper_version, section_path, level, title, body, "
        "ordinal) VALUES ('2103.00020v1:sec:2.3','2103.00020v1','2.3',2,'Method','x',0)")
    s.commit()
    return s


def _analysis(**kw):
    base = {"components": [], "claims": []}
    base.update(kw)
    return base


def test_component_citing_a_resolving_uri_is_accepted(store):
    r = record_paper_analysis(store, "2103.00020", _analysis(components=[{
        "name": "Contrastive Objective", "kind": "loss",
        "evidence": ["paperlens://paper/2103.00020/section/2.3"]}]))
    assert [c["name"] for c in r["accepted"]["components"]] == ["Contrastive Objective"]
    assert r["rejected"] == []


def test_component_citing_a_nonexistent_section_is_rejected(store):
    """The core write-back guarantee: an agent cannot invent a component by
    pointing at a section that does not exist."""
    r = record_paper_analysis(store, "2103.00020", _analysis(components=[{
        "name": "Imaginary Module", "kind": "module",
        "evidence": ["paperlens://paper/2103.00020/section/99.9"]}]))
    assert r["accepted"]["components"] == []
    assert r["rejected"][0]["name"] == "Imaginary Module"
    assert "resolve" in r["rejected"][0]["reason"]


def test_component_with_no_evidence_at_all_is_rejected(store):
    r = record_paper_analysis(store, "2103.00020", _analysis(components=[{
        "name": "Unsourced", "kind": "loss", "evidence": []}]))
    assert r["accepted"]["components"] == [] and len(r["rejected"]) == 1


def test_rejected_components_are_not_written_to_the_graph(store):
    record_paper_analysis(store, "2103.00020", _analysis(components=[{
        "name": "Imaginary", "kind": "module",
        "evidence": ["paperlens://paper/2103.00020/section/99.9"]}]))
    rows = store.all("SELECT * FROM components WHERE paper_version = '2103.00020v1'")
    assert rows == []


def test_invalid_component_kind_is_rejected(store):
    r = record_paper_analysis(store, "2103.00020", _analysis(components=[{
        "name": "X", "kind": "not-a-kind",
        "evidence": ["paperlens://paper/2103.00020/section/2.3"]}]))
    assert r["accepted"]["components"] == []
    assert "kind must be one of" in r["rejected"][0]["reason"]


def test_accepted_components_are_attributed_to_the_agent(store):
    record_paper_analysis(store, "2103.00020", _analysis(components=[{
        "name": "Image Encoder", "kind": "architecture",
        "evidence": ["paperlens://paper/2103.00020/section/2.3"]}]))
    row = store.one("SELECT source FROM components WHERE slug = 'image-encoder'")
    assert row["source"] == "AGENT"


def test_claims_follow_the_same_rule(store):
    r = record_paper_analysis(store, "2103.00020", _analysis(claims=[
        {"statement": "Supported.", "kind": "result",
         "evidence": ["paperlens://paper/2103.00020/section/2.3"]},
        {"statement": "Unsupported.", "kind": "result", "evidence": []},
    ]))
    assert len(r["accepted"]["claims"]) == 1
    assert r["accepted"]["claims"][0]["statement"] == "Supported."


def test_narrative_fields_are_stored_and_retrievable(store):
    record_paper_analysis(store, "2103.00020",
                          _analysis(problem="P", limitations=["L"], bogus_key="x"))
    got = get_analysis(store, "2103.00020")
    assert got["fields"]["problem"] == "P" and got["fields"]["limitations"] == ["L"]
    assert "bogus_key" not in got["fields"]


def test_unrecognised_fields_are_reported_not_silently_dropped(store):
    r = record_paper_analysis(store, "2103.00020", _analysis(bogus_key="x"))
    assert any("bogus_key" in n for n in r["notes"])
