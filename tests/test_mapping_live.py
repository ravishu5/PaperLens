"""map_paper_to_code against real papers and repositories.

Includes the adversarial test the build prompt requires: pointed at a repository
that is not an implementation of the paper, the mapper must invent nothing.
Run with `pytest -m network`.
"""
import pytest

from paperlens.code.indexer import index_repository
from paperlens.correlate.analysis import record_paper_analysis
from paperlens.correlate.mapper import map_paper_to_code
from paperlens.graph.store import Store
from paperlens.paper.ingest import ingest_paper
from paperlens.resources import resolve

pytestmark = pytest.mark.network

ANALYSIS = {
    "components": [
        {"name": "Image Encoder", "kind": "architecture",
         "evidence": ["paperlens://paper/2103.00020/section/2.4"]},
        {"name": "Text Encoder", "kind": "architecture",
         "evidence": ["paperlens://paper/2103.00020/section/2.4"]},
        {"name": "Contrastive Objective", "kind": "loss",
         "description": "Symmetric cross-entropy over cosine similarities.",
         "evidence": ["paperlens://paper/2103.00020/section/2.3"]},
        {"name": "Temperature Scaling", "kind": "module",
         "description": "Learnable log-parameterised temperature tau.",
         "evidence": ["paperlens://paper/2103.00020/values"]},
    ],
}


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = Store(tmp_path_factory.mktemp("map") / "g.db")
    ingest_paper(s, "2103.00020")
    record_paper_analysis(s, "2103.00020", ANALYSIS, author="agent:test")
    index_repository(s, "openai/CLIP")
    return s


@pytest.fixture(scope="module")
def clip(store):
    return {m["anchor"]["label"]: m
            for m in map_paper_to_code(store, "2103.00020", "openai/CLIP")["mappings"]}


def test_the_stated_temperature_maps_to_its_literal_in_source(clip):
    """The full Phase 0 chain, now automatic: the paper says tau = 0.07 and the
    identical literal sits at clip/model.py:295."""
    m = clip["tau = 0.07"]
    assert m["status"] == "MATCHED"
    assert m["confidence"] == "CONFIRMED"
    assert "model.py:295" in m["reasoning"]
    assert "CLIP.__init__" in (m["symbol_uri"] or "")


def test_the_match_cites_both_sides(clip):
    kinds = {e["kind"] for e in clip["tau = 0.07"]["evidence"]}
    assert "paper_stated_value" in kinds and "code_symbol" in kinds


def test_encoders_map_to_their_methods(clip):
    assert clip["Image Encoder"]["status"] == "MATCHED"
    assert "encode_image" in clip["Image Encoder"]["symbol_uri"]
    assert clip["Text Encoder"]["status"] == "MATCHED"
    assert "encode_text" in clip["Text Encoder"]["symbol_uri"]


def test_the_papers_central_method_is_reported_absent_with_citable_evidence(clip):
    """openai/CLIP ships no contrastive objective. Establishing that is the
    finding; it is what compare_paper_with_code will build on."""
    m = clip["Contrastive Objective"]
    assert m["status"] == "ABSENT" and m["confidence"] == "CONFIRMED"
    absence = [e for e in m["evidence"] if e["kind"] == "code_absence"]
    assert absence and absence[0]["stance"] == "CONTRADICTS"
    assert (absence[0].get("provenance") or {}).get("absence_ref", "").startswith("absent:")


def test_a_component_is_not_declared_absent_when_its_constant_was_located(clip):
    """`logit_scale` IS the paper's temperature. Name matching cannot see that,
    so without a cross-check the tool would confidently report the component
    missing while the constant it describes sits at model.py:295."""
    m = clip["Temperature Scaling"]
    assert m["status"] != "ABSENT"
    assert "tau" in m["reasoning"]


def test_section_headings_do_not_produce_absence_claims(clip):
    for label in ("Approach", "Experiments", "Training"):
        if label in clip:
            assert clip[label]["status"] != "ABSENT"


def test_results_are_readable_as_a_resource(store):
    got = resolve(store, "paperlens://mapping/2103.00020/openai/CLIP")
    assert got["summary"]["MATCHED"] >= 3


# ── the adversarial test ──────────────────────────────────────────────────
def test_pointed_at_an_unrelated_repository_the_mapper_invents_nothing(store):
    """The build prompt's negative test. psf/requests is an HTTP library; nothing
    in CLIP is implemented there. Every anchor must come back UNKNOWN, ABSENT or
    AMBIGUOUS -- never MATCHED.

    This caught a real defect: 'Text Encoder' matched requests' `Response.text`
    on the single token 'text' plus a models.py path.
    """
    index_repository(store, "psf/requests")
    r = map_paper_to_code(store, "2103.00020", "psf/requests")
    matched = [m for m in r["mappings"] if m["status"] == "MATCHED"]
    assert matched == [], f"invented mappings: {[m['anchor']['label'] for m in matched]}"


def test_the_unrelated_repository_is_described_as_such(store):
    index_repository(store, "psf/requests")
    r = map_paper_to_code(store, "2103.00020", "psf/requests")
    assert any("does not implement" in n for n in r["notes"])
