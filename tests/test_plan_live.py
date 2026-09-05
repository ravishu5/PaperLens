"""Reproduction planning over the real CLIP corpus.

The plan is assembled from the graph, so this exercises the whole pipeline:
ingest → analysis → discovery → indexing → mapping → comparison → plan.
Run with `pytest -m network`.
"""
import pytest

from paperlens.code.indexer import index_repository
from paperlens.correlate.analysis import record_paper_analysis
from paperlens.correlate.compare import compare_paper_with_code
from paperlens.correlate.mapper import map_paper_to_code
from paperlens.correlate.plan import build_reproduction_plan
from paperlens.graph.store import Store
from paperlens.paper.ingest import ingest_paper
from paperlens.resources import resolve

pytestmark = pytest.mark.network

ANALYSIS = {"components": [
    {"name": "Image Encoder", "kind": "architecture",
     "evidence": ["paperlens://paper/2103.00020/section/2.4"]},
    {"name": "Text Encoder", "kind": "architecture",
     "evidence": ["paperlens://paper/2103.00020/section/2.4"]},
    {"name": "Contrastive Objective", "kind": "loss",
     "description": "Symmetric cross-entropy over cosine similarities.",
     "evidence": ["paperlens://paper/2103.00020/section/2.3"]},
    {"name": "WIT Dataset", "kind": "dataset",
     "evidence": ["paperlens://paper/2103.00020/section/2.2"]},
]}


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    s = Store(tmp_path_factory.mktemp("plan") / "g.db")
    ingest_paper(s, "2103.00020")
    record_paper_analysis(s, "2103.00020", ANALYSIS, author="agent:test")
    index_repository(s, "openai/CLIP")
    map_paper_to_code(s, "2103.00020", "openai/CLIP")
    compare_paper_with_code(s, "2103.00020", "openai/CLIP")
    return s, build_reproduction_plan(s, "2103.00020", "openai/CLIP")


def test_the_absent_loss_and_missing_clamp_are_blocking_risks(plan):
    """The acceptance criterion: a CLIP plan must name both."""
    _, p = plan
    blocking = [r["risk"] for r in p["reproduction_risks"]
                if r["severity"] == "BLOCKING"]
    assert any("Contrastive Objective" in r for r in blocking), blocking
    assert any("clipping" in r for r in blocking), blocking


def test_risks_are_ordered_most_severe_first(plan):
    _, p = plan
    order = {"BLOCKING": 0, "SIGNIFICANT": 1, "MINOR": 2, "COSMETIC": 3}
    seen = [order[r["severity"]] for r in p["reproduction_risks"]]
    assert seen == sorted(seen)


def test_dependencies_are_read_from_the_repository(plan):
    _, p = plan
    dep = p["sections"]["dependencies"]
    declared = [d for it in dep["items"] for d in it["declared"]]
    assert "torch" in declared
    assert "without a version" in (dep["note"] or "")


def test_unknown_sections_say_what_would_resolve_them(plan):
    _, p = plan
    for name, sec in p["sections"].items():
        if sec["status"] == "UNKNOWN":
            assert sec.get("note"), f"{name} is UNKNOWN with no guidance"


def test_a_section_the_paper_describes_is_not_unknown(plan):
    """The paper's side can be known while the code's side is not; calling that
    UNKNOWN discards what was established."""
    _, p = plan
    assert p["sections"]["dataset"]["status"] == "PARTIAL"
    assert p["sections"]["dataset"]["items"]


def test_verification_steps_come_from_confirmed_anchors(plan):
    _, p = plan
    checks = p["verification_strategy"]["items"]
    tau = [c for c in checks if "tau" in c["check"]]
    assert tau and tau[0]["where"] == "clip/model.py:295"
    assert tau[0]["confidence"] == "CONFIRMED"


def test_absent_components_become_implementation_work(plan):
    _, p = plan
    checks = [c["check"] for c in p["verification_strategy"]["items"]]
    assert any("implement Contrastive Objective" in c for c in checks)


def test_expected_results_are_not_presented_as_predictions(plan):
    _, p = plan
    assert "not a prediction" in (p["expected_results"]["note"] or "")


def test_readiness_counts_are_consistent(plan):
    _, p = plan
    r = p["readiness"]
    assert (r["sections_known"] + r["sections_partial"] + r["sections_unknown"]
            == len(p["sections"]))
    assert r["blocking_risks"] >= 2


def test_the_plan_is_readable_as_a_resource(plan):
    store, _ = plan
    got = resolve(store, "paperlens://plan/2103.00020/openai/CLIP")
    assert got["readiness"]["blocking_risks"] >= 2


def test_planning_without_a_repository_uses_the_ranked_candidate(plan):
    store, _ = plan
    out = build_reproduction_plan(store, "2103.00020", None)
    assert "error" in out or out["repo"]


def test_planning_against_an_unindexed_repo_reports_rather_than_guesses(plan):
    store, _ = plan
    with pytest.raises(LookupError, match="has not been indexed"):
        build_reproduction_plan(store, "2103.00020", "psf/nonexistent-repo")
