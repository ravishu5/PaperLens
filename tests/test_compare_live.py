"""Comparison and gap detection against the real CLIP corpus.

Ground truth from Phase 0 (RESEARCH.md section 7). Run with `pytest -m network`.
"""
import pytest

from paperlens.code.indexer import index_repository
from paperlens.correlate.analysis import record_paper_analysis
from paperlens.correlate.compare import (compare_implementations,
                                         compare_paper_with_code,
                                         find_implementation_gaps)
from paperlens.correlate.mapper import map_paper_to_code
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
]}


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = Store(tmp_path_factory.mktemp("cmp") / "g.db")
    ingest_paper(s, "2103.00020")
    record_paper_analysis(s, "2103.00020", ANALYSIS, author="agent:test")
    index_repository(s, "openai/CLIP")
    map_paper_to_code(s, "2103.00020", "openai/CLIP")
    return s


@pytest.fixture(scope="module")
def diffs(store):
    return compare_paper_with_code(store, "2103.00020", "openai/CLIP")


def test_the_missing_temperature_clamp_is_reported(diffs):
    """The paper calls the clamp "necessary to prevent training instability" and
    openai/CLIP contains no clamp at all -- the Phase 0 gap, found automatically."""
    clip = [d for d in diffs["differences"]
            if "clipping" in d["code_does"] or "clipped" in (d["paper_states"] or "")]
    assert clip, [d["code_does"] for d in diffs["differences"]]
    d = clip[0]
    assert d["severity"] == "BLOCKING"
    assert "clipped to prevent" in d["paper_states"]


def test_the_missing_loss_is_a_blocking_difference(diffs):
    missing = [d for d in diffs["differences"] if d["kind"] == "MISSING_COMPONENT"]
    assert any("Contrastive Objective" in d["difference"] for d in missing)
    assert all(d["severity"] == "BLOCKING" for d in missing
               if "Contrastive Objective" in d["difference"])


def test_every_difference_quotes_both_sides(diffs):
    for d in diffs["differences"]:
        assert d["paper_states"] and d["code_does"]
        assert d["confidence"] in ("CONFIRMED", "LIKELY", "POSSIBLE", "UNKNOWN")


def test_heuristic_differences_are_never_confirmed(diffs):
    """Both observations are exact, but the inference joining a paper term to a
    code term is a heuristic and must not be presented as certainty."""
    behavioural = [d for d in diffs["differences"]
                   if d["kind"] in ("TRAINING", "ARCHITECTURAL",
                                    "UNDOCUMENTED_PREPROCESSING")]
    assert behavioural and all(d["confidence"] == "LIKELY" for d in behavioural)


def test_absence_of_training_code_is_stated_once_not_implied_repeatedly(diffs):
    assert any("no training code" in n for n in diffs["notes"])


def test_differences_are_readable_as_a_resource(store, diffs):
    got = resolve(store, "paperlens://difference/2103.00020/openai/CLIP")
    assert got["summary"]["BLOCKING"] >= 2


# ── gaps ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def gaps(store):
    return find_implementation_gaps(store, "2103.00020", "openai/CLIP")


def test_undocumented_normalization_constants_are_found(gaps):
    """The vision doc's own example: the normalization constants are hard-coded
    in clip/clip.py and appear nowhere in the paper."""
    hidden = [g for g in gaps["gaps"] if g["kind"] == "CONFIG_HIDDEN"]
    assert hidden, gaps["summary"]
    g = hidden[0]
    assert "0.48145466" in g["description"]
    assert g["confidence"] == "CONFIRMED"
    assert "clip.py" in (g["where"] or "")


def test_downloaded_weights_are_flagged_as_an_external_dependency(gaps):
    ext = [g for g in gaps["gaps"] if g["kind"] == "EXTERNAL_DEPENDENCY"]
    assert ext and "clip.py" in (ext[0]["where"] or "")


def test_repeated_findings_are_grouped_not_listed_individually(gaps):
    """Nine checkpoint URLs are one dependency; six constants on one line are one
    gap. Listing them separately buries everything else."""
    assert gaps["summary"]["EXTERNAL_DEPENDENCY"] == 1
    assert gaps["summary"]["CONFIG_HIDDEN"] == 1


def test_the_absent_loss_appears_as_a_reproduction_gap(gaps):
    absent = [g for g in gaps["gaps"] if g["kind"] == "ABSENT_FROM_REPOSITORY"]
    assert any("Contrastive Objective" in g["description"] for g in absent)


# ── comparing implementations ─────────────────────────────────────────────
def test_the_reproduction_beats_the_official_repo_on_substance(store):
    """openai/CLIP has tidier naming and would win on match counts, while lacking
    the contrastive objective the paper is about. Ranking must not reward that."""
    index_repository(store, "mlfoundations/open_clip")
    c = compare_implementations(store, "2103.00020",
                                ["openai/CLIP", "mlfoundations/open_clip"])
    assert c["closest_to_paper"] == "mlfoundations/open_clip"
    # The official repo carries more BLOCKING differences despite matching more
    # names, which is the whole point of ranking on substance.
    assert (c["coverage"]["openai/CLIP"]["blocking_differences"]
            > c["coverage"]["mlfoundations/open_clip"]["blocking_differences"])


def test_the_two_repositories_differ_on_the_loss(store):
    index_repository(store, "mlfoundations/open_clip")
    c = compare_implementations(store, "2103.00020",
                                ["openai/CLIP", "mlfoundations/open_clip"])
    obj = [d for d in c["differ_on"] if d["anchor"] == "Contrastive Objective"]
    assert obj and obj[0]["status"]["openai/CLIP"] == "ABSENT"


def test_comparing_one_repository_makes_no_divergence_claim(store):
    c = compare_implementations(store, "2103.00020", ["openai/CLIP", "openai/CLIP"])
    assert any("Fewer than two" in n for n in c["notes"]) or c["differ_on"] == []
