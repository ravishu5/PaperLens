"""Implementation discovery against live GitHub + arXiv.

Ground truth verified by hand in Phase 0 (RESEARCH.md section 7).
Run with `pytest -m network`.
"""
import pytest

from paperlens.correlate.discover import find_implementations
from paperlens.graph.store import Store
from paperlens.paper.ingest import ingest_paper
from paperlens.resources import resolve

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = Store(tmp_path_factory.mktemp("graph") / "g.db")
    ingest_paper(s, "2103.00020")
    ingest_paper(s, "1706.03762")
    return s


@pytest.fixture(scope="module")
def clip(store):
    return find_implementations(store, "2103.00020")


def test_official_repo_is_found_and_ranked_first(clip):
    top = clip["candidates"][0]
    assert top["repo"] == "openai/CLIP"
    assert top["relation"] == "OFFICIAL"
    assert top["confidence"] == "CONFIRMED"


def test_official_repo_confirmation_cites_the_papers_own_url(clip):
    top = clip["candidates"][0]
    kinds = {e["kind"] for e in top["evidence"] if e["stance"] == "SUPPORTS"}
    assert "paper_url" in kinds


def test_official_clip_repo_is_visibly_missing_the_training_code(clip):
    """The adversarial case. openai/CLIP is unambiguously official and ships no
    training code and no contrastive loss -- the paper's whole contribution. The
    tool must rank it first *and* show the hole."""
    top = clip["candidates"][0]
    assert "training" in top["coverage"]["missing"]
    assert top["coverage"]["score"] < 1.0
    absences = [e for e in top["evidence"] if e["kind"] == "code_absence"]
    assert absences and all(e["stance"] == "CONTRADICTS" for e in absences)


def test_the_community_reproduction_has_what_the_official_repo_lacks(clip):
    repos = {c["repo"]: c for c in clip["candidates"]}
    oc = repos.get("mlfoundations/open_clip")
    if oc is None:
        pytest.skip("open_clip not returned by GitHub search this run")
    assert "training" not in oc["coverage"]["missing"]


def test_transformer_official_repo_is_found(store):
    r = find_implementations(store, "1706.03762")
    top = r["candidates"][0]
    assert top["repo"] == "tensorflow/tensor2tensor"
    assert top["relation"] == "OFFICIAL"


def test_unrelated_repositories_are_not_returned(clip):
    for c in clip["candidates"]:
        assert c["relation"] != "UNRELATED"


def test_results_are_readable_as_a_resource(store, clip):
    got = resolve(store, "paperlens://paper/2103.00020/implementations")
    assert got["candidates"][0]["repo"] == "openai/CLIP"


def test_resource_says_so_when_no_search_has_run(store):
    ingest_paper(store, "1706.03762")
    s2 = Store(store.path.parent / "empty.db")
    ingest_paper(s2, "1706.03762")
    got = resolve(s2, "paperlens://paper/1706.03762/implementations")
    assert got["candidates"] == [] and "note" in got
