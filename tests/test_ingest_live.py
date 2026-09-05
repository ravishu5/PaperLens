"""End-to-end ingestion against real papers.

Ground truth here was verified by hand in Phase 0 (see RESEARCH.md section 7).
Marked `network`: run with `pytest -m network`.
"""
import pytest

from paperlens.graph.store import Store
from paperlens.paper.ingest import ingest_paper
from paperlens.resources import resolve

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return Store(tmp_path_factory.mktemp("graph") / "g.db")


def test_transformer_equation_numbering_matches_the_published_paper(store):
    """Attention Is All You Need numbers exactly three equations and labels
    none of them, so the numbers can only come from replaying the counter."""
    r = ingest_paper(store, "1706.03762")
    assert r.fidelity == "LATEX_EXACT"
    assert r.numbered_equations == 3

    eq1 = resolve(store, "paperlens://paper/1706.03762/equation/1")
    assert "softmax" in eq1["latex"] and "sqrt{d_k}" in eq1["latex"].replace("\\", "")
    assert eq1["latex_label"] is None          # the paper labels nothing
    assert eq1["number_confidence"] == "LIKELY"

    eq2 = resolve(store, "paperlens://paper/1706.03762/equation/2")
    assert "FFN" in eq2["latex"]


def test_transformer_hyperparameters_are_recovered_from_prose(store):
    ingest_paper(store, "1706.03762")
    vals = resolve(store, "paperlens://paper/1706.03762/values")["values"]
    got = {(v["symbol"], v["value"]) for v in vals}
    assert ("beta_2", "0.98") in got
    assert ("warmup_steps", "4000") in got
    assert ("d_ff", "2048") in got


def test_clip_has_no_equations_and_says_so(store):
    """CLIP is the counterexample to equation-keyed mapping: it contains zero
    display-math environments, and its core algorithm is a figure image."""
    r = ingest_paper(store, "2103.00020")
    assert r.equations == 0
    assert any("no display-math" in n for n in r.notes)

    algos = store.all(
        "SELECT * FROM algorithms WHERE paper_version = '2103.00020v1'")
    assert len(algos) == 1
    assert algos[0]["presentation"] == "figure_image"
    assert algos[0]["extractable"] == 0


def test_clip_temperature_constant_is_captured(store):
    """tau = 0.07 is the anchor that maps to `np.log(1 / 0.07)` in
    openai/CLIP model.py:295 -- the CONFIRMED chain from RESEARCH.md section 7."""
    ingest_paper(store, "2103.00020")
    vals = resolve(store, "paperlens://paper/2103.00020/values")["values"]
    tau = [v for v in vals if v["symbol"] == "tau"]
    assert tau and tau[0]["value"] == "0.07"


def test_official_repo_is_recovered_from_the_papers_own_url(store):
    """The signal that replaces Papers With Code. Verified 2/2 in Phase 0."""
    ingest_paper(store, "2103.00020")
    ingest_paper(store, "1706.03762")

    clip = resolve(store, "paperlens://paper/2103.00020/urls")["urls"]
    assert any(u["owner"] == "OpenAI" and u["repo"] == "CLIP" and u["in_abstract"]
               for u in clip)

    tfm = resolve(store, "paperlens://paper/1706.03762/urls")["urls"]
    assert any(u["owner"] == "tensorflow" and u["repo"] == "tensor2tensor" for u in tfm)


def test_clip_appendix_sections_use_letters(store):
    ingest_paper(store, "2103.00020")
    sec = resolve(store, "paperlens://paper/2103.00020/section/A")
    assert sec["title"] == "Linear-probe evaluation"


def test_clip_method_section_resolves(store):
    ingest_paper(store, "2103.00020")
    sec = resolve(store, "paperlens://paper/2103.00020/section/2.3")
    assert sec["title"] == "Selecting an Efficient Pre-Training Method"
    assert sec["latex_label"] == "subsection:method"


def test_asking_for_an_equation_that_does_not_exist_reports_unknown(store):
    """CLIP has no equations. The tool must say so, not invent one."""
    from paperlens.resources import ResourceNotFound
    ingest_paper(store, "2103.00020")
    with pytest.raises(ResourceNotFound, match="0 numbered equation"):
        resolve(store, "paperlens://paper/2103.00020/equation/1")


def test_a_parser_upgrade_forces_a_reparse(store, monkeypatch):
    """Bumping PARSER_VERSION is the invalidation mechanism. A store
    short-circuit that returned before checking it meant parser fixes changed
    nothing on any paper already ingested."""
    from paperlens import config
    from paperlens.paper.ingest import ingest_paper as _ingest

    _ingest(store, "2103.00020")
    row = store.paper_version_row("2103.00020v1")
    assert row["parser_version"] == config.PARSER_VERSION

    monkeypatch.setattr(config, "PARSER_VERSION", config.PARSER_VERSION + "-next")
    r = _ingest(store, "2103.00020")
    assert not any("already ingested" in n for n in r.notes), r.notes
    assert store.paper_version_row("2103.00020v1")["parser_version"].endswith("-next")
