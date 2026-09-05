"""Ranking and coverage rules for implementation discovery (offline)."""
from paperlens.correlate.discover import Candidate, _code_files, _rank_key, _shallow_coverage


def c(**kw):
    base = dict(owner="o", name="n", relation="THIRD_PARTY", confidence="POSSIBLE",
                reasoning="", coverage_score=None, coverage_kind="NONE")
    base.update(kw)
    return Candidate(**base)


def test_confirmed_official_outranks_high_coverage_stranger():
    """The bug this pins: ranking by coverage first put an unrelated repository
    full of blog assets above openai/CLIP."""
    official = c(relation="OFFICIAL", confidence="CONFIRMED", coverage_score=0.4)
    stranger = c(relation="THIRD_PARTY", confidence="POSSIBLE", coverage_score=1.0)
    assert _rank_key(official) > _rank_key(stranger)


def test_coverage_breaks_ties_within_a_confidence_tier():
    lo = c(confidence="LIKELY", coverage_score=0.2)
    hi = c(confidence="LIKELY", coverage_score=0.9)
    assert _rank_key(hi) > _rank_key(lo)


def test_only_source_files_count_toward_coverage():
    files = ["docs/model.md", "README.md", "assets/data.svg",
             "requirements-training.txt", "src/train.py"]
    assert _code_files(files) == ["src/train.py"]


def test_doc_directories_are_excluded():
    assert _code_files(["docs/train.py", "examples/loss.py", "src/loss.py"]) == ["src/loss.py"]


def test_missing_component_kinds_are_reported():
    score, matched, missing = _shallow_coverage(
        ["clip/model.py", "clip/simple_tokenizer.py"],
        ["architecture", "training", "loss"],
    )
    assert "architecture" in matched
    assert set(missing) == {"training", "loss"}
    assert score == 1 / 3


def test_coverage_is_none_when_there_is_nothing_to_measure():
    assert _shallow_coverage([], ["training"])[0] is None
    assert _shallow_coverage(["a.py"], [])[0] is None
