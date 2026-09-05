"""The evidence invariant: confidence better than UNKNOWN requires evidence."""
import pytest

from paperlens.evidence.confidence import (Evidence, Signal, assess, at_most,
                                           evidence_for, record)
from paperlens.graph.store import Store


def ev(uri="paperlens://paper/x/urls", stance="SUPPORTS"):
    return Evidence(kind="paper_url", uri=uri, stance=stance)


def test_decisive_signal_confirms():
    assert assess([Signal("a", "DECISIVE", "author-declared", [ev()])])[0] == "CONFIRMED"


def test_two_strong_signals_are_likely():
    got, _ = assess([Signal("a", "STRONG", "x", [ev()]),
                     Signal("b", "STRONG", "y", [ev()])])
    assert got == "LIKELY"


def test_a_single_strong_signal_is_only_possible():
    assert assess([Signal("a", "STRONG", "x", [ev()])])[0] == "POSSIBLE"


def test_no_signals_is_unknown():
    assert assess([])[0] == "UNKNOWN"


def test_at_most_clamps():
    assert at_most("CONFIRMED", "LIKELY") == "LIKELY"
    assert at_most("POSSIBLE", "LIKELY") == "POSSIBLE"


def test_confidence_without_supporting_evidence_is_forced_to_unknown(tmp_path):
    """The core invariant. A caller cannot assert CONFIRMED with nothing behind it."""
    s = Store(tmp_path / "g.db")
    final = record(s, "mapping", "m1", "CONFIRMED",
                   [Signal("hunch", "DECISIVE", "I think so", evidence=[])])
    assert final == "UNKNOWN"


def test_contradicting_evidence_alone_does_not_support(tmp_path):
    s = Store(tmp_path / "g.db")
    final = record(s, "mapping", "m2", "LIKELY",
                   [Signal("absent", "WEAK", "not found",
                           [ev(stance="CONTRADICTS")])])
    assert final == "UNKNOWN"


def test_unresolvable_evidence_is_rejected(tmp_path):
    """Write-back protection: cited evidence that does not resolve cannot prop
    up a claim."""
    s = Store(tmp_path / "g.db")
    final = record(s, "mapping", "m3", "CONFIRMED",
                   [Signal("a", "DECISIVE", "x", [ev(uri="paperlens://bogus")])],
                   resolver=lambda uri: False)
    assert final == "UNKNOWN"
    assert evidence_for(s, "mapping", "m3") == []


def test_evidence_is_persisted_and_retrievable(tmp_path):
    s = Store(tmp_path / "g.db")
    record(s, "mapping", "m4", "CONFIRMED",
           [Signal("a", "DECISIVE", "x", [ev(uri="paperlens://paper/2103.00020/urls")])])
    s.commit()
    got = evidence_for(s, "mapping", "m4")
    assert len(got) == 1 and got[0]["stance"] == "SUPPORTS"
