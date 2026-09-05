"""Confidence assignment and the evidence invariant.

This is the *only* place confidence is decided. Call sites gather signals and
ask; they never pick a level themselves. That is what keeps the promise in
ARCHITECTURE section 4 true as the codebase grows.

The invariant, enforced in `record`:

    Any subject with confidence better than UNKNOWN must carry at least one
    SUPPORTS evidence row whose URI resolves. Otherwise the level is forced
    down to UNKNOWN.

UNKNOWN is the only level that needs no evidence, which is exactly why it has to
be a normal, frequent, unembarrassing outcome.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal

from ..graph.store import Store, utcnow

Confidence = Literal["CONFIRMED", "LIKELY", "POSSIBLE", "UNKNOWN"]
_ORDER: dict[str, int] = {"UNKNOWN": 0, "POSSIBLE": 1, "LIKELY": 2, "CONFIRMED": 3}

EvidenceKind = Literal[
    "paper_section", "paper_equation", "paper_stated_value", "paper_url",
    "code_symbol", "code_file", "code_absence", "citation_context",
    "repo_metadata", "external_url",
]


@dataclass
class Evidence:
    kind: EvidenceKind
    uri: str
    excerpt: str | None = None
    locator: str | None = None
    stance: Literal["SUPPORTS", "CONTRADICTS"] = "SUPPORTS"
    provenance: dict[str, Any] | None = None

    @property
    def id(self) -> str:
        h = hashlib.sha256(
            f"{self.kind}|{self.uri}|{self.locator}|{self.excerpt}".encode()
        ).hexdigest()[:16]
        return f"ev:{h}"


@dataclass
class Signal:
    """One observation that bears on a claim, with the weight it carries."""
    name: str
    weight: Literal["DECISIVE", "STRONG", "WEAK"]
    detail: str
    evidence: list[Evidence] = field(default_factory=list)


def assess(signals: Iterable[Signal]) -> tuple[Confidence, str]:
    """Combine signals into a confidence level and a one-line rationale.

    DECISIVE means machine-checkable and self-evidencing: an exact numeric
    match, an identical symbol name, a URL the authors printed in their own
    paper, or a citable proof of absence. Anything an LLM merely believes is
    never DECISIVE.
    """
    sigs = list(signals)
    supporting = [s for s in sigs if not any(e.stance == "CONTRADICTS" for e in s.evidence)]
    decisive = [s for s in supporting if s.weight == "DECISIVE"]
    strong = [s for s in supporting if s.weight == "STRONG"]
    weak = [s for s in supporting if s.weight == "WEAK"]

    if decisive:
        return "CONFIRMED", "; ".join(s.detail for s in decisive)
    if len(strong) >= 2:
        return "LIKELY", "; ".join(s.detail for s in strong)
    if strong:
        # One strong signal plus corroboration is LIKELY; alone it is POSSIBLE.
        if weak:
            return "LIKELY", "; ".join(s.detail for s in strong + weak)
        return "POSSIBLE", strong[0].detail
    if weak:
        return "POSSIBLE", "; ".join(s.detail for s in weak)
    return "UNKNOWN", "no supporting evidence was found"


def at_most(level: Confidence, ceiling: Confidence) -> Confidence:
    return level if _ORDER[level] <= _ORDER[ceiling] else ceiling


def record(
    store: Store,
    subject_kind: str,
    subject_id: str,
    confidence: Confidence,
    signals: Iterable[Signal],
    *,
    resolver: Callable[[str], bool] | None = None,
) -> Confidence:
    """Persist evidence and enforce the invariant. Returns the final confidence.

    `resolver` decides whether an evidence URI actually resolves. Passing None
    skips that check, which is only appropriate for evidence the server itself
    just produced.
    """
    sigs = list(signals)
    all_ev = [e for s in sigs for e in s.evidence]
    supports = [e for e in all_ev if e.stance == "SUPPORTS"]

    if resolver is not None:
        supports = [e for e in supports if resolver(e.uri)]

    final: Confidence = confidence if supports else "UNKNOWN"

    for ev in all_ev:
        if resolver is not None and ev.stance == "SUPPORTS" and not resolver(ev.uri):
            continue
        store.execute(
            "INSERT OR REPLACE INTO evidence (id, kind, uri, excerpt, locator, "
            "retrieved_at, provenance_json) VALUES (?,?,?,?,?,?,?)",
            (ev.id, ev.kind, ev.uri, ev.excerpt, ev.locator, utcnow(),
             json.dumps(ev.provenance) if ev.provenance else None),
        )
        store.execute(
            "INSERT OR REPLACE INTO evidence_links (evidence_id, subject_kind, "
            "subject_id, stance) VALUES (?,?,?,?)",
            (ev.id, subject_kind, subject_id, ev.stance),
        )
    return final


def evidence_for(store: Store, subject_kind: str, subject_id: str) -> list[dict[str, Any]]:
    rows = store.all(
        "SELECT e.*, l.stance FROM evidence e "
        "JOIN evidence_links l ON l.evidence_id = e.id "
        "WHERE l.subject_kind = ? AND l.subject_id = ?",
        (subject_kind, subject_id),
    )
    return [
        {"kind": r["kind"], "uri": r["uri"], "excerpt": r["excerpt"],
         "locator": r["locator"], "stance": r["stance"],
         "provenance": json.loads(r["provenance_json"]) if r["provenance_json"] else None}
        for r in rows
    ]
