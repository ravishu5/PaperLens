"""Research lineage: what a paper builds on, and what built on it.

The two directions have very different evidential standing, and the output says
so rather than blurring them:

  backward  the paper's own bibliography. Exact, offline, quotable -- we can
            show the sentence in which the paper cites the work. CONFIRMED.
  forward   a third-party citation index. A database assertion we did not
            verify from source. LIKELY, or CONFIRMED where Semantic Scholar
            returns the citing sentence.

Calling something a *successor* rather than merely a citer is an interpretation,
so it is never better than LIKELY and requires a signal beyond mere citation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..graph.store import base_id, Store
from ..resources import _aid, encode_paper, paper_uri, _pv
from ..sources import openalex
from ..sources import semanticscholar as s2

_METHODOLOGY = "methodology"


@dataclass
class Node:
    paper_id: str
    title: str | None
    year: int | None
    relation: str
    confidence: str
    why: str
    contexts: list[str] = field(default_factory=list)
    citation_count: int | None = None
    is_influential: bool | None = None
    intents: list[str] = field(default_factory=list)


def _backward(store: Store, pv: str, arxiv_id: str, limit: int) -> list[Node]:
    """Predecessors from the paper's own bibliography."""
    rows = store.all(
        "SELECT b.*, ("
        "  SELECT COUNT(*) FROM citation_sites c "
        "  WHERE c.paper_version = b.paper_version AND c.bib_key = b.bib_key"
        ") AS sites FROM bib_entries b WHERE b.paper_version = ? "
        "ORDER BY sites DESC, b.year DESC", (pv,))
    out: list[Node] = []
    for r in rows[:limit]:
        contexts = [c["context"] for c in store.all(
            "SELECT context FROM citation_sites WHERE paper_version = ? AND bib_key = ? "
            "LIMIT 3", (pv, r["bib_key"]))]
        out.append(Node(
            paper_id=r["arxiv_id"] or f"bib:{r['bib_key']}",
            title=r["title"], year=r["year"],
            relation="PREDECESSOR" if r["sites"] > 1 else "CITES",
            # The paper names it and we can quote where: nothing is inferred.
            confidence="CONFIRMED",
            why=f"cited {r['sites']} time(s) in the paper's own text",
            contexts=contexts, citation_count=None,
        ))
    return out


def _forward(store: Store, arxiv_id: str, limit: int) -> tuple[list[Node], list[str]]:
    """Successors from citation indexes, enriched with S2 contexts where possible."""
    notes: list[str] = []
    ranked: list[dict] = []
    try:
        ranked = openalex.citing_works(store, arxiv_id, limit=limit)
    except openalex.OpenAlexUnavailable as exc:
        notes.append(f"OpenAlex unavailable ({exc}).")

    s2_by_arxiv: dict[str, s2.CitingPaper] = {}
    s2_by_title: dict[str, s2.CitingPaper] = {}
    try:
        for c in s2.citations(store, arxiv_id, limit=100):
            if c.arxiv_id:
                s2_by_arxiv[c.arxiv_id] = c
            if c.title:
                s2_by_title[c.title.lower()[:70]] = c
    except s2.S2Unavailable as exc:
        notes.append(
            f"Semantic Scholar unavailable ({exc}). Forward edges therefore carry "
            f"no citing sentences, only the fact of the citation."
            + ("" if s2.authenticated() else
               " Set S2_API_KEY for a guaranteed request rate.")
        )

    if not ranked and s2_by_arxiv:
        ranked = [{"title": c.title, "year": c.year, "arxiv_id": c.arxiv_id,
                   "cited_by_count": c.citation_count}
                  for c in s2_by_arxiv.values()]

    out: list[Node] = []
    for w in ranked[:limit]:
        hit = (s2_by_arxiv.get(w.get("arxiv_id") or "")
               or s2_by_title.get((w.get("title") or "").lower()[:70]))
        contexts = hit.contexts if hit else []
        intents = hit.intents if hit else []
        influential = hit.is_influential if hit else None

        # "Successor" is an interpretation and needs more than a citation edge.
        if influential or _METHODOLOGY in intents:
            relation, confidence = "SUCCESSOR", "LIKELY"
            why = ("Semantic Scholar marks the citation influential"
                   if influential else "cited as methodology")
        else:
            relation, confidence = "CITES", ("CONFIRMED" if contexts else "LIKELY")
            why = ("the citing paper's own sentences are available"
                   if contexts else "present in the citation index; not verified from source")
        out.append(Node(
            paper_id=w.get("arxiv_id") or (hit.paper_id if hit else "") or "unknown",
            title=w.get("title"), year=w.get("year"), relation=relation,
            confidence=confidence, why=why, contexts=contexts[:3],
            citation_count=w.get("cited_by_count"), is_influential=influential,
            intents=intents,
        ))
    return out, notes


def _persist(store: Store, arxiv_id: str, nodes: list[Node], forward: bool) -> None:
    for n in nodes:
        if not n.paper_id or n.paper_id.startswith("bib:"):
            continue
        frm, to = (arxiv_id, n.paper_id) if not forward else (n.paper_id, arxiv_id)
        store.execute(
            "INSERT OR REPLACE INTO lineage_edges (id, from_paper, to_paper, relation, "
            "is_influential, intents_json, contexts_json, confidence, source) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (f"{frm}->{to}:{n.relation}", frm, to, n.relation,
             None if n.is_influential is None else int(n.is_influential),
             json.dumps(n.intents) if n.intents else None,
             json.dumps(n.contexts) if n.contexts else None,
             n.confidence, "bibliography" if not forward else "citation_index"))
    store.commit()


def _pub(n: Node) -> dict[str, Any]:
    return {"paper": n.paper_id, "title": n.title, "year": n.year,
            "relation": n.relation, "confidence": n.confidence, "why": n.why,
            "contexts": n.contexts, "citation_count": n.citation_count,
            "is_influential": n.is_influential, "intents": n.intents,
            "uri": (f"paperlens://paper/{n.paper_id}"
                    if re.fullmatch(r"\d{4}\.\d{4,5}", n.paper_id) else None)}


def trace_research_lineage(store: Store, paper_id: str, direction: str = "both",
                           limit: int = 20) -> dict[str, Any]:
    pv = _pv(store, paper_id)
    arxiv_id = _aid(store, pv)
    notes: list[str] = []
    back: list[Node] = []
    fwd: list[Node] = []

    if direction in ("back", "both"):
        back = _backward(store, pv, arxiv_id, limit)
        _persist(store, arxiv_id, back, forward=False)
        if not back:
            notes.append("No bibliography was recovered for this paper, so no "
                         "predecessors can be established from its own text.")

    if direction in ("forward", "both"):
        fwd, fnotes = _forward(store, arxiv_id, limit)
        notes += fnotes
        _persist(store, arxiv_id, fwd, forward=True)
        if not fwd:
            notes.append("No citing papers were retrieved.")

    notes.append(
        "Backward edges come from the paper's own bibliography and are quotable. "
        "Forward edges come from a citation index and were not verified against "
        "the citing paper's source."
    )
    return {
        "paper_version": pv, "uri": f"paperlens://lineage/{encode_paper(arxiv_id)}",
        "predecessors": [_pub(n) for n in back],
        "successors": [_pub(n) for n in fwd],
        "counts": {"predecessors": len(back), "successors": len(fwd)},
        "notes": notes,
    }


def find_sota_successors(store: Store, paper_id: str, limit: int = 15) -> dict[str, Any]:
    pv = _pv(store, paper_id)
    arxiv_id = _aid(store, pv)
    nodes, notes = _forward(store, arxiv_id, max(limit * 3, 40))
    _persist(store, arxiv_id, nodes, forward=True)

    def rank(n: Node) -> tuple:
        return (0 if n.is_influential else 1,
                0 if _METHODOLOGY in n.intents else 1,
                -(n.citation_count or 0), -(n.year or 0))

    ranked = sorted(nodes, key=rank)[:limit]
    notes.append(
        "Ranking uses influence and citation intent where Semantic Scholar "
        "supplies them, and citation count otherwise. Citation count alone is a "
        "measure of attention, not of improvement."
    )
    notes.append(
        "What each successor changed is not asserted here. The citing sentences "
        "are returned as evidence; read them, or the successor's own sections, "
        "before claiming an improvement."
    )
    return {
        "paper_version": pv, "uri": f"paperlens://lineage/{encode_paper(arxiv_id)}",
        "successors": [_pub(n) for n in ranked],
        "notes": notes,
    }


def trace_method(store: Store, paper_id: str, method: str,
                 limit: int = 12) -> dict[str, Any]:
    """Follow one method backwards and forwards through quoted citations."""
    pv = _pv(store, paper_id)
    arxiv_id = _aid(store, pv)
    terms = [t for t in re.split(r"[^a-z0-9]+", method.lower()) if len(t) > 2]
    if not terms:
        return {"error": "ValueError", "message": "method must contain a word"}

    def mentions(text: str) -> bool:
        low = (text or "").lower()
        return all(t in low for t in terms)

    origins: list[dict[str, Any]] = []
    for r in store.all(
        "SELECT c.context, c.src_line, c.bib_key, b.title, b.year, b.arxiv_id "
        "FROM citation_sites c LEFT JOIN bib_entries b "
        "ON b.paper_version = c.paper_version AND b.bib_key = c.bib_key "
        "WHERE c.paper_version = ?", (pv,)
    ):
        if mentions(r["context"]):
            origins.append({
                "paper": r["arxiv_id"] or f"bib:{r['bib_key']}", "title": r["title"],
                "year": r["year"], "confidence": "CONFIRMED",
                "why": "the paper discusses this method in the sentence that cites "
                       "this reference",
                "context": r["context"], "src_line": r["src_line"],
                "uri": (f"paperlens://paper/{r['arxiv_id']}" if r["arxiv_id"] else None),
            })

    later: list[dict[str, Any]] = []
    nodes, notes = _forward(store, arxiv_id, 60)
    for n in nodes:
        hits = [c for c in n.contexts if mentions(c)]
        if hits:
            later.append({**_pub(n), "contexts": hits[:3],
                          "why": "the citing paper mentions this method where it "
                                 "cites the target"})

    in_paper = [{"section": r["section_path"], "title": r["title"],
                 "uri": f"paperlens://paper/{encode_paper(arxiv_id)}/section/{r['section_path']}"}
                for r in store.all(
                    "SELECT section_path, title FROM sections WHERE paper_version = ? "
                    "AND (LOWER(title) LIKE ? OR LOWER(body) LIKE ?) ORDER BY ordinal",
                    (pv, f"%{terms[0]}%", f"%{terms[0]}%"))][:6]

    notes.append(
        "Origins are drawn from the paper's own citing sentences and are quotable. "
        "Later work is matched on citing-sentence text, so a method discussed "
        "without naming it will be missed."
    )
    if not origins:
        notes.append(
            f"No sentence in this paper mentions {method!r} while citing another "
            f"work, so no origin can be established from its text."
        )
    return {
        "paper_version": pv, "method": method,
        "introduced_by": origins[:limit],
        "discussed_in_paper": in_paper,
        "carried_forward_by": later[:limit],
        "notes": notes,
    }
