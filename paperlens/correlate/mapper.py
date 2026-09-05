"""map_paper_to_code: join paper anchors to code symbols, with evidence.

This is the capability nothing else does. Everyone has paper structure or code
symbols; the correlation between them is the hard part and it is what PaperLens
is for.

The matching is deliberately anchored on things that can be *checked*, not
things that can be believed:

  * a stated numeric constant matched against a literal in source is
    machine-checkable, so it can reach CONFIRMED;
  * a component name matched against a symbol name is suggestive, so it reaches
    LIKELY at best;
  * a component with no plausible symbol at all is ABSENT, backed by the
    backend's citable absence reference -- which is a finding, not a failure.

Nothing here reaches CONFIRMED on similarity alone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .. import config
from ..code.indexer import persist_symbol, provider, require_snapshot
from ..code.provider import Symbol
from ..evidence.confidence import Evidence, Signal, assess, evidence_for, record
from ..graph.store import Store
from ..resources import _pv, symbol_uri

# Reused from implementation discovery: component kind -> file/symbol patterns.
_KIND_PATTERNS: dict[str, re.Pattern] = {
    "loss": re.compile(r"loss|criterion|objective", re.I),
    "training": re.compile(r"train|fit|optimi[sz]", re.I),
    "dataset": re.compile(r"data|dataset|loader|corpus", re.I),
    "architecture": re.compile(r"model|net|module|layer|encoder|decoder|transformer", re.I),
    "preprocessing": re.compile(r"preprocess|transform|augment|token", re.I),
    "evaluation": re.compile(r"eval|test|benchmark|metric", re.I),
    "inference": re.compile(r"infer|predict|generate|sample|decode|forward", re.I),
    "optimizer": re.compile(r"optimi[sz]|schedul", re.I),
    "module": re.compile(r"module|block|head|pool|embed", re.I),
}

# Values too common to carry information. An exact match on 0.1 or 2 says nothing.
_COMMON_VALUES = {"0", "1", "2", "3", "4", "5", "10", "100", "1000",
                  "0.0", "0.1", "0.5", "1.0", "2.0", "0.9", "0.99"}

_STOPWORDS = {"the", "a", "an", "of", "for", "and", "with", "our", "we", "using",
              "based", "to", "on", "in", "is", "are", "from", "by", "efficient",
              "selecting", "creating", "sufficiently", "large", "choosing"}


@dataclass
class Mapping:
    id: str
    anchor_kind: str
    anchor_id: str
    anchor_label: str
    anchor_uri: str
    status: str                 # MATCHED | ABSENT | AMBIGUOUS | UNKNOWN
    confidence: str
    method: str
    reasoning: str
    symbol: Symbol | None = None
    symbol_uri: str | None = None
    candidates: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)


def _stem(token: str) -> str:
    """Crude prefix stem so "encoder" matches "encode_image"."""
    return token[:5] if len(token) > 5 else token


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower())
            if len(t) > 2 and t not in _STOPWORDS]


def _distinctive(value_text: str) -> bool:
    """Would an exact match on this literal actually mean something?

    Rarity is measured empirically by how many times the literal occurs (see the
    hit-count check below), not guessed from significant figures. An earlier
    sig-fig rule rejected 0.07 -- one significant digit, but a highly distinctive
    constant -- and lost the headline mapping.
    """
    return value_text.strip() not in _COMMON_VALUES


_RESULT_HINT = re.compile(
    r"accuracy|bleu|f1\b|mAP\b|precision|recall|score|percent|%|"
    r"outperform|achieve|state-of-the-art|human", re.I)


def _looks_like_result(symbol: str | None, context: str | None) -> bool:
    """Reported metrics are not hyperparameters.

    An accuracy of 91.8 is not expected to appear in inference code, so declaring
    it ABSENT is technically true and completely uninformative."""
    return bool(_RESULT_HINT.search(f"{symbol or ''} {(context or '')[:160]}"))


def _is_named_quantity(symbol: str | None) -> bool:
    """A hyperparameter has a name; a prose phrase does not.

    "assembled collection = 27" comes from "an assembled collection of 27
    datasets". Declaring 27 absent from source says nothing about the
    implementation."""
    return bool(symbol) and len(symbol.split()) == 1


def _containing_symbol(prov, repo_key: str, file_path: str, line: int) -> Symbol | None:
    for s in prov.file_outline(repo_key, file_path):
        if s.line_start <= line <= (s.line_end or s.line_start):
            if s.kind in ("method", "function"):
                return s
    for s in prov.file_outline(repo_key, file_path):
        if s.line_start <= line <= (s.line_end or s.line_start):
            return s
    return None


# ── per-anchor strategies ─────────────────────────────────────────────────
def _map_stated_value(store: Store, prov, repo_key: str, row) -> tuple:
    """Exact numeric literal match. The only route to CONFIRMED here."""
    value = (row["value_text"] or "").strip()
    if not value:
        return "UNKNOWN", [], "no value recorded", [], None
    if _looks_like_result(row["symbol"], row["context"]):
        return ("UNKNOWN", [],
                "this reads as a reported result rather than a hyperparameter, so "
                "its absence from source would not be meaningful", [], None)
    if not _is_named_quantity(row["symbol"]):
        return ("UNKNOWN", [],
                f"{row['symbol']!r} is a prose phrase rather than a named quantity, "
                f"so no claim about the code follows from it", [], None)

    hits = prov.search_text(repo_key, rf"(?<![\w.]){re.escape(value)}(?![\w.])",
                            regex=True, limit=25)
    if not hits:
        res_absent = prov.search_symbols(repo_key, value)
        return ("ABSENT", [],
                f"the literal {value} does not appear anywhere in the repository",
                [Evidence(kind="code_absence",
                          uri=f"paperlens://repo/{repo_key}",
                          excerpt=f"literal {value!r} not found in source",
                          stance="CONTRADICTS",
                          provenance={"absence_ref": res_absent.absence_ref,
                                      "method": "literal_search"})], None)

    if not _distinctive(value):
        return ("AMBIGUOUS", [f"{h.file_path}:{h.line}" for h in hits[:5]],
                f"{value} is too common a value for an exact match to be evidence",
                [], None)

    if len(hits) > 6:
        return ("AMBIGUOUS", [f"{h.file_path}:{h.line}" for h in hits[:5]],
                f"the literal {value} appears {len(hits)} times; no single site is "
                f"identifiable", [], None)

    hit = hits[0]
    sym = _containing_symbol(prov, repo_key, hit.file_path, hit.line)
    ev = [Evidence(kind="paper_stated_value",
                   uri=f"paperlens://paper/{row['paper_version'].split('v')[0]}/values",
                   excerpt=f"{row['symbol']} = {value}",
                   locator=f"LaTeX line {row['src_line']}"),
          Evidence(kind="code_symbol",
                   uri=(symbol_uri(repo_key, sym.qualified_name) if sym
                        else f"paperlens://repo/{repo_key}/file/{hit.file_path}"),
                   excerpt=hit.text,
                   locator=f"{hit.file_path}:{hit.line}")]
    return ("MATCHED", [f"{h.file_path}:{h.line}" for h in hits],
            f"the paper states {row['symbol']} = {value} and the identical literal "
            f"appears at {hit.file_path}:{hit.line}", ev, sym)


def _matched_value_sections(store: Store, pv: str,
                            snapshot_id: str) -> tuple[set[str], set[str]]:
    """Sections whose stated values were located in this repository.

    Used to refuse an ABSENT verdict for a component drawn from such a section.
    Name matching cannot see that the paper's tau is the code's `logit_scale`, so
    without this guard "Temperature Scaling is absent" is reported confidently
    while the constant it describes sits at model.py:295 -- exactly the
    confident-wrong claim this system exists to avoid.
    """
    rows = store.all(
        "SELECT v.section_id, v.symbol FROM mappings m "
        "JOIN stated_values v ON v.id = m.anchor_id "
        "WHERE m.paper_version = ? AND m.snapshot_id = ? AND m.anchor_kind = "
        "'stated_value' AND m.status = 'MATCHED'",
        (pv, snapshot_id))
    return ({r["section_id"] for r in rows if r["section_id"]},
            {(r["symbol"] or "").lower() for r in rows if r["symbol"]})


def _map_component(store: Store, prov, repo_key: str, row,
                   located: tuple[set[str], set[str]] | None = None) -> tuple:
    """Name and kind matching. Suggestive, never decisive.

    Only agent-authored components can be declared ABSENT. A section-derived
    component is a heading such as "Experiments", which was never expected to be
    a symbol, so "no symbol resembles Experiments" is a true statement that
    asserts nothing about the implementation.
    """
    name = row["name"] or row["slug"]
    kind = row["kind"]
    derived = row["source"] == "DETERMINISTIC"
    toks = _tokens(name)
    res = prov.search_symbols(repo_key, " ".join(toks) or name, limit=10)

    if not res.found or not res.symbols:
        if derived:
            return ("UNKNOWN", [],
                    f"{name!r} is a section heading rather than a named method "
                    f"component, so its absence from code is not informative",
                    [], None)
        sections, symbols = located or (set(), set())
        described = set(_tokens(f"{name} {row['description'] or ''}"))
        overlap = described & symbols
        if (sections and row["section_id"] in sections) or overlap:
            reason = (f"a constant it describes ({', '.join(sorted(overlap))}) was "
                      f"located in this repository"
                      if overlap else
                      "a constant stated in the same section was located here")
            return ("AMBIGUOUS", [],
                    f"no symbol is named like {name!r}, but {reason}, so the "
                    f"component is probably implemented under a different name",
                    [], None)
        return ("ABSENT", [],
                f"no symbol resembling {name!r} exists in this repository",
                [Evidence(kind="code_absence", uri=f"paperlens://repo/{repo_key}",
                          excerpt=f"no symbol matches component {name!r}",
                          stance="CONTRADICTS",
                          provenance={"absence_ref": res.absence_ref,
                                      "method": res.method})], None)

    pat = _KIND_PATTERNS.get(kind)
    scored: list[tuple[float, int, Symbol]] = []
    for s in res.symbols:
        hay = f"{s.name} {s.qualified_name}".lower()
        # Every token of the component name must be represented, not just one.
        # Scoring by raw hits let "Text Encoder" match requests' Response.text
        # on the single token "text" plus a models.py path.
        covered = sum(1 for t in toks if _stem(t) in hay)
        coverage = covered / len(toks) if toks else 0.0
        bonus = 1 if (pat and pat.search(s.name)) else 0
        scored.append((coverage, bonus, s))
    scored.sort(key=lambda t: (-t[0], -t[1], t[2].qualified_name))
    coverage, bonus, best = scored[0]
    best_score = covered_desc = round(coverage * len(toks))

    if coverage < 0.5:
        return ("UNKNOWN", [s.qualified_name for _, _, s in scored[:4]],
                f"symbols were returned for {name!r} but none carry enough of its "
                f"name to identify one", [], None)

    ev = [Evidence(kind="paper_section" if row["section_id"] else "paper_url",
                   uri=f"paperlens://paper/{row['paper_version'].split('v')[0]}"
                       f"/component/{row['slug']}",
                   excerpt=f"{name} ({kind})"),
          Evidence(kind="code_symbol", uri=symbol_uri(repo_key, best.qualified_name),
                   excerpt=best.signature or best.qualified_name,
                   locator=f"{best.file_path}:{best.line_start}")]
    # A single generic token is not an identification, and a section heading
    # matching a symbol name is a coincidence of vocabulary.
    status = "MATCHED" if (coverage == 1.0 and len(toks) >= 2) else "AMBIGUOUS"
    if derived:
        status = "AMBIGUOUS"
    return (status, [s.qualified_name for _, _, s in scored[:4]],
            f"{name!r} matches {covered_desc}/{len(toks)} of its name tokens against "
            f"{best.qualified_name}", ev, best)


def _map_algorithm(store: Store, prov, repo_key: str, row) -> tuple:
    if not row["extractable"]:
        return ("UNKNOWN", [],
                "the algorithm is a figure image in the paper, so there is no text "
                "to correlate against code", [], None)
    name = row["name"] or row["slug"]
    res = prov.search_symbols(repo_key, " ".join(_tokens(name)) or name, limit=6)
    if not res.found:
        return ("ABSENT", [], f"no symbol resembling algorithm {name!r}",
                [Evidence(kind="code_absence", uri=f"paperlens://repo/{repo_key}",
                          excerpt=f"no symbol matches algorithm {name!r}",
                          stance="CONTRADICTS",
                          provenance={"absence_ref": res.absence_ref})], None)
    best = res.symbols[0]
    return ("AMBIGUOUS", [s.qualified_name for s in res.symbols[:4]],
            f"algorithm {name!r} may correspond to {best.qualified_name}, but a "
            f"name match alone does not establish it",
            [Evidence(kind="code_symbol", uri=symbol_uri(repo_key, best.qualified_name),
                      locator=f"{best.file_path}:{best.line_start}")], best)


# ── driver ────────────────────────────────────────────────────────────────
def map_paper_to_code(
    store: Store, paper_id: str, repo: str,
    anchor_kind: str | None = None, anchor_id: str | None = None,
    max_anchors: int = 60,
) -> dict[str, Any]:
    pv = _pv(store, paper_id)
    arxiv_id = pv.split("v")[0]
    snap = require_snapshot(store, repo)
    repo_key, snapshot_id = snap["repo_id"], snap["id"]
    prov = provider()

    anchors: list[tuple[str, Any]] = []
    if anchor_kind in (None, "stated_value"):
        anchors += [("stated_value", r) for r in store.all(
            "SELECT * FROM stated_values WHERE paper_version = ?"
            + (" AND id = ?" if anchor_id else ""),
            (pv, anchor_id) if anchor_id else (pv,))]
    if anchor_kind in (None, "component"):
        anchors += [("component", r) for r in store.all(
            "SELECT * FROM components WHERE paper_version = ?"
            + (" AND (id = ? OR slug = ?)" if anchor_id else ""),
            (pv, anchor_id, anchor_id) if anchor_id else (pv,))]
    if anchor_kind in (None, "algorithm"):
        anchors += [("algorithm", r) for r in store.all(
            "SELECT * FROM algorithms WHERE paper_version = ?"
            + (" AND (id = ? OR slug = ?)" if anchor_id else ""),
            (pv, anchor_id, anchor_id) if anchor_id else (pv,))]

    mappings: list[Mapping] = []
    located: tuple[set[str], set[str]] | None = None
    for kind, row in anchors[:max_anchors]:
        if kind == "component" and located is None:
            # Stated values are mapped first, so their results are available here.
            located = _matched_value_sections(store, pv, snapshot_id)
        if kind == "stated_value":
            status, cands, why, ev, sym = _map_stated_value(store, prov, repo_key, row)
            label = f"{row['symbol']} = {row['value_text']}"
            uri = f"paperlens://paper/{arxiv_id}/values"
        elif kind == "component":
            status, cands, why, ev, sym = _map_component(
                store, prov, repo_key, row, located)
            label = row["name"]
            uri = f"paperlens://paper/{arxiv_id}/component/{row['slug']}"
        else:
            status, cands, why, ev, sym = _map_algorithm(store, prov, repo_key, row)
            label = row["name"] or row["slug"]
            uri = f"paperlens://paper/{arxiv_id}/algorithm/{row['slug']}"

        weight = {"MATCHED": "DECISIVE" if kind == "stated_value" else "STRONG",
                  "AMBIGUOUS": "WEAK", "ABSENT": "DECISIVE",
                  "UNKNOWN": "WEAK"}[status]
        supporting = [e for e in ev if e.stance == "SUPPORTS"]
        if status == "ABSENT":
            # An established absence is itself a confirmed finding.
            conf, _ = ("CONFIRMED", why) if ev else ("UNKNOWN", why)
        elif supporting:
            conf, _ = assess([Signal(f"{kind}_match", weight, why, supporting)])
        else:
            conf = "UNKNOWN"

        sym_uri = None
        if sym is not None and status in ("MATCHED", "AMBIGUOUS"):
            persist_symbol(store, snapshot_id, sym)
            sym_uri = symbol_uri(repo_key, sym.qualified_name)

        mapping_id = f"{pv}|{snapshot_id}|{kind}|{row['id']}"
        final = record(store, "mapping", mapping_id, conf,
                       [Signal(f"{kind}_match", weight, why, ev)])
        if status == "ABSENT":
            final = "CONFIRMED" if ev else "UNKNOWN"

        store.execute(
            "INSERT OR REPLACE INTO mappings (id, paper_version, snapshot_id, "
            "anchor_kind, anchor_id, symbol_id, status, confidence, method, "
            "reasoning, analyzer_version) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (mapping_id, pv, snapshot_id, kind, row["id"],
             f"{snapshot_id}:{sym.qualified_name}" if sym and sym_uri else None,
             status, final, f"mapper:{kind}", why, config.ANALYZER_VERSION))

        mappings.append(Mapping(
            id=mapping_id, anchor_kind=kind, anchor_id=row["id"], anchor_label=label, anchor_uri=uri,
            status=status, confidence=final, method=f"mapper:{kind}", reasoning=why,
            symbol=sym, symbol_uri=sym_uri, candidates=cands,
            evidence=evidence_for(store, "mapping", mapping_id)))
    store.commit()

    counts = {s: sum(1 for m in mappings if m.status == s)
              for s in ("MATCHED", "ABSENT", "AMBIGUOUS", "UNKNOWN")}
    notes: list[str] = []
    if counts["MATCHED"] == 0 and mappings:
        notes.append(
            "Nothing in this paper could be located in this repository. That is the "
            "expected result when the repository does not implement the paper."
        )
    if any(m.anchor_kind == "component" and m.status == "UNKNOWN" for m in mappings):
        notes.append(
            "Some components are section-derived headings rather than named method "
            "components, which rarely correspond to a symbol. Record a paper "
            "analysis to supply finer-grained components."
        )
    absent_named = [m.anchor_label for m in mappings if m.status == "ABSENT"]
    if absent_named:
        notes.append(
            f"{len(absent_named)} anchor(s) are established as absent from this "
            f"repository, with citable evidence: {', '.join(absent_named[:4])}."
        )

    return {
        "paper_version": pv, "repo": repo_key, "commit_sha": snap["commit_sha"],
        "uri": f"paperlens://mapping/{arxiv_id}/{repo_key}",
        "summary": counts,
        "mappings": [{
            "id": m.id,
            "anchor": {"kind": m.anchor_kind, "label": m.anchor_label, "uri": m.anchor_uri},
            "status": m.status, "confidence": m.confidence, "method": m.method,
            "reasoning": m.reasoning, "symbol_uri": m.symbol_uri,
            "candidates": m.candidates[:4], "evidence": m.evidence,
        } for m in mappings],
        "notes": notes,
    }
