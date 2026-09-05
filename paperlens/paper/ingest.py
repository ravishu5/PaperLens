"""Paper ingestion: fetch -> flatten -> parse -> persist."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .. import config
from ..graph.store import Store
from ..sources import arxiv, crossref
from .citations import collect as collect_citations
from .equations import extract_equations
from .structure import (Section, extract_algorithms, extract_declared_urls,
                        extract_sections, extract_stated_values, slugify)

# Section-title keywords -> component kind. This is a coarse, deterministic
# first pass; rich components come from agent write-back in a later phase.
_KIND_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bloss(es)?\b|\bobjective\b|\bcriterion\b", re.I), "loss"),
    (re.compile(r"\barchitectur|\bmodel\b|\bnetwork\b|\bencoder\b|\bdecoder\b", re.I), "architecture"),
    (re.compile(r"\boptimi[sz]|\blearning rate\b|\bschedule\b", re.I), "optimizer"),
    (re.compile(r"\bdataset(s)?\b|\bcorpus\b|\bdata collection\b", re.I), "dataset"),
    (re.compile(r"\bpre-?process|\baugmentation\b|\btokeni[sz]", re.I), "preprocessing"),
    (re.compile(r"\btrain(ing)?\b|\bpre-?train", re.I), "training"),
    (re.compile(r"\binference\b|\bdecoding\b|\bzero-?shot\b|\bprediction\b", re.I), "inference"),
    (re.compile(r"\bevaluat|\bexperiment|\bbenchmark|\bresults?\b|\bablation", re.I), "evaluation"),
    (re.compile(r"\bapproach\b|\bmethod\b|\bcomponent\b|\bmodule\b", re.I), "module"),
]


def _kind_for(title: str) -> str | None:
    for pat, kind in _KIND_HINTS:
        if pat.search(title):
            return kind
    return None


def _locate(sections: list[Section], offset: int) -> Section | None:
    """Which section contains this character offset?"""
    best = None
    for s in sections:
        if s.char_start <= offset < s.char_end:
            # Deepest containing section wins.
            if best is None or s.level > best.level:
                best = s
    return best


@dataclass
class IngestResult:
    arxiv_id: str
    paper_version: str
    title: str
    fidelity: str
    sections: int
    equations: int
    numbered_equations: int
    algorithms: int
    unextractable_algorithms: int
    stated_values: int
    declared_urls: int
    components: int
    notes: list[str]


def enrich_references_from_crossref(store: Store, paper_version: str,
                                    doi: str) -> int:
    """Add a publisher-registered reference list to a paper that has none.

    A record ingested from a PDF carries the body text but no bibliography we can
    resolve, and a metadata-only record has no body at all. Either way Crossref
    holds the reference list the publisher registered, which is what the backward
    half of a lineage needs.
    """
    have = store.one("SELECT COUNT(*) c FROM bib_entries WHERE paper_version = ?",
                     (paper_version,))["c"]
    if have:
        return 0
    try:
        w = crossref.work(store, doi)
    except crossref.CrossrefUnavailable:
        return 0
    if w is None:
        return 0
    paper_id = store.paper_id_of(paper_version)
    for r in w.references:
        store.execute(
            "INSERT OR REPLACE INTO bib_entries (id, paper_version, bib_key, raw, "
            "authors, title, year, arxiv_id, doi, cite_count) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (f"{paper_version}:bib:{r.key}", paper_version, r.key, r.raw, None,
             r.title, r.year, None, r.doi, 0))
        if r.doi:
            store.execute(
                "INSERT OR REPLACE INTO lineage_edges (id, from_paper, to_paper, "
                "relation, is_influential, intents_json, contexts_json, confidence, "
                "source) VALUES (?,?,?,?,?,?,?,?,?)",
                (f"{paper_id}->{r.doi}:CITES", paper_id, r.doi, "CITES", None,
                 None, None, "CONFIRMED", "crossref"))
    store.commit()
    return len(w.references)


def ingest_metadata_only(store: Store, doi: str) -> IngestResult:
    """Ingest a paper that has no retrievable full text.

    Most medical-imaging work is published in closed-access journals: there is no
    LaTeX source and no readable PDF, so sections, equations and stated values
    are simply unavailable. What Crossref does register is the reference list,
    which is enough to place the paper in a research lineage. Fidelity is
    UNAVAILABLE and every downstream tool is expected to say so rather than
    infer around the gap.
    """
    w = crossref.work(store, doi)
    if w is None:
        raise LookupError(
            f"Crossref has no record for DOI {doi!r}. Check the identifier, or "
            f"supply an arXiv id if a preprint exists."
        )
    paper_version = f"{w.doi}v1"
    store.upsert_paper(w.doi, w.title, abstract=w.abstract, doi=w.doi,
                       published_at=w.published_at, authors=w.authors,
                       latest_version=paper_version)
    store.upsert_paper_version(paper_version, w.doi, 1, source_kind="NONE",
                               fidelity="UNAVAILABLE", source_sha256=None,
                               flattened_tex=None)
    store.clear_paper_artifacts(paper_version)

    for r in w.references:
        store.execute(
            "INSERT OR REPLACE INTO bib_entries (id, paper_version, bib_key, raw, "
            "authors, title, year, arxiv_id, doi, cite_count) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (f"{paper_version}:bib:{r.key}", paper_version, r.key, r.raw, None,
             r.title, r.year, None, r.doi, 0))
        if r.doi:
            store.execute(
                "INSERT OR REPLACE INTO lineage_edges (id, from_paper, to_paper, "
                "relation, is_influential, intents_json, contexts_json, confidence, "
                "source) VALUES (?,?,?,?,?,?,?,?,?)",
                (f"{w.doi}->{r.doi}:CITES", w.doi, r.doi, "CITES", None, None,
                 None, "CONFIRMED", "crossref"))
    store.commit()

    notes = [
        f"Published in {w.venue}." if w.venue else "No venue recorded.",
        "No full text is available for this paper: it is not on arXiv and the "
        "publisher record carries no open-access copy. Sections, equations, "
        "stated values and declared repository URLs are therefore UNAVAILABLE, "
        "and any tool that depends on them will say so.",
        f"{len(w.references)} reference(s) recorded from the publisher's "
        f"registered reference list; {sum(1 for r in w.references if r.doi)} "
        f"carry a DOI and became lineage edges.",
    ]
    if not w.abstract:
        notes.append("The publisher registered no abstract with Crossref.")
    return _summarize(store, paper_version, w.title, "UNAVAILABLE", notes)


def ingest_paper(store: Store, paper_id: str, force: bool = False) -> IngestResult:
    # A paper already in the store may have arrived from a non-arXiv source
    # (DOI, PDF), in which case there is nothing to fetch. Matching is exact:
    # a substring fallback here resolved any short string to an arbitrary paper.
    if not force:
        existing = store.all(
            "SELECT * FROM papers WHERE (arxiv_id = ? OR doi = ?) "
            "AND latest_version IS NOT NULL",
            (paper_id.strip(), paper_id.strip()),
        )
        if len(existing) > 1:
            raise ValueError(
                f"{paper_id!r} matches {len(existing)} ingested papers. Use a full "
                f"identifier."
            )
        if len(existing) == 1:
            pv = existing[0]["latest_version"]
            pv_row = store.paper_version_row(pv)
            # Only a version parsed by the *current* parser may be served from
            # the store. Returning here unconditionally defeated the invalidation
            # mechanism entirely: a parser fix changed nothing on any paper that
            # had already been ingested, and stale extractions kept reappearing
            # in output that had supposedly been regenerated.
            if pv_row is not None:
                stale = pv_row["parser_version"] != config.PARSER_VERSION
                # A LaTeX paper can always be re-fetched and re-parsed, so a
                # parser upgrade must invalidate it. A PDF-derived or
                # metadata-only record cannot be reproduced from anything we
                # still hold -- re-parsing it would destroy content and replace
                # it with nothing.
                reproducible = pv_row["source_kind"] == "LATEX"
                if not stale or not reproducible:
                    notes = ["loaded from local store: already ingested"]
                    if existing[0]["doi"]:
                        added = enrich_references_from_crossref(
                            store, pv, existing[0]["doi"])
                        if added:
                            notes.append(
                                f"{added} reference(s) added from the publisher's "
                                f"registered reference list, which this record did "
                                f"not carry.")
                    if stale:
                        notes.append(
                            f"Parsed by parser version {pv_row['parser_version']}, "
                            f"current is {config.PARSER_VERSION}. This record came "
                            f"from {pv_row['source_kind']} and cannot be re-derived, "
                            f"so it is served as stored rather than discarded."
                        )
                    return _summarize(store, pv, existing[0]["title"],
                                      pv_row["fidelity"], notes)

    parsed = arxiv.parse_arxiv_id(paper_id)
    if not parsed:
        # Not on arXiv: fall back to publisher metadata, which is all a
        # closed-access journal article makes available.
        if (doi := crossref.parse_doi(paper_id)):
            return ingest_metadata_only(store, doi)
        raise ValueError(
            f"{paper_id!r} is neither an arXiv identifier nor a DOI. "
            f"Use resolve_paper first."
        )
    arxiv_id, want_version = parsed

    meta = arxiv.fetch_metadata(store, arxiv_id)
    version = want_version or meta.version
    paper_version = f"{arxiv_id}v{version}"

    existing = store.paper_version_row(paper_version)
    if existing and not force and existing["parser_version"] == config.PARSER_VERSION:
        return _summarize(store, paper_version, meta.title, existing["fidelity"],
                          ["cached: already ingested with this parser version"])

    store.upsert_paper(arxiv_id, meta.title, abstract=meta.abstract, doi=meta.doi,
                       published_at=meta.published_at, authors=meta.authors,
                       latest_version=paper_version)

    notes: list[str] = []
    tex = arxiv.fetch_latex(store, arxiv_id)
    if tex:
        source_kind, fidelity = "LATEX", "LATEX_EXACT"
    else:
        source_kind, fidelity = "NONE", "UNAVAILABLE"
        notes.append(
            "No LaTeX e-print source available. PDF fallback is not implemented "
            "yet, so structure is UNAVAILABLE for this paper."
        )
        tex = ""

    sha = hashlib.sha256(tex.encode("utf-8")).hexdigest() if tex else None
    store.upsert_paper_version(paper_version, arxiv_id, version,
                               source_kind=source_kind, fidelity=fidelity,
                               source_sha256=sha, flattened_tex=tex or None)
    store.clear_paper_artifacts(paper_version)

    if not tex:
        store.commit()
        return _summarize(store, paper_version, meta.title, fidelity, notes)

    # ── sections ──────────────────────────────────────────────────────────
    sections = extract_sections(tex)
    for s in sections:
        sid = f"{paper_version}:sec:{s.section_path}"
        store.execute(
            "INSERT OR REPLACE INTO sections (id, paper_version, section_path, level, "
            "title, latex_label, body, src_line_start, src_line_end, ordinal) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (sid, paper_version, s.section_path, s.level, s.title, s.latex_label,
             s.body, s.src_line_start, s.src_line_end, s.ordinal),
        )
        store.execute(
            "INSERT INTO sections_fts (title, body, section_id, paper_version) "
            "VALUES (?,?,?,?)", (s.title, s.body, sid, paper_version),
        )

    def sec_id(offset: int) -> str | None:
        s = _locate(sections, offset)
        return f"{paper_version}:sec:{s.section_path}" if s else None

    # ── equations ─────────────────────────────────────────────────────────
    equations = extract_equations(tex)
    for e in equations:
        store.execute(
            "INSERT OR REPLACE INTO equations (id, paper_version, content_hash, latex, "
            "environment, is_numbered, derived_number, number_conf, latex_label, "
            "section_id, src_line, ordinal) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"{paper_version}:eq:{e.content_hash}", paper_version, e.content_hash,
             e.latex, e.environment, int(e.is_numbered), e.derived_number,
             e.number_conf, e.latex_label, sec_id(e.char_start), e.src_line, e.ordinal),
        )
    if equations and not any(e.is_numbered for e in equations):
        notes.append("No numbered equations in this paper; equation anchors are limited.")
    if not equations:
        notes.append(
            "This paper contains no display-math environments at all. Map to code "
            "via components and stated values rather than equations."
        )

    # ── algorithms ────────────────────────────────────────────────────────
    algos = extract_algorithms(tex)
    for a in algos:
        store.execute(
            "INSERT OR REPLACE INTO algorithms (id, paper_version, slug, name, body, "
            "presentation, extractable, section_id, src_line) VALUES (?,?,?,?,?,?,?,?,?)",
            (f"{paper_version}:algo:{a.slug}", paper_version, a.slug, a.name, a.body,
             a.presentation, int(a.extractable), sec_id(a.char_start), a.src_line),
        )
    n_unextractable = sum(1 for a in algos if not a.extractable)
    if n_unextractable:
        notes.append(
            f"{n_unextractable} algorithm(s) are figure images, not text; their "
            "content cannot be extracted from LaTeX source."
        )

    # ── stated values ─────────────────────────────────────────────────────
    values = extract_stated_values(tex)
    for i, v in enumerate(values):
        store.execute(
            "INSERT OR REPLACE INTO stated_values (id, paper_version, symbol, value_text, "
            "value_num, context, section_id, src_line) VALUES (?,?,?,?,?,?,?,?)",
            (f"{paper_version}:val:{i}", paper_version, v.symbol, v.value_text,
             v.value_num, v.context, sec_id(v.char_start), v.src_line),
        )

    # ── declared URLs ─────────────────────────────────────────────────────
    urls = extract_declared_urls(tex)
    for i, u in enumerate(urls):
        store.execute(
            "INSERT OR REPLACE INTO declared_urls (id, paper_version, url, host, owner, "
            "repo, context, section_id, in_abstract, src_line) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (f"{paper_version}:url:{i}", paper_version, u.url, u.host, u.owner, u.repo,
             u.context, sec_id(u.char_start), int(u.in_abstract), u.src_line),
        )

    # ── bibliography and citation sites ───────────────────────────────────
    # The paper carries its own references, so the backward half of the research
    # lineage needs no citation API.
    try:
        entries, sites = collect_citations(tex, config.source_cache_dir() / arxiv_id)
    except OSError:
        entries, sites = {}, []

    for key, e in entries.items():
        store.execute(
            "INSERT OR REPLACE INTO bib_entries (id, paper_version, bib_key, raw, "
            "authors, title, year, arxiv_id, doi, cite_count) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (f"{paper_version}:bib:{key}", paper_version, key, e.raw, e.authors,
             e.title, e.year, e.arxiv_id, e.doi, e.cite_count))
    for i, site in enumerate(sites):
        store.execute(
            "INSERT OR REPLACE INTO citation_sites (id, paper_version, bib_key, "
            "section_id, context, src_line, command) VALUES (?,?,?,?,?,?,?)",
            (f"{paper_version}:cite:{i}", paper_version, site.bib_key,
             sec_id(site.char_start), site.context, site.src_line, site.command))

    # A resolved reference is a CONFIRMED lineage edge: the paper names it, and
    # we can quote the sentence that does so.
    import json as _json
    for key, e in entries.items():
        if not e.arxiv_id or not e.sites:
            continue
        store.execute(
            "INSERT OR REPLACE INTO lineage_edges (id, from_paper, to_paper, relation, "
            "is_influential, intents_json, contexts_json, confidence, source) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (f"{arxiv_id}->{e.arxiv_id}:CITES", arxiv_id, e.arxiv_id, "CITES",
             None, None, _json.dumps([s.context for s in e.sites[:4]]),
             "CONFIRMED", "bibliography"))
    if entries:
        unresolved = sum(1 for e in entries.values() if not e.arxiv_id)
        notes.append(
            f"{len(entries)} bibliography entries parsed; {len(entries) - unresolved} "
            f"resolved to arXiv identifiers. Unresolved entries are still listed "
            f"with title and authors."
        )

    # ── deterministic components (coarse; enriched later by write-back) ───
    seen: set[str] = set()
    for s in sections:
        kind = _kind_for(s.title)
        if not kind:
            continue
        slug = slugify(s.title)
        if slug in seen:
            continue
        seen.add(slug)
        cid = f"{paper_version}:comp:{slug}"
        store.execute(
            "INSERT OR REPLACE INTO components (id, paper_version, slug, name, kind, "
            "description, section_id, source) VALUES (?,?,?,?,?,?,?,'DETERMINISTIC')",
            (cid, paper_version, slug, s.title, kind,
             re.sub(r"\s+", " ", s.body)[:400], f"{paper_version}:sec:{s.section_path}"),
        )
        store.execute(
            "INSERT INTO components_fts (name, description, component_id, paper_version) "
            "VALUES (?,?,?,?)", (s.title, s.body[:2000], cid, paper_version),
        )
    notes.append(
        "Components are section-derived only at this stage. Finer-grained method "
        "components come from agent analysis."
    )

    store.commit()
    return _summarize(store, paper_version, meta.title, fidelity, notes)


def _count(store: Store, table: str, pv: str, where: str = "") -> int:
    sql = f"SELECT COUNT(*) c FROM {table} WHERE paper_version = ?" + (f" AND {where}" if where else "")
    return store.one(sql, (pv,))["c"]


def _summarize(store: Store, pv: str, title: str, fidelity: str,
               notes: list[str]) -> IngestResult:
    # Looked up, not derived: a PDF-ingested record may be versioned
    # "uld-netv1" while the paper itself is keyed by DOI.
    arxiv_id = store.paper_id_of(pv)
    return IngestResult(
        arxiv_id=arxiv_id, paper_version=pv, title=title, fidelity=fidelity,
        sections=_count(store, "sections", pv),
        equations=_count(store, "equations", pv),
        numbered_equations=_count(store, "equations", pv, "is_numbered = 1"),
        algorithms=_count(store, "algorithms", pv),
        unextractable_algorithms=_count(store, "algorithms", pv, "extractable = 0"),
        stated_values=_count(store, "stated_values", pv),
        declared_urls=_count(store, "declared_urls", pv),
        components=_count(store, "components", pv),
        notes=notes,
    )
