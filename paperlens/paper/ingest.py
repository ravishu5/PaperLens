"""Paper ingestion: fetch -> flatten -> parse -> persist."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .. import config
from ..graph.store import Store
from ..sources import arxiv
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


def ingest_paper(store: Store, paper_id: str, force: bool = False) -> IngestResult:
    parsed = arxiv.parse_arxiv_id(paper_id)
    if not parsed:
        raise ValueError(
            f"{paper_id!r} is not an arXiv identifier. Use resolve_paper first."
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
    arxiv_id = pv.split("v")[0]
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
