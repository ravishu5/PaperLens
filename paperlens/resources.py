"""paperlens:// URI resolution.

One implementation serves both MCP resources and the paperlens_fetch tool, so
clients without resource support are not second-class (ARCHITECTURE 1.2, D-010).
"""
from __future__ import annotations

import json
import re
from urllib.parse import quote, unquote
from typing import Any

from .graph.store import base_id, Store
from .sources.arxiv import parse_arxiv_id


class ResourceNotFound(LookupError):
    pass


def _pv(store: Store, paper_id: str) -> str:
    """Resolve a paper identifier to a stored paper_version.

    Resolution is exact at every step. An earlier substring fallback
    (``arxiv_id LIKE '%id%'``) meant that ``_pv(".")`` or ``_pv("1")`` silently
    resolved to whichever paper SQLite returned first, and that paper then flowed
    into mappings and reports as though it had been asked for. An ambiguous
    identifier must raise, never pick.
    """
    # A DOI in a URI path segment arrives percent-encoded, because it contains
    # a slash.
    paper_id = unquote((paper_id or "").strip())
    if not paper_id:
        raise ResourceNotFound("empty paper identifier")

    # 1. An explicit arXiv identifier resolves by parsing, not by searching.
    parsed = parse_arxiv_id(paper_id)
    if parsed:
        arxiv_id, version = parsed
        pv = f"{arxiv_id}v{version}" if version else store.latest_version_of(arxiv_id)
        if pv and store.paper_version_row(pv):
            return pv
        raise ResourceNotFound(
            f"{arxiv_id} has not been ingested. Call ingest_paper first."
        )

    # 2. A stored paper_version id, verbatim (e.g. "2103.00020v1").
    if store.paper_version_row(paper_id):
        return paper_id

    # 3. Non-arXiv papers (DOI- or PDF-sourced) resolve by exact id or DOI.
    rows = store.all(
        "SELECT arxiv_id, doi, latest_version FROM papers "
        "WHERE (arxiv_id = ? OR doi = ?) AND latest_version IS NOT NULL",
        (paper_id, paper_id),
    )
    if len(rows) == 1:
        return rows[0]["latest_version"]
    if len(rows) > 1:
        ids = ", ".join(r["arxiv_id"] for r in rows[:5])
        raise ResourceNotFound(
            f"{paper_id!r} matches {len(rows)} ingested papers ({ids}). Use a full "
            f"identifier; PaperLens will not guess which one you meant."
        )
    raise ResourceNotFound(
        f"{paper_id!r} is not an arXiv identifier, and no ingested paper has that "
        f"id or DOI. Call resolve_paper to find it, then ingest_paper."
    )


def paper_overview(store: Store, paper_id: str) -> dict[str, Any]:
    pv = _pv(store, paper_id)
    row = store.paper_version_row(pv)
    paper = store.one("SELECT * FROM papers WHERE arxiv_id = ?", (row["arxiv_id"],))
    counts = {
        t: store.one(f"SELECT COUNT(*) c FROM {t} WHERE paper_version = ?", (pv,))["c"]
        for t in ("sections", "equations", "algorithms", "stated_values",
                  "declared_urls", "components")
    }
    return {
        "uri": paper_uri(row["arxiv_id"]),
        "paper_version": pv,
        "title": paper["title"],
        "authors": json.loads(paper["authors_json"] or "[]"),
        "published_at": paper["published_at"],
        "abstract": paper["abstract"],
        "fidelity": row["fidelity"],
        "counts": counts,
    }


def section(store: Store, paper_id: str, section_path: str) -> dict[str, Any]:
    pv = _pv(store, paper_id)
    row = store.one(
        "SELECT * FROM sections WHERE paper_version = ? AND section_path = ?",
        (pv, section_path),
    )
    if row is None:
        available = [r["section_path"] for r in store.all(
            "SELECT section_path FROM sections WHERE paper_version = ? ORDER BY ordinal",
            (pv,))]
        raise ResourceNotFound(
            f"No section {section_path!r} in {pv}. Available: {', '.join(available[:40])}"
        )
    return {
        "uri": paper_uri(_aid(store, pv), "section", str(section_path)),
        "paper_version": pv,
        "section_path": row["section_path"],
        "level": row["level"],
        "title": row["title"],
        "latex_label": row["latex_label"],
        "src_lines": [row["src_line_start"], row["src_line_end"]],
        "body": row["body"],
    }


_EQ_NUMBER = re.compile(r"^\d+$")


def equation(store: Store, paper_id: str, ref: str) -> dict[str, Any]:
    """Resolve an equation by content hash, or by number as a fallible alias.

    Equation numbers do not exist in LaTeX source -- they are produced by a
    counter at compile time and almost never carry a \\label. We replay the
    counter, but the result is an inference, so resolving by number returns the
    confidence of that inference alongside the equation (D-004).
    """
    pv = _pv(store, paper_id)
    if _EQ_NUMBER.match(ref):
        rows = store.all(
            "SELECT * FROM equations WHERE paper_version = ? AND derived_number = ?",
            (pv, ref),
        )
        if not rows:
            total = store.one(
                "SELECT COUNT(*) c FROM equations WHERE paper_version = ? AND is_numbered = 1",
                (pv,))["c"]
            raise ResourceNotFound(
                f"No equation numbered {ref} in {pv}: the paper has {total} numbered "
                f"equation(s). Numbers are derived by replaying the LaTeX counter and "
                f"may be UNKNOWN; address equations by content hash where possible."
            )
        row = rows[0]
        ambiguous = len(rows) > 1
    else:
        row = store.one(
            "SELECT * FROM equations WHERE paper_version = ? AND content_hash = ?",
            (pv, ref),
        )
        if row is None:
            raise ResourceNotFound(f"No equation {ref!r} in {pv}")
        ambiguous = False

    return {
        "uri": paper_uri(_aid(store, pv), "equation", str(row['content_hash'])),
        "paper_version": pv,
        "content_hash": row["content_hash"],
        "latex": row["latex"],
        "environment": row["environment"],
        "is_numbered": bool(row["is_numbered"]),
        "derived_number": row["derived_number"],
        "number_confidence": row["number_conf"],
        "latex_label": row["latex_label"],
        "section": row["section_id"],
        "src_line": row["src_line"],
        **({"warning": "Multiple equations share this derived number."} if ambiguous else {}),
    }


def component(store: Store, paper_id: str, slug: str) -> dict[str, Any]:
    pv = _pv(store, paper_id)
    row = store.one(
        "SELECT * FROM components WHERE paper_version = ? AND slug = ?", (pv, slug))
    if row is None:
        available = [r["slug"] for r in store.all(
            "SELECT slug FROM components WHERE paper_version = ?", (pv,))]
        raise ResourceNotFound(
            f"No component {slug!r} in {pv}. Available: {', '.join(available[:30])}")
    return {
        "uri": paper_uri(_aid(store, pv), "component", str(slug)),
        "paper_version": pv, "slug": row["slug"], "name": row["name"],
        "kind": row["kind"], "description": row["description"],
        "section": row["section_id"], "source": row["source"],
    }


def algorithm(store: Store, paper_id: str, slug: str) -> dict[str, Any]:
    pv = _pv(store, paper_id)
    row = store.one(
        "SELECT * FROM algorithms WHERE paper_version = ? AND slug = ?", (pv, slug))
    if row is None:
        available = [r["slug"] for r in store.all(
            "SELECT slug FROM algorithms WHERE paper_version = ?", (pv,))]
        raise ResourceNotFound(
            f"No algorithm {slug!r} in {pv}. Available: {', '.join(available[:30])}")
    out = {
        "uri": paper_uri(_aid(store, pv), "algorithm", str(slug)),
        "paper_version": pv, "slug": row["slug"], "name": row["name"],
        "presentation": row["presentation"], "extractable": bool(row["extractable"]),
        "body": row["body"], "section": row["section_id"], "src_line": row["src_line"],
    }
    if not row["extractable"]:
        out["note"] = (
            "This algorithm is a figure image, not text. Its content is not present "
            "in the LaTeX source and has not been recovered."
        )
    return out


def stated_values(store: Store, paper_id: str) -> dict[str, Any]:
    pv = _pv(store, paper_id)
    rows = store.all(
        "SELECT * FROM stated_values WHERE paper_version = ? ORDER BY src_line", (pv,))
    return {
        "uri": paper_uri(_aid(store, pv), "values"),
        "paper_version": pv,
        "values": [
            {"symbol": r["symbol"], "value": r["value_text"], "numeric": r["value_num"],
             "section": r["section_id"], "src_line": r["src_line"],
             "context": r["context"]}
            for r in rows
        ],
    }


def declared_urls(store: Store, paper_id: str) -> dict[str, Any]:
    pv = _pv(store, paper_id)
    rows = store.all(
        "SELECT * FROM declared_urls WHERE paper_version = ? ORDER BY src_line", (pv,))
    return {
        "uri": paper_uri(_aid(store, pv), "urls"),
        "paper_version": pv,
        "urls": [
            {"url": r["url"], "host": r["host"], "owner": r["owner"], "repo": r["repo"],
             "in_abstract": bool(r["in_abstract"]), "section": r["section_id"],
             "src_line": r["src_line"], "context": r["context"]}
            for r in rows
        ],
    }


def implementations(store: Store, paper_id: str) -> dict[str, Any]:
    """Stored implementation candidates. Read-only: run find_implementations to
    populate or refresh it."""
    from .evidence.confidence import evidence_for

    pv = _pv(store, paper_id)
    rows = store.all(
        "SELECT c.*, r.stars, r.license, r.archived FROM implementation_candidates c "
        "JOIN repos r ON r.id = c.repo_id WHERE c.paper_version = ? ORDER BY c.rank",
        (pv,),
    )
    if not rows:
        return {
            "uri": paper_uri(_aid(store, pv), "implementations"),
            "paper_version": pv, "candidates": [],
            "note": "No implementation search has been run for this paper yet. "
                    "Call find_implementations.",
        }
    return {
        "uri": paper_uri(_aid(store, pv), "implementations"),
        "paper_version": pv,
        "candidates": [{
            "repo": r["repo_id"], "url": f"https://github.com/{r['repo_id']}",
            "relation": r["relation"], "confidence": r["confidence"],
            "coverage_score": r["coverage_score"],
            "coverage_checked": bool(r["coverage_checked"]),
            "rank": r["rank"], "stars": r["stars"], "license": r["license"],
            "archived": bool(r["archived"]),
            "evidence": evidence_for(store, "implementation_candidate",
                                     f"{pv}|{r['repo_id']}"),
        } for r in rows],
    }


def mapping(store: Store, paper_id: str, repo: str) -> dict[str, Any]:
    """Stored paper-to-code mappings. Read-only; run map_paper_to_code to refresh."""
    from .evidence.confidence import evidence_for

    pv = _pv(store, paper_id)
    rows = store.all(
        "SELECT * FROM mappings WHERE paper_version = ? AND snapshot_id LIKE ? "
        "ORDER BY status, anchor_kind", (pv, f"{repo}@%"))
    if not rows:
        return {"uri": f"paperlens://mapping/{encode_paper(_aid(store, pv))}/{repo}",
                "paper_version": pv, "repo": repo, "mappings": [],
                "note": "No mapping has been computed for this pair. "
                        "Call map_paper_to_code."}
    return {
        "uri": f"paperlens://mapping/{encode_paper(_aid(store, pv))}/{repo}",
        "paper_version": pv, "repo": repo,
        "summary": {s: sum(1 for r in rows if r["status"] == s)
                    for s in ("MATCHED", "ABSENT", "AMBIGUOUS", "UNKNOWN")},
        "mappings": [{
            "id": r["id"], "anchor_kind": r["anchor_kind"], "anchor_id": r["anchor_id"],
            "status": r["status"], "confidence": r["confidence"],
            "method": r["method"], "reasoning": r["reasoning"],
            "evidence": evidence_for(store, "mapping", r["id"]),
        } for r in rows],
    }


def differences(store: Store, paper_id: str, repo: str) -> dict[str, Any]:
    """Stored paper-vs-code differences. Run compare_paper_with_code to refresh."""
    pv = _pv(store, paper_id)
    rows = store.all(
        "SELECT * FROM differences WHERE paper_version = ? AND snapshot_id LIKE ? "
        "ORDER BY CASE severity WHEN 'BLOCKING' THEN 0 WHEN 'SIGNIFICANT' THEN 1 "
        "WHEN 'MINOR' THEN 2 ELSE 3 END", (pv, f"{repo}@%"))
    uri = f"paperlens://difference/{encode_paper(_aid(store, pv))}/{repo}"
    if not rows:
        return {"uri": uri, "paper_version": pv, "repo": repo, "differences": [],
                "note": "No comparison has been run for this pair. "
                        "Call compare_paper_with_code."}
    return {
        "uri": uri, "paper_version": pv, "repo": repo,
        "summary": {s: sum(1 for r in rows if r["severity"] == s)
                    for s in ("BLOCKING", "SIGNIFICANT", "MINOR")},
        "differences": [{
            "kind": r["kind"], "paper_states": r["paper_states"],
            "code_does": r["code_does"], "difference": r["difference"],
            "severity": r["severity"], "confidence": r["confidence"],
        } for r in rows],
    }


def references(store: Store, paper_id: str) -> dict[str, Any]:
    """The paper's own bibliography, with where each entry is cited."""
    pv = _pv(store, paper_id)
    rows = store.all(
        "SELECT b.*, (SELECT COUNT(*) FROM citation_sites c WHERE "
        "c.paper_version = b.paper_version AND c.bib_key = b.bib_key) AS sites "
        "FROM bib_entries b WHERE b.paper_version = ? ORDER BY sites DESC", (pv,))
    aid = _aid(store, pv)
    if not rows:
        return {"uri": paper_uri(aid, "references"), "paper_version": pv,
                "references": [],
                "note": "No bibliography was recovered from this paper's source."}
    return {
        "uri": paper_uri(aid, "references"), "paper_version": pv,
        "count": len(rows),
        "resolved_to_arxiv": sum(1 for r in rows if r["arxiv_id"]),
        "references": [{
            "bib_key": r["bib_key"], "title": r["title"], "authors": r["authors"],
            "year": r["year"], "arxiv_id": r["arxiv_id"], "doi": r["doi"],
            "cited_times": r["sites"],
            "uri": f"paperlens://paper/{r['arxiv_id']}" if r["arxiv_id"] else None,
        } for r in rows],
    }


def lineage(store: Store, paper_id: str) -> dict[str, Any]:
    """Stored lineage edges. Run trace_research_lineage to populate or refresh."""
    import json as _json

    pv = _pv(store, paper_id)
    aid = _aid(store, pv)
    rows = store.all(
        "SELECT * FROM lineage_edges WHERE from_paper = ? OR to_paper = ?", (aid, aid))
    if not rows:
        return {"uri": f"paperlens://lineage/{encode_paper(aid)}", "paper_version": pv, "edges": [],
                "note": "No lineage has been traced. Call trace_research_lineage."}
    return {
        "uri": f"paperlens://lineage/{encode_paper(aid)}", "paper_version": pv,
        "edges": [{
            "from": r["from_paper"], "to": r["to_paper"], "relation": r["relation"],
            "confidence": r["confidence"], "source": r["source"],
            "is_influential": (None if r["is_influential"] is None
                               else bool(r["is_influential"])),
            "intents": _json.loads(r["intents_json"]) if r["intents_json"] else [],
            "contexts": _json.loads(r["contexts_json"]) if r["contexts_json"] else [],
        } for r in rows],
    }


def plan(store: Store, paper_id: str, repo: str) -> dict[str, Any]:
    """A reproduction plan, rebuilt from the graph on each read."""
    from .correlate.plan import build_reproduction_plan

    return build_reproduction_plan(store, paper_id, repo)


def analysis(store: Store, paper_id: str) -> dict[str, Any]:
    from .correlate.analysis import get_analysis

    got = get_analysis(store, paper_id)
    if got is None:
        pv = _pv(store, paper_id)
        return {"uri": paper_uri(_aid(store, pv), "analysis"),
                "paper_version": pv, "analysis": None,
                "note": "No analysis has been recorded. Read the skeleton, then "
                        "call record_paper_analysis."}
    return {"uri": f"paperlens://paper/{paper_id}/analysis", **got}


# ── code side ─────────────────────────────────────────────────────────────
def encode_paper(paper_id: str) -> str:
    """Percent-encode a paper identifier for a URI path segment.

    arXiv ids need no encoding; a DOI contains a slash and would otherwise split
    the path."""
    return quote(paper_id, safe="")


def paper_uri(paper_id: str, *suffix: str) -> str:
    parts = "/".join(suffix)
    return f"paperlens://paper/{encode_paper(paper_id)}" + (f"/{parts}" if parts else "")


def _aid(store: Store, pv: str) -> str:
    """The paper id for a version, looked up rather than derived."""
    return store.paper_id_of(pv)


def encode_symbol(qualified_name: str) -> str:
    """Percent-encode a symbol id for use in a URI path segment.

    Symbol ids look like `clip/model.py::CLIP.forward#method`. The `#` would be
    read as a fragment delimiter and the `/` as a path separator, so both are
    escaped. `decode_symbol` accepts either form.
    """
    return quote(qualified_name, safe="")


def decode_symbol(segment: str) -> str:
    return unquote(segment)


def symbol_uri(repo_id: str, qualified_name: str) -> str:
    return f"paperlens://repo/{repo_id}/symbol/{encode_symbol(qualified_name)}"


def repo_overview(store: Store, repo: str) -> dict[str, Any]:
    from .code.indexer import require_snapshot

    snap = require_snapshot(store, repo)
    return {
        "uri": f"paperlens://repo/{snap['repo_id']}",
        "repo": snap["repo_id"], "commit_sha": snap["commit_sha"],
        "indexer": snap["indexer"], "indexed_at": snap["indexed_at"],
        "symbol_count": snap["symbol_count"], "file_count": snap["file_count"],
    }


def repo_file(store: Store, repo: str, path: str) -> dict[str, Any]:
    """A file's symbol outline -- never its full text. The agent drills into the
    symbols it needs rather than pulling the whole file into context."""
    from .code.indexer import provider, require_snapshot

    snap = require_snapshot(store, repo)
    syms = provider().file_outline(snap["repo_id"], path)
    if not syms:
        raise ResourceNotFound(
            f"No symbols found for {path!r} in {snap['repo_id']}. The file may not "
            f"exist, or may be in a language the active backend "
            f"({snap['indexer']}) does not parse."
        )
    return {
        "uri": f"paperlens://repo/{snap['repo_id']}/file/{path}",
        "repo": snap["repo_id"], "commit_sha": snap["commit_sha"], "file_path": path,
        "symbols": [{
            "qualified_name": s.qualified_name, "name": s.name, "kind": s.kind,
            "lines": [s.line_start, s.line_end], "signature": s.signature,
            "uri": symbol_uri(snap["repo_id"], s.qualified_name),
        } for s in syms],
    }


def repo_symbol(store: Store, repo: str, qualified_name: str) -> dict[str, Any]:
    from .code.indexer import find_symbol, persist_symbol, provider, require_snapshot

    qualified_name = decode_symbol(qualified_name)
    snap = require_snapshot(store, repo)
    sym = find_symbol(store, repo, qualified_name)
    if sym is None:
        raise ResourceNotFound(
            f"No symbol {qualified_name!r} in {snap['repo_id']} at "
            f"{snap['commit_sha'][:8]}."
        )
    persist_symbol(store, snap["id"], sym)
    store.commit()
    return {
        "uri": symbol_uri(snap["repo_id"], qualified_name),
        "repo": snap["repo_id"], "commit_sha": snap["commit_sha"],
        "qualified_name": sym.qualified_name, "name": sym.name, "kind": sym.kind,
        "file_path": sym.file_path, "lines": [sym.line_start, sym.line_end],
        "signature": sym.signature,
        "source": provider().symbol_source(snap["repo_id"], qualified_name),
    }


# ── dispatch ──────────────────────────────────────────────────────────────
_ROUTES: list[tuple[re.Pattern, Any]] = [
    (re.compile(r"^paperlens://paper/([^/]+)$"), lambda s, m: paper_overview(s, m[0])),
    (re.compile(r"^paperlens://paper/([^/]+)/section/(.+)$"),
     lambda s, m: section(s, m[0], m[1])),
    (re.compile(r"^paperlens://paper/([^/]+)/equation/([^/]+)$"),
     lambda s, m: equation(s, m[0], m[1])),
    (re.compile(r"^paperlens://paper/([^/]+)/component/([^/]+)$"),
     lambda s, m: component(s, m[0], m[1])),
    (re.compile(r"^paperlens://paper/([^/]+)/algorithm/([^/]+)$"),
     lambda s, m: algorithm(s, m[0], m[1])),
    (re.compile(r"^paperlens://paper/([^/]+)/values$"), lambda s, m: stated_values(s, m[0])),
    (re.compile(r"^paperlens://paper/([^/]+)/urls$"), lambda s, m: declared_urls(s, m[0])),
    (re.compile(r"^paperlens://paper/([^/]+)/implementations$"),
     lambda s, m: implementations(s, m[0])),
    (re.compile(r"^paperlens://paper/([^/]+)/analysis$"), lambda s, m: analysis(s, m[0])),
    (re.compile(r"^paperlens://paper/([^/]+)/references$"),
     lambda s, m: references(s, m[0])),
    (re.compile(r"^paperlens://lineage/([^/]+)$"), lambda s, m: lineage(s, m[0])),
    (re.compile(r"^paperlens://lineage/(.+)$"), lambda s, m: lineage(s, m[0])),
    (re.compile(r"^paperlens://plan/([^/]+)/([^/]+/[^/]+)$"),
     lambda s, m: plan(s, m[0], m[1])),
    (re.compile(r"^paperlens://mapping/([^/]+)/([^/]+/[^/]+)$"),
     lambda s, m: mapping(s, m[0], m[1])),
    (re.compile(r"^paperlens://difference/([^/]+)/([^/]+/[^/]+)$"),
     lambda s, m: differences(s, m[0], m[1])),
    (re.compile(r"^paperlens://repo/([^/]+/[^/]+)$"), lambda s, m: repo_overview(s, m[0])),
    (re.compile(r"^paperlens://repo/([^/]+/[^/]+)/file/(.+)$"),
     lambda s, m: repo_file(s, m[0], m[1])),
    (re.compile(r"^paperlens://repo/([^/]+/[^/]+)/symbol/(.+)$"),
     lambda s, m: repo_symbol(s, m[0], m[1])),
]


def resolve(store: Store, uri: str) -> dict[str, Any]:
    for pat, fn in _ROUTES:
        m = pat.match(uri.strip())
        if m:
            return fn(store, list(m.groups()))
    raise ResourceNotFound(f"Unrecognised PaperLens URI: {uri!r}")
