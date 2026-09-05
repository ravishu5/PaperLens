"""paperlens:// URI resolution.

One implementation serves both MCP resources and the paperlens_fetch tool, so
clients without resource support are not second-class (ARCHITECTURE 1.2, D-010).
"""
from __future__ import annotations

import json
import re
from typing import Any

from .graph.store import Store
from .sources.arxiv import parse_arxiv_id


class ResourceNotFound(LookupError):
    pass


def _pv(store: Store, paper_id: str) -> str:
    """Resolve a possibly-unversioned paper id to a stored paper_version."""
    parsed = parse_arxiv_id(paper_id)
    if not parsed:
        raise ResourceNotFound(f"{paper_id!r} is not an arXiv identifier")
    arxiv_id, version = parsed
    pv = f"{arxiv_id}v{version}" if version else store.latest_version_of(arxiv_id)
    if not pv or not store.paper_version_row(pv):
        raise ResourceNotFound(
            f"{arxiv_id} has not been ingested. Call ingest_paper first."
        )
    return pv


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
        "uri": f"paperlens://paper/{row['arxiv_id']}",
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
        "uri": f"paperlens://paper/{pv.split('v')[0]}/section/{section_path}",
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
        "uri": f"paperlens://paper/{pv.split('v')[0]}/equation/{row['content_hash']}",
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
        "uri": f"paperlens://paper/{pv.split('v')[0]}/component/{slug}",
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
        "uri": f"paperlens://paper/{pv.split('v')[0]}/algorithm/{slug}",
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
        "uri": f"paperlens://paper/{pv.split('v')[0]}/values",
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
        "uri": f"paperlens://paper/{pv.split('v')[0]}/urls",
        "paper_version": pv,
        "urls": [
            {"url": r["url"], "host": r["host"], "owner": r["owner"], "repo": r["repo"],
             "in_abstract": bool(r["in_abstract"]), "section": r["section_id"],
             "src_line": r["src_line"], "context": r["context"]}
            for r in rows
        ],
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
]


def resolve(store: Store, uri: str) -> dict[str, Any]:
    for pat, fn in _ROUTES:
        m = pat.match(uri.strip())
        if m:
            return fn(store, list(m.groups()))
    raise ResourceNotFound(f"Unrecognised PaperLens URI: {uri!r}")
