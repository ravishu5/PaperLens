"""PaperLens MCP server.

Design rules that this file exists to enforce (ARCHITECTURE 1.1, 1.2):
  * No tool returns prose. Structured JSON only.
  * No tool returns a large artifact. It returns facts plus paperlens:// URIs,
    which the agent expands only where it needs to.
  * Anything not established is reported as UNKNOWN, never inferred.
"""
from __future__ import annotations

import dataclasses
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.caching import CacheHint

from . import resources as res
from .graph.store import Store
from .paper.ingest import ingest_paper as _ingest
from .sources import arxiv

INSTRUCTIONS = """\
PaperLens is a research-engineering layer over papers and their implementations.

Workflow: resolve_paper -> ingest_paper -> get_paper_skeleton, then read only the
paperlens:// URIs you actually need. Do not ask for whole papers.

Every result carries evidence and a confidence of CONFIRMED, LIKELY, POSSIBLE or
UNKNOWN. UNKNOWN is a normal answer and means exactly what it says; do not fill
the gap with an assumption.
"""

mcp = MCPServer(
    name="paperlens",
    version="0.1.0",
    instructions=INSTRUCTIONS,
    # Paper content is immutable per version, so it is safe to cache hard.
    cache_hints={
        "resources/read": CacheHint(ttl_ms=86_400_000, scope="public"),
        "resources/list": CacheHint(ttl_ms=300_000, scope="public"),
        "tools/list": CacheHint(ttl_ms=3_600_000, scope="public"),
    },
)

_store: Store | None = None


def store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


def _err(exc: Exception) -> dict[str, Any]:
    return {"error": type(exc).__name__, "message": str(exc)}


# ── tools ─────────────────────────────────────────────────────────────────
@mcp.tool(
    title="Resolve a paper",
    description="Turn an arXiv id, DOI, URL, or free-text title into candidate "
                "papers. Use before ingest_paper when you do not have an id.",
)
def resolve_paper(query: str) -> dict[str, Any]:
    parsed = arxiv.parse_arxiv_id(query)
    if parsed:
        arxiv_id, version = parsed
        try:
            meta = arxiv.fetch_metadata(store(), arxiv_id)
        except Exception as exc:
            return _err(exc)
        return {"candidates": [{
            "arxiv_id": meta.arxiv_id,
            "paper_version": f"{meta.arxiv_id}v{version or meta.version}",
            "title": meta.title,
            "authors": [a["name"] for a in meta.authors[:6]],
            "published_at": meta.published_at,
            "confidence": "CONFIRMED",
            "why": "input parsed as an explicit arXiv identifier",
            "uri": f"paperlens://paper/{meta.arxiv_id}",
        }]}
    try:
        hits = arxiv.search(store(), query)
    except Exception as exc:
        return _err(exc)
    if not hits:
        return {"candidates": [], "confidence": "UNKNOWN",
                "message": f"arXiv full-text search returned nothing for {query!r}."}
    return {"candidates": [{
        "arxiv_id": m.arxiv_id,
        "paper_version": f"{m.arxiv_id}v{m.version}",
        "title": m.title,
        "authors": [a["name"] for a in m.authors[:6]],
        "published_at": m.published_at,
        # Text search is a guess: the agent should confirm the title matches.
        "confidence": "POSSIBLE",
        "why": "arXiv full-text search match; verify the title before proceeding",
        "uri": f"paperlens://paper/{m.arxiv_id}",
    } for m in hits]}


@mcp.tool(
    title="Ingest a paper",
    description="Fetch a paper's arXiv LaTeX source, parse its structure, and store "
                "it in the research graph. Returns counts and URIs, never the text.",
)
def ingest_paper(paper_id: str, force: bool = False) -> dict[str, Any]:
    try:
        r = _ingest(store(), paper_id, force=force)
    except Exception as exc:
        return _err(exc)
    d = dataclasses.asdict(r)
    d["uri"] = f"paperlens://paper/{r.arxiv_id}"
    d["skeleton_hint"] = "call get_paper_skeleton for addressable anchors"
    return d


@mcp.tool(
    title="Get paper skeleton",
    description="The paper's addressable anchors: sections, equations, components, "
                "algorithms, stated values and declared URLs. Labels and URIs only, "
                "no bodies. This is the drill-down index.",
)
def get_paper_skeleton(paper_id: str) -> dict[str, Any]:
    try:
        pv = res._pv(store(), paper_id)
    except Exception as exc:
        return _err(exc)
    s, aid = store(), pv.split("v")[0]
    base = f"paperlens://paper/{aid}"

    sections = [
        {"path": r["section_path"], "level": r["level"], "title": r["title"],
         "uri": f"{base}/section/{r['section_path']}"}
        for r in s.all("SELECT section_path, level, title FROM sections "
                       "WHERE paper_version = ? ORDER BY ordinal", (pv,))
    ]
    equations = [
        {"hash": r["content_hash"], "number": r["derived_number"],
         "number_confidence": r["number_conf"], "environment": r["environment"],
         "preview": (r["latex"] or "")[:90],
         "uri": f"{base}/equation/{r['content_hash']}"}
        for r in s.all("SELECT * FROM equations WHERE paper_version = ? ORDER BY ordinal", (pv,))
    ]
    components = [
        {"slug": r["slug"], "name": r["name"], "kind": r["kind"], "source": r["source"],
         "uri": f"{base}/component/{r['slug']}"}
        for r in s.all("SELECT * FROM components WHERE paper_version = ?", (pv,))
    ]
    algorithms = [
        {"slug": r["slug"], "name": r["name"], "presentation": r["presentation"],
         "extractable": bool(r["extractable"]), "uri": f"{base}/algorithm/{r['slug']}"}
        for r in s.all("SELECT * FROM algorithms WHERE paper_version = ?", (pv,))
    ]
    values = [
        {"symbol": r["symbol"], "value": r["value_text"], "src_line": r["src_line"]}
        for r in s.all("SELECT * FROM stated_values WHERE paper_version = ? ORDER BY src_line", (pv,))
    ]
    urls = [
        {"url": r["url"], "owner": r["owner"], "repo": r["repo"],
         "in_abstract": bool(r["in_abstract"])}
        for r in s.all("SELECT * FROM declared_urls WHERE paper_version = ?", (pv,))
    ]

    row = s.paper_version_row(pv)
    limits: list[str] = []
    if not equations:
        limits.append("No display equations: use components and stated values as anchors.")
    if any(not a["extractable"] for a in algorithms):
        limits.append("Some algorithms are figure images; their content is unavailable.")
    if row["fidelity"] != "LATEX_EXACT":
        limits.append(f"Reduced fidelity: {row['fidelity']}.")

    return {
        "paper_version": pv, "uri": base, "fidelity": row["fidelity"],
        "sections": sections, "equations": equations, "components": components,
        "algorithms": algorithms, "stated_values": values, "declared_urls": urls,
        "limitations": limits,
    }


@mcp.tool(
    title="Fetch a PaperLens resource",
    description="Read any paperlens:// URI. Identical to the MCP resource of the "
                "same URI; provided for clients without resource support.",
)
def paperlens_fetch(uri: str) -> dict[str, Any]:
    try:
        return res.resolve(store(), uri)
    except Exception as exc:
        return _err(exc)


# ── resources ─────────────────────────────────────────────────────────────
@mcp.resource("paperlens://paper/{arxiv_id}", mime_type="application/json")
def r_paper(arxiv_id: str) -> dict[str, Any]:
    return res.paper_overview(store(), arxiv_id)


@mcp.resource("paperlens://paper/{arxiv_id}/section/{section_path}",
              mime_type="application/json")
def r_section(arxiv_id: str, section_path: str) -> dict[str, Any]:
    return res.section(store(), arxiv_id, section_path)


@mcp.resource("paperlens://paper/{arxiv_id}/equation/{ref}", mime_type="application/json")
def r_equation(arxiv_id: str, ref: str) -> dict[str, Any]:
    return res.equation(store(), arxiv_id, ref)


@mcp.resource("paperlens://paper/{arxiv_id}/component/{slug}", mime_type="application/json")
def r_component(arxiv_id: str, slug: str) -> dict[str, Any]:
    return res.component(store(), arxiv_id, slug)


@mcp.resource("paperlens://paper/{arxiv_id}/algorithm/{slug}", mime_type="application/json")
def r_algorithm(arxiv_id: str, slug: str) -> dict[str, Any]:
    return res.algorithm(store(), arxiv_id, slug)


@mcp.resource("paperlens://paper/{arxiv_id}/values", mime_type="application/json")
def r_values(arxiv_id: str) -> dict[str, Any]:
    return res.stated_values(store(), arxiv_id)


@mcp.resource("paperlens://paper/{arxiv_id}/urls", mime_type="application/json")
def r_urls(arxiv_id: str) -> dict[str, Any]:
    return res.declared_urls(store(), arxiv_id)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
