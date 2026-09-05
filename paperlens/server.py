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

from . import config
from . import resources as res
from .graph.store import Store
from .code.indexer import index_repository as _index_repo
from .code.indexer import provider as _code_provider
from .code.indexer import require_snapshot as _require_snapshot
from .correlate.analysis import record_paper_analysis as _record_analysis
from .correlate.compare import compare_implementations as _cmp_impls
from .correlate.compare import compare_paper_with_code as _cmp_code
from .correlate.compare import find_implementation_gaps as _find_gaps
from .correlate.discover import find_implementations as _discover
from .correlate.lineage import find_sota_successors as _successors
from .correlate.lineage import trace_method as _trace_method
from .correlate.lineage import trace_research_lineage as _lineage
from .correlate.plan import build_reproduction_plan as _plan
from .correlate.mapper import map_paper_to_code as _map
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

config.quiet_dependencies()

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
    title="Find implementations",
    description="Find and rank GitHub repositories implementing a paper, with "
                "evidence for each. Ranks by evidential strength and reports "
                "content coverage, so an official repository that does not "
                "actually implement the method is visibly incomplete.",
)
def find_implementations(paper_id: str, kind: str = "all",
                         verify_coverage: bool = True) -> dict[str, Any]:
    if kind not in ("all", "official", "reproduction"):
        return {"error": "ValueError",
                "message": "kind must be 'all', 'official' or 'reproduction'"}
    try:
        return _discover(store(), paper_id, kind=kind, verify_coverage=verify_coverage)
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    title="Index a repository",
    description="Clone a GitHub repository at a commit and build a code index. "
                "Returns counts and the SHA, never source. Indexing is keyed on "
                "the commit SHA, so repeat calls are free.",
)
def index_repository(repo: str, ref: str = "HEAD", force: bool = False) -> dict[str, Any]:
    import dataclasses as _dc
    try:
        return _dc.asdict(_index_repo(store(), repo, ref=ref, force=force))
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    title="Get repository outline",
    description="The code-side counterpart of get_paper_skeleton: files and their "
                "symbols as addressable URIs, with no source text.",
)
def get_repo_outline(repo: str, path_prefix: str = "", limit: int = 60) -> dict[str, Any]:
    try:
        snap = _require_snapshot(store(), repo)
    except Exception as exc:
        return _err(exc)
    prov = _code_provider()
    try:
        symbols = prov.list_symbols(snap["repo_id"])
    except Exception as exc:
        return _err(exc)

    by_file: dict[str, list] = {}
    for sym in symbols:
        if path_prefix and not sym.file_path.startswith(path_prefix):
            continue
        by_file.setdefault(sym.file_path, []).append(sym)

    base = f"paperlens://repo/{snap['repo_id']}"
    files = [{
        "path": path,
        "symbol_count": len(syms),
        "uri": f"{base}/file/{path}",
        "top_symbols": [{
            "qualified_name": s.qualified_name, "kind": s.kind,
            "uri": res.symbol_uri(snap["repo_id"], s.qualified_name),
        } for s in sorted(syms, key=lambda s: s.line_start)[:6]],
    } for path, syms in sorted(by_file.items())][:limit]

    return {
        "repo": snap["repo_id"], "commit_sha": snap["commit_sha"],
        "indexer": snap["indexer"], "uri": base,
        "files": files, "file_count": len(by_file), "symbol_count": len(symbols),
        "truncated": len(by_file) > limit,
    }


@mcp.tool(
    title="Search code",
    description="Find symbols in an indexed repository. When nothing matches, "
                "returns a citable absence record rather than an empty list, "
                "because 'it is not there' is a finding.",
)
def search_code(repo: str, query: str, limit: int = 20) -> dict[str, Any]:
    try:
        snap = _require_snapshot(store(), repo)
        res = _code_provider().search_symbols(snap["repo_id"], query, limit=limit)
    except Exception as exc:
        return _err(exc)
    base = f"paperlens://repo/{snap['repo_id']}"
    if not res.found:
        return {
            "repo": snap["repo_id"], "commit_sha": snap["commit_sha"],
            "query": query, "found": False, "confidence": "CONFIRMED",
            "finding": f"No symbol matching {query!r} exists in this repository.",
            "absence_ref": res.absence_ref, "method": res.method, "symbols": [],
        }
    return {
        "repo": snap["repo_id"], "commit_sha": snap["commit_sha"],
        "query": query, "found": True, "method": res.method,
        "symbols": [{
            "qualified_name": s.qualified_name, "name": s.name, "kind": s.kind,
            "file_path": s.file_path, "lines": [s.line_start, s.line_end],
            "signature": s.signature,
            "uri": res.symbol_uri(snap["repo_id"], s.qualified_name),
        } for s in res.symbols],
    }


@mcp.tool(
    title="Map paper to code",
    description="Join a paper's anchors to symbols in an indexed repository. "
                "Returns MATCHED, ABSENT, AMBIGUOUS or UNKNOWN per anchor with "
                "evidence. An exact constant match can reach CONFIRMED; a name "
                "resemblance cannot. Pass anchor_kind/anchor_id to drill into one.",
)
def map_paper_to_code(paper_id: str, repo: str, anchor_kind: str | None = None,
                      anchor_id: str | None = None) -> dict[str, Any]:
    if anchor_kind and anchor_kind not in ("stated_value", "component", "algorithm"):
        return {"error": "ValueError",
                "message": "anchor_kind must be stated_value, component or algorithm"}
    try:
        return _map(store(), paper_id, repo, anchor_kind=anchor_kind,
                    anchor_id=anchor_id)
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    title="Record a paper analysis",
    description="Write your structured reading of a paper back into the graph. "
                "Every component and claim MUST cite paperlens:// URIs that "
                "resolve; unsupported items are rejected, not stored. Recording "
                "real method components is what makes map_paper_to_code useful.",
)
def record_paper_analysis(paper_id: str, analysis: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(analysis, dict):
        return {"error": "ValueError", "message": "analysis must be an object"}
    try:
        return _record_analysis(store(), paper_id, analysis, author="agent")
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    title="Reverse engineer a paper",
    description="The assembled technical reconstruction: recorded analysis, "
                "structure counts, implementations and mappings. If no analysis "
                "has been recorded it says so and points at the evidence needed "
                "to produce one.",
)
def reverse_engineer_paper(paper_id: str) -> dict[str, Any]:
    from .correlate.analysis import get_analysis

    try:
        pv = res._pv(store(), paper_id)
    except Exception as exc:
        return _err(exc)
    s, aid = store(), pv.split("v")[0]
    base = f"paperlens://paper/{aid}"
    counts = {t: s.one(f"SELECT COUNT(*) c FROM {t} WHERE paper_version = ?", (pv,))["c"]
              for t in ("sections", "equations", "algorithms", "stated_values",
                        "components", "claims")}
    impls = [r["repo_id"] for r in s.all(
        "SELECT repo_id FROM implementation_candidates WHERE paper_version = ? "
        "ORDER BY rank", (pv,))]
    maps = s.all(
        "SELECT snapshot_id, status, COUNT(*) c FROM mappings WHERE paper_version = ? "
        "GROUP BY snapshot_id, status", (pv,))
    mapped: dict[str, dict[str, int]] = {}
    for r in maps:
        repo = r["snapshot_id"].split("@")[0]
        mapped.setdefault(repo, {})[r["status"]] = r["c"]

    analysis = get_analysis(s, paper_id)
    out = {
        "paper_version": pv, "uri": base, "structure": counts,
        "implementations": impls,
        "mappings": [{"repo": k, "summary": v,
                      "uri": f"paperlens://mapping/{aid}/{k}"} for k, v in mapped.items()],
        "skeleton_uri": base, "analysis_uri": f"{base}/analysis",
    }
    if analysis is None:
        out["status"] = "needs_analysis"
        out["next_step"] = (
            "No analysis has been recorded. Call get_paper_skeleton, read the "
            "sections you need, then call record_paper_analysis with components "
            "and claims that each cite a resolving paperlens:// URI."
        )
    else:
        out["status"] = "analysed"
        out["analysis"] = analysis
    return out


@mcp.tool(
    title="Compare paper with code",
    description="Where the implementation differs from the paper. Each difference "
                "pairs an exact quote from the paper with an exact search over "
                "source, ordered BLOCKING first. The inference joining them is a "
                "heuristic, so differences are LIKELY rather than CONFIRMED.",
)
def compare_paper_with_code(paper_id: str, repo: str) -> dict[str, Any]:
    try:
        return _cmp_code(store(), paper_id, repo)
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    title="Find implementation gaps",
    description="Details that would block a reproduction: constants hard-coded in "
                "source but absent from the paper, components the paper describes "
                "but the repository lacks, and pretrained artifacts downloaded "
                "rather than trained.",
)
def find_implementation_gaps(paper_id: str, repo: str) -> dict[str, Any]:
    try:
        return _find_gaps(store(), paper_id, repo)
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    title="Compare implementations",
    description="Compare repositories on what they actually implement. Ranks by "
                "BLOCKING differences and established absences, not by how many "
                "symbol names match.",
)
def compare_implementations(paper_id: str, repos: list[str]) -> dict[str, Any]:
    if not isinstance(repos, list) or len(repos) < 2:
        return {"error": "ValueError", "message": "pass at least two repositories"}
    try:
        return _cmp_impls(store(), paper_id, repos)
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    title="Trace research lineage",
    description="What the paper builds on and what built on it. Predecessors come "
                "from the paper's own bibliography and are quotable; successors "
                "come from a citation index and are not verified from source.",
)
def trace_research_lineage(paper_id: str, direction: str = "both",
                           limit: int = 20) -> dict[str, Any]:
    if direction not in ("back", "forward", "both"):
        return {"error": "ValueError",
                "message": "direction must be 'back', 'forward' or 'both'"}
    try:
        return _lineage(store(), paper_id, direction=direction, limit=limit)
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    title="Find successor work",
    description="Later papers that cite this one, ranked by citation influence and "
                "intent where available. What each successor changed is not "
                "asserted; the citing sentences are returned as evidence.",
)
def find_sota_successors(paper_id: str, limit: int = 15) -> dict[str, Any]:
    try:
        return _successors(store(), paper_id, limit=limit)
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    title="Trace a method",
    description="Follow one method through the literature: which cited work "
                "introduced it, where this paper discusses it, and which later "
                "papers carry it forward. Every step quotes a sentence.",
)
def trace_method(paper_id: str, method: str, limit: int = 12) -> dict[str, Any]:
    try:
        return _trace_method(store(), paper_id, method, limit=limit)
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    title="Build a reproduction plan",
    description="Assemble everything established about a paper and a repository "
                "into an actionable plan: environment, dependencies, dataset, "
                "architecture, training, hyperparameters, evaluation, risks, "
                "missing information and verification steps. Sections are KNOWN, "
                "PARTIAL or UNKNOWN, and UNKNOWN says what would resolve it.",
)
def build_reproduction_plan(paper_id: str, repo: str | None = None) -> dict[str, Any]:
    try:
        return _plan(store(), paper_id, repo)
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    title="Explain a confidence verdict",
    description="Why a mapping carries the confidence it does: the supporting and "
                "contradicting evidence, and what would raise it.",
)
def explain_confidence(mapping_id: str) -> dict[str, Any]:
    from .evidence.confidence import evidence_for

    row = store().one("SELECT * FROM mappings WHERE id = ?", (mapping_id,))
    if row is None:
        return {"error": "LookupError",
                "message": f"No mapping {mapping_id!r}. Ids come from map_paper_to_code."}
    ev = evidence_for(store(), "mapping", mapping_id)
    supporting = [e for e in ev if e["stance"] == "SUPPORTS"]
    raise_it = {
        "MATCHED": "Index a repository at a specific commit and confirm the symbol "
                   "still implements this anchor.",
        "AMBIGUOUS": "Narrow the anchor: map a single component or stated value, or "
                     "record a more specific component via record_paper_analysis.",
        "ABSENT": "Nothing would raise this; the absence is established. Check a "
                  "different repository, such as a community reproduction.",
        "UNKNOWN": "Record finer-grained components with record_paper_analysis, or "
                   "index a repository that actually implements the paper.",
    }[row["status"]]
    return {
        "mapping_id": mapping_id, "status": row["status"],
        "confidence": row["confidence"], "method": row["method"],
        "reasoning": row["reasoning"],
        "supporting": supporting,
        "contradicting": [e for e in ev if e["stance"] == "CONTRADICTS"],
        "what_would_raise_it": raise_it,
        "note": ("Confidence is assigned from evidence by rule, never asserted. "
                 "A verdict with no supporting evidence is forced to UNKNOWN."),
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


@mcp.resource("paperlens://paper/{arxiv_id}/implementations", mime_type="application/json")
def r_implementations(arxiv_id: str) -> dict[str, Any]:
    return res.implementations(store(), arxiv_id)


@mcp.resource("paperlens://paper/{arxiv_id}/references", mime_type="application/json")
def r_references(arxiv_id: str) -> dict[str, Any]:
    return res.references(store(), arxiv_id)


@mcp.resource("paperlens://plan/{arxiv_id}/{owner}/{repo}",
              mime_type="application/json")
def r_plan(arxiv_id: str, owner: str, repo: str) -> dict[str, Any]:
    return res.plan(store(), arxiv_id, f"{owner}/{repo}")


@mcp.resource("paperlens://lineage/{arxiv_id}", mime_type="application/json")
def r_lineage(arxiv_id: str) -> dict[str, Any]:
    return res.lineage(store(), arxiv_id)


@mcp.resource("paperlens://paper/{arxiv_id}/analysis", mime_type="application/json")
def r_analysis(arxiv_id: str) -> dict[str, Any]:
    return res.analysis(store(), arxiv_id)


@mcp.resource("paperlens://mapping/{arxiv_id}/{owner}/{repo}",
              mime_type="application/json")
def r_mapping(arxiv_id: str, owner: str, repo: str) -> dict[str, Any]:
    return res.mapping(store(), arxiv_id, f"{owner}/{repo}")


@mcp.resource("paperlens://difference/{arxiv_id}/{owner}/{repo}",
              mime_type="application/json")
def r_differences(arxiv_id: str, owner: str, repo: str) -> dict[str, Any]:
    return res.differences(store(), arxiv_id, f"{owner}/{repo}")


@mcp.resource("paperlens://repo/{owner}/{repo}", mime_type="application/json")
def r_repo(owner: str, repo: str) -> dict[str, Any]:
    return res.repo_overview(store(), f"{owner}/{repo}")


@mcp.resource("paperlens://repo/{owner}/{repo}/file/{+path}", mime_type="application/json")
def r_repo_file(owner: str, repo: str, path: str) -> dict[str, Any]:
    return res.repo_file(store(), f"{owner}/{repo}", path)


@mcp.resource("paperlens://repo/{owner}/{repo}/symbol/{+qualified_name}",
              mime_type="application/json")
def r_repo_symbol(owner: str, repo: str, qualified_name: str) -> dict[str, Any]:
    return res.repo_symbol(store(), f"{owner}/{repo}", qualified_name)


@mcp.resource("paperlens://paper/{arxiv_id}/values", mime_type="application/json")
def r_values(arxiv_id: str) -> dict[str, Any]:
    return res.stated_values(store(), arxiv_id)


@mcp.resource("paperlens://paper/{arxiv_id}/urls", mime_type="application/json")
def r_urls(arxiv_id: str) -> dict[str, Any]:
    return res.declared_urls(store(), arxiv_id)


# ── prompts ───────────────────────────────────────────────────────────────
# User-invoked workflows. These are the multi-step sequences a person starts
# deliberately, which is exactly what prompts are for -- as distinct from tools,
# which the model calls on its own initiative.

@mcp.prompt(
    name="reverse-engineer-paper",
    title="Reverse engineer a paper",
    description="Reconstruct a paper technically: structure, method components, "
                "implementations and how they differ.",
)
def reverse_engineer_paper_prompt(paper: str) -> str:
    return f"""\
Reverse engineer {paper} using PaperLens. Work in this order and stop to report
what you find rather than guessing past a gap.

1. resolve_paper, then ingest_paper.
2. get_paper_skeleton. Read only the sections you actually need, by URI.
3. record_paper_analysis with the method components you identify. Every component
   must cite a paperlens:// URI that resolves -- unsupported ones are rejected.
4. find_implementations, then index_repository for the top candidate.
5. map_paper_to_code, then compare_paper_with_code.

Report what is CONFIRMED separately from what is LIKELY, and state UNKNOWN where
the evidence does not reach. Do not describe a component as implemented because a
symbol has a similar name."""


@mcp.prompt(
    name="reproduce-paper",
    title="Plan a reproduction",
    description="Produce an actionable plan for reproducing a paper, with risks "
                "and verification steps.",
)
def reproduce_paper_prompt(paper: str, repo: str = "") -> str:
    target = f" against {repo}" if repo else ""
    return f"""\
Build a reproduction plan for {paper}{target}.

Prerequisites, in order: ingest_paper, record_paper_analysis, find_implementations,
index_repository, map_paper_to_code, compare_paper_with_code,
find_implementation_gaps. Then build_reproduction_plan.

When you write the plan up, lead with the BLOCKING risks and the missing
information -- those decide whether a reproduction is feasible at all. Keep every
UNKNOWN section marked UNKNOWN; a plan that reads complete while resting on
guesses is worse than one that admits its holes."""


@mcp.prompt(
    name="compare-implementations",
    title="Compare implementations",
    description="Compare several repositories claiming to implement the same paper.",
)
def compare_implementations_prompt(paper: str, repos: str) -> str:
    return f"""\
Compare these implementations of {paper}: {repos}.

Ingest the paper, record its method components, then index each repository and
call compare_implementations. Note that ranking is by BLOCKING differences and
established absences, not by how many symbol names match -- a repository with
tidier naming can implement less of the paper. Report what they agree on, where
they diverge, and which divergences are essential rather than stylistic."""


@mcp.prompt(
    name="trace-lineage",
    title="Trace research lineage",
    description="Trace what a paper builds on and what built on it.",
)
def trace_lineage_prompt(paper: str, method: str = "") -> str:
    focus = f"\n\nFocus on the method: {method}. Use trace_method for it." if method else ""
    return f"""\
Trace the research lineage of {paper}.

Ingest it, then call trace_research_lineage and find_sota_successors.

Predecessors come from the paper's own bibliography and are quotable -- cite the
sentence. Successors come from a citation index and were not verified against the
citing paper's source, so present them as the weaker evidence they are. Do not
claim a successor improved on the paper unless you have read what it says.{focus}"""


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
