"""record_paper_analysis: the agent write-back path.

PaperLens has no LLM of its own, and MCP sampling is deprecated (D-001), so the
structured reading of a paper comes from the agent already holding the evidence.
The server's job is to *validate and remember* it (D-009).

Validation is the whole point. Every component and claim the agent writes must
cite evidence URIs that actually resolve; unsupported items are rejected with a
reason rather than quietly downgraded. That turns "every conclusion carries
evidence" from a convention the agent is asked to honour into a property of the
store: an agent cannot write a hallucinated component, because the component has
to point at a section that exists.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from ..graph.store import Store, utcnow
from ..resources import ResourceNotFound, resolve
from ..resources import _pv
from ..paper.structure import slugify

ANALYSIS_SCHEMA_VERSION = "1"

# The narrative fields from the vision doc. Stored verbatim; the server does not
# interpret them, it only keeps them addressable and attributable.
NARRATIVE_FIELDS = (
    "problem", "motivation", "contributions", "assumptions", "methodology",
    "architecture", "datasets", "preprocessing", "training_procedure",
    "inference_procedure", "evaluation_methodology", "ablations",
    "reported_results", "limitations", "dependencies_on_previous_work",
)

_VALID_KINDS = {"architecture", "module", "loss", "optimizer", "dataset",
                "preprocessing", "training", "inference", "evaluation", "other"}
_VALID_CLAIM_KINDS = {"contribution", "result", "assumption", "limitation",
                      "requirement"}


def _resolver(store: Store):
    def _ok(uri: str) -> bool:
        try:
            resolve(store, uri)
            return True
        except (ResourceNotFound, Exception):
            return False
    return _ok


def record_paper_analysis(
    store: Store, paper_id: str, analysis: dict[str, Any],
    author: str = "agent",
) -> dict[str, Any]:
    pv = _pv(store, paper_id)
    arxiv_id = pv.split("v")[0]
    ok = _resolver(store)

    accepted_components: list[dict[str, Any]] = []
    accepted_claims: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    # ── components ────────────────────────────────────────────────────────
    for item in analysis.get("components", []) or []:
        name = (item.get("name") or "").strip()
        if not name:
            rejected.append({"kind": "component", "item": item,
                             "reason": "no name"})
            continue
        kind = (item.get("kind") or "other").strip()
        if kind not in _VALID_KINDS:
            rejected.append({"kind": "component", "name": name,
                             "reason": f"kind must be one of {sorted(_VALID_KINDS)}"})
            continue
        cites = [u for u in (item.get("evidence") or []) if isinstance(u, str)]
        good = [u for u in cites if ok(u)]
        if not good:
            rejected.append({
                "kind": "component", "name": name,
                "reason": "no cited evidence URI resolves; a component must point "
                          "at something that exists in the paper",
                "cited": cites,
            })
            continue

        slug = (item.get("slug") or slugify(name)).strip()
        section_id = next(
            (u.rsplit("/section/", 1)[-1] for u in good if "/section/" in u), None)
        store.execute(
            "INSERT OR REPLACE INTO components (id, paper_version, slug, name, kind, "
            "description, section_id, source) VALUES (?,?,?,?,?,?,?,?)",
            (f"{pv}:comp:{slug}", pv, slug, name, kind,
             (item.get("description") or "")[:2000],
             f"{pv}:sec:{section_id}" if section_id else None,
             "SERVER_LLM" if author.startswith("server_llm") else "AGENT"))
        store.execute(
            "INSERT INTO components_fts (name, description, component_id, paper_version) "
            "VALUES (?,?,?,?)",
            (name, (item.get("description") or "")[:2000], f"{pv}:comp:{slug}", pv))
        accepted_components.append({
            "slug": slug, "name": name, "kind": kind,
            "uri": f"paperlens://paper/{arxiv_id}/component/{slug}",
            "evidence": good,
        })

    # ── claims ────────────────────────────────────────────────────────────
    for item in analysis.get("claims", []) or []:
        statement = (item.get("statement") or "").strip()
        kind = (item.get("kind") or "contribution").strip()
        if not statement:
            rejected.append({"kind": "claim", "item": item, "reason": "no statement"})
            continue
        if kind not in _VALID_CLAIM_KINDS:
            rejected.append({"kind": "claim", "statement": statement[:80],
                             "reason": f"kind must be one of {sorted(_VALID_CLAIM_KINDS)}"})
            continue
        cites = [u for u in (item.get("evidence") or []) if isinstance(u, str)]
        good = [u for u in cites if ok(u)]
        if not good:
            rejected.append({
                "kind": "claim", "statement": statement[:80],
                "reason": "no cited evidence URI resolves", "cited": cites})
            continue
        cid = f"{pv}:claim:{uuid.uuid5(uuid.NAMESPACE_URL, statement).hex[:12]}"
        section_id = next(
            (u.rsplit("/section/", 1)[-1] for u in good if "/section/" in u), None)
        store.execute(
            "INSERT OR REPLACE INTO claims (id, paper_version, statement, kind, "
            "section_id, source) VALUES (?,?,?,?,?,?)",
            (cid, pv, statement, kind,
             f"{pv}:sec:{section_id}" if section_id else None,
             "SERVER_LLM" if author.startswith("server_llm") else "AGENT"))
        accepted_claims.append({"id": cid, "statement": statement, "kind": kind,
                                "evidence": good})

    # ── narrative fields, stored verbatim ─────────────────────────────────
    fields = {k: analysis[k] for k in NARRATIVE_FIELDS if k in analysis}
    unknown_keys = [k for k in analysis
                    if k not in NARRATIVE_FIELDS and k not in ("components", "claims")]
    analysis_id = f"{pv}:analysis:{uuid.uuid4().hex[:12]}"
    store.execute(
        "INSERT OR REPLACE INTO analyses (id, paper_version, schema_version, "
        "fields_json, author, validated, created_at) VALUES (?,?,?,?,?,?,?)",
        (analysis_id, pv, ANALYSIS_SCHEMA_VERSION, json.dumps(fields), author,
         1, utcnow()))
    store.commit()

    notes: list[str] = []
    if rejected:
        notes.append(
            f"{len(rejected)} item(s) were rejected. Every component and claim must "
            f"cite at least one paperlens:// URI that resolves; unsupported items "
            f"are not stored."
        )
    if unknown_keys:
        notes.append(f"Ignored unrecognised field(s): {', '.join(sorted(unknown_keys))}.")
    if not accepted_components:
        notes.append(
            "No components were accepted, so map_paper_to_code still has only "
            "section-derived anchors to work with."
        )

    return {
        "paper_version": pv,
        "analysis_uri": f"paperlens://paper/{arxiv_id}/analysis",
        "accepted": {"components": accepted_components, "claims": accepted_claims,
                     "narrative_fields": sorted(fields)},
        "rejected": rejected,
        "notes": notes,
    }


def get_analysis(store: Store, paper_id: str) -> dict[str, Any] | None:
    pv = _pv(store, paper_id)
    row = store.one(
        "SELECT * FROM analyses WHERE paper_version = ? ORDER BY created_at DESC LIMIT 1",
        (pv,))
    if row is None:
        return None
    return {"paper_version": pv, "author": row["author"], "created_at": row["created_at"],
            "schema_version": row["schema_version"],
            "fields": json.loads(row["fields_json"])}
