"""build_reproduction_plan: assemble what is known, and name what is not.

Everything here comes from the graph — structure, mappings, differences, gaps,
lineage — so the plan asserts nothing that was not established earlier and cited
somewhere. Sections are `KNOWN`, `PARTIAL` or `UNKNOWN`, and an `UNKNOWN` section
states what would resolve it rather than filling the space with plausible prose.

Prose is deliberately absent. Per D-009 the server supplies evidence-backed
structure and the agent writes the narrative.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..graph.store import Store
from ..resources import _pv, symbol_uri

_REQ_FILES = ("requirements.txt", "requirements-training.txt", "pyproject.toml",
              "setup.py", "environment.yml", "environment.yaml", "Pipfile")
_PY_VERSION = re.compile(r"python_requires\s*=\s*[\"']([^\"']+)|"
                         r"requires-python\s*=\s*[\"']([^\"']+)", re.I)
_TORCH = re.compile(r"^(torch|tensorflow|jax|flax)\b", re.I)


@dataclass
class Section:
    status: str                       # KNOWN | PARTIAL | UNKNOWN
    items: list[Any] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    note: str | None = None

    def pub(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status, "items": self.items}
        if self.evidence:
            out["evidence"] = self.evidence
        if self.note:
            out["note"] = self.note
        return out


def _ev(kind: str, uri: str, excerpt: str | None = None,
        locator: str | None = None) -> dict:
    return {"kind": kind, "uri": uri, "excerpt": excerpt, "locator": locator}


def _read(root: Path, name: str) -> str | None:
    try:
        return (root / name).read_text(errors="replace")
    except OSError:
        return None


# ── sections ──────────────────────────────────────────────────────────────
def _environment(root: Path, repo_key: str) -> Section:
    items: list[Any] = []
    evidence: list[dict] = []
    for name in ("setup.py", "pyproject.toml", ".python-version"):
        text = _read(root, name)
        if not text:
            continue
        if (m := _PY_VERSION.search(text)):
            items.append({"python": (m.group(1) or m.group(2)).strip()})
            evidence.append(_ev("code_file",
                                f"paperlens://repo/{repo_key}/file/{name}",
                                locator=name))
            break
    readme = _read(root, "README.md") or ""
    for m in re.finditer(r"(?:CUDA|cuda)[\s=]*([0-9]+\.[0-9]+)", readme):
        items.append({"cuda": m.group(1)})
        evidence.append(_ev("code_file", f"paperlens://repo/{repo_key}/file/README.md",
                            excerpt=m.group(0), locator="README.md"))
        break
    if not items:
        return Section("UNKNOWN", note=(
            "Neither the repository nor the paper states a Python or CUDA version. "
            "Pin one from the framework version below and record what you used."))
    return Section("PARTIAL" if len(items) < 2 else "KNOWN", items, evidence)


def _dependencies(root: Path, repo_key: str) -> Section:
    items: list[Any] = []
    evidence: list[dict] = []
    for name in _REQ_FILES:
        text = _read(root, name)
        if not text:
            continue
        if name.endswith((".txt", ".yml", ".yaml")):
            deps = [ln.strip() for ln in text.splitlines()
                    if ln.strip() and not ln.strip().startswith(("#", "-", "name:"))]
        else:
            deps = [d.strip(" \"'") for d in
                    re.findall(r"[\"']([A-Za-z0-9_.\-]+(?:[<>=!~]=?[^\"',]+)?)[\"']",
                               text)][:40]
        if deps:
            items.append({"file": name, "declared": deps[:40]})
            evidence.append(_ev("code_file",
                                f"paperlens://repo/{repo_key}/file/{name}",
                                locator=name))
    if not items:
        return Section("UNKNOWN", note=(
            "No dependency manifest was found in the repository."))
    unpinned = [d for it in items for d in it["declared"]
                if not re.search(r"[<>=~]", d)]
    note = None
    if unpinned:
        note = (f"{len(unpinned)} dependency(ies) are declared without a version "
                f"({', '.join(unpinned[:6])}). A reproduction should pin them, "
                f"since the paper's results predate current releases.")
    return Section("PARTIAL" if unpinned else "KNOWN", items, evidence, note)


def _from_components(store: Store, pv: str, snapshot_id: str, arxiv_id: str,
                     kinds: tuple[str, ...], repo_key: str) -> Section:
    rows = store.all(
        "SELECT c.slug, c.name, c.kind, c.description, m.status, m.confidence, "
        "m.symbol_id FROM components c "
        "LEFT JOIN mappings m ON m.anchor_id = c.id AND m.snapshot_id = ? "
        "WHERE c.paper_version = ? AND c.kind IN (%s)"
        % ",".join("?" * len(kinds)),
        (snapshot_id, pv, *kinds))
    if not rows:
        return Section("UNKNOWN", note=(
            f"No component of kind {'/'.join(kinds)} has been recorded for this "
            f"paper. Call record_paper_analysis so this section has something to "
            f"stand on."))
    items, evidence = [], []
    for r in rows:
        item = {"component": r["name"], "kind": r["kind"],
                "status_in_repo": r["status"] or "UNMAPPED",
                "confidence": r["confidence"] or "UNKNOWN",
                "uri": f"paperlens://paper/{arxiv_id}/component/{r['slug']}"}
        if r["symbol_id"]:
            qn = r["symbol_id"].split(":", 2)[-1]
            item["implemented_by"] = symbol_uri(repo_key, qn)
            evidence.append(_ev("code_symbol", item["implemented_by"], locator=qn))
        items.append(item)
    matched = sum(1 for i in items if i["status_in_repo"] == "MATCHED")
    absent = [i["component"] for i in items if i["status_in_repo"] == "ABSENT"]
    parts: list[str] = []
    if absent:
        parts.append(f"Not present in {repo_key}: {', '.join(absent)}. "
                     f"These must be written from the paper.")
    if matched == 0:
        parts.append(
            f"The paper describes {len(items)} item(s) here, but none was located "
            f"in {repo_key}, so this section rests on the paper alone.")
    elif matched < len(items):
        parts.append(f"{matched} of {len(items)} located in {repo_key}.")
    # The paper's side is known even when the code's side is not, so a section
    # with described components is PARTIAL rather than UNKNOWN -- UNKNOWN is
    # reserved for having nothing at all.
    status = "KNOWN" if matched == len(items) else "PARTIAL"
    return Section(status, items, evidence, " ".join(parts) or None)


def _hyperparameters(store: Store, pv: str, snapshot_id: str,
                     arxiv_id: str) -> Section:
    rows = store.all(
        "SELECT v.symbol, v.value_text, v.src_line, v.context, m.status, m.confidence "
        "FROM stated_values v LEFT JOIN mappings m "
        "ON m.anchor_id = v.id AND m.snapshot_id = ? "
        "WHERE v.paper_version = ? ORDER BY v.src_line", (snapshot_id, pv))
    if not rows:
        return Section("UNKNOWN", note="No hyperparameters were extracted from the paper.")
    items, evidence = [], []
    for r in rows:
        items.append({"symbol": r["symbol"], "value": r["value_text"],
                      "stated_at": f"LaTeX line {r['src_line']}",
                      "found_in_code": r["status"] or "UNMAPPED",
                      "confidence": r["confidence"] or "UNKNOWN"})
        if r["status"] == "MATCHED":
            evidence.append(_ev("paper_stated_value",
                                f"paperlens://paper/{arxiv_id}/values",
                                excerpt=f"{r['symbol']} = {r['value_text']}",
                                locator=f"LaTeX line {r['src_line']}"))
    confirmed = sum(1 for i in items if i["found_in_code"] == "MATCHED")
    return Section("PARTIAL" if confirmed else "UNKNOWN", items, evidence,
                   note=(f"{confirmed} of {len(items)} stated values were located "
                         f"in source. The rest are unverified against the code."))


def _risks(store: Store, pv: str, snapshot_id: str, repo_key: str) -> list[dict]:
    out: list[dict] = []
    for r in store.all(
        "SELECT * FROM differences WHERE paper_version = ? AND snapshot_id = ? "
        "ORDER BY CASE severity WHEN 'BLOCKING' THEN 0 WHEN 'SIGNIFICANT' THEN 1 "
        "WHEN 'MINOR' THEN 2 ELSE 3 END", (pv, snapshot_id)
    ):
        out.append({"severity": r["severity"], "kind": r["kind"],
                    "risk": r["difference"], "paper_states": r["paper_states"],
                    "code_does": r["code_does"], "confidence": r["confidence"]})
    return out


def _verification(store: Store, pv: str, snapshot_id: str, repo_key: str) -> Section:
    """Concrete checks derived from anchors that were actually confirmed."""
    checks: list[dict] = []
    for r in store.all(
        "SELECT m.reasoning, m.confidence, v.symbol, v.value_text "
        "FROM mappings m JOIN stated_values v ON v.id = m.anchor_id "
        "WHERE m.paper_version = ? AND m.snapshot_id = ? AND m.status = 'MATCHED'",
        (pv, snapshot_id)
    ):
        loc = re.search(r"appears at (\S+:\d+)", r["reasoning"] or "")
        checks.append({
            "check": f"assert the value used for {r['symbol']} equals "
                     f"{r['value_text']}",
            "where": loc.group(1) if loc else None,
            "why": "the paper states this value and the identical literal was "
                   "located in source",
            "confidence": r["confidence"]})
    for r in store.all(
        "SELECT c.name FROM mappings m JOIN components c ON c.id = m.anchor_id "
        "WHERE m.paper_version = ? AND m.snapshot_id = ? AND m.status = 'ABSENT'",
        (pv, snapshot_id)
    ):
        checks.append({
            "check": f"implement {r['name']} and test it independently",
            "where": None,
            "why": "the paper describes it and it is absent from this repository, "
                   "so nothing here can be compared against",
            "confidence": "CONFIRMED"})
    if not checks:
        return Section("UNKNOWN", note=(
            "No anchor was confirmed against this repository, so no verification "
            "step can be grounded. Run map_paper_to_code first."))
    return Section("PARTIAL", checks, note=(
        "These checks follow from anchors that were verified. They do not "
        "constitute a full test plan."))


def build_reproduction_plan(store: Store, paper_id: str,
                            repo: str | None = None) -> dict[str, Any]:
    from ..code.indexer import require_snapshot

    pv = _pv(store, paper_id)
    arxiv_id = pv.split("v")[0]
    paper = store.one("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,))
    notes: list[str] = []

    if repo is None:
        row = store.one(
            "SELECT repo_id FROM implementation_candidates WHERE paper_version = ? "
            "ORDER BY rank LIMIT 1", (pv,))
        if row is None:
            return {"error": "LookupError",
                    "message": "No repository given and none discovered. Call "
                               "find_implementations, or pass repo explicitly."}
        repo = row["repo_id"]
        notes.append(f"No repository was given; using the top-ranked candidate {repo}.")

    snap = require_snapshot(store, repo)
    repo_key, snapshot_id = snap["repo_id"], snap["id"]
    root = Path(snap["clone_path"] or "")

    sections = {
        "environment": _environment(root, repo_key),
        "dependencies": _dependencies(root, repo_key),
        "dataset": _from_components(store, pv, snapshot_id, arxiv_id,
                                    ("dataset",), repo_key),
        "dataset_preparation": _from_components(store, pv, snapshot_id, arxiv_id,
                                                ("preprocessing",), repo_key),
        "model_architecture": _from_components(store, pv, snapshot_id, arxiv_id,
                                               ("architecture", "module"), repo_key),
        "training_pipeline": _from_components(store, pv, snapshot_id, arxiv_id,
                                              ("training", "loss", "optimizer"), repo_key),
        "hyperparameters": _hyperparameters(store, pv, snapshot_id, arxiv_id),
        "evaluation": _from_components(store, pv, snapshot_id, arxiv_id,
                                       ("evaluation", "inference"), repo_key),
    }

    risks = _risks(store, pv, snapshot_id, repo_key)
    gaps_rows = store.all(
        "SELECT DISTINCT anchor_id, reasoning FROM mappings WHERE paper_version = ? "
        "AND snapshot_id = ? AND status = 'ABSENT'", (pv, snapshot_id))
    missing = [{"item": r["reasoning"], "source": "mapping"} for r in gaps_rows]

    # Expected results are reported numbers, not something to be re-derived here.
    results = store.all(
        "SELECT symbol, value_text, src_line FROM stated_values "
        "WHERE paper_version = ? AND (LOWER(symbol) LIKE '%accuracy%' "
        "OR LOWER(context) LIKE '%accuracy%' OR LOWER(context) LIKE '%bleu%') "
        "LIMIT 8", (pv,))
    expected = Section(
        "PARTIAL" if results else "UNKNOWN",
        [{"symbol": r["symbol"], "value": r["value_text"],
          "stated_at": f"LaTeX line {r['src_line']}"} for r in results],
        note=("Reported figures as stated in the paper. They are the target, not a "
              "prediction: none has been reproduced here."))

    lineage_count = store.one(
        "SELECT COUNT(*) c FROM lineage_edges WHERE from_paper = ?", (arxiv_id,))["c"]

    blocking = [r for r in risks if r["severity"] == "BLOCKING"]
    if blocking:
        notes.append(
            f"{len(blocking)} BLOCKING difference(s) stand between this repository "
            f"and the paper. Read those before estimating effort.")
    if not store.one("SELECT 1 FROM analyses WHERE paper_version = ? LIMIT 1", (pv,)):
        notes.append(
            "No paper analysis has been recorded, so component-derived sections "
            "rest on section headings alone. record_paper_analysis sharpens them.")
    notes.append(
        "Nothing in this plan is asserted beyond what was established and cited "
        "elsewhere in the graph. Sections marked UNKNOWN are genuinely unknown.")

    return {
        "paper_version": pv, "title": paper["title"] if paper else None,
        "repo": repo_key, "commit_sha": snap["commit_sha"],
        "uri": f"paperlens://plan/{arxiv_id}/{repo_key}",
        "sections": {k: v.pub() for k, v in sections.items()},
        "expected_results": expected.pub(),
        "reproduction_risks": risks,
        "missing_information": missing,
        "verification_strategy": _verification(store, pv, snapshot_id, repo_key).pub(),
        "lineage": {"predecessors_recorded": lineage_count,
                    "uri": f"paperlens://lineage/{arxiv_id}"},
        "readiness": {
            "sections_known": sum(1 for s in sections.values() if s.status == "KNOWN"),
            "sections_partial": sum(1 for s in sections.values() if s.status == "PARTIAL"),
            "sections_unknown": sum(1 for s in sections.values() if s.status == "UNKNOWN"),
            "blocking_risks": len(blocking),
        },
        "notes": notes,
    }
