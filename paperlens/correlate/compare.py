"""Phase 6: comparison and gap detection.

Three capabilities that all rest on the same idea -- a difference is only worth
reporting when both halves are checkable:

  * the paper says X            -> an exact quote from the flattened LaTeX
  * the code does Y (or not)    -> an exact search over the indexed repository
  * the inference joining them  -> stated plainly, and it caps the confidence

The join is a heuristic (the paper's "clipped" ought to appear in code as
`clamp`), so a difference is LIKELY at best even though both observations are
CONFIRMED. Saying that out loud is the difference between a useful report and a
confident wrong one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .. import config
from ..code.indexer import provider, require_snapshot
from ..evidence.confidence import Evidence, Signal, evidence_for, record
from ..graph.store import Store
from ..resources import _pv

# A behaviour the paper asserts, and the vocabulary it would leave in source.
# `paper` is matched against the flattened LaTeX; `code` against the repository.
_BEHAVIOURS: list[tuple[str, str, str, str, str]] = [
    # name, paper pattern, code pattern, difference kind, base severity
    ("value clipping", r"\bclipp?(?:ed|ing)\b|\bclamp(?:ed|ing)?\b",
     r"\bclamp\b|\.clip\(|np\.clip|torch\.clamp", "TRAINING", "SIGNIFICANT"),
    ("gradient clipping", r"gradient clipping|clip\w*\s+(?:the\s+)?gradients?",
     r"clip_grad|clip_grad_norm", "TRAINING", "SIGNIFICANT"),
    ("learning-rate warmup", r"warm-?up", r"warmup|warm_up", "TRAINING", "SIGNIFICANT"),
    ("weight decay", r"weight decay", r"weight_decay", "TRAINING", "SIGNIFICANT"),
    ("label smoothing", r"label smoothing", r"label_smooth", "TRAINING", "MINOR"),
    ("dropout", r"\bdropout\b", r"[Dd]ropout", "ARCHITECTURAL", "MINOR"),
    ("exponential moving average", r"exponential moving average|\bEMA\b",
     r"\bema\b|exponential_moving", "TRAINING", "MINOR"),
    ("mixed precision", r"mixed[- ]precision|\bfp16\b|half precision",
     r"fp16|autocast|GradScaler|\.half\(", "TRAINING", "MINOR"),
    ("data augmentation", r"data augmentation|augmentations?\b",
     r"augment|RandomCrop|RandomResized|RandomHorizontal", "TRAINING", "MINOR"),
    ("input normalization", r"normali[sz](?:ed|ation)",
     r"[Nn]ormalize|normali[sz]ation", "UNDOCUMENTED_PREPROCESSING", "MINOR"),
]

_NECESSITY = re.compile(
    r"necessary|essential|critical|required|crucial|must\b|we found (?:it|this)", re.I)

# Constants specific enough that omitting them from the paper is a real gap.
_PRECISE_CONSTANT = re.compile(r"(?<![\w.])\d\.\d{4,}(?![\w.])")
_WEIGHT_URL = re.compile(r"https?://[^\s\"']+\.(?:pt|pth|bin|ckpt|safetensors|tar|gz)")


@dataclass
class Difference:
    kind: str
    paper_states: str | None
    code_does: str | None
    difference: str
    severity: str
    confidence: str
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Gap:
    kind: str
    description: str
    where: str | None
    reproducibility_impact: str
    confidence: str
    evidence: list[dict[str, Any]] = field(default_factory=list)


def _quote(tex: str, match: re.Match, width: int = 190) -> str:
    lo = max(0, match.start() - width // 2)
    return re.sub(r"\s+", " ", tex[lo:match.end() + width // 2]).strip()


def _line_of(tex: str, offset: int) -> int:
    return tex.count("\n", 0, offset) + 1


# ── compare_paper_with_code ───────────────────────────────────────────────
def compare_paper_with_code(store: Store, paper_id: str, repo: str) -> dict[str, Any]:
    pv = _pv(store, paper_id)
    arxiv_id = pv.split("v")[0]
    snap = require_snapshot(store, repo)
    repo_key, snapshot_id = snap["repo_id"], snap["id"]
    prov = provider()
    tex = (store.paper_version_row(pv)["flattened_tex"] or "")

    store.execute("DELETE FROM differences WHERE paper_version = ? AND snapshot_id = ?",
                  (pv, snapshot_id))

    diffs: list[Difference] = []

    # 1. Components the paper defines that are absent from the repository.
    for row in store.all(
        "SELECT m.*, c.name, c.kind AS ckind FROM mappings m "
        "JOIN components c ON c.id = m.anchor_id "
        "WHERE m.paper_version = ? AND m.snapshot_id = ? AND m.anchor_kind = 'component' "
        "AND m.status = 'ABSENT'", (pv, snapshot_id)
    ):
        severity = {"loss": "BLOCKING", "training": "BLOCKING", "optimizer": "SIGNIFICANT",
                    "architecture": "SIGNIFICANT"}.get(row["ckind"], "MINOR")
        diffs.append(Difference(
            kind="MISSING_COMPONENT",
            paper_states=f"The paper defines a {row['ckind']} component: {row['name']}.",
            code_does="No symbol implementing it exists in this repository.",
            difference=f"{row['name']} is described in the paper but absent from "
                       f"{repo_key}.",
            severity=severity, confidence=row["confidence"],
            evidence=evidence_for(store, "mapping", row["id"]),
        ))

    # 2. Behaviours the paper asserts whose vocabulary is missing from source.
    behaviour_hits = 0
    for name, paper_pat, code_pat, kind, base_sev in _BEHAVIOURS:
        pm = re.search(paper_pat, tex, re.I)
        if not pm:
            continue
        behaviour_hits += 1
        found = prov.search_text(repo_key, code_pat, regex=True, limit=3)
        if found:
            continue
        quote = _quote(tex, pm)
        severity = "BLOCKING" if _NECESSITY.search(quote) else base_sev
        diffs.append(Difference(
            kind=kind,
            paper_states=quote,
            code_does=f"No occurrence of {name} vocabulary in the repository.",
            difference=f"The paper describes {name}, but nothing in {repo_key} "
                       f"appears to implement it.",
            severity=severity,
            # Both observations are exact; the inference joining them is not.
            confidence="LIKELY",
            evidence=[
                {"kind": "paper_section", "uri": f"paperlens://paper/{arxiv_id}",
                 "excerpt": quote, "locator": f"flattened LaTeX line {_line_of(tex, pm.start())}",
                 "stance": "SUPPORTS", "provenance": {"pattern": paper_pat}},
                {"kind": "code_absence", "uri": f"paperlens://repo/{repo_key}",
                 "excerpt": f"no match for /{code_pat}/ in source", "locator": None,
                 "stance": "SUPPORTS", "provenance": {"method": "regex_search",
                                                      "behaviour": name}},
            ],
        ))

    # 3. Hyperparameters stated in the paper whose literal is not in source.
    for row in store.all(
        "SELECT m.*, v.symbol, v.value_text, v.context FROM mappings m "
        "JOIN stated_values v ON v.id = m.anchor_id "
        "WHERE m.paper_version = ? AND m.snapshot_id = ? "
        "AND m.anchor_kind = 'stated_value' AND m.status = 'ABSENT'", (pv, snapshot_id)
    ):
        diffs.append(Difference(
            kind="CHANGED_HYPERPARAMETER",
            paper_states=f"{row['symbol']} = {row['value_text']}",
            code_does=f"The literal {row['value_text']} does not appear in source.",
            difference=f"The paper specifies {row['symbol']} = {row['value_text']}; "
                       f"{repo_key} does not use that value.",
            severity="SIGNIFICANT", confidence=row["confidence"],
            evidence=evidence_for(store, "mapping", row["id"]),
        ))

    order = {"BLOCKING": 0, "SIGNIFICANT": 1, "MINOR": 2, "COSMETIC": 3}
    diffs.sort(key=lambda d: order.get(d.severity, 9))

    for i, d in enumerate(diffs):
        store.execute(
            "INSERT OR REPLACE INTO differences (id, paper_version, snapshot_id, kind, "
            "paper_states, code_does, difference, severity, confidence) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (f"{pv}|{snapshot_id}|diff|{i}", pv, snapshot_id, d.kind, d.paper_states,
             d.code_does, d.difference, d.severity, d.confidence))
    store.commit()

    notes: list[str] = []
    has_training = bool(prov.search_text(repo_key, r"\.backward\(|optimizer|def train",
                                         regex=True, limit=1))
    if not has_training:
        notes.append(
            f"{repo_key} contains no training code, so every training-related "
            f"difference below follows from that single fact rather than from "
            f"independent decisions."
        )
    notes.append(
        "Each difference pairs an exact quote from the paper with an exact search "
        "over source. The inference that a paper term implies a code term is a "
        "heuristic, which is why these are LIKELY rather than CONFIRMED."
    )
    if not diffs:
        notes.append("No differences were established. That is not proof of "
                     "equivalence, only that these checks found nothing.")

    return {
        "paper_version": pv, "repo": repo_key, "commit_sha": snap["commit_sha"],
        "uri": f"paperlens://mapping/{arxiv_id}/{repo_key}",
        "summary": {s: sum(1 for d in diffs if d.severity == s)
                    for s in ("BLOCKING", "SIGNIFICANT", "MINOR")},
        "differences": [{
            "kind": d.kind, "paper_states": d.paper_states, "code_does": d.code_does,
            "difference": d.difference, "severity": d.severity,
            "confidence": d.confidence, "evidence": d.evidence,
        } for d in diffs],
        "notes": notes,
    }


# ── find_implementation_gaps ──────────────────────────────────────────────
def find_implementation_gaps(store: Store, paper_id: str, repo: str) -> dict[str, Any]:
    pv = _pv(store, paper_id)
    arxiv_id = pv.split("v")[0]
    snap = require_snapshot(store, repo)
    repo_key = snap["repo_id"]
    prov = provider()
    tex = (store.paper_version_row(pv)["flattened_tex"] or "")
    gaps: list[Gap] = []

    # G1. Precise constants baked into source that the paper never states.
    #     This is the vision doc's own example: "the normalization constants are
    #     not clearly documented in the paper".
    # Grouped by site: six normalization constants on one line are one gap, not
    # six, and listing them separately buries everything else.
    by_site: dict[str, tuple[str, list[str]]] = {}
    seen: set[str] = set()
    for hit in prov.search_text(repo_key, r"\d\.\d{4,}", regex=True, limit=40):
        undocumented = [c for c in _PRECISE_CONSTANT.findall(hit.text)
                        if c not in tex and c not in seen]
        if not undocumented:
            continue
        seen.update(undocumented)
        site = f"{hit.file_path}:{hit.line}"
        by_site[site] = (hit.text, undocumented)

    for site, (line_text, consts) in by_site.items():
        file_path = site.rsplit(":", 1)[0]
        listed = ", ".join(consts[:6]) + ("…" if len(consts) > 6 else "")
        gaps.append(Gap(
            kind="CONFIG_HIDDEN",
            description=f"{len(consts)} constant(s) hard-coded in source that appear "
                        f"nowhere in the paper: {listed}.",
            where=site,
            reproducibility_impact="A reimplementation would have to guess these "
                                   "values or copy them from this repository.",
            confidence="CONFIRMED",
            evidence=[
                {"kind": "code_file", "stance": "SUPPORTS",
                 "uri": f"paperlens://repo/{repo_key}/file/{file_path}",
                 "excerpt": line_text, "locator": site, "provenance": None},
                {"kind": "paper_url", "stance": "SUPPORTS",
                 "uri": f"paperlens://paper/{arxiv_id}",
                 "excerpt": f"none of {listed} occurs in the paper source",
                 "locator": None,
                 "provenance": {"method": "literal_absence", "constants": consts}},
            ],
        ))

    # G2. Components the paper defines that the repository does not contain.
    for row in store.all(
        "SELECT m.*, c.name, c.kind AS ckind FROM mappings m "
        "JOIN components c ON c.id = m.anchor_id "
        "WHERE m.paper_version = ? AND m.snapshot_id = ? "
        "AND m.anchor_kind = 'component' AND m.status = 'ABSENT'",
        (pv, snap["id"])
    ):
        gaps.append(Gap(
            kind="ABSENT_FROM_REPOSITORY",
            description=f"{row['name']} ({row['ckind']}) is described in the paper "
                        f"but not implemented here.",
            where=None,
            reproducibility_impact="This must be written from the paper's prose, "
                                   "which is where reproductions usually diverge.",
            confidence=row["confidence"],
            evidence=evidence_for(store, "mapping", row["id"]),
        ))

    # G3. Weights or data the repository downloads rather than builds.
    # A table of N checkpoint URLs is one dependency, reported once.
    weight_sites: dict[str, list[str]] = {}
    for hit in prov.search_text(repo_key, r"https?://\S+\.(?:pt|pth|bin|ckpt|tar\.gz)",
                                regex=True, limit=30):
        url = (_WEIGHT_URL.search(hit.text) or re.search(r"https?://\S+", hit.text))
        if url:
            weight_sites.setdefault(hit.file_path, []).append(
                f"{hit.line}: {url.group(0)[:120]}")
    for file_path, entries in weight_sites.items():
        gaps.append(Gap(
            kind="EXTERNAL_DEPENDENCY",
            description=f"The repository downloads {len(entries)} pretrained "
                        f"artifact(s) rather than producing them from the described "
                        f"procedure.",
            where=f"{file_path} ({len(entries)} URL(s))",
            reproducibility_impact="Results obtained with these artifacts do not "
                                   "demonstrate that the training procedure reproduces.",
            confidence="CONFIRMED",
            evidence=[{"kind": "code_file", "stance": "SUPPORTS",
                       "uri": f"paperlens://repo/{repo_key}/file/{file_path}",
                       "excerpt": "; ".join(entries[:4]), "locator": file_path,
                       "provenance": {"count": len(entries)}}],
        ))

    order = {"ABSENT_FROM_REPOSITORY": 0, "EXTERNAL_DEPENDENCY": 1, "CONFIG_HIDDEN": 2}
    gaps.sort(key=lambda g: order.get(g.kind, 9))

    notes = ["Gaps are established, not inferred: a hidden constant is one that "
             "occurs in source and does not occur in the paper source."]
    if not gaps:
        notes.append("No gaps were established by these checks. That is not a "
                     "claim that none exist.")
    return {
        "paper_version": pv, "repo": repo_key, "commit_sha": snap["commit_sha"],
        "summary": {k: sum(1 for g in gaps if g.kind == k)
                    for k in ("ABSENT_FROM_REPOSITORY", "EXTERNAL_DEPENDENCY",
                              "CONFIG_HIDDEN")},
        "gaps": [{
            "kind": g.kind, "description": g.description, "where": g.where,
            "reproducibility_impact": g.reproducibility_impact,
            "confidence": g.confidence, "evidence": g.evidence,
        } for g in gaps],
        "notes": notes,
    }


# ── compare_implementations ───────────────────────────────────────────────
def compare_implementations(store: Store, paper_id: str,
                            repos: list[str]) -> dict[str, Any]:
    """Compare repositories on what they actually implement.

    Ranking deliberately does not count name matches. openai/CLIP has tidier
    naming than mlfoundations/open_clip and would win on MATCHED count, while
    lacking the contrastive objective the paper is about. Established absences
    and BLOCKING differences are the substantive signal, so those lead.
    """
    from .mapper import map_paper_to_code

    pv = _pv(store, paper_id)
    # Section-derived components are headings and produce noise here, so the
    # comparison is restricted to anchors that carry meaning.
    meaningful = {
        r["name"] for r in store.all(
            "SELECT name FROM components WHERE paper_version = ? AND source != "
            "'DETERMINISTIC'", (pv,))
    }

    per_repo: dict[str, dict[str, str]] = {}
    severity: dict[str, dict[str, int]] = {}
    errors: dict[str, str] = {}
    for repo in repos:
        try:
            result = map_paper_to_code(store, paper_id, repo)
            diffs = compare_paper_with_code(store, paper_id, repo)
        except Exception as exc:
            errors[repo] = f"{type(exc).__name__}: {exc}"
            continue
        key = result["repo"]
        per_repo[key] = {
            m["anchor"]["label"]: m["status"] for m in result["mappings"]
            if m["anchor"]["kind"] == "stated_value" or m["anchor"]["label"] in meaningful
        }
        severity[key] = diffs["summary"]

    anchors = sorted({a for m in per_repo.values() for a in m})
    common, divergent = [], []
    for a in anchors:
        statuses = {r: m.get(a, "UNKNOWN") for r, m in per_repo.items()}
        (common if len(set(statuses.values())) == 1 else divergent).append(
            {"anchor": a, "status": statuses})

    def _counts(repo: str) -> dict[str, int]:
        m = per_repo[repo]
        return {"matched": sum(1 for v in m.values() if v == "MATCHED"),
                "absent": sum(1 for v in m.values() if v == "ABSENT"),
                "ambiguous": sum(1 for v in m.values() if v == "AMBIGUOUS"),
                "blocking_differences": severity[repo].get("BLOCKING", 0),
                "significant_differences": severity[repo].get("SIGNIFICANT", 0)}

    coverage = {r: _counts(r) for r in per_repo}

    def _rank(repo: str) -> tuple:
        c = coverage[repo]
        return (c["blocking_differences"], c["absent"],
                c["significant_differences"], -c["matched"])

    ranked = sorted(per_repo, key=_rank)
    closest: str | None = ranked[0] if ranked else None
    notes = [
        "Repositories are ranked by BLOCKING differences and established absences, "
        "not by how many symbol names happen to match. A repository with tidier "
        "naming can match more anchors while implementing less of the paper."
    ]
    if len(ranked) >= 2 and _rank(ranked[0]) == _rank(ranked[1]):
        closest = None
        notes.append(
            f"{ranked[0]} and {ranked[1]} score identically on these checks, so "
            f"no claim is made about which is closer to the paper."
        )
    if errors:
        notes.append(f"Not compared (index them first): {', '.join(errors)}.")
    if len(per_repo) < 2:
        notes.append("Fewer than two repositories were compared, so no divergence "
                     "can be established.")
    if not meaningful:
        notes.append(
            "No agent-authored components exist for this paper, so the comparison "
            "rests on stated values alone. Call record_paper_analysis for a "
            "sharper comparison."
        )

    return {
        "paper_version": pv,
        "repos": list(per_repo),
        "errors": errors,
        "closest_to_paper": closest,
        "coverage": coverage,
        "agree_on": common[:40],
        "differ_on": divergent[:40],
        "notes": notes,
    }
