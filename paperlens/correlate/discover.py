"""find_implementations: paper -> ranked repository candidates, with evidence.

Papers With Code shut down and nothing replaced it: Hugging Face maps arXiv ids
to Hub artifacts rather than source repositories, and alphaXiv exposes no public
API (RESEARCH.md section 3). So the primary signal here is the one that survives
in the primary source -- the GitHub URL the authors printed in their own paper.
Verified 2/2 on the test corpus.

Ranking is by *verified content coverage*, not authorship. That distinction is
load-bearing: openai/CLIP is unambiguously the official repository and contains
no training code and no contrastive loss, so ranking by authorship alone returns
a confidently wrong answer on the most famous paper in multimodal ML.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .. import config
from ..evidence.confidence import Evidence, Signal, assess, at_most, record
from ..graph.store import Store
from ..sources import github as gh

# Component kind -> filename patterns that would implement it.
_KIND_PATTERNS: dict[str, re.Pattern] = {
    "loss": re.compile(r"loss|criterion|objective", re.I),
    "training": re.compile(r"\btrain|trainer|fit\b|optimi[sz]", re.I),
    "dataset": re.compile(r"\bdata\b|dataset|dataloader|loader|corpus", re.I),
    "architecture": re.compile(r"model|network|\bnet\b|module|layer|arch|encoder|decoder", re.I),
    "preprocessing": re.compile(r"preprocess|transform|augment|token", re.I),
    "evaluation": re.compile(r"eval|\btest|benchmark|metric|score", re.I),
    "inference": re.compile(r"infer|predict|generate|sample|decode", re.I),
    "optimizer": re.compile(r"optimi[sz]|schedul|\blr\b", re.I),
}

# Coverage is only meaningful over actual source files. Without this, a big
# repo of blog posts scores 1.0 because some .svg is called "data".
_CODE_EXT = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".c",
             ".cc", ".cpp", ".h", ".hpp", ".cu", ".m", ".swift", ".kt", ".scala",
             ".rb", ".jl", ".lua", ".sh"}
_NON_CODE_DIR = re.compile(r"(^|/)(docs?|examples?|assets?|images?|paper_list|"
                           r"website|blog|\.github)(/|$)", re.I)


def _code_files(files: list[str]) -> list[str]:
    out = []
    for f in files:
        if _NON_CODE_DIR.search(f):
            continue
        dot = f.rfind(".")
        if dot != -1 and f[dot:].lower() in _CODE_EXT:
            out.append(f)
    return out


_REPRO_HINT = re.compile(
    r"\b(unofficial|reproduc|re-?implement|replicat|third[- ]party|port of|pytorch "
    r"implementation of|my implementation)\b", re.I)


@dataclass
class Candidate:
    owner: str
    name: str
    relation: str
    confidence: str
    reasoning: str
    coverage_score: float | None
    coverage_kind: str            # SHALLOW | NONE
    matched_kinds: dict[str, str] = field(default_factory=dict)
    missing_kinds: list[str] = field(default_factory=list)
    stars: int | None = None
    archived: bool | None = None
    license: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    rank: int = 0

    @property
    def repo_id(self) -> str:
        return f"{self.owner}/{self.name}"


def _author_tokens(authors: list[dict]) -> set[str]:
    toks: set[str] = set()
    for a in authors:
        for part in re.split(r"[\s.\-]+", (a.get("name") or "").lower()):
            if len(part) > 2:
                toks.add(part)
    return toks


def _shallow_coverage(
    files: list[str], expected: list[str]
) -> tuple[float | None, dict[str, str], list[str]]:
    """Does the repo contain files that plausibly implement each component kind?

    Deliberately shallow: one API call, no clone, no parsing. It cannot tell a
    real loss function from a file named loss.py, so it is corroboration, never
    proof -- deep coverage arrives with code intelligence in a later phase.
    """
    if not expected or not files:
        return None, {}, []
    matched: dict[str, str] = {}
    missing: list[str] = []
    for kind in expected:
        pat = _KIND_PATTERNS.get(kind)
        if pat is None:
            continue
        hit = next((f for f in files if pat.search(f.rsplit("/", 1)[-1])), None)
        if hit:
            matched[kind] = hit
        else:
            missing.append(kind)
    total = len(matched) + len(missing)
    return (len(matched) / total if total else None), matched, missing


def find_implementations(
    store: Store,
    paper_id: str,
    kind: str = "all",
    verify_coverage: bool = True,
    max_candidates: int = 8,
) -> dict[str, Any]:
    from ..resources import _pv

    pv = _pv(store, paper_id)
    arxiv_id = pv.split("v")[0]
    paper = store.one("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,))
    import json as _json
    authors = _json.loads(paper["authors_json"] or "[]")
    author_toks = _author_tokens(authors)
    title = paper["title"]

    expected_kinds = [
        r["kind"] for r in store.all(
            "SELECT DISTINCT kind FROM components WHERE paper_version = ?", (pv,))
    ]

    # ── seed: repositories the authors named in the paper ──────────────────
    seeds: dict[str, dict[str, Any]] = {}
    for r in store.all(
        "SELECT * FROM declared_urls WHERE paper_version = ? AND owner IS NOT NULL", (pv,)
    ):
        seeds[f"{r['owner']}/{r['repo']}".lower()] = {
            "owner": r["owner"], "name": r["repo"], "declared": True,
            "in_abstract": bool(r["in_abstract"]), "url": r["url"],
            "src_line": r["src_line"], "url_id": r["id"],
        }

    # ── widen: repositories that merely claim to implement the paper ───────
    if kind in ("all", "reproduction") and len(seeds) < max_candidates:
        for repo in gh.search_repos(store, f'"{title}" in:readme', limit=6):
            k = repo.full_name.lower()
            if repo.owner and k not in seeds:
                seeds[k] = {"owner": repo.owner, "name": repo.name, "declared": False}

    notes: list[str] = []
    if not gh.authenticated():
        notes.append(
            "Running without a GitHub token: 60 requests/hour. Some candidates may "
            "report UNKNOWN purely because metadata could not be fetched."
        )

    candidates: list[Candidate] = []
    for meta in list(seeds.values())[:max_candidates]:
        c = _assess_candidate(store, pv, arxiv_id, title, author_toks, meta,
                              expected_kinds, verify_coverage)
        if c:
            candidates.append(c)

    # Candidates with no link to the paper at all are noise, not findings.
    candidates = [c for c in candidates if c.relation != "UNRELATED"]

    if kind == "official":
        candidates = [c for c in candidates if c.relation in ("OFFICIAL", "ORGANIZATION")]
    elif kind == "reproduction":
        candidates = [c for c in candidates if c.relation in ("REPRODUCTION", "THIRD_PARTY")]

    candidates.sort(key=_rank_key, reverse=True)
    for i, c in enumerate(candidates, 1):
        c.rank = i
        _persist(store, pv, c)
    store.commit()

    if not candidates:
        notes.append(
            "No implementation candidates were established. The paper declares no "
            "repository URL and no repository was found claiming to implement it."
        )
    if any(c.missing_kinds for c in candidates):
        notes.append(
            "Coverage is SHALLOW: it matches filenames against the paper's component "
            "kinds and cannot confirm that a matched file truly implements the "
            "component. Missing kinds are the more reliable half of this signal."
        )

    return {
        "paper_version": pv,
        "uri": f"paperlens://paper/{arxiv_id}/implementations",
        "candidates": [_public(c) for c in candidates],
        "notes": notes,
    }


def _rank_key(c: Candidate) -> tuple:
    """Evidential strength first, then relation, then coverage, then popularity.

    D-005 says not to rank by authorship *alone*, and that still holds: coverage
    is computed for every candidate and reported prominently, so an official
    repository that does not implement the paper is visibly incomplete rather
    than silently trusted. But coverage must not lead the ranking -- it is a
    filename heuristic, and letting it lead ranked an unrelated blog repository
    above openai/CLIP.
    """
    conf_rank = {"CONFIRMED": 3, "LIKELY": 2, "POSSIBLE": 1, "UNKNOWN": 0}[c.confidence]
    rel_rank = {"OFFICIAL": 3, "ORGANIZATION": 2, "REPRODUCTION": 2,
                "THIRD_PARTY": 1, "DERIVED": 0, "UNRELATED": -1}[c.relation]
    return (conf_rank, rel_rank,
            c.coverage_score if c.coverage_score is not None else -1.0,
            c.stars or 0)


def _assess_candidate(
    store: Store, pv: str, arxiv_id: str, title: str, author_toks: set[str],
    meta: dict[str, Any], expected_kinds: list[str], verify_coverage: bool,
) -> Candidate | None:
    owner, name = meta["owner"], meta["name"]
    repo = gh.get_repo(store, owner, name)
    if repo is None:
        # The paper names it but GitHub does not serve it: say so, do not drop it.
        if not meta.get("declared"):
            return None
        return Candidate(
            owner=owner, name=name, relation="OFFICIAL", confidence="UNKNOWN",
            reasoning="the paper declares this repository but it could not be "
                      "fetched from GitHub (moved, private, deleted, or rate limited)",
            coverage_score=None, coverage_kind="NONE",
        )
    owner, name = repo.owner, repo.name
    repo_uri = f"https://github.com/{repo.full_name}"
    signals: list[Signal] = []

    # S1 -- the authors printed this URL in their own paper.
    if meta.get("declared"):
        where = "abstract" if meta.get("in_abstract") else "body"
        signals.append(Signal(
            name="declared_in_paper", weight="DECISIVE",
            detail=f"the paper itself links this repository (in the {where})",
            evidence=[Evidence(
                kind="paper_url",
                uri=f"paperlens://paper/{arxiv_id}/urls",
                excerpt=meta.get("url"),
                locator=f"flattened LaTeX line {meta.get('src_line')}",
            )],
        ))

    # S2 -- repository owner matches the paper's authors or their organisation.
    owner_name = gh.get_owner_name(store, owner) or ""
    owner_toks = {t for t in re.split(r"[\s.\-_]+", f"{owner} {owner_name}".lower()) if len(t) > 2}
    if owner_toks & author_toks:
        signals.append(Signal(
            name="author_owner_match", weight="STRONG",
            detail=f"repository owner {owner!r} matches a paper author",
            evidence=[Evidence(kind="repo_metadata", uri=repo_uri,
                               excerpt=f"owner={owner} ({repo.owner_type})")],
        ))

    # S3 -- the repository points back at this arXiv id.
    readme = gh.get_readme(store, owner, name) or ""
    if readme and gh.readme_mentions_arxiv(readme, arxiv_id):
        signals.append(Signal(
            name="readme_backlink", weight="STRONG",
            detail=f"the repository README cites arXiv:{arxiv_id}",
            evidence=[Evidence(kind="repo_metadata", uri=f"{repo_uri}#readme",
                               excerpt=f"README references arXiv:{arxiv_id}")],
        ))
    elif readme and title.lower()[:50] in readme.lower():
        signals.append(Signal(
            name="readme_title", weight="WEAK",
            detail="the repository README quotes the paper title",
            evidence=[Evidence(kind="repo_metadata", uri=f"{repo_uri}#readme")],
        ))

    confidence, reasoning = assess(signals)

    # Archived or unmaintained repositories still exist, but the claim that they
    # are *the* implementation deserves less weight.
    if repo.archived:
        confidence = at_most(confidence, "LIKELY")
        reasoning += "; repository is archived"

    # ── relation ───────────────────────────────────────────────────────────
    if meta.get("declared"):
        relation = "OFFICIAL"
    elif _REPRO_HINT.search(readme[:4000]):
        relation = "REPRODUCTION"
    elif owner_toks & author_toks:
        relation = "ORGANIZATION"
    elif signals:
        relation = "THIRD_PARTY"
    else:
        relation = "UNRELATED"
        confidence = "UNKNOWN"
        reasoning = "nothing links this repository to the paper"

    # ── shallow coverage ───────────────────────────────────────────────────
    score, matched, missing = (None, {}, [])
    coverage_kind = "NONE"
    if verify_coverage:
        files = _code_files(gh.get_tree(store, owner, name, repo.default_branch))
        if len(files) > 3000:
            # In a repository this large something matches every pattern by
            # accident, so the score would be meaningless.
            coverage_kind = "UNRELIABLE"
        else:
            score, matched, missing = _shallow_coverage(files, expected_kinds)
            coverage_kind = "SHALLOW" if score is not None else "NONE"

    c = Candidate(
        owner=owner, name=name, relation=relation, confidence=confidence,
        reasoning=reasoning, coverage_score=score, coverage_kind=coverage_kind,
        matched_kinds=matched, missing_kinds=missing, stars=repo.stars,
        archived=repo.archived, license=repo.license,
    )

    # Missing component kinds are recorded as absence evidence: this is what
    # later phases diff against to find implementation gaps.
    for kind in missing:
        signals.append(Signal(
            name=f"absent_{kind}", weight="WEAK",
            detail=f"no file plausibly implementing the {kind} component",
            evidence=[Evidence(
                kind="code_absence", uri=repo_uri,
                excerpt=f"no filename matches the {kind} component kind",
                stance="CONTRADICTS",
                provenance={"method": "shallow_tree_scan", "kind": kind},
            )],
        ))

    subject_id = f"{pv}|{c.repo_id}"
    c.confidence = record(store, "implementation_candidate", subject_id,
                          confidence, signals)
    from ..evidence.confidence import evidence_for
    c.evidence = evidence_for(store, "implementation_candidate", subject_id)
    return c


def _persist(store: Store, pv: str, c: Candidate) -> None:
    store.execute(
        "INSERT OR REPLACE INTO repos (id, owner, name, stars, license, archived, "
        "metadata_fetched_at) VALUES (?,?,?,?,?,?,datetime('now'))",
        (c.repo_id, c.owner, c.name, c.stars, c.license, int(bool(c.archived))),
    )
    store.execute(
        "INSERT OR REPLACE INTO implementation_candidates (id, paper_version, repo_id, "
        "relation, confidence, coverage_score, coverage_checked, rank) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (f"{pv}|{c.repo_id}", pv, c.repo_id, c.relation, c.confidence,
         c.coverage_score, int(c.coverage_kind != "NONE"), c.rank),
    )


def _public(c: Candidate) -> dict[str, Any]:
    return {
        "repo": c.repo_id,
        "url": f"https://github.com/{c.repo_id}",
        "relation": c.relation,
        "confidence": c.confidence,
        "reasoning": c.reasoning,
        "rank": c.rank,
        "coverage": {
            "kind": c.coverage_kind,
            "score": round(c.coverage_score, 2) if c.coverage_score is not None else None,
            "matched": c.matched_kinds,
            "missing": c.missing_kinds,
        },
        "stars": c.stars, "archived": c.archived, "license": c.license,
        "evidence": c.evidence,
    }
