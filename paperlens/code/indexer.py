"""Repository indexing: clone at a SHA, index via the provider, persist.

The commit SHA is the cache key. An index built at a SHA is immutable, so it is
built once and never invalidated by elapsed time (ARCHITECTURE 1.3).

Symbols are persisted lazily. jcodemunch (or the AST backend) is the authority on
the full symbol set; the graph stores only the symbols something actually
references, which keeps the database small and avoids mirroring another index.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..graph.store import Store, utcnow
from .clone import ensure_checkout
from .provider import CodeIntelligenceProvider, IndexResult, Symbol, get_provider

_REPO_RE = re.compile(
    r"^(?:https?://github\.com/)?([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+?)(?:\.git)?/?$")

_provider: CodeIntelligenceProvider | None = None


def provider() -> CodeIntelligenceProvider:
    global _provider
    if _provider is None:
        _provider = get_provider()
    return _provider


def parse_repo(text: str) -> tuple[str, str]:
    m = _REPO_RE.match(text.strip())
    if not m:
        raise ValueError(f"{text!r} is not an owner/repo identifier")
    return m.group(1), m.group(2)


@dataclass
class IndexSummary:
    repo: str
    snapshot_id: str
    commit_sha: str
    indexer: str
    symbol_count: int
    file_count: int
    languages: dict[str, int] = field(default_factory=dict)
    cached: bool = False
    warnings: list[str] = field(default_factory=list)


def index_repository(store: Store, repo: str, ref: str = "HEAD",
                     force: bool = False) -> IndexSummary:
    owner, name = parse_repo(repo)
    repo_id = f"{owner}/{name}"
    checkout = ensure_checkout(owner, name, ref)
    snapshot_id = f"{repo_id}@{checkout.commit_sha}"

    existing = store.one("SELECT * FROM repo_snapshots WHERE id = ?", (snapshot_id,))
    if existing and not force and existing["indexed_at"]:
        # Re-index in the backend anyway when it has no memory of this repo
        # (a fresh process, for the in-memory AST backend).
        _ensure_backend_knows(repo_id, checkout.path)
        return IndexSummary(
            repo=repo_id, snapshot_id=snapshot_id, commit_sha=checkout.commit_sha,
            indexer=existing["indexer"], symbol_count=existing["symbol_count"] or 0,
            file_count=existing["file_count"] or 0, cached=True,
        )

    result: IndexResult = provider().index(repo_id, str(checkout.path))

    store.execute(
        "INSERT OR IGNORE INTO repos (id, owner, name) VALUES (?,?,?)",
        (repo_id, owner, name))
    store.execute(
        "INSERT OR REPLACE INTO repo_snapshots (id, repo_id, commit_sha, indexed_at, "
        "indexer, symbol_count, file_count, clone_path) VALUES (?,?,?,?,?,?,?,?)",
        (snapshot_id, repo_id, checkout.commit_sha, utcnow(), result.indexer,
         result.symbol_count, result.file_count, str(checkout.path)))
    store.commit()

    return IndexSummary(
        repo=repo_id, snapshot_id=snapshot_id, commit_sha=checkout.commit_sha,
        indexer=result.indexer, symbol_count=result.symbol_count,
        file_count=result.file_count, languages=result.languages,
        warnings=result.warnings,
    )


_backend_seen: set[str] = set()


def _ensure_backend_knows(repo_id: str, path: Any) -> None:
    if repo_id in _backend_seen:
        return
    try:
        provider().index(repo_id, str(path))
    except Exception:
        pass
    _backend_seen.add(repo_id)


def latest_snapshot(store: Store, repo: str):
    owner, name = parse_repo(repo)
    return store.one(
        "SELECT * FROM repo_snapshots WHERE repo_id = ? ORDER BY indexed_at DESC LIMIT 1",
        (f"{owner}/{name}",))


def require_snapshot(store: Store, repo: str):
    snap = latest_snapshot(store, repo)
    if snap is None:
        owner, name = parse_repo(repo)
        raise LookupError(
            f"{owner}/{name} has not been indexed. Call index_repository first.")
    _ensure_backend_knows(snap["repo_id"], snap["clone_path"])
    return snap


def persist_symbol(store: Store, snapshot_id: str, sym: Symbol) -> str:
    """Store a symbol on first reference and return its stable id."""
    sid = f"{snapshot_id}:{sym.qualified_name}"
    store.execute(
        "INSERT OR REPLACE INTO symbols (id, snapshot_id, qualified_name, kind, "
        "file_path, line_start, line_end, signature) VALUES (?,?,?,?,?,?,?,?)",
        (sid, snapshot_id, sym.qualified_name, sym.kind, sym.file_path,
         sym.line_start, sym.line_end, sym.signature))
    return sid


def find_symbol(store: Store, repo: str, qualified_name: str) -> Symbol | None:
    snap = require_snapshot(store, repo)
    repo_id = snap["repo_id"]
    file_path = qualified_name.split("::", 1)[0] if "::" in qualified_name else None
    if file_path:
        for s in provider().file_outline(repo_id, file_path):
            if s.qualified_name == qualified_name:
                return s
    res = provider().search_symbols(repo_id, qualified_name.rsplit("::", 1)[-1])
    return next((s for s in res.symbols if s.qualified_name == qualified_name), None)
