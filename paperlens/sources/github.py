"""GitHub adapter.

Credentials are discovered opportunistically: GITHUB_TOKEN/GH_TOKEN, then the
`gh` CLI's token, then anonymous. Anonymous works but is capped at 60 requests
an hour, so the caller is told when it is running unauthenticated rather than
mysteriously hitting a wall.
"""
from __future__ import annotations

import base64
import functools
import os
import re
import subprocess
import time
from dataclasses import dataclass

import httpx

from .. import config
from ..graph.store import Store

_API = "https://api.github.com"
_DAY = 86_400


@functools.lru_cache(maxsize=1)
def token() -> str | None:
    for var in ("GITHUB_TOKEN", "GH_TOKEN", "PAPERLENS_GITHUB_TOKEN"):
        if v := os.environ.get(var):
            return v.strip()
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True,
                             text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def authenticated() -> bool:
    return token() is not None


def _headers() -> dict[str, str]:
    h = {"User-Agent": config.USER_AGENT, "Accept": "application/vnd.github+json"}
    if t := token():
        h["Authorization"] = f"Bearer {t}"
    return h


def _get(store: Store, path: str, *, params: dict | None = None,
         ttl: int | None = _DAY) -> dict | list | None:
    from datetime import datetime, timedelta, timezone

    key = f"github:{path}:{sorted((params or {}).items())}"
    if (cached := store.cache_get(key)) is not None:
        import json
        return json.loads(cached)

    wait = store.take_token("github")
    if wait > 0:
        time.sleep(min(wait, 5.0))
    url = path if path.startswith("http") else f"{_API}{path}"
    try:
        r = httpx.get(url, headers=_headers(), params=params, timeout=30.0,
                      follow_redirects=True)
    except httpx.HTTPError:
        return None
    if r.status_code == 404:
        return None
    if r.status_code in (403, 429):
        return None  # rate limited; caller degrades to UNKNOWN rather than failing
    if r.status_code >= 400:
        return None
    expires = (datetime.now(timezone.utc) + timedelta(seconds=ttl)).isoformat() if ttl else None
    store.cache_put(key, "github", str(r.url), r.content, expires)
    return r.json()


@dataclass
class Repo:
    owner: str
    name: str
    full_name: str
    description: str | None
    stars: int
    forks: int
    archived: bool
    license: str | None
    pushed_at: str | None
    default_branch: str
    owner_type: str          # "User" or "Organization"
    owner_name: str | None   # the owner's display name, for author matching
    homepage: str | None


def get_repo(store: Store, owner: str, name: str) -> Repo | None:
    d = _get(store, f"/repos/{owner}/{name}")
    if not isinstance(d, dict):
        return None
    o = d.get("owner") or {}
    return Repo(
        owner=o.get("login", owner), name=d["name"], full_name=d["full_name"],
        description=d.get("description"), stars=d.get("stargazers_count", 0),
        forks=d.get("forks_count", 0), archived=bool(d.get("archived")),
        license=((d.get("license") or {}) or {}).get("spdx_id"),
        pushed_at=d.get("pushed_at"), default_branch=d.get("default_branch", "main"),
        owner_type=o.get("type", "User"), owner_name=None,
        homepage=d.get("homepage"),
    )


def get_owner_name(store: Store, login: str) -> str | None:
    d = _get(store, f"/users/{login}")
    return d.get("name") if isinstance(d, dict) else None


def get_readme(store: Store, owner: str, name: str) -> str | None:
    d = _get(store, f"/repos/{owner}/{name}/readme")
    if not isinstance(d, dict) or "content" not in d:
        return None
    try:
        return base64.b64decode(d["content"]).decode("utf-8", errors="replace")
    except Exception:
        return None


def get_tree(store: Store, owner: str, name: str, ref: str = "HEAD") -> list[str]:
    """Full recursive file list in one API call. Cheap shallow-coverage input."""
    d = _get(store, f"/repos/{owner}/{name}/git/trees/{ref}", params={"recursive": "1"})
    if not isinstance(d, dict):
        return []
    return [e["path"] for e in d.get("tree", []) if e.get("type") == "blob"]


def search_repos(store: Store, query: str, limit: int = 10) -> list[Repo]:
    d = _get(store, "/search/repositories",
             params={"q": query, "sort": "stars", "order": "desc",
                     "per_page": min(limit, 30)}, ttl=_DAY)
    if not isinstance(d, dict):
        return []
    out: list[Repo] = []
    for item in d.get("items", [])[:limit]:
        o = item.get("owner") or {}
        out.append(Repo(
            owner=o.get("login", ""), name=item["name"], full_name=item["full_name"],
            description=item.get("description"), stars=item.get("stargazers_count", 0),
            forks=item.get("forks_count", 0), archived=bool(item.get("archived")),
            license=((item.get("license") or {}) or {}).get("spdx_id"),
            pushed_at=item.get("pushed_at"),
            default_branch=item.get("default_branch", "main"),
            owner_type=o.get("type", "User"), owner_name=None,
            homepage=item.get("homepage"),
        ))
    return out


_ARXIV_IN_TEXT = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})|arxiv[:\s]+(\d{4}\.\d{4,5})", re.I)


def readme_mentions_arxiv(readme: str, arxiv_id: str) -> bool:
    return any(arxiv_id in (m.group(1) or m.group(2) or "")
               for m in _ARXIV_IN_TEXT.finditer(readme))
