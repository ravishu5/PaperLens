"""Semantic Scholar Academic Graph.

S2's distinguishing value is `contexts` and `intents`: for an incoming citation
it returns the actual sentences in which the citing paper refers to this one.
That turns a lineage claim into a quotable one, and nothing else offers it.

The cost is access. Unauthenticated requests share one global pool and return
HTTP 429 almost immediately (observed on the second call during Phase 0), while
an API key grants a guaranteed 1 request/second. Every call here is cached for
30 days and every failure degrades to a stated absence rather than an exception.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from .. import config
from ..graph.store import Store

_BASE = "https://api.semanticscholar.org/graph/v1"
_TTL_DAYS = 30


def api_key() -> str | None:
    for var in ("S2_API_KEY", "SEMANTIC_SCHOLAR_API_KEY", "PAPERLENS_S2_API_KEY"):
        if v := os.environ.get(var):
            return v.strip()
    return None


def authenticated() -> bool:
    return api_key() is not None


class S2Unavailable(RuntimeError):
    """S2 could not be reached, or refused the request."""


@dataclass
class CitingPaper:
    paper_id: str
    title: str
    year: int | None
    arxiv_id: str | None
    is_influential: bool
    intents: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    citation_count: int | None = None


def _get(store: Store, path: str, params: dict[str, Any]) -> dict | None:
    key = f"s2:{path}:{sorted(params.items())}"
    if (cached := store.cache_get(key)) is not None:
        return json.loads(cached)

    wait = store.take_token("semanticscholar")
    if wait > 0:
        time.sleep(min(wait, 5.0))

    headers = {"User-Agent": config.USER_AGENT}
    if k := api_key():
        headers["x-api-key"] = k
    try:
        r = httpx.get(f"{_BASE}{path}", params=params, headers=headers, timeout=30.0)
    except httpx.HTTPError as exc:
        raise S2Unavailable(f"network error: {exc}") from exc
    if r.status_code == 429:
        raise S2Unavailable(
            "rate limited (HTTP 429). Unauthenticated access shares one global "
            "pool; set S2_API_KEY for a guaranteed 1 request/second."
        )
    if r.status_code == 404:
        return None
    if r.status_code >= 400:
        raise S2Unavailable(f"HTTP {r.status_code}")

    expires = (datetime.now(timezone.utc) + timedelta(days=_TTL_DAYS)).isoformat()
    store.cache_put(key, "semanticscholar", str(r.url), r.content, expires)
    return r.json()


def _to_citing(node: dict, wrapper: dict) -> CitingPaper:
    ext = node.get("externalIds") or {}
    return CitingPaper(
        paper_id=node.get("paperId") or "",
        title=node.get("title") or "",
        year=node.get("year"),
        arxiv_id=ext.get("ArXiv"),
        is_influential=bool(wrapper.get("isInfluential")),
        intents=list(wrapper.get("intents") or []),
        contexts=list(wrapper.get("contexts") or []),
        citation_count=node.get("citationCount"),
    )


def citations(store: Store, arxiv_id: str, limit: int = 100) -> list[CitingPaper]:
    """Papers citing this one, with the sentences in which they cite it."""
    data = _get(store, f"/paper/arXiv:{arxiv_id}/citations", {
        "fields": "title,year,externalIds,citationCount,intents,isInfluential,contexts",
        "limit": min(limit, 1000)})
    if not data:
        return []
    return [_to_citing(d.get("citingPaper") or {}, d) for d in data.get("data", [])]


def references(store: Store, arxiv_id: str, limit: int = 100) -> list[CitingPaper]:
    data = _get(store, f"/paper/arXiv:{arxiv_id}/references", {
        "fields": "title,year,externalIds,citationCount,intents,isInfluential,contexts",
        "limit": min(limit, 1000)})
    if not data:
        return []
    return [_to_citing(d.get("citedPaper") or {}, d) for d in data.get("data", [])]


def paper(store: Store, arxiv_id: str) -> dict | None:
    return _get(store, f"/paper/arXiv:{arxiv_id}",
                {"fields": "title,year,externalIds,citationCount,"
                           "influentialCitationCount,referenceCount"})
