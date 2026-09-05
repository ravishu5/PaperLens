"""OpenAlex: the fallback citation graph.

Works without authentication and did not throttle during Phase 0 evaluation,
which makes it the right backstop when Semantic Scholar is rate limited. What it
does not provide is citation *contexts* -- so lineage built on OpenAlex alone
carries structure without quotation, and says so.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from .. import config
from ..graph.store import Store

_BASE = "https://api.openalex.org"
_TTL_DAYS = 30
_MAILTO = "itsshankar.ravi2@gmail.com"   # polite pool


class OpenAlexUnavailable(RuntimeError):
    pass


def _get(store: Store, path: str, params: dict[str, Any]) -> dict | None:
    params = {**params, "mailto": _MAILTO}
    key = f"openalex:{path}:{sorted(params.items())}"
    if (cached := store.cache_get(key)) is not None:
        return json.loads(cached)

    wait = store.take_token("openalex")
    if wait > 0:
        time.sleep(min(wait, 5.0))
    try:
        r = httpx.get(f"{_BASE}{path}", params=params,
                      headers={"User-Agent": config.USER_AGENT}, timeout=30.0)
    except httpx.HTTPError as exc:
        raise OpenAlexUnavailable(f"network error: {exc}") from exc
    if r.status_code == 404:
        return None
    if r.status_code >= 400:
        raise OpenAlexUnavailable(f"HTTP {r.status_code}")
    expires = (datetime.now(timezone.utc) + timedelta(days=_TTL_DAYS)).isoformat()
    store.cache_put(key, "openalex", str(r.url), r.content, expires)
    return r.json()


def _arxiv_of(work: dict) -> str | None:
    for loc in (work.get("locations") or []):
        url = ((loc.get("landing_page_url") or "") + " "
               + (loc.get("pdf_url") or ""))
        if "arxiv.org/abs/" in url:
            return url.split("arxiv.org/abs/")[1].split()[0].split("v")[0]
    doi = (work.get("doi") or "")
    if "arxiv." in doi:
        return doi.rsplit("arxiv.", 1)[-1]
    return None


def work_for_arxiv(store: Store, arxiv_id: str) -> dict | None:
    return _get(store, f"/works/doi:10.48550/arXiv.{arxiv_id}", {})


def citing_works(store: Store, arxiv_id: str, limit: int = 100) -> list[dict]:
    """Papers citing this one. Structure only -- OpenAlex has no contexts."""
    work = work_for_arxiv(store, arxiv_id)
    if not work:
        return []
    data = _get(store, "/works", {"filter": f"cites:{work['id'].rsplit('/', 1)[-1]}",
                                  "per-page": min(limit, 200),
                                  "sort": "cited_by_count:desc"})
    out = []
    for w in (data or {}).get("results", []):
        out.append({"title": w.get("title") or w.get("display_name"),
                    "year": w.get("publication_year"),
                    "arxiv_id": _arxiv_of(w),
                    "cited_by_count": w.get("cited_by_count"),
                    "openalex_id": w.get("id")})
    return out
