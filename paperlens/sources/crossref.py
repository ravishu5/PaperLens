"""Crossref: metadata and reference lists for papers that are not on arXiv.

Most medical-imaging work is published in journals rather than posted as
preprints, and a closed-access article has no LaTeX source and no readable PDF.
What it does have is a registered reference list -- Crossref returns 53 entries
with titles and DOIs for a paper whose full text is behind a paywall -- which is
enough to place it in a research lineage even when nothing can be said about its
method.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from .. import config
from ..graph.store import Store

_BASE = "https://api.crossref.org"
_TTL_DAYS = 30

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>]+)", re.I)


def parse_doi(text: str) -> str | None:
    """Accept a bare DOI, a doi: prefix, or a doi.org URL."""
    t = (text or "").strip()
    t = re.sub(r"^https?://(dx\.)?doi\.org/", "", t, flags=re.I)
    t = re.sub(r"^doi:\s*", "", t, flags=re.I)
    m = DOI_RE.match(t) or DOI_RE.search(t)
    return m.group(1).rstrip(".,;)") if m else None


class CrossrefUnavailable(RuntimeError):
    pass


@dataclass
class Reference:
    key: str
    title: str | None
    doi: str | None
    year: int | None
    raw: str


@dataclass
class Work:
    doi: str
    title: str
    authors: list[dict]
    venue: str | None
    year: int | None
    published_at: str | None
    abstract: str | None
    references: list[Reference] = field(default_factory=list)


def _get(store: Store, path: str) -> dict | None:
    key = f"crossref:{path}"
    if (cached := store.cache_get(key)) is not None:
        return json.loads(cached)
    wait = store.take_token("crossref")
    if wait > 0:
        time.sleep(min(wait, 5.0))
    try:
        r = httpx.get(f"{_BASE}{path}", headers={"User-Agent": config.USER_AGENT},
                      timeout=30.0, follow_redirects=True)
    except httpx.HTTPError as exc:
        raise CrossrefUnavailable(f"network error: {exc}") from exc
    if r.status_code == 404:
        return None
    if r.status_code >= 400:
        raise CrossrefUnavailable(f"HTTP {r.status_code}")
    expires = (datetime.now(timezone.utc) + timedelta(days=_TTL_DAYS)).isoformat()
    store.cache_put(key, "crossref", str(r.url), r.content, expires)
    return r.json()


def _strip_jats(text: str | None) -> str | None:
    if not text:
        return None
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip() or None


def work(store: Store, doi: str) -> Work | None:
    data = _get(store, f"/works/{doi}")
    if not data or "message" not in data:
        return None
    m = data["message"]
    issued = (m.get("issued") or {}).get("date-parts") or [[None]]
    year = issued[0][0] if issued and issued[0] else None
    refs: list[Reference] = []
    for i, r in enumerate(m.get("reference") or []):
        title = (r.get("article-title") or r.get("volume-title")
                 or r.get("unstructured"))
        yr = r.get("year")
        refs.append(Reference(
            key=r.get("key") or f"ref{i}",
            title=re.sub(r"\s+", " ", title).strip() if title else None,
            doi=(r.get("DOI") or "").lower() or None,
            year=int(yr) if yr and str(yr).isdigit() else None,
            raw=json.dumps(r)[:1200],
        ))
    return Work(
        doi=doi.lower(),
        title=(m.get("title") or ["(untitled)"])[0],
        authors=[{"name": f"{a.get('given','')} {a.get('family','')}".strip()}
                 for a in (m.get("author") or [])],
        venue=(m.get("container-title") or [None])[0],
        year=year,
        published_at=("-".join(str(p) for p in issued[0]) if issued and issued[0] and issued[0][0] else None),
        abstract=_strip_jats(m.get("abstract")),
        references=refs,
    )


def search(store: Store, query: str, rows: int = 8) -> list[dict[str, Any]]:
    from urllib.parse import quote

    data = _get(store, f"/works?query.bibliographic={quote(query)}&rows={rows}")
    out = []
    for m in ((data or {}).get("message") or {}).get("items", []):
        out.append({"doi": (m.get("DOI") or "").lower(),
                    "title": (m.get("title") or ["(untitled)"])[0],
                    "venue": (m.get("container-title") or [None])[0],
                    "year": ((m.get("issued") or {}).get("date-parts") or [[None]])[0][0],
                    "authors": [f"{a.get('given','')} {a.get('family','')}".strip()
                                for a in (m.get("author") or [])][:6]})
    return out
