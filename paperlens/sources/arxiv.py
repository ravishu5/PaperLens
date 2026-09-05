"""arXiv adapter: metadata over the Atom API, full text over e-print LaTeX.

LaTeX source is the primary path and PDF is a marked fallback (DECISIONS D-002):
source preserves section structure, exact math and the author's own \\url{}
links, all of which PDF text extraction destroys.
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import httpx

from .. import config
from ..graph.store import Store

_ATOM = "http://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom"}

_ID_PATTERNS = [
    re.compile(r"arxiv\.org/(?:abs|pdf|e-print)/(?P<id>\d{4}\.\d{4,5})(?:v(?P<v>\d+))?", re.I),
    re.compile(r"^\s*(?:arxiv:)?(?P<id>\d{4}\.\d{4,5})(?:v(?P<v>\d+))?\s*$", re.I),
    re.compile(r"10\.48550/arxiv\.(?P<id>\d{4}\.\d{4,5})(?:v(?P<v>\d+))?", re.I),
]


def parse_arxiv_id(text: str) -> tuple[str, int | None] | None:
    """Accept a bare id, a versioned id, an arXiv URL, or an arXiv DOI."""
    for pat in _ID_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group("id"), (int(m.group("v")) if m.group("v") else None)
    return None


@dataclass
class ArxivMetadata:
    arxiv_id: str
    version: int
    title: str
    abstract: str
    authors: list[dict]
    published_at: str
    updated_at: str
    doi: str | None


def _throttle(store: Store) -> None:
    wait = store.take_token("arxiv")
    if wait > 0:
        time.sleep(wait)


def fetch_metadata(store: Store, arxiv_id: str) -> ArxivMetadata:
    key = f"arxiv:meta:{arxiv_id}"
    body = store.cache_get(key)
    if body is None:
        _throttle(store)
        r = httpx.get(_ATOM, params={"id_list": arxiv_id, "max_results": 1},
                      headers={"User-Agent": config.USER_AGENT}, timeout=30.0,
                      follow_redirects=True)
        r.raise_for_status()
        body = r.content
        store.cache_put(key, "arxiv", str(r.url), body)

    entry = ET.fromstring(body).find("a:entry", _NS)
    if entry is None:
        raise LookupError(f"arXiv returned no entry for {arxiv_id}")
    txt = lambda tag: (e.text or "").strip() if (e := entry.find(tag, _NS)) is not None else ""
    if not txt("a:id"):
        raise LookupError(f"arXiv returned no entry for {arxiv_id}")

    vm = re.search(r"v(\d+)\s*$", txt("a:id"))
    doi_el = entry.find("{http://arxiv.org/schemas/atom}doi")
    return ArxivMetadata(
        arxiv_id=arxiv_id,
        version=int(vm.group(1)) if vm else 1,
        title=re.sub(r"\s+", " ", txt("a:title")),
        abstract=re.sub(r"\s+", " ", txt("a:summary")),
        authors=[{"name": (n.text or "").strip()}
                 for n in entry.findall("a:author/a:name", _NS)],
        published_at=txt("a:published"),
        updated_at=txt("a:updated"),
        doi=(doi_el.text.strip() if doi_el is not None and doi_el.text else None),
    )


def search(store: Store, query: str, max_results: int = 8) -> list[ArxivMetadata]:
    """Title/abstract search, used by resolve_paper when the input is not an id."""
    _throttle(store)
    r = httpx.get(_ATOM,
                  params={"search_query": f'all:"{query}"', "max_results": max_results,
                          "sortBy": "relevance"},
                  headers={"User-Agent": config.USER_AGENT}, timeout=30.0,
                  follow_redirects=True)
    r.raise_for_status()
    out: list[ArxivMetadata] = []
    for entry in ET.fromstring(r.content).findall("a:entry", _NS):
        txt = lambda tag: (e.text or "").strip() if (e := entry.find(tag, _NS)) is not None else ""
        raw_id = txt("a:id")
        parsed = parse_arxiv_id(raw_id)
        if not parsed:
            continue
        aid, ver = parsed
        out.append(ArxivMetadata(
            arxiv_id=aid, version=ver or 1,
            title=re.sub(r"\s+", " ", txt("a:title")),
            abstract=re.sub(r"\s+", " ", txt("a:summary")),
            authors=[{"name": (n.text or "").strip()}
                     for n in entry.findall("a:author/a:name", _NS)],
            published_at=txt("a:published"), updated_at=txt("a:updated"), doi=None,
        ))
    return out


def fetch_latex(store: Store, arxiv_id: str) -> str | None:
    """Download and flatten the e-print source.

    Delegates to arxiv-to-prompt (D-003), which expands \\input/\\include,
    resolves author macros and strips comments. Comment stripping matters: the
    Transformer paper's only \\label{eq:attention} sits on a commented-out line,
    and two of its \\input files are commented out entirely.
    """
    import arxiv_to_prompt

    _throttle(store)
    try:
        tex = arxiv_to_prompt.process_latex_source(
            arxiv_id,
            keep_comments=False,
            use_cache=True,
            cache_dir=str(config.source_cache_dir()),
            expand_macros_flag=True,
        )
    except Exception:
        return None
    return tex or None
