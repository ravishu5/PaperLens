r"""Outgoing citations: what this paper builds on, and where it says so.

Every arXiv submission carries its own bibliography, so the backward half of a
research lineage needs no citation API at all. Extracting ``\citep{key}`` with the
sentence around it, and resolving ``key`` against the paper's ``.bbl``, gives an
edge that is exact rather than inferred -- including the quotation that justifies
it.

That matters because Semantic Scholar supplies citation *contexts* only for
incoming citations, is rate-limited to the point of unusability without a key,
and does not always resolve preprint-to-preprint edges. Here the paper is the
primary source.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_CITE = re.compile(
    r"\\(citep|citet|citealp|citealt|citeauthor|citeyear|cite)\*?\s*"
    r"(?:\[[^\]]*\]\s*)*\{([^}]+)\}")

_BIBITEM = re.compile(r"\\bibitem(?:\[[^\]]*\])?\s*\{([^}]+)\}")
_ARXIV_IN_BIB = re.compile(r"arxiv[:\s]*(?:preprint\s*)?(?:arxiv:)?\s*(\d{4}\.\d{4,5})", re.I)
_DOI_IN_BIB = re.compile(r"\b(10\.\d{4,9}/[^\s,}]+)")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")


@dataclass
class CitationSite:
    bib_key: str
    command: str
    context: str
    src_line: int
    char_start: int


@dataclass
class BibEntry:
    bib_key: str
    raw: str
    authors: str | None = None
    title: str | None = None
    year: int | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    cite_count: int = 0
    sites: list[CitationSite] = field(default_factory=list)


def _sentence_around(text: str, idx: int, width: int = 260) -> str:
    """The sentence containing this citation -- not merely a nearby one.

    Selecting the first sentence in the window that contained any \\cite
    attached the wrong quotation to a reference whenever two citations sat close
    together, which is most of a related-work section.
    """
    lo, hi = max(0, idx - width), min(len(text), idx + width)
    window = text[lo:hi]
    target = idx - lo
    pos = 0
    for part in re.split(r"(?<=[.!?])\s+", window):
        start, end = pos, pos + len(part)
        if start <= target < end + 1:
            return re.sub(r"\s+", " ", part).strip()
        pos = end + 1
    return re.sub(r"\s+", " ", window).strip()


def extract_citation_sites(tex: str) -> list[CitationSite]:
    out: list[CitationSite] = []
    for m in _CITE.finditer(tex):
        command, keys = m.group(1), m.group(2)
        context = _sentence_around(tex, m.start())
        line = tex.count("\n", 0, m.start()) + 1
        for key in (k.strip() for k in keys.split(",")):
            if key:
                out.append(CitationSite(key, command, context, line, m.start()))
    return out


def _clean(text: str) -> str:
    s = re.sub(r"\\newblock", " ", text)
    s = re.sub(r"\\(emph|textit|textbf|url|href)\s*\{([^}]*)\}", r"\2", s)
    s = re.sub(r"\\[a-zA-Z]+\s*", " ", s)
    s = s.replace("~", " ").replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def parse_bbl(text: str) -> dict[str, BibEntry]:
    r"""Parse a LaTeX ``.bbl``: a sequence of ``\bibitem{key} …`` blocks."""
    entries: dict[str, BibEntry] = {}
    matches = list(_BIBITEM.finditer(text))
    for i, m in enumerate(matches):
        key = m.group(1).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[m.end():end]
        # \newblock separates author / title / venue. Splitting on "." instead
        # breaks on author initials: "Wu, Z., Xiong, Y." became two fields.
        fields = [_clean(f) for f in re.split(r"\\newblock", block)]
        fields = [f for f in fields if f]
        raw = _clean(block)
        arx = _ARXIV_IN_BIB.search(raw)
        doi = _DOI_IN_BIB.search(raw)
        yr = _YEAR.search(raw)
        authors = fields[0] if fields else None
        title = None
        for f in fields[1:]:
            if len(f.split()) >= 3 and not f.lower().startswith("arxiv"):
                title = f.rstrip(".")
                break
        if title is None and len(fields) > 1:
            title = fields[1].rstrip(".")
        entries[key] = BibEntry(
            bib_key=key, raw=raw[:1200], authors=(authors or None),
            title=(title or None), year=int(yr.group(0)) if yr else None,
            arxiv_id=arx.group(1) if arx else None,
            doi=doi.group(1).rstrip(".") if doi else None,
        )
    return entries


def parse_bib(text: str) -> dict[str, BibEntry]:
    """Parse a BibTeX ``.bib`` file well enough to recover identity fields."""
    entries: dict[str, BibEntry] = {}
    for m in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,(.*?)(?=\n@|\Z)", text, re.S):
        key, body = m.group(1).strip(), m.group(2)

        def field_(name: str) -> str | None:
            fm = re.search(rf"{name}\s*=\s*[{{\"]([^}}\"]*)", body, re.I | re.S)
            return _clean(fm.group(1)) if fm else None

        arx = _ARXIV_IN_BIB.search(body)
        eprint = re.search(r"eprint\s*=\s*[{\"]\s*(\d{4}\.\d{4,5})", body, re.I)
        year = field_("year")
        entries[key] = BibEntry(
            bib_key=key, raw=_clean(body)[:1200], authors=field_("author"),
            title=field_("title"),
            year=int(year) if year and year.isdigit() else None,
            arxiv_id=(eprint.group(1) if eprint else (arx.group(1) if arx else None)),
            doi=field_("doi"),
        )
    return entries


def load_bibliography(source_dir: Path) -> dict[str, BibEntry]:
    """Read whichever bibliography the submission shipped: .bbl preferred."""
    entries: dict[str, BibEntry] = {}
    for path in sorted(source_dir.glob("*.bbl")):
        entries.update(parse_bbl(path.read_text(errors="replace")))
    if not entries:
        for path in sorted(source_dir.glob("*.bib")):
            entries.update(parse_bib(path.read_text(errors="replace")))
    return entries


def collect(tex: str, source_dir: Path) -> tuple[dict[str, BibEntry], list[CitationSite]]:
    """Citation sites joined to bibliography entries."""
    sites = extract_citation_sites(tex)
    entries = load_bibliography(source_dir)
    for site in sites:
        entry = entries.get(site.bib_key)
        if entry is None:
            # Cited but not in the bibliography: keep it, marked unresolved.
            entry = entries[site.bib_key] = BibEntry(bib_key=site.bib_key, raw="")
        entry.cite_count += 1
        entry.sites.append(site)
    return entries, sites
