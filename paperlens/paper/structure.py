"""Section hierarchy, algorithms, stated values and declared URLs."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .latex import iter_environments, line_of

_HEADING = re.compile(
    r"\\(section|subsection|subsubsection|paragraph)(\*)?\s*\{", re.M
)
_APPENDIX = re.compile(r"\\appendix\b")
_LEVELS = {"section": 1, "subsection": 2, "subsubsection": 3, "paragraph": 4}


def _balanced(text: str, open_at: int) -> tuple[str, int]:
    """Read a balanced {...} group starting at the '{' index."""
    depth, i, buf = 0, open_at, []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            if depth == 1:
                i += 1
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(buf), i + 1
        buf.append(ch)
        i += 1
    return "".join(buf), len(text)


def _clean_title(raw: str) -> str:
    s = re.sub(r"\\label\{[^}]*\}", "", raw)
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    return re.sub(r"[{}]", "", s).strip()


def slugify(text: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:maxlen] or "untitled"


@dataclass
class Section:
    section_path: str
    level: int
    title: str
    latex_label: str | None
    body: str
    src_line_start: int
    src_line_end: int
    ordinal: int
    char_start: int
    char_end: int


def extract_sections(tex: str) -> list[Section]:
    """Replay LaTeX's section counters, including the \\appendix switch.

    Starred headings do not increment the counter, so they get a slug-based path
    rather than a number -- which is correct: they have no number in the PDF
    either.
    """
    appendix_at = (m.start() if (m := _APPENDIX.search(tex)) else None)

    heads: list[tuple[int, int, str, bool, str | None, int]] = []
    for m in _HEADING.finditer(tex):
        kind, star = m.group(1), bool(m.group(2))
        title_raw, after = _balanced(tex, m.end() - 1)
        label = (lm.group(1) if (lm := re.search(r"\\label\{([^}]*)\}", title_raw)) else None)
        heads.append((m.start(), _LEVELS[kind], _clean_title(title_raw), star, label, after))

    counters = [0, 0, 0, 0]
    in_appendix = False
    out: list[Section] = []
    for ordinal, (start, level, title, star, label, body_from) in enumerate(heads):
        body_to = heads[ordinal + 1][0] if ordinal + 1 < len(heads) else len(tex)
        if appendix_at is not None and start > appendix_at and not in_appendix:
            # \appendix resets the section counter, so the first appendix
            # section is A -- not a letter continuing the body numbering.
            in_appendix = True
            counters = [0, 0, 0, 0]

        # \paragraph is a run-in heading with no number in the standard
        # paper classes, so numbering it would invent a path the PDF does
        # not have. Treat it like a starred heading.
        if star or level == 4:
            path = f"*{slugify(title, 40)}"
        else:
            counters[level - 1] += 1
            for j in range(level, 4):
                counters[j] = 0
            top = counters[0]
            head = chr(ord("A") + top - 1) if in_appendix else str(top)
            path = ".".join([head] + [str(counters[j]) for j in range(1, level)])

        # A trailing \label immediately after the heading belongs to the section.
        body = tex[body_from:body_to]
        if label is None:
            lm = re.match(r"\s*\\label\{([^}]*)\}", body)
            if lm:
                label = lm.group(1)

        out.append(
            Section(
                section_path=path, level=level, title=title, latex_label=label,
                body=body.strip(),
                src_line_start=line_of(tex, start), src_line_end=line_of(tex, body_to),
                ordinal=ordinal, char_start=start, char_end=body_to,
            )
        )
    return out


# ── algorithms ────────────────────────────────────────────────────────────
_ALGO_ENVS = {"algorithm", "algorithmic", "algorithm2e", "lstlisting", "verbatim",
              "minted", "Verbatim"}
_FIG_ENVS = {"figure", "figure*"}
_PSEUDO_HINT = re.compile(r"pseudo\s*-?code|algorithm", re.I)


@dataclass
class Algorithm:
    slug: str
    name: str | None
    body: str | None
    presentation: str
    extractable: bool
    src_line: int
    char_start: int


def extract_algorithms(tex: str) -> list[Algorithm]:
    """Find algorithm-bearing blocks.

    Includes figures whose caption calls itself pseudocode but whose content is
    an included image -- CLIP's core training algorithm is exactly this, a
    `pseudocode.pdf` graphic. Those are recorded with extractable=False so the
    limitation is visible rather than silently missing.
    """
    out: list[Algorithm] = []
    seen: set[str] = set()

    def add(a: Algorithm) -> None:
        base, n = a.slug, 2
        while a.slug in seen:
            a.slug = f"{base}-{n}"
            n += 1
        seen.add(a.slug)
        out.append(a)

    for b in iter_environments(tex, _ALGO_ENVS):
        cap = re.search(r"\\caption\{(.+?)\}\s*$", b.body, re.S | re.M)
        name = _clean_title(cap.group(1)) if cap else None
        add(Algorithm(
            slug=slugify(name or f"{b.name}-l{b.line}"), name=name, body=b.body.strip(),
            presentation="latex_env" if b.name.startswith("algorithm") else "listing",
            extractable=True, src_line=b.line, char_start=b.start,
        ))

    for b in iter_environments(tex, _FIG_ENVS):
        cap = re.search(r"\\caption\{(.+?)\}", b.body, re.S)
        caption = _clean_title(cap.group(1)) if cap else ""
        if not _PSEUDO_HINT.search(caption):
            continue
        if not re.search(r"\\includegraphics", b.body):
            continue
        add(Algorithm(
            slug=slugify(caption or f"figure-l{b.line}"), name=caption or None,
            body=None, presentation="figure_image", extractable=False,
            src_line=b.line, char_start=b.start,
        ))
    return out


# ── stated values ─────────────────────────────────────────────────────────
_NUM = r"(?:\d+(?:\.\d+)?(?:\s*[eE]\s*-?\d+)?|\d*\.\d+)"
_VALUE_PATTERNS = [
    # "$\tau$ was initialized to the equivalent of 0.07"
    re.compile(r"\$?\\?([A-Za-z][A-Za-z0-9_\\{}^]{0,24})\$?\s*(?:was|is|were|are)?\s*"
               r"(?:initiali[sz]ed|set)\s+to\s+(?:the\s+equivalent\s+of\s+)?"
               r"(" + _NUM + r")", re.I),
    # "d_model = 512", "\beta_2 = 0.98"
    re.compile(r"\$?\\?([A-Za-z][A-Za-z0-9_\\{}^]{0,24})\$?\s*=\s*(" + _NUM + r")"),
    # "a learning rate of 1e-4", "dropout of 0.1"
    re.compile(r"\b([a-z][a-z\s\-]{2,30}?)\s+of\s+(" + _NUM + r")\b", re.I),
]
_STOP = {"one", "two", "table", "figure", "section", "equation", "which",
         "that", "the",
         # Verbs that introduce a count in prose: "consists of 30 subjects".
         "consists", "comprises", "contains", "includes", "uses", "requires",
         "consisting", "containing", "including", "total", "set", "number"}
# Function words that mark a match as prose rather than a named quantity.
_FUNCTION_WORDS = {"to", "the", "a", "an", "and", "or", "is", "was", "were", "are",
                   "be", "been", "that", "which", "from", "with", "by", "for",
                   "in", "on", "at", "as", "it", "we", "our", "this", "these"}
# Typesetting options, not model quantities. These come from constructs like
# \includegraphics[width=1.0\columnwidth] and tabular column specs.
_LAYOUT_KEYWORDS = {"width", "height", "scale", "trim", "clip", "angle", "columnwidth",
                    "textwidth", "linewidth", "columnsep", "vspace", "hspace",
                    "baselinestretch", "arraystretch", "tabcolsep", "fboxrule",
                    "hsize", "vsize", "parskip", "parindent", "col", "row", "pt", "em",
                    "aboverulesep", "belowrulesep", "heavyrulewidth",
                    "lightrulewidth", "cmidrulewidth", "out"}


# Residue of LaTeX markup rather than a quantity: "\begin{adjustbox}{width=0.85}"
# collapses to "beginadjustboxwidth", and "$2\times2\times4$" to
# "times2times2times4". Both produced confident absence claims about literals no
# repository would ever contain.
_MARKUP_PREFIX = re.compile(r"^(begin|end|adjustbox|includegraphics|multirow|"
                            r"multicolumn|cmidrule|midrule|toprule|bottomrule)", re.I)
_REPEATED_MACRO = re.compile(r"(times|frac|cdot|quad|hspace|vspace).*\1", re.I)


def _plausible_identifier(symbol: str) -> bool:
    """Could this plausibly be a name a paper gives a quantity?"""
    if _MARKUP_PREFIX.match(symbol) or _REPEATED_MACRO.search(symbol):
        return False
    # A single run of 19+ characters with no separator is markup residue, not a
    # symbol an author would write.
    return not (len(symbol) > 18 and not any(c in symbol for c in "_ -"))


@dataclass
class StatedValue:
    symbol: str | None
    value_text: str
    value_num: float | None
    context: str
    src_line: int
    char_start: int


def extract_stated_values(tex: str, max_values: int = 400) -> list[StatedValue]:
    """Numeric constants stated in prose.

    These are the highest-precision anchors available for paper<->code
    correlation: an exact numeric match against a literal in source is
    machine-checkable, which is what CONFIRMED confidence requires.
    """
    out: list[StatedValue] = []
    seen: set[tuple[str | None, str]] = set()
    for para_start, para in _iter_paragraphs(tex):
        for pat in _VALUE_PATTERNS:
            for m in pat.finditer(para):
                sym = re.sub(r"[\\${}]", "", m.group(1)).strip()
                if not sym or sym.lower() in _STOP or len(sym) > 30:
                    continue
                words = sym.lower().split()
                if len(words) > 1 and any(w in _FUNCTION_WORDS for w in words):
                    continue  # prose, not a named quantity
                if words[-1] in _LAYOUT_KEYWORDS:
                    continue  # typesetting option, not a model quantity
                if not _plausible_identifier(sym):
                    continue
                val = m.group(2).replace(" ", "")
                key = (sym.lower(), val)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    num = float(val)
                except ValueError:
                    num = None
                out.append(StatedValue(
                    symbol=sym, value_text=val, value_num=num,
                    context=_sentence_around(para, m.start()),
                    src_line=line_of(tex, para_start + m.start()),
                    char_start=para_start + m.start(),
                ))
                if len(out) >= max_values:
                    return out
    return out


def _iter_paragraphs(tex: str):
    pos = 0
    for chunk in re.split(r"\n\s*\n", tex):
        yield pos, chunk
        pos += len(chunk) + 2


def _sentence_around(text: str, idx: int, width: int = 240) -> str:
    lo = max(0, idx - width)
    hi = min(len(text), idx + width)
    seg = text[lo:hi]
    return re.sub(r"\s+", " ", seg).strip()


# ── declared URLs (the implementation-discovery signal) ───────────────────
_URL = re.compile(
    r"(?:\\url\{|\\href\{)?(https?://[^\s{}\\,;)\]]+|(?<![\w/])github\.com/[^\s{}\\,;)\]]+)"
)
_GH = re.compile(r"github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)", re.I)


@dataclass
class DeclaredURL:
    url: str
    host: str
    owner: str | None
    repo: str | None
    context: str
    in_abstract: bool
    src_line: int
    char_start: int


def extract_declared_urls(tex: str) -> list[DeclaredURL]:
    """URLs the authors put in their own paper.

    Verified 2/2 on the test corpus: this recovers the official implementation
    for both CLIP and the Transformer paper. See RESEARCH.md section 3 -- since
    Papers With Code shut down, this is the highest-precision paper->repo signal
    that still exists, and it comes from the primary source.
    """
    abs_span = _abstract_span(tex)
    out: list[DeclaredURL] = []
    seen: set[str] = set()
    for m in _URL.finditer(tex):
        raw = m.group(1).rstrip(".,;:)")
        if raw.startswith("github.com"):
            raw = "https://" + raw
        # Trim a trailing brace-escape artefact from \url{...}
        raw = raw.split("}")[0]
        if raw in seen:
            continue
        seen.add(raw)
        gh = _GH.search(raw)
        host = re.sub(r"^https?://", "", raw).split("/")[0].lower()
        repo = gh.group(2).removesuffix(".git") if gh else None
        out.append(DeclaredURL(
            url=raw, host=host,
            owner=gh.group(1) if gh else None, repo=repo,
            context=_sentence_around(tex, m.start()),
            in_abstract=bool(abs_span and abs_span[0] <= m.start() <= abs_span[1]),
            src_line=line_of(tex, m.start()), char_start=m.start(),
        ))
    return out


def _abstract_span(tex: str) -> tuple[int, int] | None:
    blocks = iter_environments(tex, {"abstract"})
    return (blocks[0].start, blocks[0].end) if blocks else None
