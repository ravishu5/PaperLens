"""Equation extraction with LaTeX counter simulation.

Why this module exists: equation *numbers* do not appear in LaTeX source. They
are assigned by a counter at compile time, and in practice authors rarely
\\label their equations -- "Attention Is All You Need" numbers three equations
and labels none of them. So a reference to "Equation 1" cannot be resolved by
searching the source; the counter has to be replayed over the flattened
document. See DECISIONS.md D-004.

Identity is therefore the content hash. The derived number is a best-effort
attribute that carries its own confidence and may legitimately be UNKNOWN.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .latex import EnvBlock, iter_environments, line_of, normalize_math

# Environments that consume equation numbers.
_SINGLE = {"equation", "multline"}                       # exactly one number
_PER_ROW = {"align", "gather", "eqnarray", "alignat", "flalign"}  # one per row
NUMBERED_ENVS = _SINGLE | _PER_ROW
ALL_MATH_ENVS = NUMBERED_ENVS | {e + "*" for e in NUMBERED_ENVS}

_NONUM = re.compile(r"\\(nonumber|notag)\b")
_TAG = re.compile(r"\\tag\*?\{")
# Constructs that change the numbering scheme away from plain arabic.
_SCHEME_CHANGED = re.compile(
    r"\\numberwithin\s*\{\s*equation\s*\}|"
    r"\\renewcommand\s*\{?\s*\\theequation\s*\}?|"
    r"\\def\s*\\theequation|"
    r"\\counterwithin\s*\{\s*equation\s*\}"
)


@dataclass
class Equation:
    content_hash: str
    latex: str
    environment: str
    is_numbered: bool
    derived_number: str | None
    number_conf: str
    latex_label: str | None
    src_line: int
    ordinal: int
    char_start: int


def _split_rows(body: str) -> list[str]:
    r"""Split an align/gather body on top-level \\ row separators."""
    rows, depth, buf, i = [], 0, [], 0
    while i < len(body):
        ch = body[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        if body.startswith("\\\\", i) and depth == 0:
            rows.append("".join(buf))
            buf = []
            i += 2
            # consume an optional [len] spacing argument
            m = re.match(r"\s*\[[^\]]*\]", body[i:])
            if m:
                i += m.end()
            continue
        buf.append(ch)
        i += 1
    rows.append("".join(buf))
    return [r for r in rows if r.strip()]


def _hash(latex: str) -> str:
    return hashlib.sha256(normalize_math(latex).encode("utf-8")).hexdigest()[:12]


def extract_equations(tex: str) -> list[Equation]:
    """Walk display-math environments in document order, replaying the counter.

    Inline math ($...$) is intentionally not captured: CLIP alone has 91
    occurrences, none of them addressable as "Equation N", so they would be
    noise rather than anchors.
    """
    scheme_changed = bool(_SCHEME_CHANGED.search(tex))
    blocks: list[EnvBlock] = iter_environments(tex, ALL_MATH_ENVS)

    # \[ ... \] display math is never numbered, but is a legitimate anchor.
    unnumbered_display = [
        (m.start(), m.group(1)) for m in re.finditer(r"\\\[(.+?)\\\]", tex, re.S)
    ]

    items: list[tuple[int, str, str, int]] = []  # (offset, env, latex, line)
    for b in blocks:
        items.append((b.start, b.name, b.body, b.line))
    for off, body in unnumbered_display:
        items.append((off, "displaymath", body, line_of(tex, off)))
    items.sort(key=lambda t: t[0])

    counter = 0
    out: list[Equation] = []
    for ordinal, (off, env, body, line) in enumerate(items):
        starred = env.endswith("*") or env == "displaymath"
        base = env.rstrip("*")
        has_tag = bool(_TAG.search(body))

        if starred:
            numbers: list[str | None] = [None]
            rows = [body]
        elif base in _PER_ROW:
            rows = _split_rows(body)
            numbers = []
            for r in rows:
                if _NONUM.search(r):
                    numbers.append(None)
                else:
                    counter += 1
                    numbers.append(str(counter))
        else:  # equation, multline
            rows = [body]
            if _NONUM.search(body):
                numbers = [None]
            else:
                counter += 1
                numbers = [str(counter)]

        for row_latex, number in zip(rows, numbers):
            if number is None:
                conf = "UNKNOWN"
            elif has_tag or scheme_changed:
                # \tag overrides the printed number; \numberwithin changes the
                # format to e.g. "3.1". Either way our arabic count is a guess.
                conf = "POSSIBLE"
            else:
                # Never CONFIRMED from source alone: confirming a number means
                # comparing against the compiled PDF, which we have not done.
                conf = "LIKELY"
            out.append(
                Equation(
                    content_hash=_hash(row_latex),
                    latex=row_latex.strip(),
                    environment=env,
                    is_numbered=number is not None,
                    derived_number=number,
                    number_conf=conf,
                    latex_label=(re.search(r"\\label\{([^}]*)\}", row_latex).group(1)
                                 if "\\label{" in row_latex else None),
                    src_line=line,
                    ordinal=ordinal,
                    char_start=off,
                )
            )
    return out
