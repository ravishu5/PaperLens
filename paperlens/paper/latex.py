"""Small LaTeX scanning helpers shared by the structure and equation parsers.

Deliberately *not* a LaTeX parser. arXiv submissions are not well-formed
documents; they are whatever compiled once on the author's machine. We scan for
the handful of constructs we can identify reliably and report UNKNOWN elsewhere.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# \begin{env} ... \end{env}
_BEGIN = re.compile(r"\\begin\{([A-Za-z][A-Za-z0-9*]*)\}")


@dataclass(frozen=True)
class EnvBlock:
    name: str          # environment name, e.g. "align*"
    body: str          # content between \begin and \end
    start: int         # char offset of the \begin
    end: int           # char offset just past the \end
    line: int          # 1-based line of the \begin


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def iter_environments(text: str, names: set[str]) -> list[EnvBlock]:
    """Find top-level occurrences of the named environments, in document order.

    Nested same-name environments are handled by depth counting. Blocks nested
    inside another *matched* block are skipped, so an `align` inside a `figure`
    is still found but a `split` inside an `equation` is not double counted.
    """
    out: list[EnvBlock] = []
    pos = 0
    while True:
        m = _BEGIN.search(text, pos)
        if not m:
            break
        env = m.group(1)
        if env not in names:
            pos = m.end()
            continue
        end_off = _matching_end(text, env, m.end())
        if end_off is None:
            pos = m.end()
            continue
        body_end = text.rindex(f"\\end{{{env}}}", m.end(), end_off)
        out.append(
            EnvBlock(env, text[m.end():body_end], m.start(), end_off,
                     line_of(text, m.start()))
        )
        pos = end_off
    return out


def _matching_end(text: str, env: str, from_pos: int) -> int | None:
    open_tok = f"\\begin{{{env}}}"
    close_tok = f"\\end{{{env}}}"
    depth = 1
    pos = from_pos
    while depth:
        nxt_o = text.find(open_tok, pos)
        nxt_c = text.find(close_tok, pos)
        if nxt_c == -1:
            return None
        if nxt_o != -1 and nxt_o < nxt_c:
            depth += 1
            pos = nxt_o + len(open_tok)
        else:
            depth -= 1
            pos = nxt_c + len(close_tok)
    return pos


def strip_comments(text: str) -> str:
    """Remove unescaped % comments. arxiv-to-prompt does this for us, but the
    parsers must not assume it, since a PDF-derived path will not."""
    out = []
    for line in text.split("\n"):
        i, esc = 0, False
        cut = len(line)
        while i < len(line):
            ch = line[i]
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "%":
                cut = i
                break
            i += 1
        out.append(line[:cut])
    return "\n".join(out)


def extract_label(body: str) -> str | None:
    m = re.search(r"\\label\{([^}]*)\}", body)
    return m.group(1) if m else None


def normalize_math(latex: str) -> str:
    """Canonical form for content hashing: labels, tags and whitespace removed."""
    s = re.sub(r"\\(label|tag|nonumber|notag)\s*(\{[^}]*\})?", "", latex)
    s = re.sub(r"\s+", " ", s)
    return s.strip()
