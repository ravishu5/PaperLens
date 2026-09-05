"""Parser for jcodemunch's compact `#MUNCH/1` response encoding.

jcodemunch answers some actions in a token-efficient table format rather than
JSON:

    #MUNCH/1 tool=get_file_outline enc=fo1

    @1=clip/model.py::CLIP#class
    repo=openai/CLIP __tables=s:symbols:id|name|kind|signature|line|end_line|parent
    s,@1,CLIP,class,"class CLIP(nn.Module)",243,372,

`@n=` lines are aliases expanded inside rows, `__tables=` declares each table's
columns, and rows begin with the table's key. Parsing is best-effort: this is an
optional backend, so a format change must degrade rather than crash.
"""
from __future__ import annotations

import csv
import io
import json
import re
from typing import Any

_ALIAS = re.compile(r"^@(\d+)=(.*)$")
_KV = re.compile(r'(\w[\w.]*)=("(?:[^"\\]|\\.)*"|\S+)')


def is_munch(text: str) -> bool:
    return text.lstrip().startswith("#MUNCH/")


def parse(text: str) -> dict[str, Any]:
    """Return {"scalars": {...}, "tables": {name: [rowdict, ...]}}.

    The `__tables=` line separates the header from the rows. Rows are parsed as
    one CSV document rather than line by line, because a quoted field can span
    lines -- a multi-line Python signature is the common case.
    """
    lines = text.splitlines()
    header: list[str] = []
    rows_blob: list[str] = []
    seen_tables = False
    for line in lines:
        if not seen_tables:
            header.append(line)
            if "__tables=" in line:
                seen_tables = True
        else:
            rows_blob.append(line)

    aliases: dict[str, str] = {}
    table_defs: dict[str, tuple[str, list[str]]] = {}
    scalars: dict[str, Any] = {}

    for line in header:
        line = line.rstrip()
        if not line or line.startswith("#MUNCH/"):
            continue
        if (m := _ALIAS.match(line)):
            aliases[f"@{m.group(1)}"] = m.group(2)
            continue
        for km in _KV.finditer(line):
            key, val = km.group(1), km.group(2)
            if val.startswith('"') and val.endswith('"'):
                try:
                    val = json.loads(val)
                except json.JSONDecodeError:
                    val = val[1:-1]
            if key == "__tables":
                for spec in str(val).split(";"):
                    parts = spec.split(":")
                    if len(parts) >= 3:
                        table_defs[parts[0]] = (parts[1], parts[2].split("|"))
            elif not key.startswith("__"):
                scalars[key] = _coerce(val)

    tables: dict[str, list[dict[str, Any]]] = {}
    if table_defs and rows_blob:
        try:
            reader = csv.reader(io.StringIO("\n".join(rows_blob)))
            for values in reader:
                if not values:
                    continue
                key = values[0]
                if key not in table_defs:
                    continue
                name, cols = table_defs[key]
                row = {c: _expand(v, aliases) for c, v in zip(cols, values[1:])}
                tables.setdefault(name, []).append(row)
        except csv.Error:
            pass
    return {"scalars": scalars, "tables": tables}


_ALIAS_REF = re.compile(r"@(\d+)")


def _expand(value: str, aliases: dict[str, str]) -> str:
    """Aliases are substituted as prefixes, not whole cells: one row carries both
    `@1` (the file) and `@1::CLIP.forward#method` (a symbol id in that file)."""
    if not aliases or "@" not in value:
        return value
    return _ALIAS_REF.sub(lambda m: aliases.get(m.group(0), m.group(0)), value)


def _coerce(v: Any) -> Any:
    if isinstance(v, str):
        if v.isdigit():
            return int(v)
        if v in ("true", "false"):
            return v == "true"
    return v


def absence_ref(text: str) -> str | None:
    """Pull jcodemunch's citable absence marker out of an empty result."""
    m = re.search(r'absence_evidence="(\{.*?\})"', text)
    if not m:
        m = re.search(r'"ref"\s*:\s*"(absent:[0-9a-f]+)"', text)
        return m.group(1) if m else None
    try:
        blob = json.loads(m.group(1).replace('""', '"'))
        return blob.get("ref")
    except json.JSONDecodeError:
        inner = re.search(r"(absent:[0-9a-f]+)", m.group(1))
        return inner.group(1) if inner else None
