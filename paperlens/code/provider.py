"""The CodeIntelligenceProvider port.

PaperLens orchestrates code intelligence rather than reimplementing it (D-006).
jcodemunch is the preferred backend -- 70+ languages, call hierarchy, token
budgeted retrieval -- but it is proprietary, so everything behind this interface
stays swappable. A dependency-free Python AST backend ships as the default
fallback, which also keeps the port honest by having two real implementations.

Absence is a first-class result. `SearchResult.found == False` carries a
reproducible `absence_ref`, because "I looked and it is not there" is a finding
that later phases cite as evidence, not an error.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Symbol:
    qualified_name: str      # "clip/model.py::CLIP.forward#method"
    name: str                # "forward"
    kind: str                # function | class | method | constant
    file_path: str
    line_start: int
    line_end: int | None = None
    signature: str | None = None
    parent: str | None = None

    @property
    def short_id(self) -> str:
        return self.qualified_name


@dataclass
class IndexResult:
    repo_key: str
    indexer: str
    symbol_count: int
    file_count: int
    languages: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """Symbols matching a query, or a citable record that none did."""
    query: str
    found: bool
    symbols: list[Symbol] = field(default_factory=list)
    absence_ref: str | None = None
    method: str = ""

    @staticmethod
    def absent(repo_key: str, query: str, method: str, salt: str = "") -> "SearchResult":
        h = hashlib.sha256(f"{repo_key}|{query}|{method}|{salt}".encode()).hexdigest()[:12]
        return SearchResult(query=query, found=False, absence_ref=f"absent:{h}",
                            method=method)


@dataclass
class TextHit:
    file_path: str
    line: int
    text: str


@runtime_checkable
class CodeIntelligenceProvider(Protocol):
    """Everything PaperLens needs from a code-intelligence backend."""

    name: str

    def available(self) -> bool: ...

    def index(self, repo_key: str, path: str) -> IndexResult: ...

    def list_symbols(self, repo_key: str) -> list[Symbol]: ...

    def search_symbols(self, repo_key: str, query: str,
                       limit: int = 20) -> SearchResult: ...

    def file_outline(self, repo_key: str, file_path: str) -> list[Symbol]: ...

    def symbol_source(self, repo_key: str, qualified_name: str) -> str | None: ...

    def search_text(self, repo_key: str, pattern: str, *, regex: bool = False,
                    limit: int = 50) -> list[TextHit]: ...

    def callers_of(self, repo_key: str, qualified_name: str) -> list[Symbol]: ...


def get_provider(prefer: str | None = None) -> CodeIntelligenceProvider:
    """Pick a backend: jcodemunch when reachable, Python AST otherwise."""
    import os

    from .python_ast import PythonAstProvider

    choice = (prefer or os.environ.get("PAPERLENS_CODE_PROVIDER") or "auto").lower()

    if choice in ("auto", "jcodemunch"):
        try:
            from .jcodemunch_adapter import JCodeMunchProvider

            p = JCodeMunchProvider()
            if p.available():
                return p
        except Exception:
            pass
        if choice == "jcodemunch":
            raise RuntimeError(
                "PAPERLENS_CODE_PROVIDER=jcodemunch but the jcodemunch MCP server "
                "could not be started. Check PAPERLENS_JCODEMUNCH_CMD."
            )
    return PythonAstProvider()
