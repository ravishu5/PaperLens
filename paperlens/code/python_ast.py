"""Dependency-free Python code intelligence, built on the stdlib `ast` module.

The default backend. It covers Python only, which is most of the ML research
corpus, and it needs nothing installed. jcodemunch is richer -- 70+ languages,
call hierarchy, ranked retrieval -- but it is proprietary and optional, so
PaperLens must work without it.

Symbol ids match jcodemunch's format ("path::Qualified.Name#kind") so the two
backends are interchangeable behind the port.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from .provider import IndexResult, SearchResult, Symbol, TextHit

_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache",
              ".pytest_cache", "build", "dist", ".tox", ".eggs"}


class PythonAstProvider:
    name = "python-ast"

    def __init__(self) -> None:
        self._repos: dict[str, Path] = {}
        self._symbols: dict[str, list[Symbol]] = {}

    def available(self) -> bool:
        return True

    # ── indexing ──────────────────────────────────────────────────────────
    def index(self, repo_key: str, path: str) -> IndexResult:
        root = Path(path)
        self._repos[repo_key] = root
        symbols: list[Symbol] = []
        languages: dict[str, int] = {}
        warnings: list[str] = []
        files = 0

        for f in self._walk(root):
            rel = str(f.relative_to(root))
            ext = f.suffix.lower().lstrip(".")
            languages[ext] = languages.get(ext, 0) + 1
            files += 1
            if f.suffix != ".py":
                continue
            try:
                tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError as exc:
                warnings.append(f"{rel}: unparseable ({exc.msg})")
                continue
            symbols.extend(self._symbols_in(tree, rel))

        self._symbols[repo_key] = symbols
        non_py = sum(n for e, n in languages.items() if e != "py")
        if non_py:
            warnings.append(
                f"{non_py} non-Python file(s) were counted but not parsed: this "
                f"backend understands Python only. Use jcodemunch for other languages."
            )
        return IndexResult(repo_key=repo_key, indexer=f"{self.name}@1",
                           symbol_count=len(symbols), file_count=files,
                           languages=languages, warnings=warnings)

    def _walk(self, root: Path):
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            if p.stat().st_size > 2_000_000:
                continue
            yield p

    def _symbols_in(self, tree: ast.AST, rel: str) -> list[Symbol]:
        out: list[Symbol] = []

        def sig(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
            try:
                args = ast.unparse(node.args)
            except Exception:
                args = "..."
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            return f"{prefix} {node.name}({args})"

        def visit(node: ast.AST, parent: str | None) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    q = f"{parent}.{child.name}" if parent else child.name
                    out.append(Symbol(
                        qualified_name=f"{rel}::{q}#class", name=child.name,
                        kind="class", file_path=rel, line_start=child.lineno,
                        line_end=getattr(child, "end_lineno", None),
                        signature=f"class {child.name}", parent=parent))
                    visit(child, q)
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    q = f"{parent}.{child.name}" if parent else child.name
                    kind = "method" if parent else "function"
                    out.append(Symbol(
                        qualified_name=f"{rel}::{q}#{kind}", name=child.name,
                        kind=kind, file_path=rel, line_start=child.lineno,
                        line_end=getattr(child, "end_lineno", None),
                        signature=sig(child), parent=parent))
                    visit(child, q)
                elif isinstance(child, ast.Assign) and parent is None:
                    for t in child.targets:
                        if isinstance(t, ast.Name) and t.id.isupper():
                            out.append(Symbol(
                                qualified_name=f"{rel}::{t.id}#constant", name=t.id,
                                kind="constant", file_path=rel,
                                line_start=child.lineno,
                                line_end=getattr(child, "end_lineno", None)))
        visit(tree, None)
        return out

    # ── queries ───────────────────────────────────────────────────────────
    def list_symbols(self, repo_key: str) -> list[Symbol]:
        return list(self._symbols.get(repo_key, []))

    def search_symbols(self, repo_key: str, query: str, limit: int = 20) -> SearchResult:
        syms = self._symbols.get(repo_key, [])
        terms = [t for t in re.split(r"[\s_\-.]+", query.lower()) if t]
        scored: list[tuple[int, Symbol]] = []
        for s in syms:
            hay = f"{s.name} {s.qualified_name}".lower()
            hits = sum(1 for t in terms if t in hay)
            if hits:
                scored.append((hits, s))
        if not scored:
            return SearchResult.absent(repo_key, query, f"{self.name}:symbol_search")
        scored.sort(key=lambda t: (-t[0], t[1].qualified_name))
        return SearchResult(query=query, found=True,
                            symbols=[s for _, s in scored[:limit]],
                            method=f"{self.name}:symbol_search")

    def file_outline(self, repo_key: str, file_path: str) -> list[Symbol]:
        return [s for s in self._symbols.get(repo_key, []) if s.file_path == file_path]

    def symbol_source(self, repo_key: str, qualified_name: str) -> str | None:
        root = self._repos.get(repo_key)
        sym = next((s for s in self._symbols.get(repo_key, [])
                    if s.qualified_name == qualified_name), None)
        if root is None or sym is None:
            return None
        try:
            lines = (root / sym.file_path).read_text(
                encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return None
        end = sym.line_end or sym.line_start
        return "\n".join(lines[sym.line_start - 1:end])

    def search_text(self, repo_key: str, pattern: str, *, regex: bool = False,
                    limit: int = 50) -> list[TextHit]:
        root = self._repos.get(repo_key)
        if root is None:
            return []
        rx = re.compile(pattern if regex else re.escape(pattern))
        out: list[TextHit] = []
        for f in self._walk(root):
            if f.suffix not in (".py", ".pyi", ".cfg", ".toml", ".yaml", ".yml",
                                ".json", ".md", ".txt"):
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    out.append(TextHit(str(f.relative_to(root)), i, line.strip()[:240]))
                    if len(out) >= limit:
                        return out
        return out

    def callers_of(self, repo_key: str, qualified_name: str) -> list[Symbol]:
        """Name-based approximation. jcodemunch resolves this properly."""
        sym = next((s for s in self._symbols.get(repo_key, [])
                    if s.qualified_name == qualified_name), None)
        if sym is None:
            return []
        hits = self.search_text(repo_key, rf"\b{re.escape(sym.name)}\s*\(", regex=True)
        out: list[Symbol] = []
        for h in hits:
            for s in self._symbols.get(repo_key, []):
                if (s.file_path == h.file_path and s.line_start <= h.line
                        and (s.line_end or s.line_start) >= h.line
                        and s.qualified_name != qualified_name):
                    out.append(s)
                    break
        seen, uniq = set(), []
        for s in out:
            if s.qualified_name not in seen:
                seen.add(s.qualified_name)
                uniq.append(s)
        return uniq
