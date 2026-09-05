"""jcodemunch backend, driven as an MCP client over stdio.

PaperLens is an MCP server that is also an MCP *client* here: it spawns
jcodemunch as a subprocess and calls its three-verb front door. The session is
long-lived because process startup costs seconds and indexing is per-repository.

Configure with:
    PAPERLENS_JCODEMUNCH_CMD   default: "uvx jcodemunch-mcp"
    PAPERLENS_CODE_PROVIDER    "auto" (default) | "jcodemunch" | "python-ast"
"""
from __future__ import annotations

import asyncio
import json
import os
import shlex
import threading
from typing import Any

from .munch_format import absence_ref, is_munch, parse
from .provider import IndexResult, SearchResult, Symbol, TextHit

_DEFAULT_CMD = "uvx jcodemunch-mcp"
_STARTUP_TIMEOUT = 120.0
_CALL_TIMEOUT = 300.0


class _StdioBridge:
    """A background thread owning an event loop and one persistent MCP session.

    PaperLens' tool handlers are synchronous, and the MCP client is async, so the
    session lives on its own loop and calls are submitted across the boundary.
    Because the work runs on a different loop, blocking here cannot deadlock the
    server's own loop.
    """

    def __init__(self, command: str, args: list[str]) -> None:
        self._command, self._args = command, args
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._session: Any = None
        self._error: BaseException | None = None

    def start(self) -> bool:
        if self._thread is not None:
            return self._session is not None
        self._thread = threading.Thread(target=self._run, name="jcodemunch",
                                        daemon=True)
        self._thread.start()
        self._ready.wait(timeout=_STARTUP_TIMEOUT)
        return self._session is not None

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except BaseException as exc:       # noqa: BLE001 - reported to caller
            self._error = exc
        finally:
            self._ready.set()
            self._loop.close()

    async def _serve(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stop = asyncio.Event()
        params = StdioServerParameters(command=self._command, args=self._args,
                                       env={**os.environ})
        try:
            async with stdio_client(params) as (r, w):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._stop.wait()
        except BaseException as exc:        # noqa: BLE001
            self._error = exc
            self._session = None
            self._ready.set()

    def call(self, tool: str, args: dict[str, Any],
             timeout: float = _CALL_TIMEOUT) -> str:
        if self._session is None or self._loop is None:
            raise RuntimeError(f"jcodemunch session unavailable: {self._error}")

        async def _do() -> str:
            res = await self._session.call_tool(tool, args)
            parts = [c.text for c in (res.content or []) if hasattr(c, "text")]
            return "\n".join(parts)

        fut = asyncio.run_coroutine_threadsafe(_do(), self._loop)
        return fut.result(timeout=timeout)

    def stop(self) -> None:
        if self._loop and self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)


class JCodeMunchProvider:
    name = "jcodemunch"

    def __init__(self) -> None:
        cmd = shlex.split(os.environ.get("PAPERLENS_JCODEMUNCH_CMD", _DEFAULT_CMD))
        command = cmd[0] if cmd else "uvx"
        # `uvx` is commonly installed outside the PATH a server inherits.
        if command == "uvx" and not _on_path("uvx"):
            local = os.path.expanduser("~/.local/bin/uvx")
            if os.path.exists(local):
                command = local
        self._bridge = _StdioBridge(command, cmd[1:])
        self._checked: bool | None = None

    def available(self) -> bool:
        if self._checked is None:
            try:
                self._checked = self._bridge.start()
            except Exception:
                self._checked = False
        return bool(self._checked)

    def _order(self, action: str, args: dict[str, Any],
               state_changing: bool = False) -> str:
        payload: dict[str, Any] = {"action": action, "args": args}
        if state_changing:
            payload["allow_state_change"] = True
        return self._bridge.call("order", payload)

    @staticmethod
    def _rows(text: str, table: str) -> list[dict[str, Any]]:
        if is_munch(text):
            return parse(text)["tables"].get(table, [])
        try:
            blob = json.loads(text)
        except json.JSONDecodeError:
            return []
        val = blob.get(table, blob.get("results", []))
        return val if isinstance(val, list) else []

    @staticmethod
    def _to_symbol(row: dict[str, Any]) -> Symbol | None:
        qn = row.get("id") or row.get("symbol_id") or row.get("qualified_name")
        if not qn or "::" not in str(qn):
            return None
        qn = str(qn)
        file_path = qn.split("::", 1)[0]
        kind = row.get("kind") or (qn.rsplit("#", 1)[-1] if "#" in qn else "symbol")

        def as_int(v: Any) -> int | None:
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        return Symbol(
            qualified_name=qn, name=str(row.get("name") or qn.rsplit("::", 1)[-1]),
            kind=str(kind), file_path=file_path,
            line_start=as_int(row.get("line")) or 0,
            line_end=as_int(row.get("end_line")),
            signature=row.get("signature") or None,
            parent=row.get("parent") or None,
        )

    # ── provider interface ────────────────────────────────────────────────
    def index(self, repo_key: str, path: str) -> IndexResult:
        raw = self._order("index_folder", {"path": path}, state_changing=True)
        try:
            blob = json.loads(raw)
        except json.JSONDecodeError:
            blob = {}
        key = blob.get("repo", repo_key)
        if "symbol_count" not in blob:
            # An incremental run that found no changes reports only the delta,
            # so ask the index itself for the totals.
            blob = {**self._repo_outline(key), **blob}
        return IndexResult(
            repo_key=key, indexer=f"{self.name}@stdio",
            symbol_count=int(blob.get("symbol_count", 0)),
            file_count=int(blob.get("file_count", 0)),
            languages=blob.get("languages", {}) or {},
            warnings=list(blob.get("parse_warnings", []) or []),
        )

    def _repo_outline(self, repo_key: str) -> dict[str, Any]:
        try:
            return json.loads(self._order("get_repo_outline", {"repo": repo_key}))
        except (json.JSONDecodeError, RuntimeError):
            return {}

    def search_symbols(self, repo_key: str, query: str, limit: int = 20) -> SearchResult:
        raw = self._order("search_symbols", {"repo": repo_key, "query": query})
        rows = self._rows(raw, "results")
        syms = [s for s in (self._to_symbol(r) for r in rows) if s]
        if not syms:
            ref = absence_ref(raw)
            out = SearchResult.absent(repo_key, query, f"{self.name}:search_symbols")
            if ref:
                # jcodemunch's own citable absence marker beats a synthesised one.
                out.absence_ref = ref
            return out
        return SearchResult(query=query, found=True, symbols=syms[:limit],
                            method=f"{self.name}:search_symbols")

    def file_outline(self, repo_key: str, file_path: str) -> list[Symbol]:
        raw = self._order("get_file_outline", {"repo": repo_key, "file_path": file_path})
        return [s for s in (self._to_symbol(r) for r in self._rows(raw, "symbols")) if s]

    def list_symbols(self, repo_key: str, max_files: int = 200) -> list[Symbol]:
        raw = self._order("get_file_tree", {"repo": repo_key})
        try:
            files = json.loads(raw).get("files", [])
        except json.JSONDecodeError:
            files = [r.get("path") for r in self._rows(raw, "files")]
        out: list[Symbol] = []
        for f in [f for f in files if f][:max_files]:
            out.extend(self.file_outline(repo_key, f if isinstance(f, str) else f.get("path", "")))
        return out

    def symbol_source(self, repo_key: str, qualified_name: str) -> str | None:
        raw = self._order("get_symbol_source",
                          {"repo": repo_key, "symbol_id": qualified_name})
        try:
            blob = json.loads(raw)
        except json.JSONDecodeError:
            return raw or None
        return blob.get("source") or blob.get("code") or None

    def search_text(self, repo_key: str, pattern: str, *, regex: bool = False,
                    limit: int = 50) -> list[TextHit]:
        raw = self._order("search_text", {"repo": repo_key, "query": pattern,
                                          "is_regex": regex})
        hits: list[TextHit] = []
        try:
            blob = json.loads(raw)
        except json.JSONDecodeError:
            blob = {}
        # Results are grouped per file, each holding its own list of matches.
        for entry in blob.get("results", []):
            path = str(entry.get("file") or entry.get("file_path") or "")
            for m in entry.get("matches", [entry]):
                try:
                    hits.append(TextHit(path, int(m.get("line") or 0),
                                        str(m.get("text") or "").strip()[:240]))
                except (TypeError, ValueError, AttributeError):
                    continue
                if len(hits) >= limit:
                    return hits
        return hits

    def callers_of(self, repo_key: str, qualified_name: str) -> list[Symbol]:
        raw = self._order("get_call_hierarchy",
                          {"repo": repo_key, "symbol_id": qualified_name})
        rows = self._rows(raw, "callers") or self._rows(raw, "incoming")
        return [s for s in (self._to_symbol(r) for r in rows) if s]


def _on_path(name: str) -> bool:
    from shutil import which
    return which(name) is not None
