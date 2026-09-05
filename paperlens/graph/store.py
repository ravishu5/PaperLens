"""SQLite-backed research graph (ARCHITECTURE section 1.4, 2)."""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .. import config

_SCHEMA = Path(__file__).with_name("schema.sql")


_VERSION_SUFFIX = re.compile(r"v\d+$")


def base_id(paper_version: str) -> str:
    """Strip the trailing version from a paper_version.

    `pv.split("v")[0]` was fine while every identifier was an arXiv number, and
    silently truncates a DOI such as 10.1016/j.bspc.2025.108746 at any letter v.
    """
    return _VERSION_SUFFIX.sub("", paper_version)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    """Thin, explicit wrapper over sqlite3. No ORM by design: the schema *is*
    the model, and every query in this project is short enough to read."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else config.db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    # ── connection ────────────────────────────────────────────────────────
    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30.0)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA foreign_keys = ON")
            c.execute("PRAGMA journal_mode = WAL")
            c.execute("PRAGMA synchronous = NORMAL")
            self._local.conn = c
        return c

    def _init_schema(self) -> None:
        script = _SCHEMA.read_text()
        self.conn.executescript(script)
        self._migrate(script)
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('parser_version', ?)",
            (config.PARSER_VERSION,),
        )
        self.conn.commit()

    # ── migration ─────────────────────────────────────────────────────────
    _CREATE = re.compile(
        r"CREATE TABLE IF NOT EXISTS\s+(\w+)\s*\((.*?)\n\);", re.S)

    def _migrate(self, script: str) -> None:
        """Rebuild tables whose definition has drifted from schema.sql.

        `CREATE TABLE IF NOT EXISTS` is a no-op on an existing table, so a
        changed CHECK constraint is silently ignored until a write fails with
        IntegrityError against a definition nobody can see. Everything here is
        derived from immutable sources or re-recordable, so the safe repair is to
        rebuild the table and carry over whatever columns still exist.
        """
        for m in self._CREATE.finditer(script):
            name = m.group(1)
            desired = self._normalise_ddl(m.group(0))
            row = self.one(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (name,))
            if row is None or row["sql"] is None:
                continue
            if self._normalise_ddl(row["sql"]) == desired:
                continue
            self._rebuild(name, m.group(0))

    @staticmethod
    def _normalise_ddl(sql: str) -> str:
        s = re.sub(r"--[^\n]*", " ", sql)
        s = re.sub(r"IF NOT EXISTS\s+", "", s, flags=re.I)
        return re.sub(r"\s+", " ", s).strip().rstrip(";").lower()

    def _rebuild(self, name: str, create_sql: str) -> None:
        old_cols = {r["name"] for r in
                    self.all(f"PRAGMA table_info({name})")}
        tmp = f"{name}__migrating"
        self.execute("PRAGMA foreign_keys = OFF")
        try:
            self.execute(f"DROP TABLE IF EXISTS {tmp}")
            self.conn.executescript(
                create_sql.replace(f"CREATE TABLE IF NOT EXISTS {name}",
                                   f"CREATE TABLE {tmp}", 1))
            new_cols = {r["name"] for r in self.all(f"PRAGMA table_info({tmp})")}
            shared = sorted(old_cols & new_cols)
            if shared:
                cols = ", ".join(shared)
                # Rows that violate a newly tightened constraint are dropped
                # rather than blocking the migration; all of it is derived data.
                self.execute(
                    f"INSERT OR IGNORE INTO {tmp} ({cols}) SELECT {cols} FROM {name}")
            self.execute(f"DROP TABLE {name}")
            self.execute(f"ALTER TABLE {tmp} RENAME TO {name}")
            self.commit()
            logging.getLogger(__name__).info(
                "migrated table %s (kept %d column(s))", name, len(shared))
        finally:
            self.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None

    # ── primitives ────────────────────────────────────────────────────────
    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> sqlite3.Cursor:
        return self.conn.executemany(sql, rows)

    def one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def commit(self) -> None:
        self.conn.commit()

    # ── papers ────────────────────────────────────────────────────────────
    def upsert_paper(
        self,
        arxiv_id: str,
        title: str,
        *,
        abstract: str | None = None,
        doi: str | None = None,
        published_at: str | None = None,
        authors: list[dict] | None = None,
        latest_version: str | None = None,
    ) -> None:
        self.execute(
            """
            INSERT INTO papers (arxiv_id, title, abstract, doi, published_at,
                                authors_json, latest_version)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(arxiv_id) DO UPDATE SET
              title          = excluded.title,
              abstract       = COALESCE(excluded.abstract, papers.abstract),
              doi            = COALESCE(excluded.doi, papers.doi),
              published_at   = COALESCE(excluded.published_at, papers.published_at),
              authors_json   = COALESCE(excluded.authors_json, papers.authors_json),
              latest_version = COALESCE(excluded.latest_version, papers.latest_version)
            """,
            (arxiv_id, title, abstract, doi, published_at,
             json.dumps(authors) if authors is not None else None, latest_version),
        )

    def upsert_paper_version(
        self,
        paper_version: str,
        arxiv_id: str,
        version: int,
        *,
        source_kind: str,
        fidelity: str,
        source_sha256: str | None,
        flattened_tex: str | None,
    ) -> None:
        self.execute(
            """
            INSERT INTO paper_versions (paper_version, arxiv_id, version, source_kind,
                                        fidelity, source_sha256, parser_version,
                                        flattened_tex, ingested_at)
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(paper_version) DO UPDATE SET
              source_kind    = excluded.source_kind,
              fidelity       = excluded.fidelity,
              source_sha256  = excluded.source_sha256,
              parser_version = excluded.parser_version,
              flattened_tex  = excluded.flattened_tex,
              ingested_at    = excluded.ingested_at
            """,
            (paper_version, arxiv_id, version, source_kind, fidelity,
             source_sha256, config.PARSER_VERSION, flattened_tex, utcnow()),
        )

    def clear_paper_artifacts(self, paper_version: str) -> None:
        """Remove derived artifacts for a version before a re-parse.

        Agent-authored components and claims are preserved: they are expensive
        to produce and are keyed to the immutable paper version, so a parser
        upgrade must not silently discard them.
        """
        for table in ("equations", "declared_urls", "stated_values", "algorithms"):
            self.execute(f"DELETE FROM {table} WHERE paper_version = ?", (paper_version,))
        self.execute(
            "DELETE FROM components WHERE paper_version = ? AND source = 'DETERMINISTIC'",
            (paper_version,),
        )
        self.execute(
            "DELETE FROM claims WHERE paper_version = ? AND source = 'DETERMINISTIC'",
            (paper_version,),
        )
        self.execute("DELETE FROM sections_fts WHERE paper_version = ?", (paper_version,))
        self.execute("DELETE FROM sections WHERE paper_version = ?", (paper_version,))

    def paper_version_row(self, paper_version: str) -> sqlite3.Row | None:
        return self.one(
            "SELECT * FROM paper_versions WHERE paper_version = ?", (paper_version,)
        )

    def paper_id_of(self, paper_version: str) -> str:
        """The paper a version belongs to.

        Deriving it by stripping "v1" from the version string assumes the two are
        related by naming, which they need not be: a record ingested from a PDF
        may be keyed `uld-netv1` while its paper is keyed by DOI.
        """
        row = self.one(
            "SELECT arxiv_id FROM paper_versions WHERE paper_version = ?",
            (paper_version,))
        return row["arxiv_id"] if row else base_id(paper_version)

    def latest_version_of(self, arxiv_id: str) -> str | None:
        row = self.one(
            "SELECT paper_version FROM paper_versions WHERE arxiv_id = ? "
            "ORDER BY version DESC LIMIT 1",
            (arxiv_id,),
        )
        return row["paper_version"] if row else None

    # ── rate limiting (persisted token bucket) ────────────────────────────
    def take_token(self, source: str) -> float:
        """Consume one token; return seconds the caller must wait first.

        Persisted so a restart or crash loop cannot stampede an upstream API.
        """
        capacity, refill = config.RATE_LIMITS.get(source, (1.0, 1.0))
        now = datetime.now(timezone.utc)
        row = self.one("SELECT * FROM rate_limit_buckets WHERE source = ?", (source,))
        if row is None:
            self.execute(
                "INSERT INTO rate_limit_buckets (source, tokens, last_refill_at, "
                "capacity, refill_per_sec) VALUES (?,?,?,?,?)",
                (source, capacity - 1.0, now.isoformat(), capacity, refill),
            )
            self.commit()
            return 0.0

        last = datetime.fromisoformat(row["last_refill_at"])
        elapsed = max(0.0, (now - last).total_seconds())
        tokens = min(row["capacity"], row["tokens"] + elapsed * row["refill_per_sec"])
        wait = 0.0 if tokens >= 1.0 else (1.0 - tokens) / row["refill_per_sec"]
        tokens = max(0.0, tokens - 1.0)
        self.execute(
            "UPDATE rate_limit_buckets SET tokens = ?, last_refill_at = ? WHERE source = ?",
            (tokens, now.isoformat(), source),
        )
        self.commit()
        return wait

    # ── http cache ────────────────────────────────────────────────────────
    def cache_get(self, key: str) -> bytes | None:
        row = self.one("SELECT body, expires_at FROM http_cache WHERE cache_key = ?", (key,))
        if row is None:
            return None
        if row["expires_at"]:
            if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
                return None
        return row["body"]

    def cache_put(
        self, key: str, source: str, url: str, body: bytes, expires_at: str | None = None
    ) -> None:
        self.execute(
            "INSERT OR REPLACE INTO http_cache (cache_key, source, url, body, "
            "fetched_at, expires_at) VALUES (?,?,?,?,?,?)",
            (key, source, url, body, utcnow(), expires_at),
        )
        self.commit()
