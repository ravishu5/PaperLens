"""Runtime configuration. Local-first: everything lives under one directory."""
from __future__ import annotations

import os
from pathlib import Path

# Bumping either of these invalidates derived data (ARCHITECTURE 1.3).
PARSER_VERSION = "3"
ANALYZER_VERSION = "1"

USER_AGENT = "PaperLens/0.1 (https://github.com/ravi/paperlens; MCP research tool)"


def quiet_dependencies() -> None:
    """Silence third-party INFO chatter.

    httpx logs a line per request and arxiv-to-prompt logs to the root logger.
    On a stdio MCP server that noise goes to the client's stderr for every call,
    and it buries PaperLens' own warnings -- including the parser guard that
    exists precisely to be noticed.
    """
    import logging

    for name in ("httpx", "httpcore", "arxiv_to_prompt", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def home() -> Path:
    p = Path(os.environ.get("PAPERLENS_HOME", Path.home() / ".paperlens"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def db_path() -> Path:
    return Path(os.environ.get("PAPERLENS_DB", home() / "graph.db"))


def source_cache_dir() -> Path:
    p = home() / "sources"
    p.mkdir(parents=True, exist_ok=True)
    return p


# Rate limits, per RESEARCH 2. Persisted as token buckets so they survive restarts.
RATE_LIMITS: dict[str, tuple[float, float]] = {
    # source: (capacity, refill_tokens_per_second)
    "arxiv": (1.0, 1.0 / 3.0),      # arXiv asks ~3s between requests
    "semanticscholar": (1.0, 1.0),  # 1 RPS with an API key; unauthenticated 429s fast
    "openalex": (10.0, 10.0),       # polite pool via mailto=
    "github": (30.0, 5000 / 3600),  # 5,000/hr authenticated
    "huggingface": (10.0, 5.0),
}
