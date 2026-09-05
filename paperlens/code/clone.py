"""Repository checkout, pinned to a commit SHA.

The SHA is the cache key for everything downstream (ARCHITECTURE 1.3): an index
built at a SHA is immutable, so it is built once and never invalidated by time.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .. import config


class CloneError(RuntimeError):
    pass


@dataclass
class Checkout:
    repo_id: str
    path: Path
    commit_sha: str
    ref: str


def _run(args: list[str], cwd: Path | None = None, timeout: int = 300) -> str:
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise CloneError((p.stderr or p.stdout).strip()[:400])
    return p.stdout.strip()


def repos_dir() -> Path:
    d = config.home() / "repos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_checkout(owner: str, name: str, ref: str = "HEAD") -> Checkout:
    """Shallow-clone or update a repository and report the exact SHA.

    Shallow by default: PaperLens reads the working tree, it does not need
    history. A full clone of a large repository would dominate indexing time.
    """
    repo_id = f"{owner}/{name}"
    dest = repos_dir() / owner / name
    url = f"https://github.com/{owner}/{name}.git"

    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        args = ["git", "clone", "--depth", "1", "--quiet"]
        if ref and ref != "HEAD":
            args += ["--branch", ref]
        _run(args + [url, str(dest)], timeout=600)
    else:
        try:
            _run(["git", "fetch", "--depth", "1", "--quiet", "origin", ref], cwd=dest)
            _run(["git", "checkout", "--quiet", "FETCH_HEAD"], cwd=dest)
        except CloneError:
            pass  # offline or unusual ref: use what is already on disk

    sha = _run(["git", "rev-parse", "HEAD"], cwd=dest)
    return Checkout(repo_id=repo_id, path=dest, commit_sha=sha, ref=ref)
