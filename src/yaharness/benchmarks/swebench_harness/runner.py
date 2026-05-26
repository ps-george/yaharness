"""Repo checkout helper for Tier-1 SWE-bench runs.

Materialises `<repo>` at `<base_commit>` under a cache directory and exposes
the working tree as a `Path` that callers (the agent) can hand to a
scoped `FilesystemTool`. Designed to fail loudly and cancellably.

Tier 1 only does checkout + read access. We do NOT apply patches here, do
NOT run tests here — that is Tier 2 (docker-backed grading).
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

DEFAULT_GIT_TIMEOUT_SECONDS = 300.0


class RepoCheckoutError(RuntimeError):
    """Raised when a repo cannot be cloned or checked out."""


async def _run_git(
    *args: str,
    cwd: Path | None = None,
    timeout: float = DEFAULT_GIT_TIMEOUT_SECONDS,
) -> tuple[int, str, str]:
    """Run a git command. Returns (returncode, stdout, stderr). Raises on timeout."""
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise RepoCheckoutError(f"git {' '.join(args)} timed out after {timeout}s") from None
    return (
        proc.returncode or 0,
        stdout_b.decode("utf-8", "replace"),
        stderr_b.decode("utf-8", "replace"),
    )


class RepoCheckout:
    """Materialise `<repo>` at `<base_commit>` under a cache dir.

    Layout: `<cache_dir>/repos/<owner>__<name>/` is a single bare-ish working
    clone reused across commits via `git checkout`. Concurrent use across
    commits in the same repo is not supported (callers should serialise per
    repo, or use disjoint cache_dirs).
    """

    def __init__(
        self,
        *,
        repo: str,
        base_commit: str,
        cache_dir: Path,
        git_timeout_seconds: float = DEFAULT_GIT_TIMEOUT_SECONDS,
    ) -> None:
        if "/" not in repo:
            raise ValueError(f"repo must be 'owner/name', got: {repo!r}")
        self.repo = repo
        self.base_commit = base_commit
        self.cache_dir = Path(cache_dir)
        self._timeout = git_timeout_seconds
        owner, name = repo.split("/", 1)
        self.path = self.cache_dir / "repos" / f"{owner}__{name}"

    @property
    def remote_url(self) -> str:
        return f"https://github.com/{self.repo}.git"

    async def ensure_cloned(self) -> None:
        """Clone the repo if not already present."""
        if (self.path / ".git").is_dir():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rc, _, err = await _run_git(
            "clone",
            "--filter=blob:none",
            self.remote_url,
            str(self.path),
            timeout=self._timeout,
        )
        if rc != 0:
            raise RepoCheckoutError(f"git clone {self.remote_url} failed: {err.strip()}")

    async def checkout(self) -> Path:
        """Fetch (if needed) and checkout `base_commit`. Returns the working-tree path."""
        await self.ensure_cloned()
        # Try checkout first; if commit unknown, fetch then retry.
        rc, _, _ = await _run_git(
            "checkout", "--force", self.base_commit, cwd=self.path, timeout=self._timeout
        )
        if rc != 0:
            rc2, _, err2 = await _run_git(
                "fetch", "origin", self.base_commit, cwd=self.path, timeout=self._timeout
            )
            if rc2 != 0:
                raise RepoCheckoutError(f"git fetch {self.base_commit} failed: {err2.strip()}")
            rc3, _, err3 = await _run_git(
                "checkout", "--force", self.base_commit, cwd=self.path, timeout=self._timeout
            )
            if rc3 != 0:
                raise RepoCheckoutError(f"git checkout {self.base_commit} failed: {err3.strip()}")
        # Clean working tree of any leftover untracked stuff from prior runs.
        await _run_git("clean", "-fdx", cwd=self.path, timeout=self._timeout)
        return self.path

    @asynccontextmanager
    async def materialised(self) -> AsyncIterator[Path]:
        """Context-managed checkout. Does NOT delete the cache on exit."""
        path = await self.checkout()
        try:
            yield path
        finally:
            # Intentionally keep cache; cleanup is explicit via `purge()`.
            pass

    def purge(self) -> None:
        """Delete the entire local clone."""
        if self.path.exists():
            shutil.rmtree(self.path)


__all__ = ["DEFAULT_GIT_TIMEOUT_SECONDS", "RepoCheckout", "RepoCheckoutError"]
