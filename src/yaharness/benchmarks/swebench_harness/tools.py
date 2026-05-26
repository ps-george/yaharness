"""SWE-bench-specific tool registry.

Builds a `ToolRegistry` scoped to a materialised `RepoCheckout`. The agent
gets scoped read/write filesystem tools plus a shell tool restricted to an
allowlist (pytest, python, grep, ls, cat, sed, find, git diff) so it can
explore the repo and produce a unified diff.

Filesystem and shell tools all resolve paths under `checkout.path` and
refuse to escape via `..`, absolute paths, or symlinks (delegated to
`tools.filesystem._scoped`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from ...tools import Tool, ToolRegistry
from ...tools.filesystem import FindFilesTool, ListDirTool, ReadFileTool, WriteFileTool
from ...tools.shell import ShellTool
from .runner import RepoCheckout

# Allowlist of shell command prefixes the agent may run inside the checkout.
# Kept tight: read-only inspection plus pytest/python for verification.
SWEBENCH_SHELL_ALLOWLIST: tuple[str, ...] = (
    "pytest",
    "python",
    "python3",
    "grep",
    "ls",
    "cat",
    "sed",
    "find",
    "git",
)


def make_swebench_tool_registry(
    checkout: RepoCheckout,
    *,
    shell_timeout_seconds: float = 30.0,
) -> ToolRegistry:
    """Build a tool registry scoped to a materialised repo checkout.

    Registers:
      - read_file, write_file, list_dir, find_files — all scoped to
        ``checkout.path``
      - shell — allowlist of safe commands, scoped CWD = ``checkout.path``

    The checkout must already have been materialised (``await
    checkout.checkout()``) — this function only wires tools to the path;
    it does NOT perform git operations.
    """
    scope = checkout.path
    registry = ToolRegistry()
    registry.register(cast(Tool, ReadFileTool(scope_dir=scope)))
    registry.register(cast(Tool, WriteFileTool(scope_dir=scope)))
    registry.register(cast(Tool, ListDirTool(scope_dir=scope)))
    registry.register(cast(Tool, FindFilesTool(scope_dir=scope)))
    registry.register(
        cast(
            Tool,
            ShellTool(
                scope_dir=scope,
                allowed_prefixes=list(SWEBENCH_SHELL_ALLOWLIST),
                timeout_seconds=shell_timeout_seconds,
            ),
        )
    )
    return registry


DEFAULT_SWEBENCH_CACHE = Path.home() / ".cache" / "yaharness" / "swebench"


async def maybe_build_swebench_registry_from_context(
    context: dict[str, Any],
    *,
    cache_dir: Path | None = None,
) -> ToolRegistry | None:
    """If `context` describes a SWE-bench problem, materialise a checkout and
    return a scoped ``ToolRegistry``. Returns ``None`` otherwise.

    The agent adapters call this at the top of ``__call__`` and use the
    returned registry (when not ``None``) for the run. Context shape comes
    from :func:`SWEBenchVerifiedAdapter._row_to_problem`: ``repo`` +
    ``base_commit`` keys. We do NOT require an explicit ``benchmark`` flag
    — presence of both keys is enough to identify the case.
    """
    repo = context.get("repo")
    base_commit = context.get("base_commit")
    if not isinstance(repo, str) or not isinstance(base_commit, str):
        return None
    if "/" not in repo:
        return None
    cache = cache_dir or DEFAULT_SWEBENCH_CACHE
    checkout = RepoCheckout(repo=repo, base_commit=base_commit, cache_dir=cache)
    await checkout.checkout()
    return make_swebench_tool_registry(checkout)


__all__ = [
    "DEFAULT_SWEBENCH_CACHE",
    "SWEBENCH_SHELL_ALLOWLIST",
    "make_swebench_tool_registry",
    "maybe_build_swebench_registry_from_context",
]
