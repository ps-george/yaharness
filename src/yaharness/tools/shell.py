"""Generic shell runner with an allowlist of command prefixes.

A command is allowed iff its first argv token (after `shlex.split`)
appears in the allowlist passed at construction. No globbing or shell
metacharacters are expanded by us; commands are run via
`asyncio.create_subprocess_exec` (no `shell=True`).
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, ClassVar

from . import ToolResult
from .code_exec import _run


class ShellTool:
    name = "shell"
    description = "Run an allowlisted shell command (no shell metacharacters)."
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }

    def __init__(
        self,
        scope_dir: Path | str,
        allowed_prefixes: list[str],
        *,
        timeout_seconds: float = 30.0,
        max_output_bytes: int = 100_000,
    ) -> None:
        if not allowed_prefixes:
            raise ValueError("allowed_prefixes must be non-empty")
        self.scope_dir = Path(scope_dir).resolve()
        self.allowed = set(allowed_prefixes)
        self.timeout = timeout_seconds
        self.max_bytes = max_output_bytes

    async def execute(self, **kwargs: Any) -> ToolResult:
        command = kwargs.get("command")
        if not isinstance(command, str):
            return ToolResult(ok=False, output="", error="`command` must be a string")
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return ToolResult(ok=False, output="", error=f"bad shell split: {exc}")
        if not argv:
            return ToolResult(ok=False, output="", error="empty command")
        if argv[0] not in self.allowed:
            return ToolResult(
                ok=False,
                output="",
                error=f"command not allowed: {argv[0]} (allowed: {sorted(self.allowed)})",
            )
        return await _run(
            argv,
            cwd=self.scope_dir,
            timeout=self.timeout,
            max_bytes=self.max_bytes,
        )


__all__ = ["ShellTool"]
