"""Sandboxed subprocess execution with hard timeout + output cap.

Uses `asyncio.create_subprocess_exec` (NOT shell=True). On timeout the
process tree is killed. Stdout+stderr are captured and concatenated;
oversize output is truncated with a marker.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from pathlib import Path
from typing import Any, ClassVar

from . import ToolResult

_TRUNC_MARK = "\n...[truncated]..."


async def _run(
    argv: list[str],
    *,
    cwd: Path,
    timeout: float,
    max_bytes: int,
) -> ToolResult:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        return ToolResult(
            ok=False,
            output="",
            error=f"timeout after {timeout}s",
            metadata={"timeout": True},
        )

    combined = stdout_b + b"\n" + stderr_b if stderr_b else stdout_b
    truncated = False
    if len(combined) > max_bytes:
        combined = combined[:max_bytes]
        truncated = True
    text = combined.decode("utf-8", errors="replace")
    if truncated:
        text += _TRUNC_MARK
    return ToolResult(
        ok=proc.returncode == 0,
        output=text,
        error=None if proc.returncode == 0 else f"exit {proc.returncode}",
        metadata={
            "exit_code": proc.returncode,
            "truncated": truncated,
            "stdout_bytes": len(stdout_b),
            "stderr_bytes": len(stderr_b),
        },
    )


class PythonExecTool:
    name = "python_exec"
    description = "Run a short Python snippet in the same interpreter under a timeout."
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    }

    def __init__(
        self,
        scope_dir: Path | str,
        *,
        timeout_seconds: float = 30.0,
        max_output_bytes: int = 100_000,
    ) -> None:
        self.scope_dir = Path(scope_dir).resolve()
        self.timeout = timeout_seconds
        self.max_bytes = max_output_bytes

    async def execute(self, **kwargs: Any) -> ToolResult:
        code = kwargs.get("code")
        if not isinstance(code, str):
            return ToolResult(ok=False, output="", error="`code` must be a string")
        return await _run(
            [sys.executable, "-c", code],
            cwd=self.scope_dir,
            timeout=self.timeout,
            max_bytes=self.max_bytes,
        )


class BashExecTool:
    name = "bash_exec"
    description = "Run a bash snippet (no shell=True; passed via `bash -c`) under a timeout."
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    }

    def __init__(
        self,
        scope_dir: Path | str,
        *,
        timeout_seconds: float = 30.0,
        max_output_bytes: int = 100_000,
    ) -> None:
        self.scope_dir = Path(scope_dir).resolve()
        self.timeout = timeout_seconds
        self.max_bytes = max_output_bytes

    async def execute(self, **kwargs: Any) -> ToolResult:
        code = kwargs.get("code")
        if not isinstance(code, str):
            return ToolResult(ok=False, output="", error="`code` must be a string")
        bash = os.environ.get("BASH", "/bin/bash")
        return await _run(
            [bash, "-c", code],
            cwd=self.scope_dir,
            timeout=self.timeout,
            max_bytes=self.max_bytes,
        )


__all__ = ["BashExecTool", "PythonExecTool"]
