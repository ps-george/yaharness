"""Filesystem tools strictly scoped to a working directory.

All operations resolve paths via `Path.resolve()` and verify the result is
under the scope directory. Any escape attempt (`..`, absolute paths, or
symlink traversal) is rejected with `ToolResult(ok=False, error=...)`
rather than raised — callers are LLM agents and should see a structured
error they can recover from.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

import anyio

from . import ToolResult


def _scoped(scope: Path, raw: str) -> Path | None:
    """Resolve `raw` against `scope` and confirm containment. None on escape."""
    candidate = (scope / raw).resolve() if not os.path.isabs(raw) else Path(raw).resolve()
    try:
        candidate.relative_to(scope)
    except ValueError:
        return None
    return candidate


class ReadFileTool:
    name = "read_file"
    description = "Read a UTF-8 text file from the scoped working directory."
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to scope dir."},
        },
        "required": ["path"],
    }

    def __init__(self, scope_dir: Path | str) -> None:
        self.scope_dir = Path(scope_dir).resolve()

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path")
        if not isinstance(path, str):
            return ToolResult(ok=False, output="", error="`path` must be a string")
        resolved = _scoped(self.scope_dir, path)
        if resolved is None:
            return ToolResult(ok=False, output="", error=f"path escapes scope: {path}")
        if not resolved.exists():
            return ToolResult(ok=False, output="", error=f"no such file: {path}")
        if not resolved.is_file():
            return ToolResult(ok=False, output="", error=f"not a file: {path}")
        try:
            content = await anyio.Path(resolved).read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            return ToolResult(ok=False, output="", error=f"not utf-8: {exc}")
        return ToolResult(ok=True, output=content, metadata={"path": str(resolved)})


class WriteFileTool:
    name = "write_file"
    description = "Write a UTF-8 text file inside the scoped working directory."
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, scope_dir: Path | str) -> None:
        self.scope_dir = Path(scope_dir).resolve()

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path")
        content = kwargs.get("content")
        if not isinstance(path, str):
            return ToolResult(ok=False, output="", error="`path` must be a string")
        if not isinstance(content, str):
            return ToolResult(ok=False, output="", error="`content` must be a string")
        resolved = _scoped(self.scope_dir, path)
        if resolved is None:
            return ToolResult(ok=False, output="", error=f"path escapes scope: {path}")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        await anyio.Path(resolved).write_text(content, encoding="utf-8")
        return ToolResult(
            ok=True,
            output=f"wrote {len(content)} chars to {path}",
            metadata={"path": str(resolved), "bytes": len(content.encode("utf-8"))},
        )


class ListDirTool:
    name = "list_dir"
    description = "List entries in a directory inside the scoped working directory."
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "default": "."},
        },
        "required": [],
    }

    def __init__(self, scope_dir: Path | str) -> None:
        self.scope_dir = Path(scope_dir).resolve()

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path", ".")
        if not isinstance(path, str):
            return ToolResult(ok=False, output="", error="`path` must be a string")
        resolved = _scoped(self.scope_dir, path)
        if resolved is None:
            return ToolResult(ok=False, output="", error=f"path escapes scope: {path}")
        if not resolved.exists():
            return ToolResult(ok=False, output="", error=f"no such dir: {path}")
        if not resolved.is_dir():
            return ToolResult(ok=False, output="", error=f"not a dir: {path}")
        names = sorted(p.name + ("/" if p.is_dir() else "") for p in resolved.iterdir())
        return ToolResult(ok=True, output="\n".join(names), metadata={"count": len(names)})


class FindFilesTool:
    name = "find_files"
    description = "Glob for files within the scoped working directory."
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "e.g. '**/*.py'"},
        },
        "required": ["pattern"],
    }

    def __init__(self, scope_dir: Path | str) -> None:
        self.scope_dir = Path(scope_dir).resolve()

    async def execute(self, **kwargs: Any) -> ToolResult:
        pattern = kwargs.get("pattern")
        if not isinstance(pattern, str):
            return ToolResult(ok=False, output="", error="`pattern` must be a string")
        if os.path.isabs(pattern) or ".." in Path(pattern).parts:
            return ToolResult(ok=False, output="", error=f"unsafe pattern: {pattern}")
        matches = sorted(
            str(p.relative_to(self.scope_dir)) for p in self.scope_dir.glob(pattern) if p.is_file()
        )
        return ToolResult(ok=True, output="\n".join(matches), metadata={"count": len(matches)})


__all__ = ["FindFilesTool", "ListDirTool", "ReadFileTool", "WriteFileTool"]
