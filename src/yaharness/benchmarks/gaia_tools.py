"""GAIA-specific tool registry.

Bundles the tools an agent actually needs to attempt GAIA Level 1 problems:

- ``http_get`` — fetch web pages (with cache)
- ``web_search`` — find candidate pages
- ``read_file`` / ``list_dir`` / ``find_files`` — explore attached files
- ``parse_csv`` / ``parse_excel`` / ``parse_pdf`` / ``parse_eml`` — extract
  structured content from attachments

Filesystem tools are scoped read-only to the problem's attachment directory.
``WriteFileTool`` is deliberately excluded — GAIA tasks do not require
producing files, and exposing write capability widens the agent's failure
surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import httpx

from ..tools import Tool, ToolRegistry
from ..tools.filesystem import FindFilesTool, ListDirTool, ReadFileTool
from ..tools.parse import ParseCsvTool, ParseEmlTool, ParseExcelTool, ParsePdfTool
from ..tools.search import WebSearchTool
from ..tools.web import HttpGetTool


def gaia_tool_registry(
    *,
    attachments_dir: Path | str,
    cache_dir: Path | str = ".cache/web",
    search_client: httpx.AsyncClient | None = None,
) -> ToolRegistry:
    """Build the tool registry an agent uses for GAIA problems.

    Parameters
    ----------
    attachments_dir:
        Directory containing the problem's attached files. Filesystem tools
        are scoped here (read-only — no write tool registered).
    cache_dir:
        On-disk HTTP cache directory for ``http_get``.
    search_client:
        Optional shared ``httpx.AsyncClient`` for ``web_search`` (useful in
        tests / for connection pooling).
    """
    registry = ToolRegistry()
    # All tool classes use ClassVar for `parameters_schema`, which mypy's
    # protocol checker treats as a mismatch against the Protocol's instance
    # attribute. Cast keeps the runtime structural conformance intact.
    registry.register(cast(Tool, HttpGetTool(cache_dir=cache_dir)))
    registry.register(cast(Tool, WebSearchTool(client=search_client)))
    registry.register(cast(Tool, ReadFileTool(scope_dir=attachments_dir)))
    registry.register(cast(Tool, ListDirTool(scope_dir=attachments_dir)))
    registry.register(cast(Tool, FindFilesTool(scope_dir=attachments_dir)))
    registry.register(cast(Tool, ParseCsvTool()))
    registry.register(cast(Tool, ParseExcelTool()))
    registry.register(cast(Tool, ParsePdfTool()))
    registry.register(cast(Tool, ParseEmlTool()))
    return registry


__all__ = ["gaia_tool_registry"]
