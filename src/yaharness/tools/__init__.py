"""Pluggable async tool framework.

Defines the `Tool` protocol, the `ToolResult` model, and a `ToolRegistry`
that exposes function-calling schemas in the shape expected by OpenAI /
Anthropic-style APIs.

Concrete tools live in sibling modules:
  - `filesystem` — scoped file IO
  - `code_exec` — Python / bash execution with timeout
  - `web` — http_get with on-disk cache
  - `shell` — generic shell with allowlist
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class ToolResult(BaseModel):
    """Uniform result type for every tool invocation."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Tool(Protocol):
    """Async tool contract.

    Implementations must expose a canonical snake_case `name`, a
    human/LLM-readable `description`, a JSON-schema `parameters_schema`
    describing kwargs, and an async `execute` that returns a `ToolResult`.
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]

    async def execute(self, **kwargs: Any) -> ToolResult: ...


class ToolRegistry:
    """Holds tool instances and emits function-calling schemas."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def all(self) -> Sequence[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        """Function-calling schema for every registered tool.

        Shape matches the common denominator of OpenAI's `tools` parameter
        and Anthropic's `tools` parameter — a list of dicts with `name`,
        `description`, and `input_schema` (JSON schema).
        """
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters_schema,
            }
            for t in self._tools.values()
        ]


__all__ = ["Tool", "ToolRegistry", "ToolResult"]
