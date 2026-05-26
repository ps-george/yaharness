"""Tests for the allowlisted shell tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from yaharness.tools.shell import ShellTool


async def test_shell_allowed(tmp_path: Path) -> None:
    t = ShellTool(tmp_path, allowed_prefixes=["echo"])
    res = await t.execute(command="echo hello world")
    assert res.ok
    assert "hello world" in res.output


async def test_shell_disallowed(tmp_path: Path) -> None:
    t = ShellTool(tmp_path, allowed_prefixes=["echo"])
    res = await t.execute(command="rm -rf /")
    assert not res.ok
    assert res.error is not None and "not allowed" in res.error


async def test_shell_empty(tmp_path: Path) -> None:
    t = ShellTool(tmp_path, allowed_prefixes=["echo"])
    res = await t.execute(command="   ")
    assert not res.ok


async def test_shell_bad_split(tmp_path: Path) -> None:
    t = ShellTool(tmp_path, allowed_prefixes=["echo"])
    res = await t.execute(command="echo 'unterminated")
    assert not res.ok


async def test_shell_timeout(tmp_path: Path) -> None:
    t = ShellTool(tmp_path, allowed_prefixes=["sleep"], timeout_seconds=0.2)
    res = await t.execute(command="sleep 5")
    assert not res.ok
    assert res.error is not None and "timeout" in res.error


async def test_shell_bad_arg(tmp_path: Path) -> None:
    t = ShellTool(tmp_path, allowed_prefixes=["echo"])
    res = await t.execute(command=123)
    assert not res.ok


async def test_shell_requires_allowlist(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ShellTool(tmp_path, allowed_prefixes=[])
