"""Tests for SWE-bench scoped tool registry (SWE-bench scoped tool registry)."""

from __future__ import annotations

from pathlib import Path

import pytest

from yaharness.benchmarks.swebench_harness import (
    SWEBENCH_SHELL_ALLOWLIST,
    RepoCheckout,
    make_swebench_tool_registry,
)


def _fake_checkout(tmp_path: Path) -> RepoCheckout:
    """Build a RepoCheckout pointing at a pre-populated fake repo dir."""
    cache = tmp_path / "cache"
    checkout = RepoCheckout(repo="acme/widget", base_commit="deadbeef", cache_dir=cache)
    # Pre-create the working tree so checkout() isn't required.
    checkout.path.mkdir(parents=True)
    (checkout.path / "hello.py").write_text("print('hi')\n", encoding="utf-8")
    (checkout.path / "sub").mkdir()
    (checkout.path / "sub" / "nested.py").write_text("x = 1\n", encoding="utf-8")
    return checkout


def test_swebench_tool_registry_scoped_to_checkout(tmp_path: Path) -> None:
    checkout = _fake_checkout(tmp_path)
    registry = make_swebench_tool_registry(checkout)
    names = {t.name for t in registry.all()}
    assert names == {"read_file", "write_file", "list_dir", "find_files", "shell"}


@pytest.mark.asyncio
async def test_swebench_read_file_scoped(tmp_path: Path) -> None:
    checkout = _fake_checkout(tmp_path)
    registry = make_swebench_tool_registry(checkout)
    rf = registry.get("read_file")
    ok = await rf.execute(path="hello.py")
    assert ok.ok and "print('hi')" in ok.output
    escape = await rf.execute(path="../../../etc/passwd")
    assert not escape.ok


@pytest.mark.asyncio
async def test_swebench_shell_allowlist(tmp_path: Path) -> None:
    checkout = _fake_checkout(tmp_path)
    registry = make_swebench_tool_registry(checkout)
    shell = registry.get("shell")
    # `ls` is on the allowlist.
    ok = await shell.execute(command="ls")
    assert ok.ok
    # `rm` is not.
    bad = await shell.execute(command="rm -rf /")
    assert not bad.ok
    assert "not allowed" in (bad.error or "")
    # Sanity: allowlist contents.
    assert "pytest" in SWEBENCH_SHELL_ALLOWLIST
    assert "git" in SWEBENCH_SHELL_ALLOWLIST
