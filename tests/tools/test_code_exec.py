"""Tests for `python_exec` / `bash_exec`."""

from __future__ import annotations

from pathlib import Path

from yaharness.tools.code_exec import BashExecTool, PythonExecTool


async def test_python_success(tmp_path: Path) -> None:
    t = PythonExecTool(tmp_path)
    res = await t.execute(code="print('hello')")
    assert res.ok
    assert "hello" in res.output
    assert res.metadata["exit_code"] == 0


async def test_python_timeout(tmp_path: Path) -> None:
    t = PythonExecTool(tmp_path, timeout_seconds=0.3)
    res = await t.execute(code="import time; time.sleep(5)")
    assert not res.ok
    assert res.error is not None and "timeout" in res.error
    assert res.metadata.get("timeout") is True


async def test_python_stderr_capture(tmp_path: Path) -> None:
    t = PythonExecTool(tmp_path)
    res = await t.execute(code="import sys; sys.stderr.write('boom\\n'); sys.exit(2)")
    assert not res.ok
    assert "boom" in res.output
    assert res.metadata["exit_code"] == 2


async def test_python_truncation(tmp_path: Path) -> None:
    t = PythonExecTool(tmp_path, max_output_bytes=50)
    res = await t.execute(code="print('x' * 10000)")
    # Even though exit code 0, the output is capped.
    assert res.metadata["truncated"] is True
    assert "truncated" in res.output


async def test_python_bad_arg_type(tmp_path: Path) -> None:
    t = PythonExecTool(tmp_path)
    res = await t.execute(code=42)
    assert not res.ok


async def test_bash_success(tmp_path: Path) -> None:
    t = BashExecTool(tmp_path)
    res = await t.execute(code="echo hi")
    assert res.ok
    assert "hi" in res.output


async def test_bash_runs_in_scope(tmp_path: Path) -> None:
    t = BashExecTool(tmp_path)
    res = await t.execute(code="pwd")
    assert res.ok
    assert str(tmp_path.resolve()) in res.output


async def test_bash_bad_arg_type(tmp_path: Path) -> None:
    t = BashExecTool(tmp_path)
    res = await t.execute(code=None)
    assert not res.ok
