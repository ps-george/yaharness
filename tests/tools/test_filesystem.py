"""Tests for the filesystem tool family."""

from __future__ import annotations

from pathlib import Path

import pytest

from yaharness.tools import ToolRegistry
from yaharness.tools.filesystem import (
    FindFilesTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)


async def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    w = WriteFileTool(tmp_path)
    r = ReadFileTool(tmp_path)
    res = await w.execute(path="a/b.txt", content="hello")
    assert res.ok
    assert (tmp_path / "a" / "b.txt").read_text() == "hello"
    got = await r.execute(path="a/b.txt")
    assert got.ok
    assert got.output == "hello"


async def test_read_missing(tmp_path: Path) -> None:
    r = ReadFileTool(tmp_path)
    res = await r.execute(path="nope.txt")
    assert not res.ok
    assert res.error is not None and "no such file" in res.error


async def test_read_is_dir(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    r = ReadFileTool(tmp_path)
    res = await r.execute(path="sub")
    assert not res.ok
    assert res.error is not None and "not a file" in res.error


async def test_path_escape_rejected_read(tmp_path: Path) -> None:
    r = ReadFileTool(tmp_path)
    res = await r.execute(path="../escape.txt")
    assert not res.ok
    assert res.error is not None and "escapes scope" in res.error


async def test_path_escape_rejected_absolute(tmp_path: Path) -> None:
    r = ReadFileTool(tmp_path)
    res = await r.execute(path="/etc/passwd")
    assert not res.ok
    assert res.error is not None and "escapes scope" in res.error


async def test_path_escape_rejected_write(tmp_path: Path) -> None:
    w = WriteFileTool(tmp_path)
    res = await w.execute(path="../x.txt", content="bad")
    assert not res.ok


async def test_bad_argument_types(tmp_path: Path) -> None:
    r = ReadFileTool(tmp_path)
    res = await r.execute(path=123)
    assert not res.ok
    w = WriteFileTool(tmp_path)
    res2 = await w.execute(path="ok.txt", content=42)
    assert not res2.ok


async def test_list_dir_with_subdirs(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    t = ListDirTool(tmp_path)
    res = await t.execute()
    assert res.ok
    lines = res.output.splitlines()
    assert "a.txt" in lines
    assert "b.txt" in lines
    assert "sub/" in lines


async def test_list_dir_missing(tmp_path: Path) -> None:
    t = ListDirTool(tmp_path)
    res = await t.execute(path="nope")
    assert not res.ok


async def test_list_dir_on_file(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("y")
    t = ListDirTool(tmp_path)
    res = await t.execute(path="x.txt")
    assert not res.ok


async def test_find_files_glob(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("")
    (tmp_path / "c.txt").write_text("")
    t = FindFilesTool(tmp_path)
    res = await t.execute(pattern="**/*.py")
    assert res.ok
    matches = set(res.output.splitlines())
    assert "a.py" in matches
    assert "sub/b.py" in matches
    assert "c.txt" not in res.output


async def test_find_files_unsafe_pattern(tmp_path: Path) -> None:
    t = FindFilesTool(tmp_path)
    res = await t.execute(pattern="../*")
    assert not res.ok


async def test_registry_schemas(tmp_path: Path) -> None:
    reg = ToolRegistry()
    reg.register(ReadFileTool(tmp_path))
    reg.register(WriteFileTool(tmp_path))
    schemas = reg.schemas()
    assert len(schemas) == 2
    names = {s["name"] for s in schemas}
    assert names == {"read_file", "write_file"}
    for s in schemas:
        assert "input_schema" in s
        assert s["input_schema"]["type"] == "object"

    assert reg.get("read_file").name == "read_file"
    with pytest.raises(KeyError):
        reg.get("nope")
    with pytest.raises(ValueError):
        reg.register(ReadFileTool(tmp_path))
    assert len(reg.all()) == 2
