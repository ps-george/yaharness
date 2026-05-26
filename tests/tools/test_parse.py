"""Tests for parse_csv / parse_excel / parse_pdf / parse_eml — all offline."""

from __future__ import annotations

from pathlib import Path

from yaharness.tools.parse import (
    ParseCsvTool,
    ParseEmlTool,
    ParseExcelTool,
    ParsePdfTool,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "parse"


# ---- parse_csv ------------------------------------------------------------


async def test_parse_csv_basic() -> None:
    tool = ParseCsvTool()
    res = await tool.execute(path=str(FIXTURES / "sample.csv"))
    assert res.ok
    assert "name | age | city" in res.output
    assert "Alice | 30 | Paris" in res.output
    assert res.metadata["rows"] == 3
    assert res.metadata["columns"] == 3


async def test_parse_csv_max_rows() -> None:
    tool = ParseCsvTool()
    res = await tool.execute(path=str(FIXTURES / "sample.csv"), max_rows=1)
    assert res.ok
    assert "Alice" in res.output
    assert "Bob" not in res.output


async def test_parse_csv_missing() -> None:
    tool = ParseCsvTool()
    res = await tool.execute(path="/nonexistent/file.csv")
    assert not res.ok
    assert res.error is not None and "no such file" in res.error


async def test_parse_csv_bad_arg() -> None:
    tool = ParseCsvTool()
    res = await tool.execute(path=123)
    assert not res.ok
    res = await tool.execute(path=str(FIXTURES / "sample.csv"), max_rows=-1)
    assert not res.ok


# ---- parse_excel ----------------------------------------------------------


async def test_parse_excel_default_sheet() -> None:
    tool = ParseExcelTool()
    res = await tool.execute(path=str(FIXTURES / "sample.xlsx"))
    assert res.ok
    assert "item | qty | price" in res.output
    assert "apple | 3 | 1.5" in res.output
    assert res.metadata["sheet"] == "Sheet1"
    assert set(res.metadata["sheets"]) == {"Sheet1", "Other"}


async def test_parse_excel_named_sheet() -> None:
    tool = ParseExcelTool()
    res = await tool.execute(path=str(FIXTURES / "sample.xlsx"), sheet="Other")
    assert res.ok
    assert "a | b" in res.output
    assert "x | y" in res.output


async def test_parse_excel_missing_sheet() -> None:
    tool = ParseExcelTool()
    res = await tool.execute(path=str(FIXTURES / "sample.xlsx"), sheet="NoSuch")
    assert not res.ok
    assert res.error is not None


async def test_parse_excel_bad_args() -> None:
    tool = ParseExcelTool()
    assert not (await tool.execute(path=42)).ok
    assert not (await tool.execute(path=str(FIXTURES / "sample.xlsx"), sheet=99)).ok


# ---- parse_pdf ------------------------------------------------------------


async def test_parse_pdf_extracts_text() -> None:
    tool = ParsePdfTool()
    res = await tool.execute(path=str(FIXTURES / "sample.pdf"))
    assert res.ok
    assert "Hello GAIA page 1" in res.output
    assert "Page two body" in res.output
    assert res.metadata["pages"] == 2


async def test_parse_pdf_max_pages() -> None:
    tool = ParsePdfTool()
    res = await tool.execute(path=str(FIXTURES / "sample.pdf"), max_pages=1)
    assert res.ok
    assert "Hello GAIA page 1" in res.output
    assert "Page two body" not in res.output


async def test_parse_pdf_truncation() -> None:
    tool = ParsePdfTool()
    res = await tool.execute(path=str(FIXTURES / "sample.pdf"), max_chars=10)
    assert res.ok
    assert res.metadata["truncated"] is True
    assert res.output.endswith("...[truncated]")


async def test_parse_pdf_missing() -> None:
    tool = ParsePdfTool()
    res = await tool.execute(path="/nonexistent.pdf")
    assert not res.ok


# ---- parse_eml ------------------------------------------------------------


async def test_parse_eml_basic() -> None:
    tool = ParseEmlTool()
    res = await tool.execute(path=str(FIXTURES / "sample.eml"))
    assert res.ok
    assert "From: alice@example.com" in res.output
    assert "Subject: GAIA test message" in res.output
    assert "the body of the test message" in res.output
    assert res.metadata["headers"]["Subject"] == "GAIA test message"
    assert res.metadata["body_content_type"] == "text/plain"


async def test_parse_eml_truncation() -> None:
    tool = ParseEmlTool()
    res = await tool.execute(path=str(FIXTURES / "sample.eml"), max_chars=20)
    assert res.ok
    assert res.metadata["truncated"] is True


async def test_parse_eml_missing() -> None:
    tool = ParseEmlTool()
    res = await tool.execute(path="/nope.eml")
    assert not res.ok
