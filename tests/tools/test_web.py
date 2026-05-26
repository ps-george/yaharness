"""Tests for `http_get` with on-disk caching, using pytest-httpx."""

from __future__ import annotations

from pathlib import Path

from pytest_httpx import HTTPXMock

from yaharness.tools.web import HttpGetTool, html_to_text


async def test_http_get_basic(tmp_path: Path, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url="https://example.com/a", text="hi", status_code=200)
    t = HttpGetTool(cache_dir=tmp_path / "cache")
    res = await t.execute(url="https://example.com/a")
    assert res.ok
    assert res.output == "hi"
    assert res.metadata["cached"] is False


async def test_http_get_cache_hit(tmp_path: Path, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url="https://example.com/c", text="cached-body", status_code=200)
    t = HttpGetTool(cache_dir=tmp_path / "cache")
    r1 = await t.execute(url="https://example.com/c")
    assert r1.ok and r1.metadata["cached"] is False
    # Second call: no new mock response registered; cache must satisfy it.
    r2 = await t.execute(url="https://example.com/c")
    assert r2.ok
    assert r2.output == "cached-body"
    assert r2.metadata["cached"] is True


async def test_http_get_cache_ttl_expired(tmp_path: Path, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url="https://example.com/d", text="v1")
    httpx_mock.add_response(url="https://example.com/d", text="v2")
    t = HttpGetTool(cache_dir=tmp_path / "cache", ttl_seconds=-1.0)
    r1 = await t.execute(url="https://example.com/d")
    assert r1.output == "v1"
    r2 = await t.execute(url="https://example.com/d")
    # TTL is negative, so cache is always stale.
    assert r2.output == "v2"


async def test_http_get_oversize(tmp_path: Path, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url="https://example.com/big", content=b"x" * 1000)
    t = HttpGetTool(cache_dir=tmp_path / "cache", max_bytes=100)
    res = await t.execute(url="https://example.com/big")
    assert not res.ok
    assert res.error is not None and "too large" in res.error


async def test_http_get_http_error_status(tmp_path: Path, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url="https://example.com/404", status_code=404)
    t = HttpGetTool(cache_dir=tmp_path / "cache")
    res = await t.execute(url="https://example.com/404")
    assert not res.ok
    assert res.error is not None and "404" in res.error


async def test_http_get_network_error(tmp_path: Path, httpx_mock: HTTPXMock) -> None:
    import httpx

    httpx_mock.add_exception(httpx.ConnectError("boom"))
    t = HttpGetTool(cache_dir=tmp_path / "cache")
    res = await t.execute(url="https://example.com/err")
    assert not res.ok
    assert res.error is not None and "http error" in res.error


async def test_http_get_bad_arg(tmp_path: Path) -> None:
    t = HttpGetTool(cache_dir=tmp_path / "cache")
    res = await t.execute(url=123)
    assert not res.ok


def test_html_to_text_strips_boilerplate() -> None:
    html = """
    <html><head><style>.x{color:red}</style></head>
    <body>
      <script>alert(1)</script>
      <h1>Title</h1>
      <p>Hello <b>world</b>.</p>
      <p>Line two.</p>
    </body></html>
    """
    out = html_to_text(html)
    assert "Title" in out
    assert "Hello" in out and "world" in out
    assert "Line two." in out
    assert "alert(1)" not in out
    assert "color:red" not in out


def test_html_to_text_truncates() -> None:
    html = "<html><body><p>" + ("x" * 200) + "</p></body></html>"
    out = html_to_text(html, max_chars=50)
    assert out.endswith("...[truncated]")
    assert len(out) <= 50 + len("\n...[truncated]")
