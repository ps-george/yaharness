"""Tests for WebSearchTool — DDG HTML scraping + Tavily JSON, all mocked."""

from __future__ import annotations

import httpx
from pytest_httpx import HTTPXMock

from yaharness.tools.search import WebSearchTool, _parse_ddg_html

_DDG_HTML = """
<html><body>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FParis">
    Paris - Wikipedia
  </a>
  <a class="result__snippet">Paris is the capital of France.</a>
</div>
<div class="result">
  <a class="result__a" href="https://example.org/paris">
    Paris travel guide
  </a>
  <a class="result__snippet">Things to do in Paris.</a>
</div>
<div class="result">
  <a class="result__a" href="https://example.com/3">Third result</a>
  <a class="result__snippet">Snippet three.</a>
</div>
</body></html>
"""


def test_parse_ddg_html_extracts_unwrapped_urls() -> None:
    results = _parse_ddg_html(_DDG_HTML, max_results=5)
    assert len(results) == 3
    assert results[0]["title"] == "Paris - Wikipedia"
    assert results[0]["url"] == "https://en.wikipedia.org/wiki/Paris"
    assert "capital of France" in results[0]["snippet"]
    assert results[1]["url"] == "https://example.org/paris"


def test_parse_ddg_html_respects_max() -> None:
    results = _parse_ddg_html(_DDG_HTML, max_results=2)
    assert len(results) == 2


async def test_web_search_ddg(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://html.duckduckgo.com/html/",
        method="POST",
        text=_DDG_HTML,
        status_code=200,
    )
    tool = WebSearchTool(backend="ddg")
    res = await tool.execute(query="capital of france", max_results=2)
    assert res.ok
    assert res.metadata["backend"] == "ddg"
    results = res.metadata["results"]
    assert len(results) == 2
    assert "Paris - Wikipedia" in res.output
    assert "https://en.wikipedia.org/wiki/Paris" in res.output


async def test_web_search_ddg_http_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://html.duckduckgo.com/html/",
        method="POST",
        status_code=503,
    )
    tool = WebSearchTool(backend="ddg")
    res = await tool.execute(query="x")
    assert not res.ok
    assert res.error is not None and "503" in res.error


async def test_web_search_ddg_network_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ConnectError("boom"))
    tool = WebSearchTool(backend="ddg")
    res = await tool.execute(query="x")
    assert not res.ok
    assert res.error is not None and "http error" in res.error


async def test_web_search_tavily(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.tavily.com/search",
        method="POST",
        json={
            "results": [
                {
                    "title": "Paris",
                    "url": "https://en.wikipedia.org/wiki/Paris",
                    "content": "Capital of France.",
                },
                {
                    "title": "France",
                    "url": "https://en.wikipedia.org/wiki/France",
                    "content": "Country in Europe.",
                },
            ]
        },
        status_code=200,
    )
    tool = WebSearchTool(backend="tavily", api_key="test-key")
    res = await tool.execute(query="paris", max_results=2)
    assert res.ok
    assert res.metadata["backend"] == "tavily"
    assert len(res.metadata["results"]) == 2
    assert res.metadata["results"][0]["snippet"] == "Capital of France."


async def test_web_search_tavily_missing_key() -> None:
    tool = WebSearchTool(backend="tavily", api_key=None)
    # Force backend explicitly even though key absent.
    tool._api_key = None
    res = await tool.execute(query="x")
    assert not res.ok
    assert res.error is not None and "api_key" in res.error


async def test_web_search_auto_backend_no_key(monkeypatch: object) -> None:
    import os

    os.environ.pop("TAVILY_API_KEY", None)
    tool = WebSearchTool()
    assert tool.backend == "ddg"


async def test_web_search_auto_backend_with_key(monkeypatch: object) -> None:
    import os

    os.environ["TAVILY_API_KEY"] = "k"
    try:
        tool = WebSearchTool()
        assert tool.backend == "tavily"
    finally:
        os.environ.pop("TAVILY_API_KEY", None)


async def test_web_search_bad_args() -> None:
    tool = WebSearchTool(backend="ddg")
    assert not (await tool.execute(query="")).ok
    assert not (await tool.execute(query=123)).ok
    assert not (await tool.execute(query="x", max_results=0)).ok
