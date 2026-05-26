"""Web-search tool.

Backends
--------
1. ``"ddg"`` (default, no API key required) — scrapes the DuckDuckGo HTML
   endpoint (``https://html.duckduckgo.com/html/``) with ``httpx`` + parses
   results with ``selectolax``.
2. ``"tavily"`` — JSON API at ``https://api.tavily.com/search``. Active when
   ``TAVILY_API_KEY`` is present in the environment (or passed via
   ``api_key=``). Auto-selected if a key is detected.

Both backends share a unified result schema: each result is a dict with
``title``, ``url``, ``snippet``. The tool returns the results as a small
human-readable string in ``output`` and the structured list in
``metadata["results"]``.

Network operations are bounded by ``timeout`` seconds.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar, Literal
from urllib.parse import unquote, urlparse

import httpx

from . import ToolResult

Backend = Literal["ddg", "tavily"]


def _parse_ddg_html(html: str, max_results: int) -> list[dict[str, str]]:
    """Parse DuckDuckGo HTML results. Returns up to ``max_results`` items."""
    from selectolax.parser import HTMLParser

    tree = HTMLParser(html)
    results: list[dict[str, str]] = []
    for node in tree.css("div.result"):
        if len(results) >= max_results:
            break
        a = node.css_first("a.result__a")
        if a is None:
            continue
        href = a.attributes.get("href") or ""
        # DDG wraps real URLs in a redirect: //duckduckgo.com/l/?uddg=<encoded>
        url = _unwrap_ddg_redirect(href)
        title = a.text(strip=True)
        snippet_node = node.css_first("a.result__snippet") or node.css_first(".result__snippet")
        snippet = snippet_node.text(strip=True) if snippet_node is not None else ""
        if title and url:
            results.append({"title": title, "url": url, "snippet": snippet})
    return results


def _unwrap_ddg_redirect(href: str) -> str:
    if not href:
        return ""
    parsed = urlparse(href if "://" in href else f"https:{href}")
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        # Find uddg=...
        for kv in parsed.query.split("&"):
            if kv.startswith("uddg="):
                return unquote(kv[len("uddg=") :])
    return href


def _format_results(results: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}")
    return "\n".join(lines)


class WebSearchTool:
    name = "web_search"
    description = "Search the web; returns ranked title/url/snippet results."
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    def __init__(
        self,
        *,
        backend: Backend | None = None,
        api_key: str | None = None,
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if backend is None:
            backend = "tavily" if self._api_key else "ddg"
        self.backend: Backend = backend
        self.timeout = timeout
        self._owns_client = client is None
        self._client = client

    async def _ddg(self, client: httpx.AsyncClient, query: str, max_results: int) -> ToolResult:
        try:
            resp = await client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers={"User-Agent": "Mozilla/5.0 yaharness"},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            return ToolResult(ok=False, output="", error=f"search http error: {exc}")
        if resp.status_code >= 400:
            return ToolResult(
                ok=False,
                output="",
                error=f"search http {resp.status_code}",
                metadata={"status_code": resp.status_code},
            )
        results = _parse_ddg_html(resp.text, max_results)
        return ToolResult(
            ok=True,
            output=_format_results(results),
            metadata={"results": results, "backend": "ddg"},
        )

    async def _tavily(self, client: httpx.AsyncClient, query: str, max_results: int) -> ToolResult:
        if not self._api_key:
            return ToolResult(ok=False, output="", error="tavily backend requires api_key")
        try:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "max_results": max_results,
                },
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            return ToolResult(ok=False, output="", error=f"search http error: {exc}")
        if resp.status_code >= 400:
            return ToolResult(
                ok=False,
                output="",
                error=f"search http {resp.status_code}",
                metadata={"status_code": resp.status_code},
            )
        try:
            payload = resp.json()
        except ValueError as exc:
            return ToolResult(ok=False, output="", error=f"bad tavily json: {exc}")
        raw = payload.get("results", []) if isinstance(payload, dict) else []
        results: list[dict[str, str]] = []
        for item in raw[:max_results]:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": str(item.get("title", "")),
                    "url": str(item.get("url", "")),
                    "snippet": str(item.get("content", "") or item.get("snippet", "")),
                }
            )
        return ToolResult(
            ok=True,
            output=_format_results(results),
            metadata={"results": results, "backend": "tavily"},
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query")
        max_results = kwargs.get("max_results", 5)
        if not isinstance(query, str) or not query.strip():
            return ToolResult(ok=False, output="", error="`query` must be a non-empty string")
        if not isinstance(max_results, int) or max_results <= 0:
            return ToolResult(ok=False, output="", error="`max_results` must be a positive int")
        client = self._client or httpx.AsyncClient()
        try:
            if self.backend == "tavily":
                return await self._tavily(client, query, max_results)
            return await self._ddg(client, query, max_results)
        finally:
            if self._owns_client:
                await client.aclose()


__all__ = ["WebSearchTool"]
