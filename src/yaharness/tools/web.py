"""HTTP GET with on-disk cache.

Cache key is sha256 of the URL. Cache entries store body + timestamp +
content-type as JSON. TTL is enforced on read. Bodies larger than
`max_bytes` are rejected (the partial content is discarded and `ok=False`
is returned).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, ClassVar

import httpx

from . import ToolResult

_DEFAULT_TTL = 7 * 24 * 3600  # 7 days
_DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MB


class HttpGetTool:
    name = "http_get"
    description = "GET a URL with on-disk caching; returns body text (or error)."
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    }

    def __init__(
        self,
        *,
        cache_dir: Path | str = ".cache/web",
        ttl_seconds: float = _DEFAULT_TTL,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl_seconds
        self.max_bytes = max_bytes
        self._owns_client = client is None
        self._client = client

    def _cache_path(self, url: str) -> Path:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, url: str) -> ToolResult | None:
        p = self._cache_path(url)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if time.time() - float(data["ts"]) > self.ttl:
            return None
        return ToolResult(
            ok=True,
            output=str(data["body"]),
            metadata={
                "cached": True,
                "content_type": data.get("content_type", ""),
                "status_code": int(data.get("status_code", 200)),
            },
        )

    def _write_cache(self, url: str, body: str, content_type: str, status_code: int) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(url).write_text(
            json.dumps(
                {
                    "ts": time.time(),
                    "body": body,
                    "content_type": content_type,
                    "status_code": status_code,
                }
            ),
            encoding="utf-8",
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        url = kwargs.get("url")
        if not isinstance(url, str):
            return ToolResult(ok=False, output="", error="`url` must be a string")

        cached = self._read_cache(url)
        if cached is not None:
            return cached

        client = self._client or httpx.AsyncClient()
        try:
            resp = await client.get(url, follow_redirects=True)
        except httpx.HTTPError as exc:
            if self._owns_client:
                await client.aclose()
            return ToolResult(ok=False, output="", error=f"http error: {exc}")

        try:
            if resp.status_code >= 400:
                return ToolResult(
                    ok=False,
                    output="",
                    error=f"http {resp.status_code}",
                    metadata={"status_code": resp.status_code},
                )
            content = resp.content
            if len(content) > self.max_bytes:
                return ToolResult(
                    ok=False,
                    output="",
                    error=f"response too large: {len(content)} > {self.max_bytes}",
                    metadata={"status_code": resp.status_code, "bytes": len(content)},
                )
            try:
                text = content.decode(resp.encoding or "utf-8", errors="replace")
            except LookupError:
                text = content.decode("utf-8", errors="replace")
            ct = resp.headers.get("content-type", "")
            self._write_cache(url, text, ct, resp.status_code)
            return ToolResult(
                ok=True,
                output=text,
                metadata={
                    "cached": False,
                    "content_type": ct,
                    "status_code": resp.status_code,
                },
            )
        finally:
            if self._owns_client:
                await client.aclose()


def html_to_text(html: str, *, max_chars: int = 50_000) -> str:
    """Extract readable text from an HTML document.

    Strips ``<script>``, ``<style>``, ``<noscript>`` and similar boilerplate,
    then returns the body text with internal whitespace collapsed and lines
    trimmed. Output is capped at ``max_chars`` to keep LLM context tractable
    (truncation is marked with a trailing ``"\\n...[truncated]"``).
    """
    from selectolax.parser import HTMLParser

    tree = HTMLParser(html)
    for tag in ("script", "style", "noscript", "template", "svg"):
        for node in tree.css(tag):
            node.decompose()
    body = tree.body if tree.body is not None else tree.root
    if body is None:
        return ""
    text = body.text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "\n...[truncated]"
    return cleaned


__all__ = ["HttpGetTool", "html_to_text"]
