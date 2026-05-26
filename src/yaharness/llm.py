"""LLM client abstraction with a mock implementation for tests and an
OpenRouter-backed real client for production.

No real network calls are exercised in this brief; the OpenRouter client
is here so downstream briefs can wire it up without re-touching the
abstraction.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """A single LLM completion response, normalised across providers."""

    model_config = ConfigDict(frozen=True)

    content: str
    cost_usd: float
    model: str
    tokens_in: int
    tokens_out: int


@runtime_checkable
class LLMClient(Protocol):
    """Minimal protocol all LLM clients implement.

    `messages` is the standard role/content shape (OpenAI/OpenRouter
    compatible). `system` is passed separately because some providers
    treat it specially.
    """

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse: ...


class MockLLMClientExhausted(RuntimeError):  # noqa: N818 - "Exhausted" reads better than "ExhaustedError"
    """Raised when a MockLLMClient is called with no queued responses left."""


class MockLLMClient:
    """Deterministic LLM client for tests.

    Plays back pre-queued responses in FIFO order. Records every call's
    `(system, messages)` for assertions. Raises
    :class:`MockLLMClientExhausted` if the queue empties — tests should
    queue exactly the responses they expect.
    """

    def __init__(self, responses: Sequence[LLMResponse] | None = None) -> None:
        self._queue: deque[LLMResponse] = deque(responses or ())
        self._calls: list[tuple[str, list[dict[str, str]]]] = []

    def queue_response(self, response: LLMResponse) -> None:
        self._queue.append(response)

    def queue_text(
        self,
        text: str,
        *,
        cost_usd: float = 0.001,
        model: str = "mock",
        tokens_in: int = 10,
        tokens_out: int = 10,
    ) -> None:
        self._queue.append(
            LLMResponse(
                content=text,
                cost_usd=cost_usd,
                model=model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
        )

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        # Record call for test introspection.
        self._calls.append((system, [dict(m) for m in messages]))
        if not self._queue:
            raise MockLLMClientExhausted(
                f"MockLLMClient has no queued responses (call #{len(self._calls)})"
            )
        return self._queue.popleft()

    def call_count(self) -> int:
        return len(self._calls)

    def calls(self) -> Sequence[tuple[str, list[dict[str, str]]]]:
        return tuple(self._calls)

    def remaining(self) -> int:
        return len(self._queue)


# --- OpenRouter ---------------------------------------------------------

# Pricing is the unified table in cost.py; this module delegates so there is
# exactly one source of truth.
from yaharness.cost import MODEL_PRICES, estimate_cost  # noqa: E402


def _price_for(model: str, tokens_in: int, tokens_out: int) -> float:
    if model not in MODEL_PRICES:
        logger.warning("No price entry for model %r; reporting cost_usd=0.0", model)
        return 0.0
    return estimate_cost(model, tokens_in, tokens_out)


class OpenRouterClient:
    """Real LLM client backed by OpenRouter via httpx.AsyncClient."""

    _ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OpenRouterClient requires an api_key or OPENROUTER_API_KEY env var")
        self._api_key = key
        self._model = model
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=timeout)

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        resp = await self._http.post(self._ENDPOINT, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        tokens_in = int(usage.get("prompt_tokens", 0))
        tokens_out = int(usage.get("completion_tokens", 0))
        cost = _price_for(self._model, tokens_in, tokens_out)
        return LLMResponse(
            content=content,
            cost_usd=cost,
            model=self._model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()
