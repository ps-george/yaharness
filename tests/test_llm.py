"""Tests for the LLM client abstraction (MockLLMClient only — no real calls)."""

from __future__ import annotations

import pytest

from yaharness.llm import LLMResponse, MockLLMClient, MockLLMClientExhausted


async def test_queue_response_fifo_playback() -> None:
    mock = MockLLMClient()
    mock.queue_text("first")
    mock.queue_text("second")
    r1 = await mock.complete(system="s", messages=[{"role": "user", "content": "x"}])
    r2 = await mock.complete(system="s", messages=[{"role": "user", "content": "y"}])
    assert r1.content == "first"
    assert r2.content == "second"


async def test_queue_via_response_object() -> None:
    resp = LLMResponse(content="hi", cost_usd=0.5, model="m", tokens_in=1, tokens_out=2)
    mock = MockLLMClient([resp])
    out = await mock.complete(system="s", messages=[])
    assert out is resp


async def test_exhaustion_raises() -> None:
    mock = MockLLMClient()
    mock.queue_text("only")
    await mock.complete(system="s", messages=[])
    with pytest.raises(MockLLMClientExhausted):
        await mock.complete(system="s", messages=[])


async def test_calls_recorded() -> None:
    mock = MockLLMClient()
    mock.queue_text("a")
    mock.queue_text("b")
    await mock.complete(system="sys1", messages=[{"role": "user", "content": "m1"}])
    await mock.complete(system="sys2", messages=[{"role": "user", "content": "m2"}])
    assert mock.call_count() == 2
    calls = mock.calls()
    assert calls[0][0] == "sys1"
    assert calls[1][1] == [{"role": "user", "content": "m2"}]


async def test_remaining_decrements() -> None:
    mock = MockLLMClient()
    mock.queue_text("a")
    mock.queue_text("b")
    assert mock.remaining() == 2
    await mock.complete(system="s", messages=[])
    assert mock.remaining() == 1
