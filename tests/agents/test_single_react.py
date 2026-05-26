"""Tests for `SingleReActSystem`."""

from __future__ import annotations

import json
from typing import Any, ClassVar

import pytest

from yaharness.agents import AGENT_SYSTEMS, SingleReActSystem
from yaharness.agents._protocol import AgentSystem, AgentSystemResult
from yaharness.cost import CostBudget
from yaharness.llm import LLMResponse, MockLLMClient
from yaharness.tools import ToolRegistry, ToolResult


class _EchoTool:
    name = "echo"
    description = "Echo back the provided text."
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(ok=True, output=str(kwargs.get("text", "")))


def _step(
    *,
    action: str,
    final_answer: str = "",
    tool_name: str = "",
    tool_args: dict[str, Any] | None = None,
    cost: float = 0.001,
) -> LLMResponse:
    payload = {
        "thought": "thinking",
        "action": action,
        "tool_name": tool_name,
        "tool_args": tool_args or {},
        "final_answer": final_answer,
    }
    return LLMResponse(
        content=json.dumps(payload),
        cost_usd=cost,
        model="mock",
        tokens_in=10,
        tokens_out=10,
    )


@pytest.mark.asyncio
async def test_satisfies_agent_system_protocol() -> None:
    sys: AgentSystem = SingleReActSystem(llm_client=MockLLMClient())
    assert callable(sys)


@pytest.mark.asyncio
async def test_trivial_one_step_final_answer() -> None:
    llm = MockLLMClient([_step(action="final_answer", final_answer="42")])
    system = SingleReActSystem(llm_client=llm)
    result = await system("what's 6*7?", {}, cost_budget=CostBudget(1.0))
    assert isinstance(result, AgentSystemResult)
    assert result.completed is True
    assert result.final_answer == "42"
    assert result.n_steps == 1
    assert result.raw_trace["termination_reason"] == "final_answer"


@pytest.mark.asyncio
async def test_multi_step_with_tools() -> None:
    llm = MockLLMClient(
        [
            _step(action="tool_call", tool_name="echo", tool_args={"text": "hello"}),
            _step(action="tool_call", tool_name="echo", tool_args={"text": "world"}),
            _step(action="final_answer", final_answer="done"),
        ]
    )
    tools = ToolRegistry()
    tools.register(_EchoTool())
    system = SingleReActSystem(llm_client=llm, tools=tools)
    result = await system("multi step", {"k": "v"}, cost_budget=CostBudget(1.0))
    assert result.completed is True
    assert result.n_steps == 3
    assert result.final_answer == "done"
    observations = [s.get("observation") for s in result.raw_trace["steps"] if "observation" in s]
    assert observations == ["hello", "world"]


@pytest.mark.asyncio
async def test_max_steps_exceeded() -> None:
    # Queue many tool_call steps; never returns final_answer.
    llm = MockLLMClient([_step(action="tool_call", tool_name="echo", tool_args={"text": "x"})] * 5)
    tools = ToolRegistry()
    tools.register(_EchoTool())
    system = SingleReActSystem(llm_client=llm, tools=tools)
    result = await system("loop", {}, cost_budget=CostBudget(1.0), max_steps=3)
    assert result.completed is False
    assert result.n_steps == 3
    assert result.raw_trace["termination_reason"] == "max_steps"


@pytest.mark.asyncio
async def test_budget_exhausted() -> None:
    llm = MockLLMClient(
        [_step(action="tool_call", tool_name="echo", tool_args={"text": "x"}, cost=0.5)]
    )
    tools = ToolRegistry()
    tools.register(_EchoTool())
    system = SingleReActSystem(llm_client=llm, tools=tools)
    result = await system("expensive", {}, cost_budget=CostBudget(0.1))
    assert result.completed is False
    assert result.raw_trace["termination_reason"] == "budget"


@pytest.mark.asyncio
async def test_unknown_tool_recovery() -> None:
    llm = MockLLMClient(
        [
            _step(action="tool_call", tool_name="nope", tool_args={}),
            _step(action="final_answer", final_answer="recovered"),
        ]
    )
    system = SingleReActSystem(llm_client=llm, tools=ToolRegistry())
    result = await system("recover", {}, cost_budget=CostBudget(1.0))
    assert result.completed is True
    assert result.final_answer == "recovered"
    err_steps = [
        s for s in result.raw_trace["steps"] if "TOOL_ERROR" in str(s.get("observation", ""))
    ]
    assert err_steps, "expected TOOL_ERROR observation"


@pytest.mark.asyncio
async def test_parse_error_recovery() -> None:
    llm = MockLLMClient(
        [
            LLMResponse(
                content="not json at all", cost_usd=0.001, model="mock", tokens_in=1, tokens_out=1
            ),
            _step(action="final_answer", final_answer="ok"),
        ]
    )
    system = SingleReActSystem(llm_client=llm)
    result = await system("garbage", {}, cost_budget=CostBudget(1.0))
    assert result.completed is True
    assert result.final_answer == "ok"


@pytest.mark.asyncio
async def test_unknown_action_recovery() -> None:
    llm = MockLLMClient(
        [
            _step(action="weird"),
            _step(action="final_answer", final_answer="ok"),
        ]
    )
    system = SingleReActSystem(llm_client=llm)
    result = await system("unknown", {}, cost_budget=CostBudget(1.0))
    assert result.completed is True


@pytest.mark.asyncio
async def test_json_fenced_response() -> None:
    fenced = (
        "```json\n" + json.dumps({"action": "final_answer", "final_answer": "fenced"}) + "\n```"
    )
    llm = MockLLMClient(
        [LLMResponse(content=fenced, cost_usd=0.001, model="mock", tokens_in=1, tokens_out=1)]
    )
    system = SingleReActSystem(llm_client=llm)
    result = await system("fence", {}, cost_budget=CostBudget(1.0))
    assert result.completed is True
    assert result.final_answer == "fenced"


@pytest.mark.asyncio
async def test_single_react_nudge_toward_final_answer() -> None:
    """As n_steps approaches max_steps the OBSERVATION carries a system nudge."""
    llm = MockLLMClient([_step(action="tool_call", tool_name="echo", tool_args={"text": "x"})] * 10)
    tools = ToolRegistry()
    tools.register(_EchoTool())
    system = SingleReActSystem(llm_client=llm, tools=tools)
    await system("loop", {}, cost_budget=CostBudget(1.0), max_steps=5)
    # Inspect the messages that were sent to the LLM on later calls.
    user_blob = "\n".join(
        m["content"] for _s, msgs in llm.calls() for m in msgs if m.get("role") == "user"
    )
    assert "[SYSTEM] Step" in user_blob
    assert "final_answer" in user_blob


def test_registry_contains_single_react() -> None:
    assert AGENT_SYSTEMS["single_react"] is SingleReActSystem
