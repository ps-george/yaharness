"""Tests for `PlannerWorkerSystem`."""

from __future__ import annotations

import json
from typing import Any, ClassVar

import pytest

from yaharness.agents import AGENT_SYSTEMS, PlannerWorkerSystem
from yaharness.agents._protocol import AgentSystem
from yaharness.cost import CostBudget
from yaharness.llm import LLMResponse, MockLLMClient
from yaharness.tools import ToolRegistry, ToolResult


class _EchoTool:
    name = "echo"
    description = "Echo."
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(ok=True, output=str(kwargs.get("text", "")))


def _plan(steps: list[str], *, cost: float = 0.001) -> LLMResponse:
    return LLMResponse(
        content=json.dumps({"plan": steps}),
        cost_usd=cost,
        model="mock",
        tokens_in=10,
        tokens_out=10,
    )


def _worker_step(
    *,
    action: str,
    final_answer: str = "",
    tool_name: str = "",
    tool_args: dict[str, Any] | None = None,
    reason: str = "",
    cost: float = 0.001,
) -> LLMResponse:
    return LLMResponse(
        content=json.dumps(
            {
                "thought": "t",
                "action": action,
                "tool_name": tool_name,
                "tool_args": tool_args or {},
                "final_answer": final_answer,
                "reason": reason,
            }
        ),
        cost_usd=cost,
        model="mock",
        tokens_in=10,
        tokens_out=10,
    )


@pytest.mark.asyncio
async def test_satisfies_agent_system_protocol() -> None:
    sys: AgentSystem = PlannerWorkerSystem(llm_client=MockLLMClient())
    assert callable(sys)


@pytest.mark.asyncio
async def test_planner_then_worker_final_answer() -> None:
    llm = MockLLMClient(
        [
            _plan(["do thing", "report"]),
            _worker_step(action="final_answer", final_answer="done"),
        ]
    )
    system = PlannerWorkerSystem(llm_client=llm)
    result = await system("task", {}, cost_budget=CostBudget(1.0))
    assert result.completed is True
    assert result.final_answer == "done"
    assert result.n_steps == 2
    assert result.raw_trace["termination_reason"] == "final_answer"


@pytest.mark.asyncio
async def test_worker_uses_tools_then_answers() -> None:
    llm = MockLLMClient(
        [
            _plan(["echo something", "answer"]),
            _worker_step(action="tool_call", tool_name="echo", tool_args={"text": "hi"}),
            _worker_step(action="final_answer", final_answer="hi-back"),
        ]
    )
    tools = ToolRegistry()
    tools.register(_EchoTool())
    system = PlannerWorkerSystem(llm_client=llm, tools=tools)
    result = await system("task", {}, cost_budget=CostBudget(1.0))
    assert result.completed is True
    assert result.final_answer == "hi-back"
    assert result.n_steps == 3


@pytest.mark.asyncio
async def test_stuck_triggers_replan_then_completes() -> None:
    llm = MockLLMClient(
        [
            _plan(["step a"]),
            _worker_step(action="stuck", reason="no progress"),
            _plan(["step b"]),
            _worker_step(action="final_answer", final_answer="after-replan"),
        ]
    )
    system = PlannerWorkerSystem(llm_client=llm, max_replanning=2)
    result = await system("task", {}, cost_budget=CostBudget(1.0))
    assert result.completed is True
    assert result.final_answer == "after-replan"
    assert result.raw_trace["replans_used"] == 1
    assert result.raw_trace["final_plan"] == ["step b"]


@pytest.mark.asyncio
async def test_replans_exhausted() -> None:
    llm = MockLLMClient(
        [
            _plan(["p0"]),
            _worker_step(action="stuck", reason="r1"),
            _plan(["p1"]),
            _worker_step(action="stuck", reason="r2"),
        ]
    )
    system = PlannerWorkerSystem(llm_client=llm, max_replanning=1)
    result = await system("task", {}, cost_budget=CostBudget(1.0))
    assert result.completed is False
    assert result.raw_trace["termination_reason"] == "replans_exhausted"


@pytest.mark.asyncio
async def test_max_steps_exceeded() -> None:
    llm = MockLLMClient(
        [_plan(["x"])]
        + [_worker_step(action="tool_call", tool_name="echo", tool_args={"text": "x"})] * 10
    )
    tools = ToolRegistry()
    tools.register(_EchoTool())
    system = PlannerWorkerSystem(llm_client=llm, tools=tools)
    result = await system("loop", {}, cost_budget=CostBudget(1.0), max_steps=4)
    assert result.completed is False
    assert result.n_steps == 4
    assert result.raw_trace["termination_reason"] == "max_steps"


@pytest.mark.asyncio
async def test_budget_exhausted_at_planner() -> None:
    llm = MockLLMClient([_plan(["x"], cost=0.5)])
    system = PlannerWorkerSystem(llm_client=llm)
    result = await system("task", {}, cost_budget=CostBudget(0.1))
    assert result.completed is False
    assert result.raw_trace["termination_reason"] == "budget"


@pytest.mark.asyncio
async def test_planner_parse_error_terminates() -> None:
    llm = MockLLMClient(
        [LLMResponse(content="not a plan", cost_usd=0.001, model="mock", tokens_in=1, tokens_out=1)]
    )
    system = PlannerWorkerSystem(llm_client=llm)
    result = await system("task", {}, cost_budget=CostBudget(1.0))
    assert result.completed is False
    assert "planner_parse_error" in result.raw_trace["termination_reason"]


@pytest.mark.asyncio
async def test_worker_unknown_tool_recovery() -> None:
    llm = MockLLMClient(
        [
            _plan(["x"]),
            _worker_step(action="tool_call", tool_name="nope", tool_args={}),
            _worker_step(action="final_answer", final_answer="ok"),
        ]
    )
    system = PlannerWorkerSystem(llm_client=llm, tools=ToolRegistry())
    result = await system("t", {}, cost_budget=CostBudget(1.0))
    assert result.completed is True


def test_registry_contains_planner_worker() -> None:
    assert AGENT_SYSTEMS["planner_worker"] is PlannerWorkerSystem
