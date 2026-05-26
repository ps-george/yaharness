"""Tests for the LangGraph agent system.

Covers the four requirements from the langgraph adapter brief:

1. Protocol conformance + end-to-end with a mock model
2. Cost callback attributes token usage into our ``CostBudget``
3. Budget overrun aborts cleanly mid-graph
4. Tool bridging from our ``ToolRegistry`` to LangChain ``BaseTool``

Live OpenRouter route verification is in
``scripts/verify_openrouter_route.py`` (one short prompt, ~$0.01); see
the brief's report for the result. We do not run it from pytest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from yaharness.agents import AGENT_SYSTEMS, LangGraphSystem
from yaharness.agents._protocol import AgentSystem
from yaharness.agents.langgraph import (
    CostBudgetCallback,
    bridge_registry,
    bridge_tool,
)
from yaharness.cost import BudgetExceededError, CostBudget
from yaharness.tools import ToolRegistry, ToolResult
from yaharness.tools.filesystem import ReadFileTool

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _EchoTool:
    name = "echo"
    description = "Echo back the text."
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def execute(self, **kw: Any) -> ToolResult:
        return ToolResult(ok=True, output=f"echoed:{kw.get('text', '')}")


def _plan_msg(steps: list[str]) -> AIMessage:
    import json

    return AIMessage(content=json.dumps({"plan": steps}))


def _tool_call_msg(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _final_msg(text: str) -> AIMessage:
    return AIMessage(content=text)


# ---------------------------------------------------------------------------
# Registry / protocol
# ---------------------------------------------------------------------------


def test_registry_contains_langgraph() -> None:
    assert AGENT_SYSTEMS["langgraph"] is LangGraphSystem


@pytest.mark.asyncio
async def test_satisfies_agent_system_protocol() -> None:
    fake = FakeMessagesListChatModel(responses=[_final_msg("x")])
    sys: AgentSystem = LangGraphSystem(chat_model=fake)
    assert callable(sys)
    assert sys.name == "langgraph"


@pytest.mark.asyncio
async def test_missing_chat_model_raises() -> None:
    sys = LangGraphSystem()
    with pytest.raises(RuntimeError, match="requires a chat_model"):
        await sys("task", {}, cost_budget=CostBudget(1.0))


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_langgraph_runs_through_protocol() -> None:
    """Planner -> worker -> tool -> worker -> done."""
    tools = ToolRegistry()
    tools.register(_EchoTool())
    fake = FakeMessagesListChatModel(
        responses=[
            _plan_msg(["echo it", "answer"]),
            _tool_call_msg("echo", {"text": "hello"}),
            _final_msg("done-with-echo"),
        ]
    )
    sys = LangGraphSystem(chat_model=fake, tools=tools)
    result = await sys("echo task", {}, cost_budget=CostBudget(1.0))

    assert result.completed is True
    assert result.final_answer == "done-with-echo"
    assert result.raw_trace["termination_reason"] == "final_answer"
    assert result.raw_trace["plan"] == ["echo it", "answer"]
    # planner + worker + tool-observation + worker
    msg_types = [m["type"] for m in result.raw_trace["messages"]]
    assert "ToolMessage" in msg_types
    assert msg_types.count("AIMessage") == 3


@pytest.mark.asyncio
async def test_runs_without_tools() -> None:
    """No tools registered: planner -> worker -> done in one shot."""
    fake = FakeMessagesListChatModel(
        responses=[_plan_msg(["just answer"]), _final_msg("the-answer")]
    )
    sys = LangGraphSystem(chat_model=fake)
    result = await sys("toolless", {"k": "v"}, cost_budget=CostBudget(1.0))

    assert result.completed is True
    assert result.final_answer == "the-answer"
    assert result.raw_trace["context"] == {"k": "v"}


# ---------------------------------------------------------------------------
# Cost callback
# ---------------------------------------------------------------------------


def test_cost_callback_attribution_from_llm_output() -> None:
    """Synthesised LLMResult with usage drives correct cost attribution."""
    budget = CostBudget(1.0)
    cb = CostBudgetCallback(budget, default_model="openai/gpt-5-mini")
    # gpt-5-mini: $0.25/M in, $1.00/M out  -> 1000 in + 500 out
    # = 1000/1e6 * 0.25 + 500/1e6 * 1.00 = 0.00025 + 0.0005 = 0.00075
    fake_result = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="hi"))]],
        llm_output={
            "model_name": "openai/gpt-5-mini",
            "token_usage": {"prompt_tokens": 1000, "completion_tokens": 500},
        },
    )
    cb.on_llm_end(fake_result)
    assert len(cb.records) == 1
    rec = cb.records[0]
    assert rec["tokens_in"] == 1000
    assert rec["tokens_out"] == 500
    assert rec["model"] == "openai/gpt-5-mini"
    assert abs(rec["cost_usd"] - 0.00075) < 1e-9
    assert abs(budget.spent_usd - 0.00075) < 1e-9


def test_cost_callback_raises_when_over_budget() -> None:
    budget = CostBudget(0.0001)
    cb = CostBudgetCallback(budget, default_model="openai/gpt-5-mini")
    fake_result = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="hi"))]],
        llm_output={
            "model_name": "openai/gpt-5-mini",
            "token_usage": {"prompt_tokens": 10_000, "completion_tokens": 0},
        },
    )
    with pytest.raises(BudgetExceededError):
        cb.on_llm_end(fake_result)


# ---------------------------------------------------------------------------
# Budget overrun mid-graph
# ---------------------------------------------------------------------------


class _UsageEmittingFakeModel(FakeMessagesListChatModel):
    """Fake model that injects token usage into ``llm_output``.

    The base ``FakeMessagesListChatModel._generate`` returns a
    ``ChatResult`` with no ``llm_output``; we override to attach a
    realistic OpenAI-shape usage block so the cost callback fires.
    """

    tokens_per_call: int = 1_000_000  # huge by default — used to exceed budgets

    def _generate(  # type: ignore[override]
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        result = super()._generate(messages, stop, run_manager, **kwargs)
        result.llm_output = {
            "model_name": "openai/gpt-5-mini",
            "token_usage": {
                "prompt_tokens": self.tokens_per_call,
                "completion_tokens": 0,
            },
        }
        return result


@pytest.mark.asyncio
async def test_budget_overrun_aborts_mid_graph() -> None:
    """Tiny budget + token-heavy fake model -> clean abort, no crash."""
    fake = _UsageEmittingFakeModel(
        responses=[
            _plan_msg(["a", "b"]),
            _final_msg("should-never-reach-this"),
        ],
        tokens_per_call=10_000_000,  # at $0.25/M = $2.50 — way over $0.001 budget
    )
    sys = LangGraphSystem(chat_model=fake)
    result = await sys("anything", {}, cost_budget=CostBudget(0.001))
    assert result.completed is False
    assert result.raw_trace["termination_reason"] == "budget"
    # Budget aborted before the worker could produce the final answer
    assert result.final_answer == ""


@pytest.mark.asyncio
async def test_cost_records_present_in_trace() -> None:
    fake = _UsageEmittingFakeModel(
        responses=[_plan_msg(["x"]), _final_msg("y")],
        tokens_per_call=100,  # cheap
    )
    sys = LangGraphSystem(chat_model=fake)
    result = await sys("task", {}, cost_budget=CostBudget(1.0))
    assert result.completed is True
    assert result.total_cost_usd > 0
    assert len(result.raw_trace["cost_records"]) == 2  # planner + worker


# ---------------------------------------------------------------------------
# Tool bridging
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_bridge_filesystem_read(tmp_path: Path) -> None:
    """Bridge the real ReadFileTool through LangChain and execute it."""
    (tmp_path / "hello.txt").write_text("hello world", encoding="utf-8")
    our_tool = ReadFileTool(scope_dir=tmp_path)
    bridged = bridge_tool(our_tool)

    assert bridged.name == "read_file"
    assert "Read a UTF-8" in bridged.description
    result = await bridged.ainvoke({"path": "hello.txt"})
    assert result == "hello world"


@pytest.mark.asyncio
async def test_tool_bridge_propagates_errors() -> None:
    """Tool errors come back as TOOL_ERROR string, not exceptions."""

    class _BoomTool:
        name = "boom"
        description = "Always fails."
        parameters_schema: ClassVar[dict[str, Any]] = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        async def execute(self, **kw: Any) -> ToolResult:
            return ToolResult(ok=False, output="", error="kaboom")

    bridged = bridge_tool(_BoomTool())
    result = await bridged.ainvoke({})
    assert "TOOL_ERROR" in result
    assert "kaboom" in result


@pytest.mark.asyncio
async def test_tools_bridge_via_graph(tmp_path: Path) -> None:
    """Tool bridged through the real ToolNode produces a ToolMessage."""
    (tmp_path / "data.txt").write_text("contents-of-data", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(scope_dir=tmp_path))

    fake = FakeMessagesListChatModel(
        responses=[
            _plan_msg(["read data.txt", "answer"]),
            _tool_call_msg("read_file", {"path": "data.txt"}),
            _final_msg("read it"),
        ]
    )
    sys = LangGraphSystem(chat_model=fake, tools=tools)
    result = await sys("read data.txt", {}, cost_budget=CostBudget(1.0))
    assert result.completed is True
    tool_messages = [m for m in result.raw_trace["messages"] if m["type"] == "ToolMessage"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "contents-of-data"


def test_bridge_registry_preserves_order() -> None:
    tools = ToolRegistry()
    tools.register(_EchoTool())
    bridged = bridge_registry(tools)
    assert [t.name for t in bridged] == ["echo"]
