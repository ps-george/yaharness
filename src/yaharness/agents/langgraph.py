"""LangGraph agent system — planner -> worker -> tools loop via `StateGraph`.

This wraps the canonical LangGraph multi-agent pattern so it can be
compared against the harness at fixed model + cost. The point is
*not* that LangGraph is the best framework — only that it's the most
common one a user would reach for, so beating it is a publishable
result.

Architecture
------------

The graph has four nodes:

* ``planner`` — one LLM call producing a JSON plan (a list of steps).
* ``worker``  — the ReAct-style executor; each visit issues one LLM call
  bound to the tool set. Tool calls in the response route to ``tools``;
  a non-tool response routes to ``done``.
* ``tools``   — LangGraph's prebuilt ``ToolNode`` dispatches each
  ``ToolMessage`` by name; the wrapped tools delegate to our
  :class:`~yaharness.tools.ToolRegistry`.
* ``done``    — terminal; copies the worker's last message into the
  graph state's ``final_answer`` and marks ``completed=True``.

Cost tracking
-------------

LangChain hides every LLM call behind a ``Runnable``. We wire a
:class:`BaseCallbackHandler` that reads ``LLMResult.llm_output`` /
``response_metadata`` for token usage on ``on_llm_end`` and pushes the
cost into the supplied :class:`~yaharness.cost.CostBudget`. If
the budget raises :class:`BudgetExceededError`, the exception
propagates out of the graph; the system's ``__call__`` catches it and
returns a clean ``AgentSystemResult`` with ``termination_reason="budget"``.

LLM client bridging
-------------------

LangGraph requires a LangChain ``BaseChatModel``. Two routes are
supported by ``build_chat_model``:

* ``openrouter``: ``ChatOpenAI`` pointed at the OpenRouter
  OpenAI-compatible endpoint (``base_url="https://openrouter.ai/api/v1"``,
  ``api_key=$OPENROUTER_API_KEY``). This is the production route. The
  OpenRouter base-URL trick has bitten people before — the
  ``scripts/verify_openrouter_route.py`` helper exists to confirm it
  routes correctly with one short, cheap (~$0.01) prompt before any
  primary evaluation.
* Pass a pre-built ``BaseChatModel`` directly. Tests use
  :class:`FakeMessagesListChatModel` so the graph is exercised
  end-to-end without network.

Dependency versions
-------------------

Pinned (via ``pyproject.toml``) to currently-stable releases:

* ``langgraph>=1.2,<2`` (1.x is the post-public-API series; 2.x not
  yet released)
* ``langchain-core>=1.4,<2``
* ``langchain-openai>=1.2,<2``
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Annotated, Any, ClassVar, TypedDict, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, ConfigDict, Field, create_model

from ..benchmarks.outcome import AgentSystemResult
from ..cost import BudgetExceededError, CostBudget, estimate_cost
from ..tools import Tool, ToolRegistry

_PLANNER_SYSTEM = (
    "You are the PLANNER. Given a task, output exactly one JSON object "
    'of the form {{"plan": ["step 1", "step 2", ...]}}. No prose.'
)

_WORKER_SYSTEM = (
    "You are the WORKER. Follow the plan to solve the task. Call tools "
    "when useful. When done, respond with the final answer as plain text "
    "and do NOT call any tools."
)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class _GraphState(TypedDict, total=False):
    """Mutable graph state passed between nodes.

    ``messages`` is annotated with ``add_messages`` so LangGraph appends
    rather than overwrites — the standard pattern for chat loops.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    plan: list[str]
    task_text: str
    final_answer: str
    completed: bool
    step_count: int


# ---------------------------------------------------------------------------
# Cost callback
# ---------------------------------------------------------------------------


class CostBudgetCallback(BaseCallbackHandler):
    """LangChain callback that attributes LLM token usage to a CostBudget.

    Reads usage from ``LLMResult.llm_output`` (OpenAI/OpenRouter shape:
    ``{"token_usage": {...}, "model_name": "..."}``) or, failing that,
    from the per-generation ``response_metadata``. ``estimate_cost`` is
    our single pricing source of truth.
    """

    # Tell LangChain's callback manager to re-raise exceptions from our
    # handlers (default behaviour swallows them, which would let a
    # BudgetExceededError be silently logged instead of aborting the graph).
    raise_error: bool = True

    def __init__(self, budget: CostBudget, *, default_model: str = "mock") -> None:
        super().__init__()
        self._budget = budget
        self._default_model = default_model
        self.records: list[dict[str, Any]] = []

    def _extract_usage(self, response: LLMResult) -> tuple[str, int, int]:
        model = self._default_model
        tokens_in = 0
        tokens_out = 0
        llm_output = response.llm_output
        if isinstance(llm_output, dict):
            model = str(llm_output.get("model_name") or llm_output.get("model") or model)
            usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
            if isinstance(usage, dict):
                tokens_in = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
                tokens_out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        # Fallback: scan generations' response_metadata / usage_metadata
        if tokens_in == 0 and tokens_out == 0:
            for gens in response.generations:
                for gen in gens:
                    meta = getattr(gen, "message", None)
                    if meta is not None:
                        um = getattr(meta, "usage_metadata", None)
                        if isinstance(um, dict):
                            tokens_in = int(um.get("input_tokens", 0))
                            tokens_out = int(um.get("output_tokens", 0))
                            rm = getattr(meta, "response_metadata", {}) or {}
                            if isinstance(rm, dict):
                                model = str(rm.get("model_name") or rm.get("model") or model)
                            break
        return model, tokens_in, tokens_out

    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any,
    ) -> None:
        model, tokens_in, tokens_out = self._extract_usage(response)
        cost = estimate_cost(model, tokens_in, tokens_out)
        self.records.append(
            {
                "model": model,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost,
            }
        )
        # `add` will raise BudgetExceededError if we're over budget; that
        # exception propagates through LangGraph and is caught in __call__.
        self._budget.add(cost)


# ---------------------------------------------------------------------------
# Tool bridging
# ---------------------------------------------------------------------------


def _args_model_from_schema(tool_name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Build a tiny pydantic model from an OpenAI-style ``input_schema``.

    Only the shape we need: object schema with string-typed properties
    (matches every tool in ``yaharness.tools``). Extra properties
    fall back to ``str`` so unusual schemas don't crash the bridge.
    """
    properties: dict[str, Any] = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    fields: dict[str, Any] = {}
    type_map: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    for prop_name, prop_schema in properties.items():
        py_type = type_map.get(str(prop_schema.get("type", "string")), str)
        description = str(prop_schema.get("description", ""))
        default = prop_schema.get("default", ... if prop_name in required else None)
        fields[prop_name] = (py_type, Field(default=default, description=description))
    if not fields:
        # pydantic refuses empty models; give it a single optional field
        fields["noop"] = (str | None, Field(default=None))
    return cast(
        type[BaseModel],
        create_model(f"{tool_name}_Args", __config__=ConfigDict(extra="allow"), **fields),
    )


def bridge_tool(tool: Tool) -> BaseTool:
    """Wrap one of our `Tool`s as a LangChain `BaseTool`.

    The wrapper is async (LangGraph's ``ToolNode`` awaits coroutines).
    The result string is the tool's ``output`` on success or the
    ``error`` message on failure — keeps the LLM-visible surface uniform
    with the other agent systems.
    """
    args_schema = _args_model_from_schema(tool.name, tool.parameters_schema)

    async def _run(**kwargs: Any) -> str:
        kwargs.pop("noop", None)
        result = await tool.execute(**kwargs)
        if result.ok:
            return result.output
        return f"TOOL_ERROR: {result.error or 'tool failed'}"

    return StructuredTool.from_function(
        coroutine=_run,
        name=tool.name,
        description=tool.description,
        args_schema=args_schema,
    )


def bridge_registry(registry: ToolRegistry) -> list[BaseTool]:
    """Bridge every tool in the registry, preserving order."""
    return [bridge_tool(t) for t in registry.all()]


# ---------------------------------------------------------------------------
# LLM client bridging
# ---------------------------------------------------------------------------


def build_chat_model(
    *,
    model: str,
    api_key: str | None = None,
    base_url: str = "https://openrouter.ai/api/v1",
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> BaseChatModel:
    """Build a `ChatOpenAI` pointed at OpenRouter.

    Importing inside the function keeps the import cost off the
    module's hot path and lets tests that pass in a pre-built fake
    model avoid the dependency.
    """
    from langchain_openai import ChatOpenAI

    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("build_chat_model requires `api_key` or OPENROUTER_API_KEY env var")
    # `ChatOpenAI` accepts `base_url` and `api_key`; passing them by name
    # is robust across the 1.x line.
    return ChatOpenAI(  # type: ignore[call-arg]
        model=model,
        api_key=key,  # type: ignore[arg-type]
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Plan parsing
# ---------------------------------------------------------------------------


def _parse_plan(content: str) -> list[str]:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or "plan" not in parsed:
        raise ValueError(f"plan JSON must have a 'plan' key: {parsed!r}")
    plan = parsed["plan"]
    if not isinstance(plan, list) or not all(isinstance(s, str) for s in plan):
        raise ValueError(f"plan must be a list of strings: {plan!r}")
    return list(plan)


# ---------------------------------------------------------------------------
# The system
# ---------------------------------------------------------------------------


class LangGraphSystem:
    """LangGraph multi-agent agent system: planner -> worker -> tool loop.

    Implements the `AgentSystem` protocol so it slots into the benchmark
    runner identically to the other agent systems.
    """

    name: ClassVar[str] = "langgraph"

    def __init__(
        self,
        *,
        chat_model: BaseChatModel | None = None,
        tools: ToolRegistry | None = None,
        max_planning_rounds: int = 2,
        default_model_for_cost: str = "openai/gpt-5-mini",
    ) -> None:
        self._chat = chat_model
        self._tools = tools or ToolRegistry()
        self._bridged_tools: list[BaseTool] = bridge_registry(self._tools)
        self._max_planning_rounds = max_planning_rounds
        self._default_model_for_cost = default_model_for_cost

    # ---- nodes ----------------------------------------------------------

    def _make_planner_node(self, model: BaseChatModel) -> Any:
        async def planner(state: _GraphState, config: RunnableConfig) -> dict[str, Any]:
            task_text = state.get("task_text", "")
            response = await model.ainvoke(
                [
                    SystemMessage(content=_PLANNER_SYSTEM),
                    HumanMessage(content=f"Task: {task_text}"),
                ],
                config=config,
            )
            try:
                plan = _parse_plan(str(response.content))
            except (ValueError, json.JSONDecodeError):
                plan = [task_text]  # degrade gracefully
            return {
                "plan": plan,
                "messages": [response],
                "step_count": state.get("step_count", 0) + 1,
            }

        return planner

    def _make_worker_node(self, model: BaseChatModel, tools: list[BaseTool]) -> Any:
        try:
            bound = model.bind_tools(tools) if tools else model
        except NotImplementedError:
            # Fake chat models for tests don't implement bind_tools; the
            # responses they emit already include `tool_calls` directly, so
            # falling back to the unbound model is the right behaviour.
            bound = model

        async def worker(state: _GraphState, config: RunnableConfig) -> dict[str, Any]:
            plan = state.get("plan", [])
            plan_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan))
            system = SystemMessage(
                content=(
                    f"{_WORKER_SYSTEM}\n\nTask: {state.get('task_text', '')}\nPlan:\n{plan_text}"
                )
            )
            history = list(state.get("messages", []))
            response = await bound.ainvoke([system, *history], config=config)
            return {
                "messages": [response],
                "step_count": state.get("step_count", 0) + 1,
            }

        return worker

    def _make_done_node(self) -> Any:
        async def done(state: _GraphState) -> dict[str, Any]:
            messages = state.get("messages", [])
            answer = ""
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    answer = str(msg.content)
                    break
            return {"final_answer": answer, "completed": True}

        return done

    @staticmethod
    def _route_after_worker(state: _GraphState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "done"
        last = messages[-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        return "tools" if tool_calls else "done"

    def _build_graph(
        self,
        model: BaseChatModel,
        tools: list[BaseTool],
        max_steps: int,
    ) -> Any:
        graph: StateGraph[_GraphState, None, _GraphState, _GraphState] = StateGraph(_GraphState)
        graph.add_node("planner", self._make_planner_node(model))
        graph.add_node("worker", self._make_worker_node(model, tools))
        if tools:
            graph.add_node("tools", ToolNode(tools))
        graph.add_node("done", self._make_done_node())

        graph.add_edge(START, "planner")
        graph.add_edge("planner", "worker")
        if tools:
            graph.add_conditional_edges(
                "worker",
                self._route_after_worker,
                {"tools": "tools", "done": "done"},
            )
            graph.add_edge("tools", "worker")
        else:
            graph.add_edge("worker", "done")
        graph.add_edge("done", END)
        return graph.compile()

    # ---- protocol -------------------------------------------------------

    async def __call__(
        self,
        task_text: str,
        context: dict[str, Any],
        *,
        cost_budget: CostBudget,
        max_steps: int = 50,
    ) -> AgentSystemResult:
        if self._chat is None:
            raise RuntimeError(
                "LangGraphSystem requires a chat_model. Use build_chat_model() "
                "or pass a fake model in tests."
            )

        cb = CostBudgetCallback(cost_budget, default_model=self._default_model_for_cost)
        config: RunnableConfig = {
            "callbacks": [cb],
            "recursion_limit": max(2 * max_steps + 4, 25),
        }

        compiled = self._build_graph(self._chat, self._bridged_tools, max_steps)
        initial: _GraphState = {
            "messages": [],
            "plan": [],
            "task_text": _augment_with_context(task_text, context),
            "final_answer": "",
            "completed": False,
            "step_count": 0,
        }

        termination_reason = "final_answer"
        try:
            final_state = await compiled.ainvoke(initial, config=config)
        except BudgetExceededError:
            return AgentSystemResult(
                final_answer="",
                completed=False,
                n_steps=len(cb.records),
                total_cost_usd=sum(r["cost_usd"] for r in cb.records),
                raw_trace={
                    "termination_reason": "budget",
                    "cost_records": cb.records,
                    "context": context,
                },
            )
        except Exception as exc:
            return AgentSystemResult(
                final_answer="",
                completed=False,
                n_steps=len(cb.records),
                total_cost_usd=sum(r["cost_usd"] for r in cb.records),
                raw_trace={
                    "termination_reason": f"error:{type(exc).__name__}:{exc}",
                    "cost_records": cb.records,
                    "context": context,
                },
            )

        completed = bool(final_state.get("completed", False))
        final_answer = str(final_state.get("final_answer", ""))
        if not completed:
            termination_reason = "max_steps"

        n_steps = int(final_state.get("step_count", len(cb.records)))
        total_cost = sum(r["cost_usd"] for r in cb.records)

        trace_messages: list[dict[str, str]] = []
        for msg in final_state.get("messages", []):
            trace_messages.append(
                {
                    "type": type(msg).__name__,
                    "content": str(getattr(msg, "content", "")),
                }
            )

        return AgentSystemResult(
            final_answer=final_answer,
            completed=completed,
            n_steps=n_steps,
            total_cost_usd=total_cost,
            raw_trace={
                "termination_reason": termination_reason,
                "plan": final_state.get("plan", []),
                "messages": trace_messages,
                "cost_records": cb.records,
                "context": context,
            },
        )


def _augment_with_context(task_text: str, context: dict[str, Any]) -> str:
    if not context:
        return task_text
    return f"{task_text}\n\nContext: {json.dumps(context, sort_keys=True)}"


__all__ = [
    "CostBudgetCallback",
    "LangGraphSystem",
    "bridge_registry",
    "bridge_tool",
    "build_chat_model",
]


# Convenience aliases for tests that want to construct the system with
# minimal ceremony. Tests import `FakeMessagesListChatModel` directly.
def _resolve_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Identity helper kept for symmetry with future LLM-bridging code."""
    return list(messages)


# Re-exports used in tests
__all__.append("_resolve_messages")


# Helper imports for tests to avoid having to know LangChain layout.
# Kept at the bottom so the public API above stays uncluttered.
def _unused_keep_imports_alive() -> None:  # pragma: no cover
    _ = (HumanMessage, AIMessage, ToolMessage, SystemMessage)
