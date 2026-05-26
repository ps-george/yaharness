"""Sequential two-agent system: planner + worker.

The planner produces a plan once (and may revise it up to
`max_replanning` times if the worker signals stuck). The worker executes
ReAct-style against the plan.
"""

from __future__ import annotations

import json
from typing import Any

from ..cost import BudgetExceededError, CostBudget
from ..llm import LLMClient
from ..tools import ToolRegistry
from ._protocol import AgentSystemResult
from .single_react import _find_balanced_object, _parse_step

_PLANNER_PROMPT = (
    "You are the PLANNER agent. Given a task, output ONE JSON object.\n\n"
    "STRICT OUTPUT CONTRACT:\n"
    "- Respond with a single JSON object only (optionally wrapped in ```json ... ```).\n"
    "- NO XML markup: do not emit <function_calls>, <invoke>, <parameter> or any XML tags.\n"
    "- NO prose before or after the JSON.\n\n"
    'Schema: {{ "plan": ["step 1", "step 2", ...] }}\n\n'
    "Task: {task_text}"
)

_REPLAN_PROMPT = (
    "You are the PLANNER agent. Your previous plan stalled. Revise it.\n\n"
    "STRICT OUTPUT CONTRACT: single JSON object only. No XML. No prose.\n\n"
    "Task: {task_text}\n"
    "Previous plan: {prev_plan}\n"
    "Worker report: {report}\n\n"
    'Output JSON: {{ "plan": ["step 1", ...] }}'
)

_WORKER_PROMPT = (
    "You are the WORKER agent executing a plan to solve the task.\n\n"
    "Task: {task_text}\n"
    "Plan:\n{plan_text}\n\n"
    "You work in a loop: THINK, ACT, OBSERVE. At each step output JSON: "
    '{{ "thought": "...", "action": "tool_call|final_answer|stuck", '
    '"tool_name": "...", "tool_args": {{}}, "final_answer": "...", "reason": "..." }}.\n\n'
    'Use action="stuck" with a "reason" if you cannot make progress and need replanning. '
    'Use action="final_answer" when done.'
)


def _parse_plan(content: str) -> list[str]:
    """Parse a planner response into a list of plan steps.

    Tolerates prose prefix, fenced JSON, and Haiku-style XML markup
    prepended to the JSON body — we balanced-brace-scan past any leading
    noise.
    """
    parsed: Any | None = None
    last_exc: Exception | None = None

    text = content.strip()
    # 1. Full-text fenced.
    if text.startswith("```"):
        inner = text.split("```", 2)[1]
        if inner.startswith("json"):
            inner = inner[4:]
        inner = inner.strip()
        if inner.endswith("```"):
            inner = inner[:-3].strip()
        try:
            parsed = json.loads(inner)
        except json.JSONDecodeError as exc:
            last_exc = exc

    # 2. Bare JSON.
    if parsed is None and text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            last_exc = exc

    # 3. Balanced-brace scan — tolerates XML / prose prefix.
    if parsed is None:
        block = _find_balanced_object(content)
        if block is not None:
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError as exc:
                last_exc = exc

    if parsed is None:
        raise ValueError(f"could not parse plan JSON: {last_exc}: {content!r}") from last_exc
    if not isinstance(parsed, dict) or "plan" not in parsed:
        raise ValueError(f"plan JSON must have a 'plan' key: {parsed!r}")
    plan = parsed["plan"]
    if not isinstance(plan, list) or not all(isinstance(s, str) for s in plan):
        raise ValueError(f"plan must be a list of strings: {plan!r}")
    return list(plan)


class PlannerWorkerSystem:
    """Two-agent sequential system. Implements the `AgentSystem` protocol."""

    name: str = "planner_worker"

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tools: ToolRegistry | None = None,
        max_replanning: int = 2,
    ) -> None:
        self._llm = llm_client
        self._tools = tools or ToolRegistry()
        self._max_replanning = max_replanning

    async def _ask_planner(
        self, prompt: str, cost_budget: CostBudget
    ) -> tuple[list[str], float, str]:
        resp = await self._llm.complete(system=prompt, messages=[])
        cost_budget.add(resp.cost_usd)
        plan = _parse_plan(resp.content)
        return plan, resp.cost_usd, resp.content

    async def __call__(
        self,
        task_text: str,
        context: dict[str, Any],
        *,
        cost_budget: CostBudget,
        max_steps: int = 50,
    ) -> AgentSystemResult:
        trace: list[dict[str, Any]] = []
        total_cost = 0.0
        n_steps = 0
        completed = False
        final_answer = ""
        termination_reason = "max_steps"
        replans_used = 0

        if not self._tools.all() and context:
            from ..benchmarks.swebench_harness import (
                maybe_build_swebench_registry_from_context,
            )

            registry = await maybe_build_swebench_registry_from_context(context)
            if registry is not None:
                self._tools = registry

        # --- Planner: initial plan -----------------------------------------
        try:
            plan, cost, raw = await self._ask_planner(
                _PLANNER_PROMPT.format(task_text=task_text), cost_budget
            )
        except BudgetExceededError:
            return AgentSystemResult(
                final_answer="",
                completed=False,
                n_steps=0,
                total_cost_usd=total_cost,
                raw_trace={"termination_reason": "budget", "stage": "planner"},
            )
        except ValueError as exc:
            return AgentSystemResult(
                final_answer="",
                completed=False,
                n_steps=0,
                total_cost_usd=total_cost,
                raw_trace={"termination_reason": f"planner_parse_error:{exc}"},
            )
        total_cost += cost
        n_steps += 1
        trace.append({"step": n_steps, "agent": "planner", "plan": plan, "raw": raw})

        # --- Worker loop ---------------------------------------------------
        worker_messages: list[dict[str, str]] = []
        stuck_streak = 0

        def _worker_system() -> str:
            plan_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan))
            base = _WORKER_PROMPT.format(task_text=task_text, plan_text=plan_text)
            if context:
                base = f"{base}\n\nContext: {json.dumps(context, sort_keys=True)}"
            return base

        while n_steps < max_steps:
            if cost_budget.remaining_usd <= 0:
                termination_reason = "budget"
                break

            try:
                resp = await self._llm.complete(system=_worker_system(), messages=worker_messages)
            except Exception as exc:  # pragma: no cover - defensive
                termination_reason = f"llm_error:{type(exc).__name__}"
                trace.append({"step": n_steps + 1, "agent": "worker", "error": str(exc)})
                break

            n_steps += 1
            total_cost += resp.cost_usd
            try:
                cost_budget.add(resp.cost_usd)
            except BudgetExceededError:
                termination_reason = "budget"
                trace.append({"step": n_steps, "agent": "worker", "note": "budget_exceeded"})
                break

            try:
                step = _parse_step(resp.content)
            except ValueError as exc:
                observation = f"PARSE_ERROR: {exc}"
                trace.append(
                    {
                        "step": n_steps,
                        "agent": "worker",
                        "parse_error": str(exc),
                        "raw": resp.content,
                    }
                )
                worker_messages.append({"role": "assistant", "content": resp.content})
                worker_messages.append({"role": "user", "content": observation})
                continue

            worker_messages.append({"role": "assistant", "content": resp.content})
            action = step.get("action")

            if action == "final_answer":
                final_answer = str(step.get("final_answer", ""))
                completed = True
                termination_reason = "final_answer"
                trace.append({"step": n_steps, "agent": "worker", "answer": final_answer})
                break

            if action == "stuck":
                stuck_streak += 1
                reason = str(step.get("reason", "no reason"))
                trace.append(
                    {"step": n_steps, "agent": "worker", "action": "stuck", "reason": reason}
                )
                if replans_used >= self._max_replanning:
                    termination_reason = "replans_exhausted"
                    break
                try:
                    plan, cost, raw = await self._ask_planner(
                        _REPLAN_PROMPT.format(
                            task_text=task_text,
                            prev_plan=plan,
                            report=reason,
                        ),
                        cost_budget,
                    )
                except BudgetExceededError:
                    termination_reason = "budget"
                    break
                except ValueError as exc:
                    termination_reason = f"planner_parse_error:{exc}"
                    break
                total_cost += cost
                n_steps += 1
                replans_used += 1
                stuck_streak = 0
                worker_messages = []
                trace.append(
                    {
                        "step": n_steps,
                        "agent": "planner",
                        "plan": plan,
                        "raw": raw,
                        "replan_index": replans_used,
                    }
                )
                continue

            if action == "tool_call":
                stuck_streak = 0
                tool_name = str(step.get("tool_name", ""))
                tool_args = step.get("tool_args") or {}
                if not isinstance(tool_args, dict):
                    observation = (
                        f"TOOL_ERROR: tool_args must be an object, got {type(tool_args).__name__}"
                    )
                else:
                    try:
                        tool = self._tools.get(tool_name)
                    except KeyError:
                        observation = f"TOOL_ERROR: unknown tool {tool_name!r}"
                    else:
                        try:
                            result = await tool.execute(**tool_args)
                            observation = (
                                result.output
                                if result.ok
                                else f"TOOL_ERROR: {result.error or 'tool failed'}"
                            )
                        except Exception as exc:
                            observation = f"TOOL_ERROR: {type(exc).__name__}: {exc}"
                trace.append(
                    {
                        "step": n_steps,
                        "agent": "worker",
                        "tool_name": tool_name,
                        "observation": observation,
                    }
                )
                worker_messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})
                continue

            observation = f"ACTION_ERROR: unknown action {action!r}"
            trace.append({"step": n_steps, "agent": "worker", "action_error": observation})
            worker_messages.append({"role": "user", "content": observation})

        return AgentSystemResult(
            final_answer=final_answer,
            completed=completed,
            n_steps=n_steps,
            total_cost_usd=total_cost,
            raw_trace={
                "termination_reason": termination_reason,
                "replans_used": replans_used,
                "final_plan": plan,
                "steps": trace,
                "context": context,
            },
        )


__all__ = ["PlannerWorkerSystem"]
