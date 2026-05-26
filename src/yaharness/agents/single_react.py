"""ReAct-style single-agent agent system.

A standard think -> act -> observe loop run by a single LLM agent. This
is the textbook unilateral agent system against which the iterative
agent must justify itself.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..cost import BudgetExceededError, CostBudget
from ..llm import LLMClient
from ..tools import ToolRegistry
from ._protocol import AgentSystemResult

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*\n(\{.*?\})\s*\n```", re.DOTALL)


def _find_balanced_object(text: str) -> str | None:
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    return text[start : i + 1]
    return None


_DEFAULT_SYSTEM_PROMPT = (
    "You are an autonomous agent solving the following task: {task_text}\n\n"
    "You work in a loop: THINK (analyse), ACT (call a tool or commit an answer), "
    "OBSERVE (see the result). Repeat until the task is solved.\n\n"
    'At each step, output JSON: {{ "thought": "...", "action": "tool_call|final_answer", '
    '"tool_name": "...", "tool_args": {{}}, "final_answer": "..." }}.\n\n'
    'When you have the final answer, output action: "final_answer" and stop.'
)


def _parse_step(content: str) -> dict[str, Any]:
    """Parse a model step, tolerating prose prefix, ```json fences, and inline JSON.

    Strategy in order:
      1. ```json (or ```) fenced JSON anywhere in the text.
      2. Stripped-fence form (entire response is fenced).
      3. Bare JSON (response starts with `{`).
      4. Balanced-brace scan over the full text (tolerates prose prefix).
    """
    last_exc: Exception | None = None

    # 1. Fenced JSON anywhere.
    fenced = _FENCED_JSON_RE.search(content)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            last_exc = exc

    text = content.strip()
    # 2. Full-text fenced.
    if text.startswith("```"):
        inner = text.split("```", 2)[1]
        if inner.startswith("json"):
            inner = inner[4:]
        inner = inner.strip()
        if inner.endswith("```"):
            inner = inner[:-3].strip()
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            last_exc = exc

    # 3. Bare JSON starting at index 0.
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            last_exc = exc

    # 4. Balanced-brace scan over the original (tolerates prose prefix).
    block = _find_balanced_object(content)
    if block is not None:
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            last_exc = exc

    raise ValueError(f"could not parse step JSON: {last_exc}: {content!r}") from last_exc


class SingleReActSystem:
    """Single-agent ReAct agent system. Implements the `AgentSystem` protocol."""

    name: str = "single_react"

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tools: ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_thought_tokens: int = 2048,
    ) -> None:
        self._llm = llm_client
        self._tools = tools or ToolRegistry()
        self._system_prompt_template = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._max_thought_tokens = max_thought_tokens

    @staticmethod
    def _nudge(
        n_steps: int,
        max_steps: int,
        tools_used: int,
        nudge_threshold: int,
        urgent_threshold: int,
    ) -> str | None:
        """Soft system-style nudge toward final_answer as the budget runs down."""
        if n_steps >= urgent_threshold:
            return (
                f"[SYSTEM] Step {n_steps} of {max_steps} (tools used: {tools_used}). "
                "Budget nearly exhausted. If you have enough information, output "
                'action: "final_answer" now. For SWE-bench / patch tasks the '
                "final_answer must be the unified diff (starts with `diff --git`)."
            )
        if n_steps >= nudge_threshold:
            return (
                f"[SYSTEM] Step {n_steps} of {max_steps} (tools used: {tools_used}). "
                "Consider whether you have enough information to answer. If yes, "
                'output action: "final_answer".'
            )
        return None

    async def __call__(
        self,
        task_text: str,
        context: dict[str, Any],
        *,
        cost_budget: CostBudget,
        max_steps: int = 50,
    ) -> AgentSystemResult:
        # Auto-materialise a SWE-bench scoped registry if the caller did
        # not provide one and the context looks like a SWE-bench problem.
        if not self._tools.all() and context:
            from ..benchmarks.swebench_harness import (
                maybe_build_swebench_registry_from_context,
            )

            registry = await maybe_build_swebench_registry_from_context(context)
            if registry is not None:
                self._tools = registry

        system = self._system_prompt_template.format(task_text=task_text)
        if context:
            system = f"{system}\n\nContext: {json.dumps(context, sort_keys=True)}"

        messages: list[dict[str, str]] = []
        trace: list[dict[str, Any]] = []
        total_cost = 0.0
        n_steps = 0
        tools_used = 0
        termination_reason = "max_steps"
        final_answer = ""
        completed = False

        # Soft-nudge thresholds: half-way and 80% of the budget.
        nudge_threshold = max(1, max_steps // 2)
        urgent_threshold = max(1, (max_steps * 4) // 5)

        while n_steps < max_steps:
            if cost_budget.remaining_usd <= 0:
                termination_reason = "budget"
                break

            try:
                resp = await self._llm.complete(
                    system=system,
                    messages=messages,
                    max_tokens=self._max_thought_tokens,
                )
            except Exception as exc:  # pragma: no cover - defensive
                termination_reason = f"llm_error:{type(exc).__name__}"
                trace.append({"step": n_steps, "error": str(exc)})
                break

            n_steps += 1
            total_cost += resp.cost_usd
            try:
                cost_budget.add(resp.cost_usd)
            except BudgetExceededError:
                trace.append({"step": n_steps, "raw": resp.content, "note": "budget_exceeded"})
                termination_reason = "budget"
                break

            try:
                step = _parse_step(resp.content)
            except ValueError as exc:
                observation = f"PARSE_ERROR: {exc}"
                trace.append({"step": n_steps, "raw": resp.content, "parse_error": str(exc)})
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": observation})
                continue

            messages.append({"role": "assistant", "content": resp.content})
            action = step.get("action")

            if action == "final_answer":
                final_answer = str(step.get("final_answer", ""))
                completed = True
                termination_reason = "final_answer"
                trace.append({"step": n_steps, "action": "final_answer", "answer": final_answer})
                break

            if action == "tool_call":
                tools_used += 1
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
                        "action": "tool_call",
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "observation": observation,
                    }
                )
                obs_msg = f"OBSERVATION: {observation}"
                nudge = self._nudge(
                    n_steps, max_steps, tools_used, nudge_threshold, urgent_threshold
                )
                if nudge:
                    obs_msg = f"{obs_msg}\n\n{nudge}"
                messages.append({"role": "user", "content": obs_msg})
                continue

            # Unknown action — surface to model and continue.
            observation = f"ACTION_ERROR: unknown action {action!r}"
            trace.append({"step": n_steps, "raw": resp.content, "action_error": observation})
            messages.append({"role": "user", "content": observation})

        return AgentSystemResult(
            final_answer=final_answer,
            completed=completed,
            n_steps=n_steps,
            total_cost_usd=total_cost,
            raw_trace={
                "termination_reason": termination_reason,
                "steps": trace,
                "context": context,
            },
        )


__all__ = ["SingleReActSystem"]
