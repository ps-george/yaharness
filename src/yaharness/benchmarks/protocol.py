"""Protocols defining the contract between agent systems and benchmark adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from yaharness.cost import CostBudget

from .outcome import AgentSystemResult, Problem, ProblemOutcome


@runtime_checkable
class AgentSystem(Protocol):
    """A callable that runs a single task end-to-end.

    Both the harness and agent system wrappers (this iteration) implement this protocol.
    """

    name: str

    async def __call__(
        self,
        task_text: str,
        context: dict[str, Any],
        *,
        cost_budget: CostBudget,
        max_steps: int = 50,
    ) -> AgentSystemResult: ...


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """Loads benchmark problems and grades agent outputs."""

    name: str

    def load_problems(
        self,
        *,
        subset: str | None = None,
        limit: int | None = None,
    ) -> Sequence[Problem]: ...

    async def grade(
        self,
        problem: Problem,
        agent_result: AgentSystemResult,
    ) -> ProblemOutcome: ...


__all__ = ["AgentSystem", "BenchmarkAdapter"]
