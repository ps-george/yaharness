"""Protocol smoke tests: ensure our adapters and an inline AgentSystem
runtime-check against the Protocol classes."""

from __future__ import annotations

from typing import Any

from yaharness.benchmarks import (
    AgentSystem,
    AgentSystemResult,
    BenchmarkAdapter,
    GaiaAdapter,
    ToyBenchAdapter,
)
from yaharness.cost import CostBudget


class _NullAgent:
    name = "null"

    async def __call__(
        self,
        task_text: str,
        context: dict[str, Any],
        *,
        cost_budget: CostBudget,
        max_steps: int = 50,
    ) -> AgentSystemResult:
        return AgentSystemResult(final_answer="", completed=False, n_steps=0, total_cost_usd=0.0)


def test_toy_bench_adapter_satisfies_protocol() -> None:
    assert isinstance(ToyBenchAdapter(), BenchmarkAdapter)


def test_gaia_adapter_satisfies_protocol() -> None:
    assert isinstance(GaiaAdapter(), BenchmarkAdapter)


def test_agent_system_protocol_runtime_check() -> None:
    assert isinstance(_NullAgent(), AgentSystem)
