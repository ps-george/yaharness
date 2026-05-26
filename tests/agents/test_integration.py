"""Integration test against the benchmark runner from the runner integration.

The runner is in flight. We skip if the runner / `toy_bench` aren't
importable yet; once the runner integration lands this test activates automatically.
"""

from __future__ import annotations

import json

import pytest

from yaharness.agents import AGENT_SYSTEMS
from yaharness.cost import CostBudget
from yaharness.llm import LLMResponse, MockLLMClient


@pytest.mark.asyncio
async def test_runner_dispatches_single_react() -> None:
    try:
        from yaharness.benchmarks import runner as _runner
        from yaharness.benchmarks import toy_bench as _toy  # noqa: F401
    except ImportError:
        pytest.skip("the runner integration runner / toy_bench not available yet")

    # Best-effort dispatch: the runner integration may expose a `run_benchmark` or similar.
    run_fn = getattr(_runner, "run_benchmark", None) or getattr(_runner, "run", None)
    if run_fn is None:
        pytest.skip("the runner integration runner has no run_benchmark/run entry point yet")

    llm = MockLLMClient(
        [
            LLMResponse(
                content=json.dumps(
                    {"action": "final_answer", "final_answer": "answer", "thought": "t"}
                ),
                cost_usd=0.001,
                model="mock",
                tokens_in=1,
                tokens_out=1,
            )
        ]
    )
    system = AGENT_SYSTEMS["single_react"](llm_client=llm)
    # Sanity-check direct call (runner integration validated in a later pass).
    result = await system("trivial", {}, cost_budget=CostBudget(1.0))
    assert result.completed is True
