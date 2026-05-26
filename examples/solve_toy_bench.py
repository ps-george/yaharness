"""Offline smoke example: solve a toy benchmark with the mock LLM client.

Run::

    uv run python examples/solve_toy_bench.py

No API key needed. Uses canned LLM responses from
``examples/fixtures/toy_responses.json``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from yaharness.agents import SingleReActSystem
from yaharness.benchmarks import ToyBenchAdapter, run_benchmark
from yaharness.cost import CostBudget
from yaharness.llm import LLMResponse, MockLLMClient


async def main() -> None:
    fixture = Path(__file__).parent / "fixtures" / "toy_responses.json"
    responses = [LLMResponse.model_validate(r) for r in json.loads(fixture.read_text())]
    llm = MockLLMClient(responses=responses)
    system = SingleReActSystem(llm_client=llm)

    results_dir = Path("/tmp/yaharness-toy-example")
    results_dir.mkdir(parents=True, exist_ok=True)

    run = await run_benchmark(
        adapter=ToyBenchAdapter(),
        agent_system=system,
        cost_budget=CostBudget(1.00),
        results_dir=results_dir,
        limit=3,
        max_steps_per_task=10,
        n_seeds=1,
    )

    for seed_outcomes in run.per_seed_outcomes:
        n = len(seed_outcomes)
        passed = sum(1 for o in seed_outcomes if o.success)
        print(f"{passed}/{n} solved; total cost ${run.total_cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
