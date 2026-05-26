"""Solve a single SWE-bench Verified problem with the single_react agent.

Requires a real LLM. Run::

    export OPENROUTER_API_KEY=...
    uv run python examples/solve_swebench_single.py

This script loads one problem from SWE-bench Verified, runs
`single_react` against it, and prints the resulting patch. The patch is
NOT graded — use `yagrade` for Tier-2 docker grading.
"""

from __future__ import annotations

import asyncio

from yaharness.agents import SingleReActSystem
from yaharness.benchmarks import SWEBenchVerifiedAdapter
from yaharness.cost import CostBudget
from yaharness.llm import OpenRouterClient


async def main() -> None:
    adapter = SWEBenchVerifiedAdapter()
    problems = adapter.load_problems(limit=1)
    if not problems:
        raise SystemExit("no problems loaded; check your HuggingFace cache")

    problem = problems[0]
    print(f"problem: {problem.problem_id}")
    print(f"task:\n{problem.task_text[:400]}...\n")

    llm = OpenRouterClient(model="anthropic/claude-haiku-4.5")
    system = SingleReActSystem(llm_client=llm)
    result = await system(
        problem.task_text,
        problem.context,
        cost_budget=CostBudget(2.00),
        max_steps=30,
    )
    print(f"completed: {result.completed}")
    print(f"steps:     {result.n_steps}")
    print(f"cost:      ${result.total_cost_usd:.4f}")
    print("patch (first 800 chars):")
    print(result.final_answer[:800])


if __name__ == "__main__":
    asyncio.run(main())
