"""Plug a custom benchmark into the harness.

This example defines an in-memory benchmark with two arithmetic problems
and runs `single_react` against it. Copy this file as a starting point
for your own benchmark adapter.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from yaharness.agents import SingleReActSystem
from yaharness.benchmarks import (
    AgentSystemResult,
    Problem,
    ProblemOutcome,
    run_benchmark,
)
from yaharness.cost import CostBudget
from yaharness.llm import LLMResponse, MockLLMClient


class ArithmeticAdapter:
    """Minimal benchmark adapter — two integer-arithmetic problems."""

    name = "arithmetic"

    def load_problems(
        self,
        *,
        subset: str | None = None,
        limit: int | None = None,
    ) -> Sequence[Problem]:
        problems = [
            Problem(
                problem_id="arith-001",
                task_text="What is 2 + 3?",
                context={},
                expected_answer="5",
                metadata={},
            ),
            Problem(
                problem_id="arith-002",
                task_text="What is 7 times 6?",
                context={},
                expected_answer="42",
                metadata={},
            ),
        ]
        if limit is not None:
            problems = problems[:limit]
        return problems

    async def grade(
        self,
        problem: Problem,
        agent_result: AgentSystemResult,
    ) -> ProblemOutcome:
        expected = (problem.expected_answer or "").strip()
        got = agent_result.final_answer.strip()
        ok = expected == got
        return ProblemOutcome(
            problem_id=problem.problem_id,
            success=ok,
            completed=agent_result.completed,
            false_positive_completion=agent_result.completed and not ok,
            n_steps=agent_result.n_steps,
            cost_usd=agent_result.total_cost_usd,
            grader_notes=f"expected={expected!r} got={got!r}",
            final_answer=agent_result.final_answer,
            error=None,
        )


async def main() -> None:
    llm = MockLLMClient(
        responses=[
            LLMResponse(
                content='{"thought": "2+3=5", "action": "final_answer", "final_answer": "5"}',
                cost_usd=0.0001,
                model="mock",
                tokens_in=10,
                tokens_out=10,
            ),
            LLMResponse(
                content='{"thought": "7*6=42", "action": "final_answer", "final_answer": "42"}',
                cost_usd=0.0001,
                model="mock",
                tokens_in=10,
                tokens_out=10,
            ),
        ]
    )
    system = SingleReActSystem(llm_client=llm)
    results_dir = Path("/tmp/yaharness-custom-bench")
    results_dir.mkdir(parents=True, exist_ok=True)
    run = await run_benchmark(
        adapter=ArithmeticAdapter(),
        agent_system=system,
        cost_budget=CostBudget(1.00),
        results_dir=results_dir,
        max_steps_per_task=5,
        n_seeds=1,
    )
    for seed_outcomes in run.per_seed_outcomes:
        for outcome in seed_outcomes:
            status = "PASS" if outcome.success else "FAIL"
            print(f"{status}  {outcome.problem_id}  {outcome.grader_notes}")


if __name__ == "__main__":
    asyncio.run(main())
