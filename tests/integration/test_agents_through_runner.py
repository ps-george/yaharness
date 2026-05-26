"""Integration: real agent systems (single_react, planner_worker) through the runner."""

from __future__ import annotations

import json
from pathlib import Path

from yaharness.agents import PlannerWorkerSystem, SingleReActSystem
from yaharness.benchmarks import ToyBenchAdapter, run_benchmark
from yaharness.cost import CostBudget
from yaharness.llm import MockLLMClient


def _react_final_answer(answer: str) -> str:
    return json.dumps({"thought": "done", "action": "final_answer", "final_answer": answer})


def _plan(steps: list[str]) -> str:
    return json.dumps({"plan": steps})


async def test_single_react_through_runner(tmp_path: Path) -> None:
    mock = MockLLMClient()
    # 3 problems, 1 call each (final_answer immediately).
    for ans in ("4", "Paris", "olleh"):
        mock.queue_text(_react_final_answer(ans))
    system = SingleReActSystem(llm_client=mock)
    run = await run_benchmark(
        adapter=ToyBenchAdapter(),
        agent_system=system,
        cost_budget=CostBudget(5.0),
        results_dir=tmp_path,
        limit=3,
    )
    outcomes = run.per_seed_outcomes[0]
    assert len(outcomes) == 3
    assert all(o.success for o in outcomes), [(o.problem_id, o.grader_notes) for o in outcomes]
    assert run.agent_system_name == "single_react"


async def test_planner_worker_through_runner(tmp_path: Path) -> None:
    mock = MockLLMClient()
    # For each problem: 1 planner call + 1 worker call returning final_answer.
    for ans in ("4", "Paris", "olleh"):
        mock.queue_text(_plan(["produce the answer"]))
        mock.queue_text(_react_final_answer(ans))
    system = PlannerWorkerSystem(llm_client=mock)
    run = await run_benchmark(
        adapter=ToyBenchAdapter(),
        agent_system=system,
        cost_budget=CostBudget(5.0),
        results_dir=tmp_path,
        limit=3,
    )
    outcomes = run.per_seed_outcomes[0]
    assert len(outcomes) == 3
    assert all(o.success for o in outcomes), [(o.problem_id, o.grader_notes) for o in outcomes]
