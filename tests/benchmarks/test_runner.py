"""Runner end-to-end tests over toy_bench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from yaharness.benchmarks import (
    AgentSystemResult,
    ToyBenchAdapter,
    run_benchmark,
    toy_answers,
)
from yaharness.cost import CostBudget


class CorrectMockAgent:
    """Deterministic agent that returns the canonical toy_bench answer keyed
    by task_text — but in practice we lookup via problem_id, so we accept
    an answers map at construction time."""

    name = "correct-mock"

    def __init__(self, answers_by_text: dict[str, str]) -> None:
        self._by_text = answers_by_text

    async def __call__(
        self,
        task_text: str,
        context: dict[str, Any],
        *,
        cost_budget: CostBudget,
        max_steps: int = 50,
    ) -> AgentSystemResult:
        return AgentSystemResult(
            final_answer=self._by_text.get(task_text, ""),
            completed=True,
            n_steps=1,
            total_cost_usd=0.0,
            raw_trace={"task_text": task_text},
        )


class WrongMockAgent:
    name = "wrong-mock"

    async def __call__(
        self,
        task_text: str,
        context: dict[str, Any],
        *,
        cost_budget: CostBudget,
        max_steps: int = 50,
    ) -> AgentSystemResult:
        return AgentSystemResult(
            final_answer="WRONG", completed=True, n_steps=1, total_cost_usd=0.0
        )


class CrashOnFifthAgent:
    name = "crash-on-5"

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(
        self,
        task_text: str,
        context: dict[str, Any],
        *,
        cost_budget: CostBudget,
        max_steps: int = 50,
    ) -> AgentSystemResult:
        self.calls += 1
        if self.calls == 5:
            raise RuntimeError("boom")
        return AgentSystemResult(
            final_answer="placeholder", completed=False, n_steps=1, total_cost_usd=0.0
        )


def _answers_by_text(adapter: ToyBenchAdapter) -> dict[str, str]:
    return {p.task_text: p.expected_answer or "" for p in adapter.load_problems()}


async def test_runner_all_pass(tmp_path: Path) -> None:
    adapter = ToyBenchAdapter()
    agent = CorrectMockAgent(_answers_by_text(adapter))
    run = await run_benchmark(
        adapter=adapter,
        agent_system=agent,
        cost_budget=CostBudget(1.0),
        results_dir=tmp_path,
    )
    assert run.n_problems == 10
    assert run.n_seeds == 1
    assert len(run.per_seed_outcomes) == 1
    outcomes = run.per_seed_outcomes[0]
    assert len(outcomes) == 10
    assert all(o.success for o in outcomes)
    assert run.total_cost_usd == 0.0
    # Per-problem JSON written.
    written = list((tmp_path / "toy_bench-seed-0").glob("*.json"))
    assert len(written) == 10
    # Summary file written.
    summary = json.loads((tmp_path / "toy_bench-summary.json").read_text())
    assert summary["n_problems"] == 10


async def test_runner_all_fail(tmp_path: Path) -> None:
    run = await run_benchmark(
        adapter=ToyBenchAdapter(),
        agent_system=WrongMockAgent(),
        cost_budget=CostBudget(1.0),
        results_dir=tmp_path,
    )
    outcomes = run.per_seed_outcomes[0]
    assert len(outcomes) == 10
    assert all(not o.success for o in outcomes)
    assert all(o.false_positive_completion for o in outcomes)


async def test_runner_records_agent_crash(tmp_path: Path) -> None:
    run = await run_benchmark(
        adapter=ToyBenchAdapter(),
        agent_system=CrashOnFifthAgent(),
        cost_budget=CostBudget(1.0),
        results_dir=tmp_path,
    )
    outcomes = run.per_seed_outcomes[0]
    assert len(outcomes) == 10  # all 10 attempted, none silently dropped
    crashed = [o for o in outcomes if o.error is not None]
    assert len(crashed) == 1
    assert "boom" in (crashed[0].error or "")
    assert crashed[0].grader_notes == "agent crashed"
    assert crashed[0].success is False


async def test_runner_multi_seed(tmp_path: Path) -> None:
    adapter = ToyBenchAdapter()
    agent = CorrectMockAgent(_answers_by_text(adapter))
    run = await run_benchmark(
        adapter=adapter,
        agent_system=agent,
        cost_budget=CostBudget(1.0),
        results_dir=tmp_path,
        n_seeds=3,
        seed_offset=7,
    )
    assert run.n_seeds == 3
    assert len(run.per_seed_outcomes) == 3
    for i in range(3):
        seed = 7 + i
        seed_dir = tmp_path / f"toy_bench-seed-{seed}"
        assert seed_dir.exists()
        assert len(list(seed_dir.glob("*.json"))) == 10


async def test_runner_e2e_smoke_all_outcomes_produced(tmp_path: Path) -> None:
    """Self-soil step 5: run toy_bench end-to-end via runner with a
    deterministic MockAgentSystem and verify every outcome is produced."""
    adapter = ToyBenchAdapter()
    agent = CorrectMockAgent(_answers_by_text(adapter))
    run = await run_benchmark(
        adapter=adapter,
        agent_system=agent,
        cost_budget=CostBudget(1.0),
        results_dir=tmp_path,
    )
    expected_ids = set(toy_answers().keys())
    got_ids = {o.problem_id for o in run.per_seed_outcomes[0]}
    assert got_ids == expected_ids


def test_runner_n_seeds_must_be_positive(tmp_path: Path) -> None:
    import asyncio

    with pytest.raises(ValueError):
        asyncio.run(
            run_benchmark(
                adapter=ToyBenchAdapter(),
                agent_system=WrongMockAgent(),
                cost_budget=CostBudget(1.0),
                results_dir=tmp_path,
                n_seeds=0,
            )
        )
