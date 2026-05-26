"""Generic benchmark runner.

Bridges a `BenchmarkAdapter` and an `AgentSystem`: loads problems, runs the
agent on each, grades the result, writes incremental per-problem JSON, and
returns an aggregated `BenchmarkRun`.

Per-problem outcomes are written to
`results_dir/<adapter.name>-seed-{i}/<problem_id>.json` as they complete so
the run is safe to interrupt and resume (caller-level resume not yet
implemented; this is just durable progress).
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path

from yaharness.cost import CostBudget

from .outcome import BenchmarkRun, Problem, ProblemOutcome
from .protocol import AgentSystem, BenchmarkAdapter

logger = logging.getLogger(__name__)


def _safe_filename(problem_id: str) -> str:
    """Make a problem_id safe to use as a filename."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in problem_id)


async def run_benchmark(
    *,
    adapter: BenchmarkAdapter,
    agent_system: AgentSystem,
    cost_budget: CostBudget,
    results_dir: Path,
    subset: str | None = None,
    limit: int | None = None,
    max_steps_per_task: int = 50,
    n_seeds: int = 1,
    seed_offset: int = 0,
) -> BenchmarkRun:
    """Run an agent system on a benchmark. Returns aggregated results."""
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1")
    problems = list(adapter.load_problems(subset=subset, limit=limit))
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    per_seed_outcomes: list[list[ProblemOutcome]] = []
    total_cost = 0.0

    for i in range(n_seeds):
        seed = seed_offset + i
        seed_dir = results_dir / f"{adapter.name}-seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        outcomes: list[ProblemOutcome] = []
        for problem in problems:
            outcome = await _run_one(
                adapter=adapter,
                agent_system=agent_system,
                problem=problem,
                cost_budget=cost_budget,
                max_steps=max_steps_per_task,
            )
            total_cost += outcome.cost_usd
            outcomes.append(outcome)
            out_path = seed_dir / f"{_safe_filename(problem.problem_id)}.json"
            out_path.write_text(outcome.model_dump_json(indent=2), encoding="utf-8")
        per_seed_outcomes.append(outcomes)

    completed_at = time.time()
    run = BenchmarkRun(
        benchmark_name=adapter.name,
        agent_system_name=getattr(agent_system, "name", agent_system.__class__.__name__),
        subset=subset,
        n_problems=len(problems),
        n_seeds=n_seeds,
        per_seed_outcomes=per_seed_outcomes,
        total_cost_usd=total_cost,
        started_at=started_at,
        completed_at=completed_at,
    )
    summary_path = results_dir / f"{adapter.name}-summary.json"
    summary_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return run


async def _run_one(
    *,
    adapter: BenchmarkAdapter,
    agent_system: AgentSystem,
    problem: Problem,
    cost_budget: CostBudget,
    max_steps: int,
) -> ProblemOutcome:
    """Run agent on one problem; capture crashes as error outcomes."""
    try:
        result = await agent_system(
            problem.task_text,
            problem.context,
            cost_budget=cost_budget,
            max_steps=max_steps,
        )
    except Exception as e:
        logger.exception("Agent crashed on problem %s", problem.problem_id)
        return ProblemOutcome(
            problem_id=problem.problem_id,
            success=False,
            completed=False,
            false_positive_completion=False,
            n_steps=0,
            cost_usd=0.0,
            grader_notes="agent crashed",
            raw_trace={},
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )
    return await adapter.grade(problem, result)


__all__ = ["run_benchmark"]
