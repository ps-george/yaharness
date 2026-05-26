"""Data models for benchmark problems, agent results, and per-problem outcomes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Problem(BaseModel):
    """A single benchmark problem as loaded by an adapter."""

    model_config = ConfigDict(frozen=True)

    problem_id: str
    task_text: str
    context: dict[str, Any] = Field(default_factory=dict)
    expected_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentSystemResult(BaseModel):
    """What an `AgentSystem` returns from a single task invocation."""

    model_config = ConfigDict(frozen=True)

    final_answer: str
    completed: bool
    n_steps: int
    total_cost_usd: float
    raw_trace: dict[str, Any] = Field(default_factory=dict)


class ProblemOutcome(BaseModel):
    """The graded outcome of running an `AgentSystem` on one `Problem`."""

    problem_id: str
    success: bool
    completed: bool
    false_positive_completion: bool
    n_steps: int
    cost_usd: float
    grader_notes: str = ""
    raw_trace: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None  # populated if the agent crashed on this problem


class BenchmarkRun(BaseModel):
    """Aggregated results of running an agent system on a benchmark."""

    benchmark_name: str
    agent_system_name: str
    subset: str | None
    n_problems: int
    n_seeds: int
    per_seed_outcomes: list[list[ProblemOutcome]]
    total_cost_usd: float
    started_at: float
    completed_at: float


__all__ = [
    "AgentSystemResult",
    "BenchmarkRun",
    "Problem",
    "ProblemOutcome",
]
