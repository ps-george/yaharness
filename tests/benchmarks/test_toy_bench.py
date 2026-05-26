"""ToyBench unit tests: loading + grading."""

from __future__ import annotations

import pytest

from yaharness.benchmarks import AgentSystemResult, ToyBenchAdapter, toy_answers


def test_load_returns_ten_problems() -> None:
    adapter = ToyBenchAdapter()
    problems = adapter.load_problems()
    assert len(problems) == 10
    assert {p.problem_id for p in problems} == set(toy_answers().keys())
    for p in problems:
        assert p.expected_answer  # all problems have a canonical answer
        assert p.metadata["source"] == "toy_bench"


def test_load_with_limit() -> None:
    adapter = ToyBenchAdapter()
    problems = adapter.load_problems(limit=3)
    assert len(problems) == 3


def test_load_unknown_subset_raises() -> None:
    adapter = ToyBenchAdapter()
    with pytest.raises(ValueError):
        adapter.load_problems(subset="level_1")


def test_load_all_subset_ok() -> None:
    adapter = ToyBenchAdapter()
    problems = adapter.load_problems(subset="all")
    assert len(problems) == 10


async def test_grade_correct_answer() -> None:
    adapter = ToyBenchAdapter()
    problem = adapter.load_problems(limit=1)[0]
    result = AgentSystemResult(
        final_answer=problem.expected_answer or "",
        completed=True,
        n_steps=1,
        total_cost_usd=0.0,
    )
    outcome = await adapter.grade(problem, result)
    assert outcome.success is True
    assert outcome.completed is True
    assert outcome.false_positive_completion is False


async def test_grade_wrong_answer_completed_flags_false_positive() -> None:
    adapter = ToyBenchAdapter()
    problem = adapter.load_problems(limit=1)[0]
    result = AgentSystemResult(
        final_answer="not the answer", completed=True, n_steps=3, total_cost_usd=0.01
    )
    outcome = await adapter.grade(problem, result)
    assert outcome.success is False
    assert outcome.false_positive_completion is True


async def test_grade_normalises_whitespace_and_case() -> None:
    adapter = ToyBenchAdapter()
    problem = adapter.load_problems(limit=1)[0]  # 2+2 → "4"
    result = AgentSystemResult(final_answer="  4  ", completed=True, n_steps=1, total_cost_usd=0.0)
    outcome = await adapter.grade(problem, result)
    assert outcome.success is True
