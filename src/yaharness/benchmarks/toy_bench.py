"""ToyBench — a tiny deterministic smoke benchmark for harness validation.

Ten trivial problems with exact-match grading after light normalisation. No
network, no LLM dependency. Used to verify the runner / adapter contract
end-to-end without spending budget on a real benchmark.
"""

from __future__ import annotations

from collections.abc import Sequence

from .outcome import AgentSystemResult, Problem, ProblemOutcome

_PROBLEMS: tuple[tuple[str, str, str], ...] = (
    ("toy-001", "What is 2+2?", "4"),
    ("toy-002", "What is the capital of France?", "Paris"),
    ("toy-003", "Reverse the string 'hello'.", "olleh"),
    ("toy-004", "How many letters are in 'harness'?", "7"),
    ("toy-005", "What is 7 times 6?", "42"),
    ("toy-006", "Name the largest planet in our solar system.", "Jupiter"),
    ("toy-007", "What colour do you get by mixing blue and yellow?", "green"),
    ("toy-008", "What is the square root of 81?", "9"),
    ("toy-009", "How many continents are there?", "7"),
    ("toy-010", "What is the chemical symbol for water?", "H2O"),
)


def _normalise(s: str) -> str:
    """Lowercase, strip, collapse whitespace, strip trailing sentence punctuation."""
    return " ".join(s.strip().lower().rstrip(".!?").split())


def _is_match(got: str, expected: str) -> bool:
    """Exact match OR expected appears as a whole-word substring of got.

    The substring path is what lets natural-language answers like
    'The sum of 2 and 2 is 4' match the expected '4' — closer to how
    real benchmarks (GAIA, etc.) grade.
    """
    g = _normalise(got)
    e = _normalise(expected)
    if not e:
        return False
    if g == e:
        return True
    # Whole-word substring: pad with spaces so 'is 4' matches '4' but
    # 'is 42' does not.
    return f" {e} " in f" {g} "


class ToyBenchAdapter:
    """Deterministic smoke benchmark adapter."""

    name: str = "toy_bench"

    def load_problems(
        self,
        *,
        subset: str | None = None,
        limit: int | None = None,
    ) -> Sequence[Problem]:
        if subset is not None and subset != "all":
            raise ValueError(f"toy_bench has no subset {subset!r}; only 'all' (or None)")
        problems = [
            Problem(
                problem_id=pid,
                task_text=text,
                expected_answer=ans,
                context={},
                metadata={"source": "toy_bench"},
            )
            for pid, text, ans in _PROBLEMS
        ]
        if limit is not None:
            problems = problems[:limit]
        return problems

    async def grade(
        self,
        problem: Problem,
        agent_result: AgentSystemResult,
    ) -> ProblemOutcome:
        expected = problem.expected_answer or ""
        got = agent_result.final_answer
        success = _is_match(got, expected)
        false_positive = agent_result.completed and not success
        notes = (
            "match (normalised, whole-word substring ok)"
            if success
            else f"expected {expected!r}, got {got!r} (normalised mismatch)"
        )
        return ProblemOutcome(
            problem_id=problem.problem_id,
            success=success,
            completed=agent_result.completed,
            false_positive_completion=false_positive,
            n_steps=agent_result.n_steps,
            cost_usd=agent_result.total_cost_usd,
            grader_notes=notes,
            raw_trace=agent_result.raw_trace,
        )


def toy_answers() -> dict[str, str]:
    """Canonical correct answers, for use by `MockAgentSystem` in tests."""
    return {pid: ans for pid, _, ans in _PROBLEMS}


__all__ = ["ToyBenchAdapter", "toy_answers"]
