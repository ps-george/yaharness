"""Benchmark adapters and runner.

Public surface:
- Protocols: `AgentSystem`, `BenchmarkAdapter`
- Data models: `Problem`, `AgentSystemResult`, `ProblemOutcome`, `BenchmarkRun`
- Adapters: `ToyBenchAdapter`, `GaiaAdapter`, `SWEBenchVerifiedAdapter`
- Runner: `run_benchmark`

How to add a new benchmark adapter
----------------------------------
Create a class with a `name: str` class attribute and two methods:

    def load_problems(self, *, subset=None, limit=None) -> Sequence[Problem]:
        # parse your source format into Problem(problem_id, task_text,
        # context, expected_answer, metadata) and return a deterministic
        # ordered sequence.

    async def grade(self, problem, agent_result) -> ProblemOutcome:
        # compare agent_result.final_answer to problem.expected_answer (or
        # call a model-based grader). Set success, completed,
        # false_positive_completion, n_steps, cost_usd, grader_notes.

Register it by importing from `yaharness.benchmarks`. Drop into
`run_benchmark(adapter=YourAdapter(), agent_system=..., ...)` — no further
plumbing is required.
"""

from .gaia import GaiaAdapter, GaiaLoadError, gaia_answers_match, gaia_normalise
from .outcome import AgentSystemResult, BenchmarkRun, Problem, ProblemOutcome
from .protocol import AgentSystem, BenchmarkAdapter
from .runner import run_benchmark
from .swebench import (
    SWEBenchLoadError,
    SWEBenchVerifiedAdapter,
    is_valid_unified_diff,
)
from .toy_bench import ToyBenchAdapter, toy_answers

__all__ = [
    "AgentSystem",
    "AgentSystemResult",
    "BenchmarkAdapter",
    "BenchmarkRun",
    "GaiaAdapter",
    "GaiaLoadError",
    "Problem",
    "ProblemOutcome",
    "SWEBenchLoadError",
    "SWEBenchVerifiedAdapter",
    "ToyBenchAdapter",
    "gaia_answers_match",
    "gaia_normalise",
    "is_valid_unified_diff",
    "run_benchmark",
    "toy_answers",
]
