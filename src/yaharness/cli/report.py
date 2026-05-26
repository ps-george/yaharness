"""`lma-report` — aggregate per-system BenchmarkRun JSON dumps into a
markdown comparison table via :mod:`yaharness.analysis.reporting`.

Usage::

    uv run lma-report --benchmark toy --output SUMMARY.md results/*.json

Each input file must be a :class:`BenchmarkRun` dump (as written by
``lma-bench``). All inputs are expected to describe the SAME benchmark; if
they don't, the script aborts (mixing benchmarks in one table is a
category error and silently doing it would mislead).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yaharness.analysis.reporting import SystemResults, benchmark_results_table
from yaharness.benchmarks.outcome import BenchmarkRun


def _run_to_system_results(run: BenchmarkRun) -> SystemResults:
    per_seed_outcomes: list[list[bool]] = [
        [o.success for o in seed] for seed in run.per_seed_outcomes
    ]
    per_seed_costs: list[list[float]] = [
        [o.cost_usd for o in seed] for seed in run.per_seed_outcomes
    ]
    false_positives: list[bool] = [
        o.false_positive_completion for seed in run.per_seed_outcomes for o in seed
    ]
    steps_per_success: list[float] = [
        float(o.n_steps) for seed in run.per_seed_outcomes for o in seed if o.success
    ]
    return SystemResults(
        system_name=run.agent_system_name,
        per_seed_outcomes=per_seed_outcomes,
        per_seed_costs_usd=per_seed_costs,
        per_step_outcomes=[],
        false_positive_completions=false_positives,
        steps_per_success=steps_per_success,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lma-report",
        description="Aggregate BenchmarkRun JSON dumps into a comparison markdown table.",
    )
    p.add_argument("inputs", nargs="+", type=Path, help="BenchmarkRun JSON files")
    p.add_argument("--output", "-o", type=Path, required=True, help="output markdown file")
    p.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="expected benchmark name; aborts on mismatch",
    )
    p.add_argument("--base-model", type=str, default="unknown")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    systems: dict[str, SystemResults] = {}
    seen_benchmark: str | None = args.benchmark
    for path in args.inputs:
        if not path.exists():
            print(f"ERROR: input not found: {path}", file=sys.stderr)
            return 2
        run = BenchmarkRun.model_validate(json.loads(path.read_text(encoding="utf-8")))
        if seen_benchmark is None:
            seen_benchmark = run.benchmark_name
        elif seen_benchmark != run.benchmark_name:
            print(
                f"ERROR: mixed benchmarks: expected {seen_benchmark!r}, "
                f"got {run.benchmark_name!r} in {path}",
                file=sys.stderr,
            )
            return 2
        systems[run.agent_system_name] = _run_to_system_results(run)

    if not systems:
        print("ERROR: no inputs supplied", file=sys.stderr)
        return 2

    md = benchmark_results_table(
        benchmark_name=seen_benchmark or "benchmark",
        systems=systems,
        base_model=args.base_model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


def _entry() -> None:
    sys.exit(main())


if __name__ == "__main__":  # pragma: no cover
    _entry()


__all__ = ["build_parser", "main"]
