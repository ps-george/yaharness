"""Markdown reporting from per-system benchmark results.

Generates the committed table format from BENCHMARKS.md.
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

from pydantic import BaseModel, Field

from yaharness.analysis.bootstrap import paired_bootstrap
from yaharness.analysis.degradation import per_step_success_curve


class SystemResults(BaseModel):
    """Per-system, per-seed outcomes used to render the results table."""

    system_name: str
    per_seed_outcomes: list[list[bool]] = Field(
        description="Outer: seeds. Inner: per-problem success/fail."
    )
    per_seed_costs_usd: list[list[float]] = Field(
        description="Outer: seeds. Inner: per-problem $ cost."
    )
    per_step_outcomes: list[list[bool]] = Field(
        default_factory=list,
        description="Trajectories of per-step outcomes (one per problem*seed).",
    )
    false_positive_completions: list[bool] = Field(
        default_factory=list,
        description="Per-problem: agent claimed done but grader said fail.",
    )
    steps_per_success: list[float] = Field(
        default_factory=list,
        description="Steps used per successful problem (across seeds).",
    )


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _seed_pass_rate(seed_outcomes: list[bool]) -> float:
    if not seed_outcomes:
        return 0.0
    return sum(1 for x in seed_outcomes if x) / len(seed_outcomes)


def _seed_dollar_per_success(outcomes: list[bool], costs: list[float]) -> float | None:
    successes = sum(1 for x in outcomes if x)
    if successes == 0:
        return None
    return sum(costs) / successes


def _format_mean_std(xs: list[float], fmt: str = "{:.3f}") -> str:
    if not xs:
        return "n/a"
    return f"{fmt.format(_mean(xs))} ± {fmt.format(_std(xs))}"


def _format_money(x: float) -> str:
    """Format USD with adaptive precision; small numbers stay legible."""
    if x == 0:
        return "$0"
    if abs(x) >= 1:
        return f"${x:.2f}"
    if abs(x) >= 0.01:
        return f"${x:.3f}"
    return f"${x:.5f}"


def _format_money_mean_std(xs: list[float]) -> str:
    present = [x for x in xs if x is not None]
    if not present:
        return "n/a"
    return f"{_format_money(_mean(present))} ± {_format_money(_std(present))}"


def _pooled_outcomes_aligned(per_seed: list[list[bool]]) -> list[bool]:
    """Pool by averaging per-problem across seeds, then majority-threshold.

    For pairwise tests we want a per-problem outcome aligned across systems.
    With multiple seeds, we take the per-problem mean across seeds (a value in
    [0,1]) and threshold at >=0.5. This keeps the paired-by-problem structure
    that the methodology document commits to.
    """
    if not per_seed:
        return []
    n_problems = len(per_seed[0])
    n_seeds = len(per_seed)
    out: list[bool] = []
    for j in range(n_problems):
        s = sum(1 for seed in per_seed if j < len(seed) and seed[j])
        out.append(s / n_seeds >= 0.5)
    return out


def _primary_table(systems_sorted: list[SystemResults]) -> str:
    lines = [
        "| System | Pass rate (mean ± std) | $/success (mean ± std) | "
        "False-pos rate | Mean steps/success |",
        "|---|---|---|---|---|",
    ]
    for s in systems_sorted:
        pass_rates = [_seed_pass_rate(seed) for seed in s.per_seed_outcomes]
        dps = [
            d
            for d in (
                _seed_dollar_per_success(o, c)
                for o, c in zip(s.per_seed_outcomes, s.per_seed_costs_usd, strict=False)
            )
            if d is not None
        ]
        fp_rate = (
            sum(1 for x in s.false_positive_completions if x) / len(s.false_positive_completions)
            if s.false_positive_completions
            else 0.0
        )
        mean_steps = _mean(s.steps_per_success) if s.steps_per_success else 0.0
        lines.append(
            f"| {s.system_name} | "
            f"{_format_mean_std(pass_rates)} | "
            f"{_format_money_mean_std(dps)} | "
            f"{fp_rate:.3f} | "
            f"{mean_steps:.1f} |"
        )
    return "\n".join(lines)


def _verdict(p_value: float, ci_low: float, ci_high: float, alpha: float = 0.05) -> str:
    if p_value < alpha and ci_low * ci_high > 0:
        return "A > B" if ci_low > 0 else "B > A"
    return "inconclusive"


def _pairwise_table(systems: list[SystemResults]) -> str:
    lines = [
        "| A | B | Difference | 95% CI | p-value | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for a, b in combinations(systems, 2):
        a_out = _pooled_outcomes_aligned(a.per_seed_outcomes)
        b_out = _pooled_outcomes_aligned(b.per_seed_outcomes)
        if not a_out or not b_out or len(a_out) != len(b_out):
            continue
        r = paired_bootstrap(a_out, b_out, n_resamples=2000)
        lines.append(
            f"| {a.system_name} | {b.system_name} | "
            f"{r.difference:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | "
            f"{r.p_value:.4f} | {_verdict(r.p_value, r.ci_low, r.ci_high)} |"
        )
    return "\n".join(lines)


def _degradation_section(systems: list[SystemResults]) -> str:
    lines = ["| System | Step | Conditional success | N reached |", "|---|---|---|---|"]
    any_rows = False
    for s in systems:
        if not s.per_step_outcomes:
            continue
        curve = per_step_success_curve(s.per_step_outcomes)
        for step, rate, n in curve:
            any_rows = True
            lines.append(f"| {s.system_name} | {step} | {rate:.3f} | {n} |")
    if not any_rows:
        return "(no per-step trajectories provided)"
    return "\n".join(lines)


def benchmark_results_table(
    benchmark_name: str,
    systems: dict[str, SystemResults],
    *,
    base_model: str = "unknown",
) -> str:
    """Render the standard `results/<benchmark>-<date>.md` table block.

    Sorts systems by mean pass-rate descending. Generates the three sections
    committed in `BENCHMARKS.md`: primary results, pairwise statistical tests,
    per-step degradation.
    """
    if not systems:
        raise ValueError("`systems` must be non-empty")

    systems_list = list(systems.values())
    systems_sorted = sorted(
        systems_list,
        key=lambda s: _mean([_seed_pass_rate(o) for o in s.per_seed_outcomes]),
        reverse=True,
    )

    n_problems = (
        len(systems_sorted[0].per_seed_outcomes[0]) if systems_sorted[0].per_seed_outcomes else 0
    )
    n_seeds = len(systems_sorted[0].per_seed_outcomes)

    parts = [
        f"## Primary results — {benchmark_name} "
        f"({n_problems} problems, {n_seeds} seeds, base model: {base_model})",
        "",
        _primary_table(systems_sorted),
        "",
        "## Pairwise statistical tests (paired bootstrap, alpha=0.05)",
        "",
        _pairwise_table(systems_sorted),
        "",
        "## Per-step degradation curve",
        "",
        _degradation_section(systems_sorted),
        "",
    ]
    return "\n".join(parts)


def load_results(path: Path) -> dict[str, SystemResults]:
    """Load a JSON results dump into a ``{system_name: SystemResults}`` mapping.

    Expected JSON shape:

    ```
    {
      "systems": [
        { "system_name": "...", "per_seed_outcomes": [[...]], ... },
        ...
      ]
    }
    ```
    """
    data = json.loads(Path(path).read_text())
    return {s["system_name"]: SystemResults.model_validate(s) for s in data["systems"]}
