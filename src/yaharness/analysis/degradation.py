"""Per-step success-rate degradation curve + linear-regression slope.

Per BENCHMARKS.md: per-step success conditional on previous steps having
succeeded.
"""

from __future__ import annotations

from collections.abc import Sequence


def per_step_success_curve(
    per_step_outcomes: Sequence[Sequence[bool]],
    max_steps: int | None = None,
) -> list[tuple[int, float, int]]:
    """Per-step conditional success rate.

    ``per_step_outcomes`` is a list of trajectories — one per problem/seed pair.
    Each trajectory is the sequence of per-step outcomes for that run; the
    trajectory ends naturally when the run halts (the final step's outcome
    represents whether that step succeeded).

    Returns ``[(step_index, conditional_success_rate, n_reached), ...]`` where
    ``conditional_success_rate`` at step ``i`` is the fraction of trajectories
    that reached step ``i`` and succeeded on that step.
    """
    if not per_step_outcomes:
        return []
    max_observed = max((len(t) for t in per_step_outcomes), default=0)
    limit = max_observed if max_steps is None else min(max_steps, max_observed)
    curve: list[tuple[int, float, int]] = []
    for i in range(limit):
        n_reached = sum(1 for t in per_step_outcomes if len(t) > i)
        if n_reached == 0:
            continue
        n_success = sum(1 for t in per_step_outcomes if len(t) > i and t[i])
        curve.append((i, n_success / n_reached, n_reached))
    return curve


def fit_degradation_slope(
    curve: Sequence[tuple[int, float, int]],
    min_n_reached: int = 10,
) -> float:
    """Ordinary-least-squares slope of conditional-success vs step index.

    Restricted to steps with ``n_reached >= min_n_reached``. Returns 0.0 if
    fewer than two qualifying points (no slope can be fit).
    """
    pts = [(float(i), r) for i, r, n in curve if n >= min_n_reached]
    if len(pts) < 2:
        return 0.0
    n = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxy = sum(x * y for x, y in pts)
    sxx = sum(x * x for x, _ in pts)
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0
    return (n * sxy - sx * sy) / denom
