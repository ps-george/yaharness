"""Paired bootstrap test at problem level.

Per EVALUATION-METHODOLOGY.md failure-mode 2: statistical significance via paired
bootstrap at problem level (not aggregate level), p < 0.05 with >=1000 resamples.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field


class BootstrapResult(BaseModel):
    """Outcome of a paired-bootstrap comparison between two systems."""

    a_pass_rate: float
    b_pass_rate: float
    difference: float = Field(description="a_pass_rate - b_pass_rate")
    ci_low: float
    ci_high: float
    p_value: float
    n_problems: int
    n_resamples: int


def _mean_bool(xs: Sequence[bool]) -> float:
    if not xs:
        return 0.0
    return sum(1 for x in xs if x) / len(xs)


def paired_bootstrap(
    system_a_outcomes: Sequence[bool],
    system_b_outcomes: Sequence[bool],
    *,
    n_resamples: int = 10_000,
    alternative: Literal["greater", "less", "two-sided"] = "two-sided",
    confidence: float = 0.95,
    random_seed: int | None = 42,
) -> BootstrapResult:
    """Paired bootstrap at problem level.

    Tests H0: mean(A_outcomes) == mean(B_outcomes). Each resample draws problem
    indices WITH replacement; the same indices are used for both systems (paired),
    and we compute A_pass_rate - B_pass_rate on that resample. The distribution
    of differences gives the CI; the p-value is the bootstrap-shifted estimate.

    Outcomes must be the same length and aligned (same problem at same index).
    """
    if len(system_a_outcomes) != len(system_b_outcomes):
        raise ValueError(
            f"Outcome lists must be same length: {len(system_a_outcomes)} vs "
            f"{len(system_b_outcomes)}"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0,1); got {confidence}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >=1; got {n_resamples}")

    n = len(system_a_outcomes)
    if n == 0:
        raise ValueError("Outcome lists must be non-empty")

    a = [bool(x) for x in system_a_outcomes]
    b = [bool(x) for x in system_b_outcomes]

    a_rate = _mean_bool(a)
    b_rate = _mean_bool(b)
    observed_diff = a_rate - b_rate

    rng = random.Random(random_seed)
    diffs: list[float] = []
    for _ in range(n_resamples):
        a_sum = 0
        b_sum = 0
        for _i in range(n):
            idx = rng.randrange(n)
            if a[idx]:
                a_sum += 1
            if b[idx]:
                b_sum += 1
        diffs.append((a_sum - b_sum) / n)

    diffs.sort()
    alpha = 1.0 - confidence
    lo_idx = int((alpha / 2.0) * n_resamples)
    hi_idx = min(n_resamples - 1, int((1.0 - alpha / 2.0) * n_resamples))
    ci_low = diffs[lo_idx]
    ci_high = diffs[hi_idx]

    # P-value via bootstrap-shifted null: shift the resampled diffs so the mean
    # is zero (i.e. the null H0: difference == 0), then ask how often the
    # shifted distribution is as or more extreme than the observed difference.
    mean_diff = sum(diffs) / n_resamples
    shifted = [d - mean_diff for d in diffs]
    if alternative == "greater":
        p_value = sum(1 for d in shifted if d >= observed_diff) / n_resamples
    elif alternative == "less":
        p_value = sum(1 for d in shifted if d <= observed_diff) / n_resamples
    else:  # two-sided
        p_value = sum(1 for d in shifted if abs(d) >= abs(observed_diff)) / n_resamples

    return BootstrapResult(
        a_pass_rate=a_rate,
        b_pass_rate=b_rate,
        difference=observed_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n_problems=n,
        n_resamples=n_resamples,
    )
