"""Per-step success curve + linear-regression e."""

from __future__ import annotations

import math

from yaharness.analysis.degradation import (
    fit_degradation_slope,
    per_step_success_curve,
)


def test_per_step_curve_hand_constructed() -> None:
    # 4 trajectories of varying lengths.
    trajs = [
        [True, True, True, False],  # reached steps 0..3
        [True, True, False],  # reached steps 0..2
        [True, False],  # reached steps 0..1
        [False],  # reached step 0
    ]
    curve = per_step_success_curve(trajs)
    # step 0: 3/4 successes, n=4
    # step 1: 2/3 successes, n=3
    # step 2: 1/2 successes, n=2
    # step 3: 0/1 successes, n=1
    assert curve == [
        (0, 0.75, 4),
        (1, 2 / 3, 3),
        (2, 0.5, 2),
        (3, 0.0, 1),
    ]


def test_per_step_curve_empty() -> None:
    assert per_step_success_curve([]) == []


def test_per_step_curve_max_steps_caps() -> None:
    trajs = [[True, True, True, True, True] for _ in range(3)]
    curve = per_step_success_curve(trajs, max_steps=2)
    assert len(curve) == 2
    assert all(r == 1.0 and n == 3 for _, r, n in curve)


def test_fit_degradation_slope_decreasing() -> None:
    # Hand-made: success rate drops linearly from 1.0 to 0.4 over steps 0..3,
    # all with enough n_reached.
    curve = [
        (0, 1.0, 100),
        (1, 0.8, 100),
        (2, 0.6, 100),
        (3, 0.4, 100),
    ]
    e = fit_degradation_slope(curve, min_n_reached=10)
    assert math.isclose(e, -0.2, abs_tol=1e-9)


def test_fit_degradation_slope_filters_low_n() -> None:
    # First three points have plenty of data; last has too few — should be ignored.
    curve = [
        (0, 1.0, 50),
        (1, 0.5, 50),
        (2, 0.0, 50),
        (3, 1.0, 2),  # outlier; filtered
    ]
    e = fit_degradation_slope(curve, min_n_reached=10)
    assert math.isclose(e, -0.5, abs_tol=1e-9)


def test_fit_degradation_slope_insufficient_points_returns_zero() -> None:
    curve = [(0, 1.0, 100)]
    assert fit_degradation_slope(curve, min_n_reached=10) == 0.0


def test_fit_degradation_slope_all_filtered_returns_zero() -> None:
    curve = [(0, 1.0, 1), (1, 0.5, 1)]
    assert fit_degradation_slope(curve, min_n_reached=10) == 0.0
