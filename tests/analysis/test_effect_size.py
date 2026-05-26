"""Cohen's h + proportion-difference CI."""

from __future__ import annotations

import math

import pytest

from yaharness.analysis.effect_size import cohens_h, proportion_diff_ci


def test_cohens_h_zero_when_equal() -> None:
    assert cohens_h(0.5, 0.5) == 0.0


def test_cohens_h_known_value() -> None:
    # 2*arcsin(sqrt(0.5)) - 2*arcsin(sqrt(0.4)) ~= 0.20135...
    h = cohens_h(0.5, 0.4)
    expected = 2 * math.asin(math.sqrt(0.5)) - 2 * math.asin(math.sqrt(0.4))
    assert math.isclose(h, expected, rel_tol=1e-12)
    assert math.isclose(h, 0.20135792, abs_tol=1e-6)


def test_cohens_h_antisymmetry() -> None:
    assert math.isclose(cohens_h(0.7, 0.3), -cohens_h(0.3, 0.7), rel_tol=1e-12)


def test_cohens_h_endpoints() -> None:
    # 0 → 0; 1 → π.  h(1,0) = π
    assert math.isclose(cohens_h(1.0, 0.0), math.pi, rel_tol=1e-12)


def test_cohens_h_validates_range() -> None:
    with pytest.raises(ValueError):
        cohens_h(-0.1, 0.5)
    with pytest.raises(ValueError):
        cohens_h(0.5, 1.1)


def test_proportion_diff_ci_point_estimate() -> None:
    a = [True] * 60 + [False] * 40
    b = [True] * 40 + [False] * 60
    diff, lo, hi = proportion_diff_ci(a, b)
    assert math.isclose(diff, 0.2, abs_tol=1e-9)
    assert lo < diff < hi


def test_proportion_diff_ci_matches_known_scipy_result() -> None:
    # Pre-computed once via scipy (not added as a dep):
    # p1=0.7 (n=100), p2=0.5 (n=100). diff=0.2,
    # se = sqrt(0.7*0.3/100 + 0.5*0.5/100) = sqrt(0.0046) ≈ 0.067823
    # z_0.95 = 1.95996398...
    # CI = 0.2 ± 1.95996398 * 0.067823 = [0.06707, 0.33293]
    a = [True] * 70 + [False] * 30
    b = [True] * 50 + [False] * 50
    diff, lo, hi = proportion_diff_ci(a, b, confidence=0.95)
    assert math.isclose(diff, 0.2, abs_tol=1e-9)
    assert math.isclose(lo, 0.06707, abs_tol=1e-3)
    assert math.isclose(hi, 0.33293, abs_tol=1e-3)


def test_proportion_diff_ci_99_pct_wider_than_95() -> None:
    a = [True] * 70 + [False] * 30
    b = [True] * 50 + [False] * 50
    _, lo95, hi95 = proportion_diff_ci(a, b, confidence=0.95)
    _, lo99, hi99 = proportion_diff_ci(a, b, confidence=0.99)
    assert lo99 < lo95
    assert hi99 > hi95


def test_proportion_diff_ci_empty_raises() -> None:
    with pytest.raises(ValueError):
        proportion_diff_ci([], [True])
