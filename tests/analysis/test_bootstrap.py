"""Paired-bootstrap tests: known effect, no effect, determinism, validation."""

from __future__ import annotations

import random

import pytest

from yaharness.analysis.bootstrap import paired_bootstrap


def test_known_effect_detected() -> None:
    # 100 trials, A passes 70%, B passes 30% — paired (same problem indices).
    rng = random.Random(0)
    a = [rng.random() < 0.7 for _ in range(100)]
    b = [rng.random() < 0.3 for _ in range(100)]

    r = paired_bootstrap(a, b, n_resamples=2000, random_seed=1)

    assert r.p_value < 0.01
    assert r.ci_low > 0  # CI strictly excludes zero on the positive side
    assert r.difference > 0.2
    assert r.n_problems == 100
    assert r.n_resamples == 2000


def test_no_effect_p_value_large_and_ci_brackets_zero() -> None:
    rng = random.Random(7)
    a = [rng.random() < 0.5 for _ in range(200)]
    b = [rng.random() < 0.5 for _ in range(200)]

    r = paired_bootstrap(a, b, n_resamples=2000, random_seed=2)

    assert r.p_value > 0.1
    assert r.ci_low <= 0.0 <= r.ci_high


def test_determinism_with_fixed_seed() -> None:
    rng = random.Random(99)
    a = [rng.random() < 0.6 for _ in range(50)]
    b = [rng.random() < 0.4 for _ in range(50)]

    r1 = paired_bootstrap(a, b, n_resamples=500, random_seed=123)
    r2 = paired_bootstrap(a, b, n_resamples=500, random_seed=123)

    assert r1.model_dump() == r2.model_dump()


def test_different_seeds_produce_different_resamples() -> None:
    rng = random.Random(100)
    a = [rng.random() < 0.55 for _ in range(50)]
    b = [rng.random() < 0.45 for _ in range(50)]

    r1 = paired_bootstrap(a, b, n_resamples=500, random_seed=1)
    r2 = paired_bootstrap(a, b, n_resamples=500, random_seed=2)

    # CIs will differ even though observed difference is identical.
    assert r1.difference == r2.difference
    assert (r1.ci_low, r1.ci_high) != (r2.ci_low, r2.ci_high)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap([True, False], [True, False, True], n_resamples=10)


def test_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        paired_bootstrap([], [], n_resamples=10)


def test_alternative_greater_and_less() -> None:
    # A clearly better than B.
    a = [True] * 80 + [False] * 20
    b = [True] * 20 + [False] * 80

    r_greater = paired_bootstrap(a, b, n_resamples=500, alternative="greater")
    r_less = paired_bootstrap(a, b, n_resamples=500, alternative="less")

    assert r_greater.p_value < 0.05
    assert r_less.p_value > 0.5


def test_invalid_confidence_raises() -> None:
    with pytest.raises(ValueError, match="confidence"):
        paired_bootstrap([True], [False], n_resamples=10, confidence=1.5)


def test_invalid_n_resamples_raises() -> None:
    with pytest.raises(ValueError, match="n_resamples"):
        paired_bootstrap([True], [False], n_resamples=0)
