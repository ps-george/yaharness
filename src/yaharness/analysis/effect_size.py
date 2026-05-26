"""Effect-size measures for proportion differences.

Cohen's h (φ = 2·arcsin(√p)) and normal-approximation CI for the unpaired
difference in proportions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# Z critical values for common confidence levels. (Kept tiny; we look up the
# requested confidence and approximate via the inverse normal if needed.)
_Z_TABLE: dict[float, float] = {
    0.80: 1.2815515655446004,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.99: 2.5758293035489004,
}


def _inverse_normal(p: float) -> float:
    """Inverse-CDF of the standard normal, via Acklam's approximation.

    Used only when the requested confidence isn't in `_Z_TABLE`.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1); got {p}")
    # Coefficients
    a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239,
    ]
    b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ]
    c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ]
    d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996,
        3.754408661907416,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )


def _z_for(confidence: float) -> float:
    if confidence in _Z_TABLE:
        return _Z_TABLE[confidence]
    return _inverse_normal(1.0 - (1.0 - confidence) / 2.0)


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for the difference between two proportions.

    h = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2)). Conventional magnitudes: 0.2 small,
    0.5 medium, 0.8 large.
    """
    for name, p in (("p1", p1), ("p2", p2)):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{name} must be in [0,1]; got {p}")
    phi1 = 2.0 * math.asin(math.sqrt(p1))
    phi2 = 2.0 * math.asin(math.sqrt(p2))
    return phi1 - phi2


def proportion_diff_ci(
    a_outcomes: Sequence[bool],
    b_outcomes: Sequence[bool],
    *,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Wald (normal-approx) confidence interval for an unpaired proportion difference.

    Returns ``(point_estimate, ci_low, ci_high)`` where the point estimate is
    ``mean(a) - mean(b)``. Standard error uses ``sqrt(p1(1-p1)/n1 + p2(1-p2)/n2)``.
    """
    if not a_outcomes or not b_outcomes:
        raise ValueError("Both outcome lists must be non-empty")
    n_a = len(a_outcomes)
    n_b = len(b_outcomes)
    p_a = sum(1 for x in a_outcomes if x) / n_a
    p_b = sum(1 for x in b_outcomes if x) / n_b
    diff = p_a - p_b
    se = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z = _z_for(confidence)
    return diff, diff - z * se, diff + z * se
