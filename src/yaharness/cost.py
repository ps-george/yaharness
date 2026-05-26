"""Cost budget + per-component cost attribution.

Pricing is in USD per million tokens, separately for input and output, in
line with the public price sheets for the listed models. Unknown models
fall back to a conservative default and emit a `warnings.warn`.
"""

from __future__ import annotations

import warnings
from typing import Any

# (input_per_million_usd, output_per_million_usd)
# Verified against OpenRouter pricing as of 2026-05-26.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    # Anthropic
    "anthropic/claude-haiku-4.5": (0.80, 4.00),
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
    "anthropic/claude-opus-4.7": (15.00, 75.00),
    # Google — current cheap/flash family
    "google/gemini-3.1-flash-lite": (0.10, 0.40),
    "google/gemini-3.5-flash": (0.30, 1.20),
    # OpenAI
    "openai/gpt-5-mini": (0.25, 1.00),
    "openai/gpt-5": (2.50, 10.00),
    # Open-weight, ultra-cheap
    "deepseek/deepseek-v4-flash": (0.04, 0.16),
    "meta-llama/llama-3.1-70b-instruct": (0.59, 0.79),
    # Legacy aliases (kept so older configs don't crash)
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "anthropic/claude-3-haiku": (0.25, 1.25),
    "google/gemini-flash-1.5": (0.075, 0.30),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
}

# Conservative fallback (~ Claude-3.5-Sonnet) when model is unknown.
_FALLBACK_PRICE: tuple[float, float] = (3.00, 15.00)


class BudgetExceededError(RuntimeError):
    """Raised by `CostBudget.check` (and `add`) when budget is exhausted."""

    def __init__(self, spent_usd: float, budget_usd: float) -> None:
        super().__init__(f"budget exceeded: spent ${spent_usd:.4f} > ${budget_usd:.4f}")
        self.spent_usd = spent_usd
        self.budget_usd = budget_usd


class CostBudget:
    """Tracks cumulative cost across all LLM calls in a run."""

    def __init__(self, budget_usd: float, *, hard_abort: bool = True) -> None:
        if budget_usd < 0:
            raise ValueError("budget_usd must be >= 0")
        self._budget_usd = budget_usd
        self._hard_abort = hard_abort
        self._spent_usd = 0.0

    def add(self, cost_usd: float) -> None:
        if cost_usd < 0:
            raise ValueError("cost_usd must be >= 0")
        self._spent_usd += cost_usd
        self.check()

    @property
    def spent_usd(self) -> float:
        return self._spent_usd

    @property
    def remaining_usd(self) -> float:
        return self._budget_usd - self._spent_usd

    @property
    def budget_usd(self) -> float:
        return self._budget_usd

    def check(self) -> None:
        if self._hard_abort and self._spent_usd > self._budget_usd:
            raise BudgetExceededError(self._spent_usd, self._budget_usd)


class CostTracker:
    """Per-component cost attribution wrapping a `CostBudget`."""

    def __init__(self, budget: CostBudget) -> None:
        self._budget = budget
        self._by_component: dict[str, float] = {}
        self._records: list[dict[str, Any]] = []

    def record(
        self,
        component: str,
        cost_usd: float,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if cost_usd < 0:
            raise ValueError("cost_usd must be >= 0")
        self._by_component[component] = self._by_component.get(component, 0.0) + cost_usd
        self._records.append(
            {
                "component": component,
                "cost_usd": cost_usd,
                "metadata": dict(metadata) if metadata else {},
            }
        )
        self._budget.add(cost_usd)

    def total(self) -> float:
        return sum(self._by_component.values())

    def by_component(self) -> dict[str, float]:
        return dict(self._by_component)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_usd": self.total(),
            "budget_usd": self._budget.budget_usd,
            "remaining_usd": self._budget.remaining_usd,
            "by_component": self.by_component(),
            "records": list(self._records),
        }


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate USD cost of a single LLM call.

    Unknown models fall back to a conservative price and emit a warning so
    that surprises are visible in test logs and CI output.
    """
    if tokens_in < 0 or tokens_out < 0:
        raise ValueError("token counts must be >= 0")
    price = MODEL_PRICES.get(model)
    if price is None:
        warnings.warn(
            f"unknown model {model!r}; using conservative fallback price",
            stacklevel=2,
        )
        price = _FALLBACK_PRICE
    in_per_m, out_per_m = price
    return (tokens_in / 1_000_000) * in_per_m + (tokens_out / 1_000_000) * out_per_m


__all__ = [
    "MODEL_PRICES",
    "BudgetExceededError",
    "CostBudget",
    "CostTracker",
    "estimate_cost",
]
