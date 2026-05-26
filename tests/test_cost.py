"""Tests for the cost budget + tracker + pricing helpers."""

from __future__ import annotations

import json
import warnings

import pytest

from yaharness.cost import (
    MODEL_PRICES,
    BudgetExceededError,
    CostBudget,
    CostTracker,
    estimate_cost,
)


def test_budget_add_and_remaining() -> None:
    b = CostBudget(1.00)
    b.add(0.25)
    assert b.spent_usd == pytest.approx(0.25)
    assert b.remaining_usd == pytest.approx(0.75)
    assert b.budget_usd == 1.00


def test_budget_hard_abort() -> None:
    b = CostBudget(0.10, hard_abort=True)
    b.add(0.05)
    with pytest.raises(BudgetExceededError) as excinfo:
        b.add(0.10)
    assert excinfo.value.spent_usd == pytest.approx(0.15)
    assert excinfo.value.budget_usd == 0.10


def test_budget_soft_does_not_raise() -> None:
    b = CostBudget(0.10, hard_abort=False)
    b.add(1.00)
    assert b.spent_usd == pytest.approx(1.00)
    b.check()  # should not raise


def test_budget_rejects_negative() -> None:
    with pytest.raises(ValueError):
        CostBudget(-1.0)
    b = CostBudget(1.0)
    with pytest.raises(ValueError):
        b.add(-0.01)


def test_tracker_aggregates_by_component() -> None:
    b = CostBudget(10.0)
    t = CostTracker(b)
    t.record("planner", 0.10)
    t.record("critic", 0.05)
    t.record("planner", 0.20, metadata={"step": 3})
    assert t.total() == pytest.approx(0.35)
    assert t.by_component() == {"planner": pytest.approx(0.30), "critic": pytest.approx(0.05)}
    assert b.spent_usd == pytest.approx(0.35)


def test_tracker_to_dict_json_serialisable() -> None:
    b = CostBudget(5.0)
    t = CostTracker(b)
    t.record("planner", 0.10, metadata={"model": "openai/gpt-4o-mini"})
    d = t.to_dict()
    assert d["total_usd"] == pytest.approx(0.10)
    assert d["budget_usd"] == 5.0
    assert d["remaining_usd"] == pytest.approx(4.90)
    assert d["by_component"]["planner"] == pytest.approx(0.10)
    # round-trip through JSON
    s = json.dumps(d)
    d2 = json.loads(s)
    assert d2["records"][0]["metadata"]["model"] == "openai/gpt-4o-mini"


def test_tracker_propagates_budget_abort() -> None:
    b = CostBudget(0.10)
    t = CostTracker(b)
    with pytest.raises(BudgetExceededError):
        t.record("planner", 0.20)


def test_tracker_rejects_negative() -> None:
    t = CostTracker(CostBudget(1.0))
    with pytest.raises(ValueError):
        t.record("x", -0.01)


def test_estimate_cost_known() -> None:
    # openai/gpt-4o-mini: (0.15, 0.60) per 1M tokens
    got = estimate_cost("openai/gpt-4o-mini", 1_000_000, 1_000_000)
    assert got == pytest.approx(0.15 + 0.60)


def test_estimate_cost_unknown_warns() -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        got = estimate_cost("unknown/model", 1_000, 1_000)
        assert any("unknown" in str(x.message) for x in w)
    assert got > 0


def test_estimate_cost_negative_tokens() -> None:
    with pytest.raises(ValueError):
        estimate_cost("openai/gpt-4o-mini", -1, 0)


def test_model_prices_required_keys() -> None:
    required = {
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3-haiku",
        "google/gemini-flash-1.5",
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "meta-llama/llama-3.1-70b-instruct",
    }
    assert required.issubset(MODEL_PRICES.keys())
