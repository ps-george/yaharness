"""GAIA adapter tests: load from fixture, grader normalisation."""

from __future__ import annotations

from pathlib import Path

import pytest

from yaharness.benchmarks import (
    AgentSystemResult,
    GaiaAdapter,
    GaiaLoadError,
    gaia_normalise,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "gaia_mini.jsonl"


def test_load_problems_from_fixture() -> None:
    adapter = GaiaAdapter(metadata_path=FIXTURE)
    problems = adapter.load_problems()
    assert len(problems) == 3
    # Deterministic sort.
    assert [p.problem_id for p in problems] == ["gaia-001", "gaia-002", "gaia-003"]
    assert problems[0].expected_answer == "Paris"
    # File attachment captured in context.
    assert problems[2].context.get("files") == ["sample.txt"]


def test_load_problems_subset_filter() -> None:
    adapter = GaiaAdapter(metadata_path=FIXTURE)
    only_l1 = adapter.load_problems(subset="level_1")
    assert {p.problem_id for p in only_l1} == {"gaia-001", "gaia-002"}
    only_l2 = adapter.load_problems(subset="level_2")
    assert {p.problem_id for p in only_l2} == {"gaia-003"}


def test_load_problems_limit() -> None:
    adapter = GaiaAdapter(metadata_path=FIXTURE)
    problems = adapter.load_problems(limit=1)
    assert len(problems) == 1
    assert problems[0].problem_id == "gaia-001"


def test_missing_metadata_path_raises() -> None:
    adapter = GaiaAdapter(metadata_path=Path("/tmp/does-not-exist-yaharness.jsonl"))
    with pytest.raises(GaiaLoadError):
        adapter.load_problems()


def test_no_cache_no_hf_raises_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With empty cache_dir and huggingface_hub unavailable, raise GaiaLoadError."""
    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    adapter = GaiaAdapter(cache_dir=tmp_path / "gaia")
    with pytest.raises(GaiaLoadError):
        adapter.load_problems()


def test_gaia_normalise() -> None:
    assert gaia_normalise("  Paris  ") == "paris"
    assert gaia_normalise("Paris.") == "paris"
    assert gaia_normalise("The Eiffel Tower") == "eiffel tower"
    assert gaia_normalise("a   cat") == "cat"


async def test_grader_normalises_whitespace() -> None:
    adapter = GaiaAdapter(metadata_path=FIXTURE)
    problem = adapter.load_problems(limit=1)[0]  # Paris
    result = AgentSystemResult(
        final_answer="  paris  ", completed=True, n_steps=1, total_cost_usd=0.0
    )
    outcome = await adapter.grade(problem, result)
    assert outcome.success is True


async def test_grader_strips_articles_and_punctuation() -> None:
    adapter = GaiaAdapter(metadata_path=FIXTURE)
    problem = adapter.load_problems(limit=1)[0]  # Paris
    result = AgentSystemResult(
        final_answer="The Paris.", completed=True, n_steps=1, total_cost_usd=0.0
    )
    outcome = await adapter.grade(problem, result)
    assert outcome.success is True


async def test_grader_flags_false_positive_completion() -> None:
    adapter = GaiaAdapter(metadata_path=FIXTURE)
    problem = adapter.load_problems(limit=1)[0]
    result = AgentSystemResult(
        final_answer="London", completed=True, n_steps=5, total_cost_usd=0.02
    )
    outcome = await adapter.grade(problem, result)
    assert outcome.success is False
    assert outcome.false_positive_completion is True
