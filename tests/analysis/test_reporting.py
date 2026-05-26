"""Markdown reporting tests + end-to-end synthetic flow + load_results glue."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yaharness.analysis.reporting import (
    SystemResults,
    benchmark_results_table,
    load_results,
)


def _make_system(
    name: str,
    pass_rate: float,
    n_problems: int = 20,
    n_seeds: int = 3,
    cost_per_problem: float = 0.05,
) -> SystemResults:
    # Deterministic: first int(pass_rate*n_problems) succeed, rest fail.
    n_pass = int(pass_rate * n_problems)
    outcomes = [True] * n_pass + [False] * (n_problems - n_pass)
    per_seed_outcomes = [list(outcomes) for _ in range(n_seeds)]
    per_seed_costs = [[cost_per_problem] * n_problems for _ in range(n_seeds)]
    # Per-step trajectories: one per problem*seed, longer when system is better.
    per_step = [[True] * 5 + [False] for _ in range(n_problems * n_seeds)]
    return SystemResults(
        system_name=name,
        per_seed_outcomes=per_seed_outcomes,
        per_seed_costs_usd=per_seed_costs,
        per_step_outcomes=per_step,
        false_positive_completions=[False] * n_problems,
        steps_per_success=[5.0] * (n_pass * n_seeds),
    )


def test_table_well_formed_and_sorted() -> None:
    systems = {
        "weak": _make_system("weak", 0.3),
        "strong": _make_system("strong", 0.8),
        "mid": _make_system("mid", 0.55),
    }
    md = benchmark_results_table("synth-bench", systems, base_model="claude-test")

    assert "## Primary results — synth-bench (20 problems, 3 seeds" in md
    assert "claude-test" in md
    assert "## Pairwise statistical tests" in md
    assert "## Per-step degradation curve" in md

    # Sorted descending by pass rate: 'strong' before 'mid' before 'weak'.
    strong_idx = md.index("| strong |")
    mid_idx = md.index("| mid |")
    weak_idx = md.index("| weak |")
    assert strong_idx < mid_idx < weak_idx

    # Pairwise rows exist (C(3,2)=3 comparisons).
    pair_section = md.split("## Pairwise statistical tests")[1].split(
        "## Per-step degradation curve"
    )[0]
    pair_rows = [line for line in pair_section.splitlines() if line.startswith("|")]
    # header + separator + 3 data rows
    assert len(pair_rows) == 2 + 3

    # Degradation rows reference each system at step 0.
    assert "| strong | 0 |" in md or "strong | 0 " in md


def test_table_raises_on_empty_systems() -> None:
    with pytest.raises(ValueError):
        benchmark_results_table("x", {})


def test_load_results_roundtrip(tmp_path: Path) -> None:
    systems = {
        "alpha": _make_system("alpha", 0.6, n_problems=10, n_seeds=2),
        "beta": _make_system("beta", 0.3, n_problems=10, n_seeds=2),
    }
    payload = {"systems": [s.model_dump() for s in systems.values()]}
    out = tmp_path / "results.json"
    out.write_text(json.dumps(payload))

    loaded = load_results(out)
    assert set(loaded.keys()) == {"alpha", "beta"}
    assert loaded["alpha"].per_seed_outcomes == systems["alpha"].per_seed_outcomes

    md = benchmark_results_table("loaded", loaded)
    assert "alpha" in md and "beta" in md


def test_end_to_end_renders_clean_markdown(tmp_path: Path) -> None:
    """The brief's self-soil: construct synthetic results, dump JSON,
    load via the glue helper, produce a markdown table, and verify it
    renders cleanly (well-formed pipes, three sections, no exceptions)."""
    systems = {
        "rubber_stamp": _make_system("rubber_stamp", 0.4, n_problems=15, n_seeds=3),
        "iterative": _make_system("iterative", 0.7, n_problems=15, n_seeds=3),
        "single_agent": _make_system("single_agent", 0.5, n_problems=15, n_seeds=3),
    }
    json_path = tmp_path / "swe.json"
    json_path.write_text(json.dumps({"systems": [s.model_dump() for s in systems.values()]}))

    loaded = load_results(json_path)
    md = benchmark_results_table("swe-bench-verified", loaded, base_model="claude-x")
    out_md = tmp_path / "out.md"
    out_md.write_text(md)

    # Re-read and structurally check.
    text = out_md.read_text()
    assert text.startswith("## Primary results — swe-bench-verified")
    assert "## Pairwise statistical tests (paired bootstrap, alpha=0.05)" in text
    assert "## Per-step degradation curve" in text

    # Every table line should be a valid pipe-delimited row.
    for line in text.splitlines():
        if line.startswith("|"):
            assert line.rstrip().endswith("|"), f"malformed table row: {line!r}"

    # Sort order check: iterative (0.7) > single_agent (0.5) > rubber_stamp (0.4).
    bi_idx = text.index("| iterative |")
    sa_idx = text.index("| single_agent |")
    rs_idx = text.index("| rubber_stamp |")
    assert bi_idx < sa_idx < rs_idx
