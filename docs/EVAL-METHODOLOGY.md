# Evaluation methodology

The conventions in this document are what `yaharness.analysis` assumes
and what `yareport` surfaces.

## Unit of analysis

The unit is the **problem-seed pair**. A run with `n_problems=20,
n_seeds=3` produces 60 outcomes. Pooled success rates are over all
60; per-seed rates are over 20.

## Comparing two systems

When comparing two agent systems on the same benchmark:

1. Run both under the **same cost budget per problem**.
2. Use the **same seeds**. Seeds determine sampling temperature and any
   stochastic tie-breaking; matching seeds across systems makes pairing
   meaningful.
3. Treat the per-(problem, seed) outcome as the paired observation.

`yaharness.analysis.paired_bootstrap` returns a bootstrap distribution
of the success-rate difference and a 95% percentile CI. Default 10 000
resamples; reproducible by passing `seed=...`.

`cohens_h(p1, p2)` and `proportion_diff_ci(p1, p2, n1, n2)` give
non-bootstrap effect-size and Wald-interval baselines.

## Per-step degradation

`per_step_success_curve(per_step_outcomes)` returns
`[(step_index, conditional_success, n_reached), ...]` —
the fraction of trajectories that reached step `i` and succeeded on
that step. `fit_degradation_slope(curve, min_n_reached=10)` is the
OLS slope of conditional success vs step index, restricted to steps
with enough trajectories to be meaningful. Strongly negative slopes
indicate trajectory degradation.

## What the harness does NOT claim

- It does not claim its grading is more correct than the upstream
  benchmark's grading. Tier-2 SWE-bench grading via the official
  `swebench` docker harness is the ground truth; everything else is
  best-effort.
- It does not pool across benchmarks or interpolate missing seeds.
- It does not run statistical tests for you; you decide what's
  significant.

## Reproducibility checklist

- Pin the model version (OpenRouter sometimes routes to multiple
  underlying providers — `scripts/verify_openrouter_route.py` documents
  one way to confirm).
- Record the cost budget and `max_steps` per task.
- Use a deterministic seed list and report them.
- Commit the sidecar JSONs alongside the report markdown.
