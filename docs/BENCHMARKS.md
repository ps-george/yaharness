# Benchmarks

Three benchmark adapters ship with `yaharness`:

- `toy` — ten trivial offline problems, exact-match grading.
- `gaia` — GAIA Level 1/2/3 with normalised exact-match grading.
- `swebench` — SWE-bench Verified with optional Tier-2 docker grading
  via `yagrade`.

## Toy

Pure-Python, no network. Used as the canary for the runner. Pass a
`MockLLMClient` (or any other client) and a budget. See
`examples/solve_toy_bench.py`.

```bash
uv run yabench \
  --benchmark toy --system single_react \
  --seeds 1 --limit 10 --budget 1.00 \
  --model mock:examples/fixtures/toy_responses.json \
  --results-dir /tmp/yaharness-toy
```

## GAIA

GAIA metadata is downloaded on first use to
`~/.cache/yaharness/gaia/metadata.jsonl`. You'll need a
HuggingFace account with access to `gaia-benchmark/GAIA`.

```bash
huggingface-cli login
uv run yabench \
  --benchmark gaia --subset level_1 \
  --system single_react \
  --seeds 3 --limit 20 --budget 5.00 \
  --model openrouter:anthropic/claude-haiku-4.5 \
  --results-dir results/gaia/
```

Subsets: `level_1`, `level_2`, `level_3`. Grading uses
`gaia_answers_match`, which performs light normalisation
(case-folding, punctuation stripping, alias collapse) before
exact-match.

## SWE-bench Verified

Two phases: **predict** (agent produces patches) and **grade** (Tier-2
docker harness from `swebench` produces resolved/not-resolved).

### Predict

```bash
uv run yabench \
  --benchmark swebench \
  --system single_react \
  --seeds 1 --limit 5 --budget 10.00 \
  --model openrouter:anthropic/claude-sonnet-4.6 \
  --results-dir results/swebench/
```

Each problem produces a sidecar JSON with `instance_id`, `system`,
`seed`, and `final_answer` (the patch). The runner also writes the
combined per-run JSON.

### Grade

Requires Docker. `yagrade` invokes
`swebench.harness.run_evaluation`, caches environment images, and
merges `resolved: bool` back into each sidecar.

```bash
uv run yagrade \
  --eval-dir results/swebench/ \
  --dataset-name princeton-nlp/SWE-bench_Verified \
  --split test \
  --max-workers 2 \
  --timeout 1800
```

Grading is slow (5-20 min per instance on a cold cache) and
docker-image-heavy (1-3 GB per instance). Subsequent runs reuse cached
environment images.

## Reporting

`yareport` aggregates JSON sidecars or a `BenchmarkRun` JSON into a
markdown table with per-seed and pooled success rates, total cost, and
per-step degradation slopes where available.

```bash
uv run yareport --results-dir results/swebench/ --out report.md
```

See `docs/EVAL-METHODOLOGY.md` for the statistical conventions used in
the report.
