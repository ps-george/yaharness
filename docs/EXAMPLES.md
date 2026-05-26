# Examples

Runnable scripts live in `examples/`. This document is a tour.

## Offline smoke

```bash
uv run python examples/solve_toy_bench.py
```

Solves three problems from the toy benchmark using `MockLLMClient` and
canned LLM responses. No API key required. Useful to confirm the
package is installed and importable.

## CLI smoke

```bash
uv run yabench \
  --benchmark toy --system single_react \
  --seeds 1 --limit 3 --budget 1.00 \
  --model mock:examples/fixtures/toy_responses.json \
  --results-dir /tmp/yaharness-cli-smoke
```

Same workload as the offline smoke, but via the CLI. Produces a
`BenchmarkRun` JSON and per-problem sidecars in `--results-dir`.

## Solve a SWE-bench Verified problem

```bash
export OPENROUTER_API_KEY=...
uv run python examples/solve_swebench_single.py
```

Loads one SWE-bench Verified problem, runs `single_react`, prints the
patch. NOT graded — use `yagrade` for Tier-2 grading.

## Plug in a custom benchmark

```bash
uv run python examples/custom_benchmark.py
```

Defines a tiny in-memory `ArithmeticAdapter` and runs it through the
generic runner. Copy this file as a starting point for your own
adapter — implement `name`, `load_problems`, and `grade`, then drop
the instance into `run_benchmark(adapter=..., ...)`.

## Compare two systems

```bash
for sys in single_react planner_worker; do
  uv run yabench \
    --benchmark gaia --subset level_1 \
    --system "$sys" --seeds 3 --limit 20 --budget 5.00 \
    --model openrouter:anthropic/claude-haiku-4.5 \
    --results-dir "results/gaia/$sys"
done
uv run yareport --results-dir results/gaia/ --out report.md
```

`yareport` discovers sidecars and groups by system. See
`docs/EVAL-METHODOLOGY.md` for the statistical conventions used.
