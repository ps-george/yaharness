# yaharness

A clean, modern Python reference implementation of a ReAct agent harness
with SWE-bench grading support.

The name is honest about positioning: this is **yet another harness**, not
a novel architecture and not a SOTA system. It is meant to be small enough
to read in an afternoon, and structured to be a comfortable starting point
for your own experiments.

## What this is

- ReAct loop (Yao et al. 2022) with tool dispatch and observation injection
- Scoped tool registry (filesystem, shell, web, parsing, code execution)
  bindable to a working directory
- SWE-bench Verified adapter with Tier-2 docker grading via the official
  `swebench` package
- GAIA adapter with normalised exact-match grading
- A toy benchmark for offline smoke-testing (no API key needed)
- Statistical analysis (paired bootstrap, effect sizes, per-step degradation)
  and comparative reporting
- CLIs: `yabench` to run, `yagrade` to evaluate SWE-bench patches,
  `yareport` to aggregate results
- A `MockLLMClient` for testing without API spend
- Cost tracking against a current OpenRouter pricing table

Three agent systems are bundled as reference implementations:

- `single_react` — a single-agent ReAct loop.
- `planner_worker` — a two-agent sequential planner/worker system.
- `langgraph` — a thin LangGraph wrapper provided as a comparison option.

## What this is not

- A novel agent architecture (this is ReAct, plus a small planner/worker
  variant for comparison).
- A drop-in production alternative to OpenHands or SWE-Agent. Those have
  much broader tool ecosystems and have been battle-tested at scale.
- A SOTA-chasing project. No benchmark numbers are claimed.

## Why it might be useful

- You want to understand how an agent harness actually works without
  reading 10k+ lines of code.
- You want a small clean fork-base for your own experiments.
- You want a reference implementation of ReAct + SWE-bench grading that
  follows modern Python conventions (uv, ruff, mypy strict, pytest, anyio).
- You want to plug in a new benchmark and see how the adapter protocol works.

## Quickstart

```bash
uv sync
uv run yabench \
  --benchmark toy \
  --system single_react \
  --seeds 1 --limit 3 \
  --budget 1.00 \
  --model mock:examples/fixtures/toy_responses.json \
  --results-dir /tmp/yaharness-smoke
```

To run against a real model via OpenRouter:

```bash
export OPENROUTER_API_KEY=...
uv run yabench \
  --benchmark gaia --subset level_1 \
  --system single_react \
  --seeds 3 --limit 20 --budget 5.00 \
  --model openrouter:anthropic/claude-haiku-4.5 \
  --results-dir results/
```

See `docs/EXAMPLES.md` for more invocations and `examples/` for runnable
scripts.

## Documentation

- `docs/ARCHITECTURE.md` — module layout and the agent/benchmark protocols
- `docs/BENCHMARKS.md` — how to run toy / GAIA / SWE-bench
- `docs/EVAL-METHODOLOGY.md` — statistical methodology for comparisons
- `docs/EXAMPLES.md` — annotated example invocations

## Citation

If you use this in academic work, please cite the underlying papers
(ReAct, SWE-bench, GAIA) rather than this repo.

## Acknowledgements

- ReAct: Yao et al., 2022 — "ReAct: Synergizing Reasoning and Acting in
  Language Models"
- SWE-bench: Jimenez et al., 2023 — "SWE-bench: Can Language Models
  Resolve Real-World GitHub Issues?"
- GAIA: Mialon et al., 2023 — "GAIA: A Benchmark for General AI Assistants"
- OpenHands and SWE-Agent for showing the wider design space.

## License

MIT — see `LICENSE`.
