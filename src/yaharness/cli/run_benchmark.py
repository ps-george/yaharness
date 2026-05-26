"""`yabench` — run an agent system against a benchmark, write JSON results.

Resolves three things from CLI args:
  - the benchmark (toy | gaia | swebench)
  - the agent system (single_react | planner_worker | langgraph)
  - the LLM client (``mock:<fixture_path>`` for offline / CI runs,
    ``openrouter:<model_id>`` for real API spend)

Then constructs a :class:`CostBudget`, invokes :func:`run_benchmark`, and
writes ``<results_dir>/<benchmark>-<system>-<timestamp>.json`` containing
the full :class:`BenchmarkRun` dump. Per-problem outcome files are still
written incrementally by the runner.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from yaharness.agents import (
    PlannerWorkerSystem,
    SingleReActSystem,
)
from yaharness.benchmarks import (
    BenchmarkAdapter,
    BenchmarkRun,
    GaiaAdapter,
    ToyBenchAdapter,
    run_benchmark,
)
from yaharness.benchmarks.protocol import AgentSystem
from yaharness.cost import CostBudget
from yaharness.llm import LLMClient, LLMResponse, MockLLMClient, OpenRouterClient

logger = logging.getLogger(__name__)


# --- resolvers ---------------------------------------------------------


def _resolve_benchmark(name: str) -> BenchmarkAdapter:
    if name == "toy":
        return ToyBenchAdapter()
    if name == "gaia":
        return GaiaAdapter()
    raise SystemExit(f"unknown benchmark: {name!r} (expected: toy | gaia)")


def _resolve_llm(spec: str) -> LLMClient:
    """`mock:<fixture_path>` or `openrouter:<model_id>`."""
    if ":" not in spec:
        raise SystemExit(f"invalid --model {spec!r}; expected '<provider>:<id>'")
    provider, ident = spec.split(":", 1)
    if provider == "mock":
        path = Path(ident)
        if not path.exists():
            raise SystemExit(f"mock fixture not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise SystemExit(f"mock fixture must be a JSON list of responses: {path}")
        responses = [LLMResponse.model_validate(r) for r in raw]
        return MockLLMClient(responses=responses)
    if provider == "openrouter":
        return OpenRouterClient(model=ident)
    raise SystemExit(f"unknown LLM provider {provider!r} (expected: mock | openrouter)")


def _resolve_system(name: str, llm: LLMClient) -> AgentSystem:
    if name == "single_react":
        return SingleReActSystem(llm_client=llm)
    if name == "planner_worker":
        return PlannerWorkerSystem(llm_client=llm)
    raise SystemExit(f"unknown system: {name!r} (expected: single_react | planner_worker)")


# --- summary printing --------------------------------------------------


def _summary_table(run: BenchmarkRun) -> str:
    lines = [
        f"benchmark: {run.benchmark_name}",
        f"system:    {run.agent_system_name}",
        f"problems:  {run.n_problems}   seeds: {run.n_seeds}",
        f"total $:   {run.total_cost_usd:.4f}",
        f"elapsed:   {run.completed_at - run.started_at:.1f}s",
    ]
    for i, seed_outcomes in enumerate(run.per_seed_outcomes):
        n = len(seed_outcomes)
        passed = sum(1 for o in seed_outcomes if o.success)
        fp = sum(1 for o in seed_outcomes if o.false_positive_completion)
        errs = sum(1 for o in seed_outcomes if o.error is not None)
        lines.append(f"  seed {i}: {passed}/{n} pass  |  {fp} false-pos  |  {errs} errors")
    return "\n".join(lines)


# --- main --------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yabench",
        description="Run a yaharness agent system against a benchmark.",
    )
    p.add_argument("--benchmark", required=True, choices=["toy", "gaia"])
    p.add_argument(
        "--system",
        required=True,
        choices=["single_react", "planner_worker"],
    )
    p.add_argument("--seeds", type=int, default=1, help="number of seeds (default: 1)")
    p.add_argument("--limit", type=int, default=None, help="max problems (default: all)")
    p.add_argument("--subset", type=str, default=None, help="benchmark subset (e.g. level_1)")
    p.add_argument("--budget", type=float, required=True, help="cost budget in USD")
    p.add_argument(
        "--model",
        required=True,
        help="LLM spec: mock:<fixture.json> or openrouter:<model_id>",
    )
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--verbose", "-v", action="store_true")
    return p


async def _run(args: argparse.Namespace) -> BenchmarkRun:
    adapter = _resolve_benchmark(args.benchmark)
    llm = _resolve_llm(args.model)
    system = _resolve_system(args.system, llm)
    budget = CostBudget(args.budget)
    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    run = await run_benchmark(
        adapter=adapter,
        agent_system=system,
        cost_budget=budget,
        results_dir=results_dir,
        subset=args.subset,
        limit=args.limit,
        max_steps_per_task=args.max_steps,
        n_seeds=args.seeds,
    )
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = results_dir / f"{args.benchmark}-{args.system}-{timestamp}.json"
    out_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return run


def main(argv: list[str] | None = None) -> int:
    # Autoload .env from CWD so OPENROUTER_API_KEY etc. are available without
    # manual sourcing. No-op if .env doesn't exist.
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        run = asyncio.run(_run(args))
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive top-level
        logger.exception("benchmark run failed")
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(_summary_table(run))
    return 0


def _entry() -> Any:
    """Console-script wrapper that propagates the integer exit code."""
    sys.exit(main())


if __name__ == "__main__":  # pragma: no cover
    _entry()


__all__ = ["build_parser", "main"]
