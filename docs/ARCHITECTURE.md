# Architecture

`yaharness` is organised around a small set of protocols and four module
groups. The package layout under `src/yaharness/`:

```
agents/        single_react, planner_worker, langgraph
benchmarks/    protocol, outcome, toy_bench, gaia, swebench, runner
                 + swebench_harness/ (RepoCheckout + scoped tools + prompts)
tools/         filesystem, code_exec, shell, web, parse, search
analysis/      bootstrap, effect_size, degradation, reporting
cli/           run_benchmark (yabench), report (yareport), grade_swebench (yagrade)
llm.py         LLMClient protocol + MockLLMClient + OpenRouterClient
cost.py        CostBudget, CostTracker, MODEL_PRICES, estimate_cost
```

## Core protocols

Defined in `yaharness.benchmarks.protocol`:

- **`AgentSystem`** — the unit the runner dispatches. Anything with
  `name: str` and an async `__call__(task_text, context, *, cost_budget,
  max_steps) -> AgentSystemResult` satisfies it.
- **`BenchmarkAdapter`** — anything with `name`, a synchronous
  `load_problems(...)` returning a sequence of `Problem`, and an async
  `grade(problem, agent_result) -> ProblemOutcome`.

Bundled agent systems satisfy `AgentSystem`; bundled benchmark adapters
satisfy `BenchmarkAdapter`. The runner is generic over both.

## Data models (`benchmarks.outcome`)

- `Problem(problem_id, task_text, context, expected_answer, metadata)`
- `AgentSystemResult(final_answer, completed, n_steps, total_cost_usd,
   raw_trace)`
- `ProblemOutcome(problem_id, success, completed,
   false_positive_completion, n_steps, cost_usd, grader_notes,
   final_answer, error)`
- `BenchmarkRun(benchmark_name, agent_system_name, n_problems, n_seeds,
   per_seed_outcomes, total_cost_usd, started_at, completed_at)`

## Runner

`yaharness.benchmarks.runner.run_benchmark` iterates `(seed, problem)`
pairs, invokes the agent system under the supplied cost budget, grades
each result through the adapter, and writes per-problem JSON sidecars as
it goes (so a partial run is recoverable).

## Tool registry

`yaharness.tools.ToolRegistry` holds async `Tool` instances. Each tool
declares a `name`, `description`, `parameters_schema` (JSON Schema), and
an async `execute(**kwargs) -> ToolResult`. `ToolRegistry.schemas()`
emits the function-calling shape consumed by OpenAI-style and
Anthropic-style APIs.

The SWE-bench harness binds a working directory to a scoped subset of
tools via `swebench_harness.tools.maybe_build_swebench_registry_from_context`.

## LLM clients

`yaharness.llm.LLMClient` is a small async protocol with a single
`complete(system, messages, ...) -> LLMResponse` method.

- `MockLLMClient` — replays a list of `LLMResponse`. Used for tests and
  for offline smoke runs.
- `OpenRouterClient` — wraps OpenRouter's OpenAI-compatible endpoint,
  with retry, token counting, and per-call cost estimation against
  `cost.MODEL_PRICES`.

## Cost tracking

`CostBudget` is a single-account ledger that raises
`BudgetExceededError` when overdrawn. Every agent system threads a
`CostBudget` into each LLM call and aborts cleanly when the budget is
exhausted.

## Analysis

`yaharness.analysis` provides:

- `paired_bootstrap` over problem-level outcomes;
- `cohens_h` and `proportion_diff_ci` for effect sizes on success-rate
  differences;
- `per_step_success_curve` and `fit_degradation_slope` for trajectory
  diagnostics;
- `benchmark_results_table` for the markdown comparison shape used in
  `docs/BENCHMARKS.md`.
