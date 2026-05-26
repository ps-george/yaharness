# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-26

Initial public release.

### Added

- `single_react`, `planner_worker`, and `langgraph` reference agent systems.
- Toy, GAIA, and SWE-bench Verified benchmark adapters.
- Tier-2 SWE-bench docker grading via `yagrade`.
- Scoped tool registry (filesystem, shell, code execution, web, parsing,
  search).
- `MockLLMClient` and `OpenRouterClient`.
- Statistical analysis: paired bootstrap, Cohen's h, per-step degradation
  curves.
- `yabench`, `yareport`, `yagrade` CLI entry points.
