# Contributing

PRs welcome. Before opening one, please:

1. `uv sync`
2. `uv run ruff check . && uv run ruff format .`
3. `uv run mypy src/`
4. `uv run pytest`

All four must be green. Tests are pure-Python by default (no network, no
docker); SWE-bench grading and OpenRouter integration tests are skipped
unless their environment variables are set.

## Scope

This repo aims to stay small and readable. Please open an issue before
contributing:

- a new benchmark adapter (we may want to keep the core set tight),
- a new tool that brings heavy dependencies,
- a new agent system that duplicates an existing one.

Bug fixes, documentation improvements, and quality-of-life tweaks are
always welcome without prior discussion.

## Style

- Type-annotated, mypy-strict.
- `ruff` lint and format.
- Docstrings on public functions; comments for non-obvious decisions only.
- No emoji in code or comments.
