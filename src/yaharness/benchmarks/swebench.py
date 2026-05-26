"""SWE-bench Verified adapter — Tier 1 (patch-producing).

Loads instances from the HuggingFace `princeton-nlp/SWE-bench_Verified`
dataset (or a local JSONL fixture for tests), frames each as a `Problem`
whose `task_text` instructs the agent to emit a unified-diff patch.

Tier 1 grading is *syntactic*: we check that `final_answer` parses as a
non-empty unified diff. The actual `fail_to_pass` / `pass_to_pass` test
execution is Tier 2 (the `lma-grade-swebench` CLI, separate brief) and runs
inside the official `swebench/sweb.eval.x86_64.<instance_id>` docker image.

Cache layout: `~/.cache/yaharness/swebench/` holds the HF
metadata snapshot. Repo checkouts (used by the agent when the agent
needs to read source) live under `<cache_dir>/repos/<owner>__<name>/`.

Expected disk usage: each Python repo in scope (django, sympy,
scikit-learn, etc.) is 100MB-1GB cloned with `--filter=blob:none`. Plan
for ~10GB total cache for the full 500-problem suite.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .outcome import AgentSystemResult, Problem, ProblemOutcome
from .swebench_harness.prompts import build_swebench_task_text, extract_relevant_files

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "yaharness" / "swebench"

# Curated subset of small-but-representative instances. Populated empirically;
# placeholder until measured. Callers passing `subset="small"` get whatever
# subset of the loaded instances matches.
SMALL_SUBSET_IDS: frozenset[str] = frozenset()


class SWEBenchLoadError(RuntimeError):
    """Raised when SWE-bench Verified instances cannot be loaded."""


_DIFF_GIT_RE = re.compile(r"^diff --git a/.+ b/.+$", re.MULTILINE)
_HUNK_RE = re.compile(r"^@@ .+ @@", re.MULTILINE)
_MINUS_HEADER_RE = re.compile(r"^--- a/.+$", re.MULTILINE)
_PLUS_HEADER_RE = re.compile(r"^\+\+\+ b/.+$", re.MULTILINE)


def is_valid_unified_diff(text: str) -> tuple[bool, str]:
    """Best-effort check: does `text` look like a unified-diff patch?

    Returns (is_valid, reason). We only validate structural shape, not
    that the patch applies to any particular tree — that's Tier 2.
    """
    if not text or not text.strip():
        return False, "empty patch"
    if not _DIFF_GIT_RE.search(text):
        return False, "missing 'diff --git a/... b/...' header"
    if not _MINUS_HEADER_RE.search(text):
        return False, "missing '--- a/...' file header"
    if not _PLUS_HEADER_RE.search(text):
        return False, "missing '+++ b/...' file header"
    if not _HUNK_RE.search(text):
        return False, "missing '@@ ... @@' hunk header"
    has_change = any(
        (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
        for line in text.splitlines()
    )
    if not has_change:
        return False, "no +/- change lines"
    return True, "valid unified diff"


def _hf_download(cache_dir: Path, revision: str) -> Path:
    """Download SWE-bench Verified metadata into `cache_dir`. Raises on failure."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SWEBenchLoadError(
            "huggingface_hub not installed; cannot fetch SWE-bench Verified. "
            "Install `huggingface_hub` or pass `metadata_path=` to SWEBenchVerifiedAdapter."
        ) from e
    # SWE-bench Verified ships as a Parquet file; for offline simplicity we
    # let callers point at a converted JSONL. Real network path: download the
    # parquet and convert. We defer parquet handling to a later integration
    # path — Tier 1 tests use a JSONL fixture exclusively.
    try:
        path = hf_hub_download(
            repo_id="princeton-nlp/SWE-bench_Verified",
            filename="data/test-00000-of-00001.parquet",
            repo_type="dataset",
            revision=revision,
            cache_dir=str(cache_dir),
        )
    except Exception as e:
        raise SWEBenchLoadError(
            f"Failed to download SWE-bench Verified from HuggingFace: {e}"
        ) from e
    return Path(path)


def _load_parquet_as_rows(parquet_path: Path) -> list[dict[str, Any]]:
    """Load a parquet file to a list of dicts. Requires `pyarrow`."""
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found,unused-ignore]
    except ImportError as e:
        raise SWEBenchLoadError(
            "pyarrow not installed; cannot read SWE-bench Verified parquet. "
            "Install `pyarrow` or pass a JSONL `metadata_path=`."
        ) from e
    table = pq.read_table(str(parquet_path))  # type: ignore[no-untyped-call]
    return [dict(row) for row in table.to_pylist()]


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_to_problem(row: dict[str, Any]) -> Problem:
    instance_id = str(row["instance_id"])
    repo = str(row["repo"])
    base_commit = str(row["base_commit"])
    problem_statement = str(row.get("problem_statement") or "")
    hints_text = str(row.get("hints_text") or "")
    relevant_files = extract_relevant_files(row)
    task_text = build_swebench_task_text(
        problem_statement=problem_statement,
        repo=repo,
        base_commit=base_commit,
        hints_text=hints_text,
        relevant_files=relevant_files,
    )
    fail_to_pass = row.get("FAIL_TO_PASS") or row.get("fail_to_pass") or []
    pass_to_pass = row.get("PASS_TO_PASS") or row.get("pass_to_pass") or []
    if isinstance(fail_to_pass, str):
        try:
            fail_to_pass = json.loads(fail_to_pass)
        except json.JSONDecodeError:
            fail_to_pass = [fail_to_pass]
    if isinstance(pass_to_pass, str):
        try:
            pass_to_pass = json.loads(pass_to_pass)
        except json.JSONDecodeError:
            pass_to_pass = [pass_to_pass]
    context: dict[str, Any] = {
        "repo": repo,
        "base_commit": base_commit,
        "problem_statement": problem_statement,
        "hints_text": hints_text,
        "relevant_files": relevant_files,
    }
    metadata: dict[str, Any] = {
        "source": "swebench_verified",
        "repo": repo,
        "base_commit": base_commit,
        "fail_to_pass": list(fail_to_pass),
        "pass_to_pass": list(pass_to_pass),
        "version": row.get("version"),
        # Golden patch is included in metadata but MUST NOT be exposed via
        # context (where it could leak into the agent's prompt).
        "golden_patch": row.get("patch") or row.get("golden_patch"),
        "test_patch": row.get("test_patch"),
    }
    return Problem(
        problem_id=instance_id,
        task_text=task_text,
        context=context,
        expected_answer=None,  # No single ground-truth string for SWE-bench
        metadata=metadata,
    )


class SWEBenchVerifiedAdapter:
    """SWE-bench Verified adapter — Tier 1, patch-producing only.

    Parameters
    ----------
    metadata_path:
        Optional JSONL file with pre-fetched instances (used by tests and
        callers who pre-downloaded). When provided, no network access.
    cache_dir:
        Where HF downloads are cached.
        Defaults to `~/.cache/yaharness/swebench/`.
    dataset_revision:
        HuggingFace dataset revision (branch/tag/sha). Defaults to `main`.
    """

    name: str = "swebench_verified"

    def __init__(
        self,
        *,
        metadata_path: Path | None = None,
        cache_dir: Path | None = None,
        dataset_revision: str = "main",
    ) -> None:
        self._metadata_path = metadata_path
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._dataset_revision = dataset_revision

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def _load_rows(self) -> list[dict[str, Any]]:
        if self._metadata_path is not None:
            if not self._metadata_path.exists():
                raise SWEBenchLoadError(f"metadata_path does not exist: {self._metadata_path}")
            suffix = self._metadata_path.suffix.lower()
            if suffix in {".jsonl", ".json"}:
                return _load_jsonl_rows(self._metadata_path)
            if suffix == ".parquet":
                return _load_parquet_as_rows(self._metadata_path)
            raise SWEBenchLoadError(
                f"unsupported metadata_path suffix: {suffix} (want .jsonl or .parquet)"
            )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        parquet = _hf_download(self._cache_dir, self._dataset_revision)
        return _load_parquet_as_rows(parquet)

    def load_problems(
        self,
        *,
        subset: str | None = None,
        limit: int | None = None,
    ) -> Sequence[Problem]:
        """Load SWE-bench Verified problems.

        Parameters
        ----------
        subset:
            - None or "all": return everything.
            - "small": return only instances in `SMALL_SUBSET_IDS`.
            - A repo owner or repo name fragment (e.g. "django",
              "sympy", "scikit-learn"): filter `repo` field by substring.
        limit:
            Cap the number of problems returned (after sort+filter).
        """
        rows = self._load_rows()
        problems = [_row_to_problem(row) for row in rows]
        if subset and subset != "all":
            if subset == "small":
                problems = [p for p in problems if p.problem_id in SMALL_SUBSET_IDS]
            else:
                needle = subset.lower()
                problems = [
                    p for p in problems if needle in str(p.metadata.get("repo", "")).lower()
                ]
        problems.sort(key=lambda p: p.problem_id)
        if limit is not None:
            problems = problems[:limit]
        return problems

    async def grade(
        self,
        problem: Problem,
        agent_result: AgentSystemResult,
    ) -> ProblemOutcome:
        """Tier-1 grading: validate that `final_answer` is a unified diff.

        We deliberately do NOT execute the patch here. Real pass/fail
        determination is the responsibility of the Tier-2 docker grader
        (`lma-grade-swebench`). A `success=True` outcome here means the
        agent produced a syntactically valid patch — necessary but not
        sufficient for actual benchmark success.
        """
        text = agent_result.final_answer
        valid, reason = is_valid_unified_diff(text)
        notes = f"tier1-syntactic-check: {reason}"
        if valid:
            notes += " (NOTE: real pass/fail requires Tier-2 docker grading)"
        false_positive = agent_result.completed and not valid
        return ProblemOutcome(
            problem_id=problem.problem_id,
            success=valid,
            completed=agent_result.completed,
            false_positive_completion=false_positive,
            n_steps=agent_result.n_steps,
            cost_usd=agent_result.total_cost_usd,
            grader_notes=notes,
            raw_trace=agent_result.raw_trace,
        )


__all__ = [
    "DEFAULT_CACHE_DIR",
    "SMALL_SUBSET_IDS",
    "SWEBenchLoadError",
    "SWEBenchVerifiedAdapter",
    "is_valid_unified_diff",
]
