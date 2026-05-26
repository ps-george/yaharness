"""GAIA benchmark adapter.

GAIA: https://huggingface.co/datasets/gaia-benchmark/GAIA — 466 validation
problems across 3 difficulty levels. We deliberately avoid the heavy
`datasets` dependency: GAIA's `metadata.jsonl` is a small file we can fetch
directly via `huggingface_hub.hf_hub_download` and parse with `json`.

Loading strategy
----------------
1. Look for an already-downloaded cache file under
   `~/.cache/yaharness/gaia/metadata.jsonl`. If present, parse it.
2. Otherwise, try `huggingface_hub.hf_hub_download` for
   `2023/validation/metadata.jsonl` into the cache dir.
3. Tests can sidestep all of this by passing `metadata_path=` directly to
   `GaiaAdapter(...)`, pointing at a fixture file.

If the dataset cannot be fetched (no network) and no fixture / cache is
provided, `load_problems()` raises a clear `GaiaLoadError`. The smoke
benchmark (`toy_bench`) exists for offline harness testing.

Grading
-------
GAIA uses exact-match after normalisation. We implement defensive
normalisation: whitespace collapse, lowercase, strip trailing punctuation,
and strip leading articles ("the ", "a ", "an "). We do **not** implement
number-word ↔ digit aliasing (defensible per brief — documented here).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .outcome import AgentSystemResult, Problem, ProblemOutcome

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "yaharness" / "gaia"


class GaiaLoadError(RuntimeError):
    """Raised when GAIA problems cannot be loaded (no fixture, no cache, no network)."""


_PUNCT_RE = re.compile(r"[\s\.,;:!\?\"'`]+$")
_LEAD_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)

# Number-words → digits. Covers cardinals up to 999 via composition.
_UNITS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS: dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALES: dict[str, int] = {"hundred": 100, "thousand": 1000, "million": 1_000_000}


def _word_to_int(token: str) -> int | None:
    """Convert a single number-word token to int.

    Handles hyphenated compounds ("twenty-one") and single tokens. Returns
    None if not a number-word.
    """
    t = token.lower().replace("-", " ")
    parts = t.split()
    if not parts:
        return None
    total = 0
    current = 0
    matched = False
    for p in parts:
        if p in _UNITS:
            current += _UNITS[p]
            matched = True
        elif p in _TENS:
            current += _TENS[p]
            matched = True
        elif p in _SCALES:
            scale = _SCALES[p]
            if current == 0:
                current = 1
            if scale == 100:
                current *= 100
            else:
                total += current * scale
                current = 0
            matched = True
        elif p == "and":
            continue
        else:
            return None
    return total + current if matched else None


def _replace_number_words(s: str) -> str:
    """Replace runs of number-words (incl. hyphenated) with digit strings."""
    tokens = re.findall(r"[A-Za-z\-]+|[^A-Za-z\-]+", s)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if re.match(r"^[A-Za-z\-]+$", tok) and tok.lower() not in {"a", "an", "and", "the"}:
            # Try greedy consume of consecutive word-tokens separated by spaces.
            j = i
            run: list[str] = []
            while j < len(tokens):
                t = tokens[j]
                if re.match(r"^[A-Za-z\-]+$", t):
                    run.append(t)
                    j += 1
                elif (
                    j + 1 < len(tokens)
                    and tokens[j].strip() == ""
                    and re.match(r"^[A-Za-z\-]+$", tokens[j + 1])
                ):
                    run.append(tokens[j + 1])
                    j += 2
                else:
                    break
            # Try the longest valid number-word run.
            best_len = 0
            best_value: int | None = None
            for k in range(len(run), 0, -1):
                candidate = " ".join(run[:k])
                val = _word_to_int(candidate)
                if val is not None:
                    best_len = k
                    best_value = val
                    break
            if best_value is not None and best_len > 0:
                out.append(str(best_value))
                # Skip the consumed tokens: we consumed best_len word-tokens which
                # in the original ``tokens`` stream may include separators.
                consumed_words = 0
                k = i
                while k < len(tokens) and consumed_words < best_len:
                    if re.match(r"^[A-Za-z\-]+$", tokens[k]):
                        consumed_words += 1
                    k += 1
                i = k
                continue
        out.append(tok)
        i += 1
    return "".join(out)


def gaia_normalise(s: str) -> str:
    """Normalise a GAIA answer for exact-match comparison.

    - Lowercase
    - Collapse internal whitespace
    - Strip leading/trailing whitespace
    - Strip trailing punctuation (.,;:!?'"`)
    - Strip leading articles ("the ", "a ", "an ")
    - Number-words → digits ("twelve" → "12", "twenty-one" → "21")

    Note: set-comparison for comma-separated lists is handled by
    ``gaia_answers_match`` (not by this function, which preserves order).
    """
    s = s.strip().lower()
    s = _replace_number_words(s)
    s = " ".join(s.split())
    s = _PUNCT_RE.sub("", s)
    s = _LEAD_ARTICLE_RE.sub("", s)
    return s


def gaia_answers_match(expected: str, got: str) -> bool:
    """Official-spec GAIA match: scalar exact-match OR set-equality for lists.

    A comma-separated answer is treated as an unordered set of normalised
    items. Both sides must be comma-separated for set comparison to apply
    (otherwise the canonical exact-match path is used).
    """
    if "," in expected and "," in got:
        exp_set = {gaia_normalise(p) for p in expected.split(",") if p.strip()}
        got_set = {gaia_normalise(p) for p in got.split(",") if p.strip()}
        return exp_set == got_set
    return gaia_normalise(expected) == gaia_normalise(got)


def _hf_download(cache_dir: Path) -> Path:
    """Download GAIA metadata.jsonl from HuggingFace into `cache_dir`. Raises GaiaLoadError on failure."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise GaiaLoadError(
            "huggingface_hub not installed; cannot fetch GAIA. "
            "Install `huggingface_hub` or pass `metadata_path=` to GaiaAdapter."
        ) from e
    try:
        path = hf_hub_download(
            repo_id="gaia-benchmark/GAIA",
            filename="2023/validation/metadata.jsonl",
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
    except Exception as e:
        raise GaiaLoadError(f"Failed to download GAIA from HuggingFace: {e}") from e
    return Path(path)


class GaiaAdapter:
    """GAIA benchmark adapter.

    Parameters
    ----------
    metadata_path:
        If provided, load problems from this JSONL file directly (used by
        tests with fixtures and by callers who pre-downloaded).
    cache_dir:
        Where to cache HF downloads. Defaults to
        `~/.cache/yaharness/gaia/`.
    """

    name: str = "gaia"

    def __init__(
        self,
        *,
        metadata_path: Path | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self._metadata_path = metadata_path
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR

    def _resolve_metadata_path(self) -> Path:
        if self._metadata_path is not None:
            if not self._metadata_path.exists():
                raise GaiaLoadError(f"GAIA metadata_path does not exist: {self._metadata_path}")
            return self._metadata_path
        cached = self._cache_dir / "metadata.jsonl"
        if cached.exists():
            return cached
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        return _hf_download(self._cache_dir)

    def load_problems(
        self,
        *,
        subset: str | None = None,
        limit: int | None = None,
    ) -> Sequence[Problem]:
        path = self._resolve_metadata_path()
        problems: list[Problem] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                level = row.get("Level") or row.get("level")
                if subset and subset != "all" and str(level) != subset.removeprefix("level_"):
                    continue
                pid = str(row.get("task_id") or row.get("problem_id"))
                question = str(row.get("Question") or row.get("question") or "")
                final = row.get("Final answer") or row.get("final_answer")
                file_name = row.get("file_name") or ""
                context: dict[str, Any] = {}
                if file_name:
                    context["files"] = [file_name]
                problems.append(
                    Problem(
                        problem_id=pid,
                        task_text=question,
                        expected_answer=str(final) if final is not None else None,
                        context=context,
                        metadata={"level": level, "source": "gaia"},
                    )
                )
        # Deterministic ordering by problem_id.
        problems.sort(key=lambda p: p.problem_id)
        if limit is not None:
            problems = problems[:limit]
        return problems

    async def grade(
        self,
        problem: Problem,
        agent_result: AgentSystemResult,
    ) -> ProblemOutcome:
        expected = problem.expected_answer or ""
        got = agent_result.final_answer
        success = gaia_answers_match(expected, got)
        false_positive = agent_result.completed and not success
        notes = (
            "exact match (gaia_normalise)"
            if success
            else f"expected {expected!r}, got {got!r} (gaia_normalise mismatch)"
        )
        return ProblemOutcome(
            problem_id=problem.problem_id,
            success=success,
            completed=agent_result.completed,
            false_positive_completion=false_positive,
            n_steps=agent_result.n_steps,
            cost_usd=agent_result.total_cost_usd,
            grader_notes=notes,
            raw_trace=agent_result.raw_trace,
        )


__all__ = [
    "DEFAULT_CACHE_DIR",
    "GaiaAdapter",
    "GaiaLoadError",
    "gaia_answers_match",
    "gaia_normalise",
]
