"""Task-text framing for SWE-bench problems.

The agent receives `task_text` plus a `context` dict. We bake the
patch-output contract into task_text so the agent knows what shape its
final answer must take.
"""

from __future__ import annotations

from typing import Any

PATCH_CONTRACT = """\
Your final answer MUST be a unified-diff patch (the output of `git diff`) \
that, when applied to the repository at the given base_commit, fixes the issue.

Format requirements:
- Start with `diff --git a/<path> b/<path>` headers.
- Include `--- a/<path>` and `+++ b/<path>` lines.
- Include at least one hunk (`@@ ... @@`) with `+`/`-` lines.
- Do NOT wrap the patch in markdown code fences in the final answer.
- Do NOT include any prose alongside the patch in the final answer.
"""


def build_swebench_task_text(
    *,
    problem_statement: str,
    repo: str,
    base_commit: str,
    hints_text: str = "",
    relevant_files: list[str] | None = None,
) -> str:
    """Compose the task_text fed to the agent for a SWE-bench instance."""
    parts: list[str] = [
        f"You are fixing an issue in the `{repo}` repository at commit `{base_commit}`.",
        "",
        "## Problem statement",
        problem_statement.strip(),
    ]
    if hints_text.strip():
        parts.extend(["", "## Hints", hints_text.strip()])
    if relevant_files:
        parts.extend(
            [
                "",
                "## Likely-relevant files",
                "\n".join(f"- {fp}" for fp in relevant_files),
            ]
        )
    parts.extend(
        [
            "",
            "## How to work",
            "Use the read_file / list_dir / find_files tools (scoped to the repo "
            "checkout) to read source. Reason about the fix. Then produce the patch.",
            "",
            "## Output contract",
            PATCH_CONTRACT,
        ]
    )
    return "\n".join(parts)


def extract_relevant_files(instance: dict[str, Any]) -> list[str]:
    """Heuristically pull file paths out of the test_patch (best signal of what's touched).

    Falls back to empty list if test_patch is missing/malformed.
    """
    test_patch = instance.get("test_patch") or ""
    files: list[str] = []
    for line in test_patch.splitlines():
        if line.startswith("diff --git a/"):
            # "diff --git a/path b/path"
            try:
                a_part = line.split(" a/", 1)[1]
                path = a_part.split(" b/", 1)[0]
                if path and path not in files:
                    files.append(path)
            except IndexError:
                continue
    return files


__all__ = ["PATCH_CONTRACT", "build_swebench_task_text", "extract_relevant_files"]
