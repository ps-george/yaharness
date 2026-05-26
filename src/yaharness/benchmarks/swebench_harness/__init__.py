"""SWE-bench Tier-1 harness glue: repo checkout and prompt framing.

Tier 1 is patch-producing only. Docker-backed grading is Tier 2 and is a
separate CLI (`lma-grade-swebench`) not in this module.
"""

from .prompts import build_swebench_task_text
from .runner import RepoCheckout, RepoCheckoutError
from .tools import (
    SWEBENCH_SHELL_ALLOWLIST,
    make_swebench_tool_registry,
    maybe_build_swebench_registry_from_context,
)

__all__ = [
    "SWEBENCH_SHELL_ALLOWLIST",
    "RepoCheckout",
    "RepoCheckoutError",
    "build_swebench_task_text",
    "make_swebench_tool_registry",
    "maybe_build_swebench_registry_from_context",
]
