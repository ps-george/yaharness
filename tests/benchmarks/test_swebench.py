"""SWE-bench Verified adapter (Tier 1) tests — fully offline."""

from __future__ import annotations

from pathlib import Path

import pytest

from yaharness.benchmarks import (
    AgentSystemResult,
    SWEBenchLoadError,
    SWEBenchVerifiedAdapter,
    is_valid_unified_diff,
)
from yaharness.benchmarks.swebench_harness import (
    RepoCheckout,
    build_swebench_task_text,
)
from yaharness.benchmarks.swebench_harness.prompts import extract_relevant_files
from yaharness.tools.filesystem import ReadFileTool

FIXTURE = Path(__file__).parent.parent / "fixtures" / "swebench_mini.jsonl"


def test_load_problems_offline() -> None:
    adapter = SWEBenchVerifiedAdapter(metadata_path=FIXTURE)
    problems = adapter.load_problems()
    assert len(problems) == 3
    assert [p.problem_id for p in problems] == [
        "django__django-12345",
        "scikit-learn__scikit-learn-34567",
        "sympy__sympy-23456",
    ]


def test_problem_context_shape() -> None:
    adapter = SWEBenchVerifiedAdapter(metadata_path=FIXTURE)
    p = adapter.load_problems(limit=1)[0]
    assert p.context["repo"] == "django/django"
    assert p.context["base_commit"].startswith("abc1234")
    assert "problem_statement" in p.context
    assert "relevant_files" in p.context
    assert isinstance(p.context["relevant_files"], list)
    # task_text should bake in the patch contract.
    assert "unified-diff" in p.task_text
    assert "diff --git" in p.task_text
    # Metadata holds fail/pass and golden patch (not exposed via context).
    assert p.metadata["fail_to_pass"]
    assert p.metadata["pass_to_pass"]
    assert p.metadata["golden_patch"]
    assert "golden_patch" not in p.context  # leakage guard


def test_load_problems_subset_filter_by_repo() -> None:
    adapter = SWEBenchVerifiedAdapter(metadata_path=FIXTURE)
    django_only = adapter.load_problems(subset="django")
    assert {p.problem_id for p in django_only} == {"django__django-12345"}
    sympy_only = adapter.load_problems(subset="sympy")
    assert {p.problem_id for p in sympy_only} == {"sympy__sympy-23456"}


def test_load_problems_limit() -> None:
    adapter = SWEBenchVerifiedAdapter(metadata_path=FIXTURE)
    problems = adapter.load_problems(limit=2)
    assert len(problems) == 2


def test_missing_metadata_path_raises() -> None:
    adapter = SWEBenchVerifiedAdapter(metadata_path=Path("/tmp/does-not-exist-swe.jsonl"))
    with pytest.raises(SWEBenchLoadError):
        adapter.load_problems()


def test_no_cache_no_hf_raises_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    adapter = SWEBenchVerifiedAdapter(cache_dir=tmp_path / "swebench")
    with pytest.raises(SWEBenchLoadError):
        adapter.load_problems()


VALID_DIFF = """diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,3 @@
 line1
-old
+new
 line3
"""


def test_is_valid_unified_diff_positive() -> None:
    ok, _ = is_valid_unified_diff(VALID_DIFF)
    assert ok is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "this is just prose, no patch",
        "diff --git a/x b/x\nbut no hunk header",
        "--- a/x\n+++ b/x\n@@ ... @@\n",  # no diff --git header
    ],
)
def test_is_valid_unified_diff_negative(text: str) -> None:
    ok, _ = is_valid_unified_diff(text)
    assert ok is False


async def test_grade_valid_patch() -> None:
    adapter = SWEBenchVerifiedAdapter(metadata_path=FIXTURE)
    problem = adapter.load_problems(limit=1)[0]
    result = AgentSystemResult(
        final_answer=VALID_DIFF, completed=True, n_steps=12, total_cost_usd=0.05
    )
    outcome = await adapter.grade(problem, result)
    assert outcome.success is True
    assert outcome.false_positive_completion is False
    assert "Tier-2" in outcome.grader_notes


async def test_grade_non_patch_output() -> None:
    adapter = SWEBenchVerifiedAdapter(metadata_path=FIXTURE)
    problem = adapter.load_problems(limit=1)[0]
    result = AgentSystemResult(
        final_answer="I think you should fix the bug in query.py",
        completed=True,
        n_steps=4,
        total_cost_usd=0.01,
    )
    outcome = await adapter.grade(problem, result)
    assert outcome.success is False
    assert outcome.false_positive_completion is True
    assert "missing" in outcome.grader_notes.lower()


async def test_grade_empty_answer_not_false_positive_if_not_completed() -> None:
    adapter = SWEBenchVerifiedAdapter(metadata_path=FIXTURE)
    problem = adapter.load_problems(limit=1)[0]
    result = AgentSystemResult(final_answer="", completed=False, n_steps=0, total_cost_usd=0.0)
    outcome = await adapter.grade(problem, result)
    assert outcome.success is False
    assert outcome.false_positive_completion is False


def test_extract_relevant_files_from_test_patch() -> None:
    instance = {
        "test_patch": (
            "diff --git a/foo/bar.py b/foo/bar.py\n"
            "--- a/foo/bar.py\n"
            "+++ b/foo/bar.py\n"
            "@@\n+x\n"
            "diff --git a/tests/test_x.py b/tests/test_x.py\n"
            "--- a/tests/test_x.py\n"
            "+++ b/tests/test_x.py\n"
        )
    }
    files = extract_relevant_files(instance)
    assert files == ["foo/bar.py", "tests/test_x.py"]


def test_extract_relevant_files_empty() -> None:
    assert extract_relevant_files({}) == []
    assert extract_relevant_files({"test_patch": ""}) == []


def test_build_swebench_task_text_includes_all_parts() -> None:
    text = build_swebench_task_text(
        problem_statement="Bug: foo",
        repo="django/django",
        base_commit="abc123",
        hints_text="Look at bar",
        relevant_files=["src/foo.py"],
    )
    assert "django/django" in text
    assert "abc123" in text
    assert "Bug: foo" in text
    assert "Look at bar" in text
    assert "src/foo.py" in text
    assert "unified-diff" in text


def test_repo_checkout_rejects_bad_repo(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RepoCheckout(repo="not-a-slash", base_commit="abc", cache_dir=tmp_path)


def test_repo_checkout_path_layout(tmp_path: Path) -> None:
    """Verify the path layout for a checkout — the agent depends on this
    being a stable Path it can hand to FilesystemTool."""
    co = RepoCheckout(repo="django/django", base_commit="abc123", cache_dir=tmp_path)
    assert co.path == tmp_path / "repos" / "django__django"
    assert co.remote_url == "https://github.com/django/django.git"


async def test_filesystem_tool_scopes_to_checkout_path(tmp_path: Path) -> None:
    """Self-soil: simulate a materialised checkout and confirm a FilesystemTool
    scoped to it can read files at the right path — this is what the agent
    uses to inspect the repo during reasoning."""
    fake_checkout = tmp_path / "repos" / "django__django"
    (fake_checkout / "django" / "db" / "models").mkdir(parents=True)
    target = fake_checkout / "django" / "db" / "models" / "query.py"
    target.write_text("class QuerySet:\n    pass\n", encoding="utf-8")
    tool = ReadFileTool(scope_dir=fake_checkout)
    result = await tool.execute(path="django/db/models/query.py")
    assert result.ok is True
    assert "QuerySet" in result.output
    # Escape attempt blocked.
    bad = await tool.execute(path="../../../etc/passwd")
    assert bad.ok is False


async def test_end_to_end_trace_offline(tmp_path: Path) -> None:
    """Self-soil end-to-end trace: load → task_text → simulated agent → grade.

    Verifies the full Tier-1 flow works offline with no docker / no network.
    """
    adapter = SWEBenchVerifiedAdapter(metadata_path=FIXTURE)
    problems = adapter.load_problems(limit=1)
    p = problems[0]
    # Simulate the agent handing the agent a scoped filesystem tool
    # pointing at a "materialised" checkout for this problem.
    fake_checkout = tmp_path / "repos" / p.context["repo"].replace("/", "__")
    fake_checkout.mkdir(parents=True)
    (fake_checkout / "README.md").write_text("# repo", encoding="utf-8")
    tool = ReadFileTool(scope_dir=fake_checkout)
    readme = await tool.execute(path="README.md")
    assert readme.ok is True
    # Agent emits a (fake) patch as final_answer.
    result = AgentSystemResult(
        final_answer=VALID_DIFF, completed=True, n_steps=8, total_cost_usd=0.04
    )
    outcome = await adapter.grade(p, result)
    assert outcome.success is True
    assert outcome.problem_id == p.problem_id
