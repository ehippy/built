"""Regression tests for git_tools.py diff truncation.

Confirmed in production: `review_diff` (Reviewer's `diff_against_ref`) had no output
size cap at all — unlike every other tool (bash's MAX_OUTPUT_CHARS, read_file's
MAX_FILE_BYTES/pagination, grep/glob's MAX_MATCHES) — so a branch touching one large
file produced a single multi-megabyte tool result that blew straight through the
model's context window (an 8.4M-token request against a 248K-token budget) before
agent/context_window.py's compaction ever got a chance to run.
"""

import subprocess

from built.tools import git_tools


def _run(repo_dir, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True)


async def test_diff_against_ref_truncates_oversized_diff(toy_repo_remote):
    _run(toy_repo_remote, "branch", "base")
    (toy_repo_remote / "big.txt").write_text("x" * 500_000)
    _run(toy_repo_remote, "add", "-A")
    _run(toy_repo_remote, "commit", "-q", "-m", "add a huge file")

    result = await git_tools.diff_against_ref(toy_repo_remote, "base")

    assert len(result) <= git_tools.MAX_DIFF_CHARS + 300
    assert "truncated" in result


async def test_diff_against_ref_leaves_small_diffs_untouched(toy_repo_remote):
    _run(toy_repo_remote, "branch", "base")
    (toy_repo_remote / "app.py").write_text("def greet():\n    return 'hello'\n")
    _run(toy_repo_remote, "add", "-A")
    _run(toy_repo_remote, "commit", "-q", "-m", "tweak greet()")

    result = await git_tools.diff_against_ref(toy_repo_remote, "base")

    assert "truncated" not in result
    assert "hello" in result


async def test_diff_truncates_oversized_uncommitted_diff(toy_repo_remote):
    # `git diff` (unlike diff_against_ref) only shows tracked-file changes, not new
    # untracked files — so the oversized change has to land in a file already
    # committed by the toy_repo_remote fixture, not a brand new one.
    (toy_repo_remote / "app.py").write_text("x" * 500_000)

    result = await git_tools.diff(toy_repo_remote)

    assert len(result) <= git_tools.MAX_DIFF_CHARS + 300
    assert "truncated" in result
