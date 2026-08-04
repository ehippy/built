"""Regression tests for git_tools.py diff truncation.

Confirmed in production: `review_diff` (Reviewer's `diff_against_ref`) had no output
size cap at all — unlike every other tool (bash's MAX_OUTPUT_CHARS, read_file's
MAX_FILE_BYTES/pagination, grep/glob's MAX_MATCHES) — so a branch touching one large
file produced a single multi-megabyte tool result that blew straight through the
model's context window (an 8.4M-token request against a 248K-token budget) before
agent/context_window.py's compaction ever got a chance to run.
"""

import base64
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


async def test_diff_shortstat_against_ref_counts_insertions_and_deletions(toy_repo_remote):
    _run(toy_repo_remote, "branch", "base")
    new_content = "def greet():\n    return 'hi there'\n\n\ndef farewell():\n    pass\n"
    (toy_repo_remote / "app.py").write_text(new_content)
    _run(toy_repo_remote, "add", "-A")
    _run(toy_repo_remote, "commit", "-q", "-m", "tweak greet, add farewell")

    insertions, deletions = await git_tools.diff_shortstat_against_ref(toy_repo_remote, "base")

    assert insertions == 5
    assert deletions == 1


async def test_diff_shortstat_against_ref_is_zero_when_nothing_changed(toy_repo_remote):
    _run(toy_repo_remote, "branch", "base")

    insertions, deletions = await git_tools.diff_shortstat_against_ref(toy_repo_remote, "base")

    assert (insertions, deletions) == (0, 0)


def test_token_auth_config_scopes_the_token_and_clears_credential_helpers():
    """The deploy push fix ("authenticate deploy git push with the configured
    GitHub token"): the token travels as a Basic-auth header via
    http.https://github.com/.extraheader — scoped to github.com only, passed as
    -c config for that one invocation (never written to any on-disk gitconfig),
    and base64-encoded so the literal token never appears in argv where ps could
    see it. credential.helper= is cleared so an ambient helper (e.g. gh's, holding
    a differently-scoped token) can't silently win over the explicit header."""
    args = git_tools._token_auth_config("sekrit-token")

    expected_basic = base64.b64encode(b"x-access-token:sekrit-token").decode()
    assert args == (
        "-c",
        "credential.helper=",
        "-c",
        f"http.https://github.com/.extraheader=AUTHORIZATION: basic {expected_basic}",
    )
    assert "sekrit-token" not in " ".join(args)


async def test_run_git_token_scoped_push_completes(tmp_path):
    """A token-scoped push: run_git(..., token=...) must accept the token and push
    successfully. The fake github.com URL is rewritten (url.<base>.insteadOf) to a
    local receiver so no network is touched; the auth -c args ride along harmlessly.
    Guards against the auth flags ever breaking the push command they're meant to
    secure."""
    receiver = tmp_path / "receiver.git"
    subprocess.run(["git", "init", "-q", "-b", "main", "--bare", str(receiver)], check=True)
    work = tmp_path / "work"
    work.mkdir()
    for args in (
        ["init", "-q", "-b", "main"],
        ["config", "user.email", "test@example.com"],
        ["config", "user.name", "Test"],
        ["remote", "add", "origin", "https://github.com/fake-owner/fake-repo.git"],
        ["config", f"url.{receiver}.insteadOf", "https://github.com/fake-owner/fake-repo.git"],
    ):
        subprocess.run(["git", *args], cwd=work, check=True, capture_output=True)
    (work / "app.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "c1"], cwd=work, check=True, capture_output=True)

    await git_tools.run_git("push", "origin", "main:main", cwd=work, token="sekrit-token")

    log = subprocess.run(
        ["git", "log", "--oneline", "main"],
        cwd=receiver,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "c1" in log
