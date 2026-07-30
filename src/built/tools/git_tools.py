"""Thin async wrapper over the `git` CLI (not GitPython — one less dependency, and the
CLI's behavior is what we've actually verified empirically for bare-repo worktrees)."""

import asyncio
import re
from pathlib import Path

_COMMIT_AUTHOR_NAME = "built-agent"
_COMMIT_AUTHOR_EMAIL = "agent@built.local"

# Unlike every other tool (bash's MAX_OUTPUT_CHARS, read_file's MAX_FILE_BYTES/pagination,
# grep/glob's MAX_MATCHES), diff output had no cap at all — a branch that touches a
# generated/vendored file or a lockfile can produce a multi-megabyte diff in one tool call,
# which blows straight through the model's context window in a single message before
# agent/context_window.py's compaction ever gets a chance to run. Same order of magnitude as
# bash's cap, for consistency.
MAX_DIFF_CHARS = 20_000


def _truncate_diff(diff_text: str) -> str:
    if len(diff_text) <= MAX_DIFF_CHARS:
        return diff_text
    return (
        diff_text[:MAX_DIFF_CHARS] + f"\n... [diff truncated, {len(diff_text)} chars total — this "
        "change is too large to review in one shot; use bash to diff individual files or "
        "directories instead, e.g. `git diff <ref>...HEAD -- path/to/file`]"
    )


class GitCommandError(Exception):
    # git writes some of its most useful failure detail to stdout, not stderr — e.g.
    # `git merge` conflict summaries ("CONFLICT (content): Merge conflict in ...") are
    # on stdout. Keep both so callers reporting the error don't silently drop it.
    def __init__(self, args: tuple[str, ...], returncode: int, stdout: str, stderr: str):
        self.args_ = args
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        detail = (stdout.strip() + "\n" + stderr.strip()).strip()
        super().__init__(f"git {' '.join(args)} failed ({returncode}): {detail}")


async def _run(*args: str, cwd: Path) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return process.returncode or 0, stdout.decode(), stderr.decode()


async def run_git(*args: str, cwd: Path) -> str:
    returncode, stdout, stderr = await _run(*args, cwd=cwd)
    if returncode != 0:
        raise GitCommandError(args, returncode, stdout, stderr)
    return stdout


async def status(worktree: Path) -> str:
    return await run_git("status", "--short", cwd=worktree)


async def diff(worktree: Path, *, staged: bool = False) -> str:
    args = ["diff", "--staged"] if staged else ["diff"]
    return _truncate_diff(await run_git(*args, cwd=worktree))


async def diff_against_ref(worktree: Path, ref: str) -> str:
    """Diff of everything HEAD has done since it diverged from `ref` (three-dot,
    against the merge-base — not a straight two-dot diff, which would also include
    commits `ref` picked up on its own side since the branch point). What Reviewer
    actually needs: an ordinary `git diff` only shows uncommitted changes, and this
    pipeline auto-commits after every tool call, so there's normally nothing there."""
    return _truncate_diff(await run_git("diff", f"{ref}...HEAD", cwd=worktree))


_CONFLICT_STATUS_CODES = {"UU", "AA", "DD", "AU", "UA", "DU", "UD"}
# Classic git conflict marker lines: '<<<<<<< HEAD', a bare '=======', '>>>>>>> branch'.
_CONFLICT_MARKER_RE = re.compile(r"^<{7} |^={7}$|^>{7} ", re.MULTILINE)


async def merge_in_progress(worktree: Path) -> bool:
    returncode, _, _ = await _run("rev-parse", "-q", "--verify", "MERGE_HEAD", cwd=worktree)
    return returncode == 0


async def conflicted_paths(worktree: Path) -> list[str]:
    """Paths git still considers unmerged (per porcelain status) whose current
    on-disk content still contains literal conflict markers. Git's unmerged status
    alone isn't a reliable "still needs work" signal here: nothing but `git add`
    clears it, and this flow deliberately never auto-stages a file just because some
    other conflicted file got fixed — so the real test of "has this one actually
    been resolved" is whether the markers are gone, not the index state."""
    output = await run_git("status", "--porcelain", cwd=worktree)
    unmerged = [line[3:] for line in output.splitlines() if line[:2] in _CONFLICT_STATUS_CODES]
    still_conflicted = []
    for rel_path in unmerged:
        try:
            content = (worktree / rel_path).read_text(errors="replace")
        except OSError:
            still_conflicted.append(rel_path)
            continue
        if _CONFLICT_MARKER_RE.search(content):
            still_conflicted.append(rel_path)
    return still_conflicted


async def complete_merge(worktree: Path, *, message: str, author_name: str, author_email: str) -> str:
    """Stages everything and commits to finish an in-progress merge once
    conflicted_paths() is empty. Doesn't reuse commit_all's "anything to commit"
    short-circuit — a merge commit is required even when it introduces no content
    diff of its own, since it still needs two parents to actually close out
    MERGE_HEAD."""
    await run_git("add", "-A", cwd=worktree)
    await run_git(
        "-c",
        f"user.name={author_name}",
        "-c",
        f"user.email={author_email}",
        "commit",
        "-m",
        message,
        cwd=worktree,
    )
    sha = await run_git("rev-parse", "HEAD", cwd=worktree)
    return sha.strip()


async def commit_all(worktree: Path, *, message: str) -> str | None:
    """Stage everything and commit under a fixed agent identity. Returns the new
    commit SHA, or None if the tool call didn't actually change anything on disk —
    CardEvent records the SHA, not a full diff, so "nothing to commit" must be
    distinguishable from "committed"."""
    await run_git("add", "-A", cwd=worktree)
    returncode, _, _ = await _run("diff", "--cached", "--quiet", cwd=worktree)
    if returncode == 0:
        return None
    await run_git(
        "-c",
        f"user.name={_COMMIT_AUTHOR_NAME}",
        "-c",
        f"user.email={_COMMIT_AUTHOR_EMAIL}",
        "commit",
        "-m",
        message,
        cwd=worktree,
    )
    sha = await run_git("rev-parse", "HEAD", cwd=worktree)
    return sha.strip()
