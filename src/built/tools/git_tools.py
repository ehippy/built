"""Thin async wrapper over the `git` CLI (not GitPython — one less dependency, and the
CLI's behavior is what we've actually verified empirically for bare-repo worktrees)."""

import asyncio
from pathlib import Path

_COMMIT_AUTHOR_NAME = "built-agent"
_COMMIT_AUTHOR_EMAIL = "agent@built.local"


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
    return await run_git(*args, cwd=worktree)


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
