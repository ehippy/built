"""One bare, service-owned clone per project; one git worktree per card, created once
and reused for the card's whole lifetime across every column it visits. Bare-repo
branches live directly under refs/heads/* (verified empirically — `git clone --bare`
does not use refs/remotes/origin/*), so worktrees are created off the plain branch
name, not `origin/<branch>`."""

from pathlib import Path

from built.config import settings
from built.db.models import Card, Project
from built.tools.git_tools import GitCommandError, run_git


def bare_repo_path(project: Project) -> Path:
    return settings.data_dir / "repos" / f"{project.id}.git"


def worktree_path(card: Card) -> Path:
    return settings.data_dir / "worktrees" / card.id


async def ensure_managed_clone(project: Project) -> Path:
    """Clone the project's remote into the service-managed bare repo if it doesn't
    exist yet; otherwise fetch the latest. Idempotent — safe to call before every
    column visit.

    The explicit refspec on fetch matters: `git clone --bare` sets up no fetch
    refspec at all (verified empirically), so a plain `git fetch origin` only updates
    FETCH_HEAD — refs/heads/* silently never advances. That stayed hidden as long as
    the only thing moving refs/heads/<default_branch> was a merge commit made while
    that branch was directly checked out (a side effect of the merge itself, not of
    fetch) — it broke the moment default_branch needed to be refreshed without a
    merge happening on it directly (e.g. resetting a tool's own dedicated branch to
    the current upstream tip)."""
    bare_path = bare_repo_path(project)
    if bare_path.exists():
        await run_git("fetch", "origin", "+refs/heads/*:refs/heads/*", cwd=bare_path)
        return bare_path
    bare_path.parent.mkdir(parents=True, exist_ok=True)
    await run_git("clone", "--bare", project.repo_remote_url, str(bare_path), cwd=bare_path.parent)
    return bare_path


def tool_worktree_path(project: Project, *, tool: str) -> Path:
    return settings.data_dir / "worktrees" / f"_tool_{tool}" / project.id


def _tool_branch_name(tool: str) -> str:
    return f"_tool/{tool}"


async def ensure_tool_worktree(project: Project, *, tool: str) -> Path:
    """A worktree on its own dedicated branch, reset to the tip of default_branch
    before every use — not the literal default branch itself. Git refuses to check
    out the same branch in two worktrees at once (a hard error, not a race), so
    Deployer's auto-main merges, PM discovery, and the Tender's AGENTS.md edits each
    get their own independent worktree + branch rather than fighting over one.
    Pushing `HEAD:{default_branch}` from here still lands changes on the real
    default branch, regardless of what this local branch is actually named."""
    bare_path = await ensure_managed_clone(project)
    wt_path = tool_worktree_path(project, tool=tool)
    branch = _tool_branch_name(tool)
    if not wt_path.exists():
        wt_path.parent.mkdir(parents=True, exist_ok=True)
        await run_git("worktree", "add", "-b", branch, str(wt_path), project.default_branch, cwd=bare_path)
    else:
        # Bare-repo branches live under refs/heads/*, not refs/remotes/origin/* (see
        # module docstring) — ensure_managed_clone's fetch above already updated
        # refs/heads/<default_branch> directly, so reset targets the plain branch
        # name, not origin/<branch>.
        await run_git("checkout", branch, cwd=wt_path)
        await run_git("reset", "--hard", project.default_branch, cwd=wt_path)
    return wt_path


async def read_default_branch_file(project: Project, path: str) -> str | None:
    """Reads a file straight from the tip of default_branch in the bare repo (`git
    show`) — no worktree checkout needed, so unlike ensure_tool_worktree this is safe
    to call from anywhere at any time with no risk of two concurrent callers stepping
    on the same working directory. Returns None if the file (or the branch) doesn't
    exist yet."""
    bare_path = await ensure_managed_clone(project)
    try:
        return await run_git("show", f"{project.default_branch}:{path}", cwd=bare_path)
    except GitCommandError:
        return None


async def create_card_worktree(project: Project, card: Card) -> Path:
    """Create (or reuse) the card's dedicated worktree + branch, off the project's
    default branch. A card's worktree persists across revision loops — Developer's
    second attempt after `request_changes` picks up right where the first left off."""
    bare_path = await ensure_managed_clone(project)
    wt_path = worktree_path(card)
    if wt_path.exists():
        return wt_path
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    assert card.branch_name is not None, "card.branch_name must be set at creation time"
    await run_git(
        "worktree", "add", "-b", card.branch_name, str(wt_path), project.default_branch, cwd=bare_path
    )
    return wt_path


async def remove_card_worktree(project: Project, card: Card) -> None:
    """Not wired to any endpoint yet — used for cleanup once a card reaches a
    terminal state (Phase 4+)."""
    bare_path = bare_repo_path(project)
    wt_path = worktree_path(card)
    if not wt_path.exists():
        return
    await run_git("worktree", "remove", "--force", str(wt_path), cwd=bare_path)
