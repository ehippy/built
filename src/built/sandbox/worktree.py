"""One bare, service-owned clone per project; one git worktree per card, created once
and reused for the card's whole lifetime across every column it visits. Bare-repo
branches live directly under refs/heads/* (verified empirically — `git clone --bare`
does not use refs/remotes/origin/*), so worktrees are created off the plain branch
name, not `origin/<branch>`."""

from pathlib import Path

from built.config import settings
from built.db.models import Card, Project
from built.tools.git_tools import run_git


def bare_repo_path(project: Project) -> Path:
    return settings.data_dir / "repos" / f"{project.id}.git"


def worktree_path(card: Card) -> Path:
    return settings.data_dir / "worktrees" / card.id


async def ensure_managed_clone(project: Project) -> Path:
    """Clone the project's remote into the service-managed bare repo if it doesn't
    exist yet; otherwise fetch the latest. Idempotent — safe to call before every
    column visit."""
    bare_path = bare_repo_path(project)
    if bare_path.exists():
        await run_git("fetch", "origin", cwd=bare_path)
        return bare_path
    bare_path.parent.mkdir(parents=True, exist_ok=True)
    await run_git("clone", "--bare", project.repo_remote_url, str(bare_path), cwd=bare_path.parent)
    return bare_path


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
