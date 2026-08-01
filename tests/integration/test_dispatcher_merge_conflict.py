"""tools/dispatcher.py's auto-commit becomes merge-aware when a card's worktree is
mid-merge — sandbox/worktree.sync_card_branch_with_default leaves one unresolved at
the start of a Developer visit (see orchestrator/worker.py). A partial fix spread
across several tool calls must never bake a still-conflicted file into a commit.
Real git, real filesystem — see tests/integration/test_worktree.py for the lower-level
sync_card_branch_with_default behavior this builds on."""

import subprocess

from built.db.models import Card, Project
from built.sandbox import worktree
from built.sandbox.container import CommandResult
from built.tools import git_tools
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher
from tests.unit.fakes import FakeCommandExecutor


def _project(remote_path, suffix: str) -> Project:
    return Project(id=f"proj-merge-commit-{suffix}", repo_remote_url=str(remote_path), default_branch="main")


def _card(project: Project, suffix: str) -> Card:
    return Card(id=f"card-merge-commit-{suffix}", project_id=project.id, branch_name=f"card/merge-{suffix}")


def _dispatcher(card_id: str, wt_path) -> ToolDispatcher:
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    return ToolDispatcher(ctx=ToolContext(card_id=card_id, worktree_root=wt_path), executor=executor)


async def _make_two_file_conflict(toy_repo_remote, suffix: str):
    """Card branch changes app.py and adds card_only.py; main independently changes
    app.py in a conflicting way and adds other.py — two unrelated new files plus one
    genuine conflict, so a fix to app.py can be observed not committing the merge on
    its own while other already-merged-in content sits alongside it untouched.

    suffix must be unique per test: data_dir is one shared tmp dir for the whole
    test session (tests/conftest.py), keyed by project/card id, so two tests
    reusing the same id would silently share (and fight over) the same bare clone
    and worktree on disk."""
    project = _project(toy_repo_remote, suffix)
    card = _card(project, suffix)
    wt_path = await worktree.create_card_worktree(project, card)

    (wt_path / "app.py").write_text("def greet():\n    return 'from the card'\n")
    (wt_path / "card_only.py").write_text("value = 'card'\n")
    await git_tools.commit_all(wt_path, message="card change")

    (toy_repo_remote / "app.py").write_text("def greet():\n    return 'from main'\n")
    (toy_repo_remote / "other.py").write_text("value = 'main'\n")
    subprocess.run(["git", "add", "-A"], cwd=toy_repo_remote, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "main change"], cwd=toy_repo_remote, check=True, capture_output=True
    )
    await worktree.ensure_managed_clone(project)

    conflicted = await worktree.sync_card_branch_with_default(project, wt_path)
    assert conflicted == ["app.py"]
    return project, card, wt_path


async def test_partial_fix_does_not_commit_while_conflict_remains(toy_repo_remote):
    project, card, wt_path = await _make_two_file_conflict(toy_repo_remote, "partial")
    dispatcher = _dispatcher(card.id, wt_path)

    # A write to an unrelated, non-conflicted file — this alone must not trigger a
    # commit, or it would silently bake app.py's raw conflict markers into it.
    outcome = await dispatcher.dispatch("write_file", {"path": "untouched.py", "content": "x = 1\n"})

    assert not outcome.result.is_error
    assert outcome.commit_sha is None
    assert await git_tools.merge_in_progress(wt_path)
    assert await git_tools.conflicted_paths(wt_path) == ["app.py"]


async def test_resolving_the_last_conflict_completes_the_merge_commit(toy_repo_remote):
    project, card, wt_path = await _make_two_file_conflict(toy_repo_remote, "resolve")
    dispatcher = _dispatcher(card.id, wt_path)

    resolved_content = "def greet():\n    return 'from the card and main'\n"
    outcome = await dispatcher.dispatch("write_file", {"path": "app.py", "content": resolved_content})

    assert not outcome.result.is_error
    assert outcome.commit_sha is not None
    assert not await git_tools.merge_in_progress(wt_path)
    assert await git_tools.status(wt_path) == ""
    assert (wt_path / "app.py").read_text() == resolved_content
    # Content the merge already brought in cleanly survived too — not just the one
    # path that happened to get the final fix.
    assert (wt_path / "card_only.py").read_text() == "value = 'card'\n"
    assert (wt_path / "other.py").read_text() == "value = 'main'\n"


async def test_normal_auto_commit_resumes_once_the_merge_is_settled(toy_repo_remote):
    project, card, wt_path = await _make_two_file_conflict(toy_repo_remote, "resume")
    dispatcher = _dispatcher(card.id, wt_path)
    await dispatcher.dispatch(
        "write_file", {"path": "app.py", "content": "def greet():\n    return 'merged'\n"}
    )

    outcome = await dispatcher.dispatch("write_file", {"path": "later.py", "content": "y = 2\n"})

    assert outcome.commit_sha is not None
    assert not await git_tools.merge_in_progress(wt_path)
