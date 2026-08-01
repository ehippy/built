"""get_card_diff_stat: the +insertions/-deletions readout shown next to a card's
branch name on the card detail page and slideout."""

from built.sandbox import worktree
from built.services import card_service, project_service
from built.tools import git_tools


async def _make_card_with_worktree(db_session, toy_repo_remote, **project_overrides):
    defaults = {
        "name": "diff-stat",
        "overarching_goal": "Ship a thing.",
        "repo_remote_url": str(toy_repo_remote),
    }
    defaults.update(project_overrides)
    project = await project_service.create_project(db_session, **defaults)
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await db_session.flush()
    wt_path = await worktree.create_card_worktree(project, card)
    card.worktree_path = str(wt_path)
    await db_session.flush()
    return project, card, wt_path


async def test_no_diff_stat_without_a_worktree(db_session):
    project = await project_service.create_project(
        db_session, name="no-worktree", overarching_goal="x", repo_remote_url="https://example.invalid/r.git"
    )
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")

    assert await card_service.get_card_diff_stat(db_session, card.id) is None


async def test_diff_stat_is_zero_before_any_commits(db_session, toy_repo_remote):
    _, card, _ = await _make_card_with_worktree(db_session, toy_repo_remote)

    assert await card_service.get_card_diff_stat(db_session, card.id) == (0, 0)


async def test_diff_stat_reflects_committed_changes(db_session, toy_repo_remote):
    _, card, wt_path = await _make_card_with_worktree(db_session, toy_repo_remote)
    (wt_path / "app.py").write_text("def greet():\n    return 'hi there'\n\n\ndef farewell():\n    pass\n")
    await git_tools.commit_all(wt_path, message="tweak greet, add farewell")

    assert await card_service.get_card_diff_stat(db_session, card.id) == (5, 1)


async def test_diff_stat_is_none_once_the_worktree_is_gone(db_session, toy_repo_remote):
    """Mirrors orchestrator/archiver.py clearing worktree_path on archive — but even if a
    caller passed a stale path through some other route, a missing directory shouldn't
    surface as a 500."""
    project = await project_service.create_project(
        db_session, name="stale-worktree", overarching_goal="x", repo_remote_url=str(toy_repo_remote)
    )
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    card.worktree_path = "/nonexistent/path/for/this/test"
    await db_session.flush()

    assert await card_service.get_card_diff_stat(db_session, card.id) is None
