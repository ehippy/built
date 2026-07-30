"""orchestrator/archiver.py — a deterministic (no LLM) sweep that archives DONE
cards once they've sat idle past the configured threshold, so the board doesn't
accumulate finished work forever without a human archiving it by hand. Also cleans
up on-disk git worktrees for cards that are archived or DONE."""

from datetime import UTC, datetime, timedelta

from built.config import settings
from built.domain import transitions
from built.orchestrator.archiver import cleanup_stale_worktrees_once, run_archiver_once
from built.sandbox import worktree
from built.services import card_service, project_service


async def _make_done_card(session, *, updated_at, title="done card"):
    project = await project_service.create_project(
        session,
        name=f"archiver-{title}-{id(session)}",
        overarching_goal="goal",
        repo_remote_url="https://example.invalid/repo.git",
    )
    card = await card_service.create_card(session, project.id, title=title, raw_request="r")
    visit = await transitions.start_visit(session, card)
    await transitions.complete_deployer_visit(session, card, visit, success=True, summary="Deployed")
    card.updated_at = updated_at
    await session.commit()
    return card


async def test_archives_a_done_card_past_the_idle_threshold(db_session, monkeypatch):
    monkeypatch.setattr(settings, "auto_archive_done_after_days", 7)
    stale = datetime.now(UTC) - timedelta(days=8)
    card = await _make_done_card(db_session, updated_at=stale)

    archived_count = await run_archiver_once()

    assert archived_count == 1
    await db_session.refresh(card)
    assert card.archived_at is not None


async def test_leaves_a_recently_done_card_alone(db_session, monkeypatch):
    monkeypatch.setattr(settings, "auto_archive_done_after_days", 7)
    recent = datetime.now(UTC) - timedelta(days=1)
    card = await _make_done_card(db_session, updated_at=recent)

    archived_count = await run_archiver_once()

    assert archived_count == 0
    await db_session.refresh(card)
    assert card.archived_at is None


async def test_leaves_active_cards_alone_regardless_of_age(db_session, monkeypatch):
    monkeypatch.setattr(settings, "auto_archive_done_after_days", 7)
    project = await project_service.create_project(
        db_session,
        name="archiver-active",
        overarching_goal="goal",
        repo_remote_url="https://example.invalid/repo.git",
    )
    card = await card_service.create_card(db_session, project.id, title="still active", raw_request="r")
    card.updated_at = datetime.now(UTC) - timedelta(days=30)
    await db_session.commit()

    archived_count = await run_archiver_once()

    assert archived_count == 0
    await db_session.refresh(card)
    assert card.archived_at is None


async def test_does_not_re_archive_an_already_archived_done_card(db_session, monkeypatch):
    monkeypatch.setattr(settings, "auto_archive_done_after_days", 7)
    stale = datetime.now(UTC) - timedelta(days=8)
    card = await _make_done_card(db_session, updated_at=stale)
    await card_service.archive_card(db_session, card.id)
    await db_session.commit()

    archived_count = await run_archiver_once()

    assert archived_count == 0


async def _make_card_with_worktree(session, toy_repo_remote, *, title):
    project = await project_service.create_project(
        session,
        name=f"archiver-wt-{title}",
        overarching_goal="goal",
        repo_remote_url=str(toy_repo_remote),
    )
    card = await card_service.create_card(session, project.id, title=title, raw_request="r")
    wt_path = await worktree.create_card_worktree(project, card)
    card.worktree_path = str(wt_path)
    await session.commit()
    return project, card, wt_path


async def test_cleanup_removes_worktree_for_an_archived_card(db_session, toy_repo_remote):
    project, card, wt_path = await _make_card_with_worktree(db_session, toy_repo_remote, title="archived")
    await card_service.archive_card(db_session, card.id)
    await db_session.commit()

    cleaned = await cleanup_stale_worktrees_once()

    assert cleaned == 1
    assert not wt_path.exists()
    await db_session.refresh(card)
    assert card.worktree_path is None


async def test_cleanup_removes_worktree_for_a_done_card(db_session, toy_repo_remote):
    project, card, wt_path = await _make_card_with_worktree(db_session, toy_repo_remote, title="done")
    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_deployer_visit(db_session, card, visit, success=True, summary="Deployed")
    await db_session.commit()

    cleaned = await cleanup_stale_worktrees_once()

    assert cleaned == 1
    assert not wt_path.exists()
    await db_session.refresh(card)
    assert card.worktree_path is None


async def test_cleanup_leaves_an_active_cards_worktree_alone(db_session, toy_repo_remote):
    project, card, wt_path = await _make_card_with_worktree(db_session, toy_repo_remote, title="active")

    cleaned = await cleanup_stale_worktrees_once()

    assert cleaned == 0
    assert wt_path.exists()
    await db_session.refresh(card)
    assert card.worktree_path == str(wt_path)


async def test_cleanup_ignores_archived_cards_with_no_recorded_worktree(db_session):
    project = await project_service.create_project(
        db_session,
        name="archiver-wt-none",
        overarching_goal="goal",
        repo_remote_url="https://example.invalid/repo.git",
    )
    card = await card_service.create_card(db_session, project.id, title="never worked", raw_request="r")
    await card_service.archive_card(db_session, card.id)
    await db_session.commit()

    cleaned = await cleanup_stale_worktrees_once()

    assert cleaned == 0
