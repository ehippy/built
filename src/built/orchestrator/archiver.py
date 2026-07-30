"""Runs a deterministic sweep on a timer — no LLM involved, unlike Reviver/Curator.
Archives DONE cards that have sat idle past settings.auto_archive_done_after_days,
so the board doesn't accumulate finished work forever without a human remembering
to archive it by hand. Mirrors orchestrator/reviver.py's loop shape.

Also cleans up on-disk git worktrees for any card that's archived or DONE — a card
in either state won't be worked on again, so its worktree (a full checkout, possibly
with a node_modules or similar dependency tree inside it) is pure disk waste from
that point on. This is a separate sweep from the archiving above rather than being
called at each individual place archived_at or DONE gets set (a human archiving a
card, split_into_subtasks retiring an oversized one, the archive sweep itself,
Deployer reaching a terminal success) — a periodic query for "archived-or-done and
still has a worktree" covers every current and future such place for free, instead
of every one of them having to remember to trigger cleanup itself."""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import or_, select

from built.config import settings
from built.db.base import async_session_factory
from built.db.models import Card, Project
from built.domain.enums import LifecycleState
from built.sandbox import worktree
from built.services import card_service

logger = logging.getLogger(__name__)


async def run_archiver_once() -> int:
    """One pass, in its own session — archives every DONE, non-archived card whose
    updated_at (the moment it went DONE; nothing touches a card afterward) is older
    than the configured threshold. Returns how many were archived."""
    cutoff = datetime.now(UTC) - timedelta(days=settings.auto_archive_done_after_days)
    async with async_session_factory() as session:
        stmt = select(Card.id).where(
            Card.lifecycle_state == LifecycleState.DONE,
            Card.archived_at.is_(None),
            Card.updated_at < cutoff,
        )
        stale_ids = list((await session.scalars(stmt)).all())
        for card_id in stale_ids:
            await card_service.archive_card(session, card_id)
        if stale_ids:
            await session.commit()
            logger.info(
                "archiver: archived %d done card(s) idle > %dd",
                len(stale_ids),
                settings.auto_archive_done_after_days,
            )
        return len(stale_ids)


async def cleanup_stale_worktrees_once() -> int:
    """One pass, in its own session — removes the on-disk worktree for every card
    that's archived or DONE and still has one recorded. Returns how many were
    cleaned up. A worktree that's already gone from disk (e.g. removed by hand) is
    a no-op, not an error — see remove_card_worktree."""
    async with async_session_factory() as session:
        stmt = select(Card).where(
            Card.worktree_path.is_not(None),
            or_(Card.archived_at.is_not(None), Card.lifecycle_state == LifecycleState.DONE),
        )
        cards = list((await session.scalars(stmt)).all())
        cleaned = 0
        for card in cards:
            project = await session.get(Project, card.project_id)
            if project is None:
                continue
            try:
                await worktree.remove_card_worktree(project, card)
            except Exception:
                # One card's worktree being unremovable (e.g. a stray lock file)
                # shouldn't stop the rest of the sweep from cleaning up.
                logger.exception("worktree cleanup failed for card %s", card.id)
                continue
            card.worktree_path = None
            cleaned += 1
        if cleaned:
            await session.commit()
            logger.info("archiver: cleaned up %d stale worktree(s)", cleaned)
        return cleaned


async def run_archiver_loop(*, stop_event: asyncio.Event, poll_interval: float) -> None:
    while not stop_event.is_set():
        try:
            await run_archiver_once()
            await cleanup_stale_worktrees_once()
        except Exception:
            # Crash-isolation: one bad pass shouldn't kill the archiver forever — log
            # and try again at the next scheduled wake.
            logger.exception("archiver: unhandled error during pass")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
        except TimeoutError:
            pass
