"""Runs a deterministic sweep on a timer — no LLM involved, unlike Reviver/Curator.
Archives DONE cards that have sat idle past settings.auto_archive_done_after_days,
so the board doesn't accumulate finished work forever without a human remembering
to archive it by hand. Mirrors orchestrator/reviver.py's loop shape."""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from built.config import settings
from built.db.base import async_session_factory
from built.db.models import Card
from built.domain.enums import LifecycleState
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


async def run_archiver_loop(*, stop_event: asyncio.Event, poll_interval: float) -> None:
    while not stop_event.is_set():
        try:
            await run_archiver_once()
        except Exception:
            # Crash-isolation: one bad pass shouldn't kill the archiver forever — log
            # and try again at the next scheduled wake.
            logger.exception("archiver: unhandled error during pass")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
        except TimeoutError:
            pass
