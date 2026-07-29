"""Runs the Reviver (agent/reviver.py) on a timer — wakes on its own, no manual
trigger. Mirrors orchestrator/worker.py's _worker_loop shape (sleep, do one unit of
work, repeat) but the "unit of work" here is a pass over every project's stuck cards
at once, not a single card claim, so it doesn't need claim/lease machinery."""

import asyncio
import logging

from built.agent.reviver import run_reviver_pass
from built.config import settings
from built.db.base import async_session_factory
from built.domain.enums import Column
from built.llm.client import FallbackLLMClient
from built.services import endpoint_service

logger = logging.getLogger(__name__)


async def run_reviver_once() -> dict:
    """One pass, in its own session — meant to be called by the timer loop below, but
    kept separate so it can also be driven directly (tests, a future manual trigger)."""
    async with async_session_factory() as session:
        try:
            chain = await endpoint_service.get_resolved_chain(session, project_id=None, role=Column.PM)
            llm_client = FallbackLLMClient(chain)
        except Exception:
            logger.exception("reviver: no usable global endpoint configured")
            return {"revived": 0, "left_blocked": 0, "errors": 0}

        counts = await run_reviver_pass(
            session, llm_client=llm_client, max_iterations=settings.reviver_max_iterations
        )
        logger.info(
            "reviver pass: revived=%d left_blocked=%d errors=%d",
            counts["revived"],
            counts["left_blocked"],
            counts["errors"],
        )
        return counts


async def run_reviver_loop(*, stop_event: asyncio.Event, poll_interval: float) -> None:
    """Runs a pass immediately on startup (no reason to make already-stuck cards wait
    a full interval after a restart), then on every poll_interval after that."""
    while not stop_event.is_set():
        try:
            await run_reviver_once()
        except Exception:
            # Crash-isolation: one bad pass shouldn't kill the reviver forever — log
            # and try again at the next scheduled wake.
            logger.exception("reviver: unhandled error during pass")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
        except TimeoutError:
            pass
