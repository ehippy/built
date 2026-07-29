"""Runs a deterministic sweep on a timer — no LLM involved, same shape as
orchestrator/archiver.py. Polls GitHub's Checks API for every card whose
auto_main deploy pushed successfully but hasn't been confirmed yet
(Card.deploying_commit_sha — see domain/transitions.complete_deployer_visit):
the Deployer's own job, the git-level push, is done, but the card's overall
completion isn't until CI actually confirms it."""

import asyncio
import logging
from datetime import UTC, datetime

from sqlalchemy import select

from built.config import settings
from built.db.base import async_session_factory
from built.db.models import Card, Project
from built.domain import transitions
from built.sandbox import deploy_runner

logger = logging.getLogger(__name__)


def _as_utc(dt: datetime) -> datetime:
    # SQLite doesn't reliably round-trip tzinfo — see card_service._as_utc for the
    # same guard against the same issue elsewhere in this codebase.
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


async def _check_one(session, card: Card) -> str:
    """Returns which bucket this card landed in, for the pass-level summary log."""
    project = await session.get(Project, card.project_id)
    elapsed = (datetime.now(UTC) - _as_utc(card.deploying_since)).total_seconds()

    try:
        check_runs = await deploy_runner.fetch_check_runs(project, card.deploying_commit_sha)
    except deploy_runner.CIStatusUnavailableError as exc:
        logger.warning("ci watcher: couldn't check card %s this pass: %s", card.id, exc)
        return "still_pending"

    if check_runs is None:
        await transitions.confirm_ci_passed(
            session,
            card,
            note="no CI status available (no GitHub token configured, or not a github.com repo)",
        )
        return "confirmed"

    if not check_runs:
        if elapsed < settings.ci_watcher_grace_period_seconds:
            return "still_pending"  # give GitHub a moment to register the push
        await transitions.confirm_ci_passed(session, card, note="no CI checks reported for this commit")
        return "confirmed"

    if any(run.status != "completed" for run in check_runs):
        if elapsed > settings.ci_watcher_timeout_seconds:
            pending = ", ".join(r.name for r in check_runs if r.status != "completed")
            await transitions.mark_ci_wait_timed_out(
                session, card, note=f"CI still running after {elapsed / 60:.0f}m: {pending}"
            )
            return "timed_out"
        return "still_pending"

    failing = [r for r in check_runs if r.conclusion in deploy_runner.FAILING_CONCLUSIONS]
    if failing:
        lines = "\n".join(f"- {r.name}: {r.conclusion} ({r.html_url or 'no URL'})" for r in failing)
        summary = f"CI failed on commit {card.deploying_commit_sha[:8]}:\n{lines}"
        await transitions.bounce_deployed_card_to_developer(session, card, ci_summary=summary)
        return "bounced"

    await transitions.confirm_ci_passed(session, card, note=f"all {len(check_runs)} check(s) passed")
    return "confirmed"


async def run_ci_watcher_once() -> dict:
    """One pass, in its own session — checks every card currently waiting on CI.
    Returns per-outcome counts."""
    counts = {"confirmed": 0, "bounced": 0, "timed_out": 0, "still_pending": 0}
    async with async_session_factory() as session:
        cards = list(
            (await session.scalars(select(Card).where(Card.deploying_commit_sha.is_not(None)))).all()
        )
        for card in cards:
            outcome = await _check_one(session, card)
            counts[outcome] += 1
        if counts["confirmed"] or counts["bounced"] or counts["timed_out"]:
            await session.commit()
            logger.info(
                "ci watcher: confirmed=%d bounced=%d timed_out=%d still_pending=%d",
                counts["confirmed"],
                counts["bounced"],
                counts["timed_out"],
                counts["still_pending"],
            )
    return counts


async def run_ci_watcher_loop(*, stop_event: asyncio.Event, poll_interval: float) -> None:
    while not stop_event.is_set():
        try:
            await run_ci_watcher_once()
        except Exception:
            # Crash-isolation: one bad pass shouldn't kill the watcher forever —
            # log and try again at the next scheduled wake.
            logger.exception("ci watcher: unhandled error during pass")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
        except TimeoutError:
            pass
