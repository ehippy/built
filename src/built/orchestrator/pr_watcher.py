"""Runs a deterministic sweep on a timer — no LLM involved, same shape as
orchestrator/ci_watcher.py. Polls GitHub for every card whose pr_to_operator
deploy opened a PR but hasn't been confirmed yet (Card.pr_number — see
domain/transitions.complete_deployer_visit): the Deployer's own job, pushing the
branch and opening the PR, is done, but the card's overall completion isn't until
the PR is actually reviewed and merged. Per PR:

  - merged (by this watcher after an approving review, or by a human) → card DONE.
  - an approving review → the watcher merges it via the merge API (the one place
    the orchestrator acts on GitHub, never the LLM).
  - a "changes requested" review → the card bounces back to Developer with the
    review feedback, exactly like Tester/Reviewer's request_changes; when it flows
    back through to Deployer, open_pull_request re-uses the still-open PR.
  - closed without merging (human) → card BLOCKED, for a human to decide.
  - approved but not mergeable (default_branch advanced into a conflict) → card
    BLOCKED — an autonomous rebase of shared history is out of scope.
  - no review within the configured window → card BLOCKED (reviver may pick it up).

A transient fetch failure is logged and left still_pending for the next pass — not
treated as a resolution."""

import asyncio
import logging
from datetime import UTC, datetime

from sqlalchemy import select

from built.agent import summarizer
from built.config import settings
from built.db.base import async_session_factory
from built.db.models import Card, Project
from built.domain import transitions
from built.domain.enums import LifecycleState
from built.sandbox import deploy_runner

logger = logging.getLogger(__name__)


def _as_utc(dt: datetime) -> datetime:
    # SQLite doesn't reliably round-trip tzinfo — see card_service._as_utc for the
    # same guard against the same issue elsewhere in this codebase.
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


async def _maybe_write_postmortem(session, project: Project, card: Card) -> None:
    """Same postmortem hook as orchestrator/ci_watcher.py's — the other place a
    pr_to_operator card reaches DONE (confirm_pr_merged). Runs before the caller's
    own commit, so the postmortem lands in the same commit as the closure it's
    about, exactly like agent/loop.py's deploy terminal handlers."""
    if card.lifecycle_state not in (LifecycleState.DONE, LifecycleState.FAILED):
        return
    await summarizer.write_card_postmortem(session, project, card, outcome=card.lifecycle_state)


async def _check_one(session, card: Card) -> str:
    """Returns which bucket this card landed in, for the pass-level summary log."""
    project = await session.get(Project, card.project_id)
    elapsed = (datetime.now(UTC) - _as_utc(card.pr_waiting_since)).total_seconds()

    try:
        status = await deploy_runner.fetch_pr_status(project, card)
    except deploy_runner.PrStatusUnavailableError as exc:
        logger.warning("pr watcher: couldn't check card %s this pass: %s", card.id, exc)
        return "still_pending"

    if status.merged:
        await transitions.confirm_pr_merged(session, card, note="the PR is merged")
        await _maybe_write_postmortem(session, project, card)
        return "merged"

    if status.state != "open":
        await transitions.mark_pr_closed_unmerged(
            session, card, note="the PR was closed without being merged"
        )
        return "closed_unmerged"

    if status.review_decision == "changes_requested":
        feedback = status.feedback or "a reviewer requested changes on the PR (no feedback text given)"
        await transitions.request_pr_changes(
            session,
            card,
            feedback=feedback,
            note=f"a reviewer requested changes on PR #{card.pr_number}",
        )
        return "changes_requested"

    if status.review_decision == "approved":
        result = await deploy_runner.merge_pull_request(project, card)
        if result.success:
            await transitions.confirm_pr_merged(session, card, note=result.message)
            await _maybe_write_postmortem(session, project, card)
            return "merged"
        if "not mergeable" in result.message:
            await transitions.mark_pr_merge_conflicted(session, card, note=result.message)
            return "merge_conflicted"
        # A transient merge failure (rate limit, GitHub hiccup) is not a verdict —
        # retry next pass, within the timeout valve below.
        logger.warning("pr watcher: merge failed for card %s this pass: %s", card.id, result.message)
        return "still_pending"

    # No substantive review yet.
    if elapsed > settings.pr_watcher_timeout_seconds:
        await transitions.mark_pr_wait_timed_out(
            session, card, note=f"PR never reviewed within {elapsed / 3600:.0f}h"
        )
        return "timed_out"
    return "still_pending"


async def run_pr_watcher_once() -> dict:
    """One pass, in its own session — checks every card currently waiting on a PR.
    Returns per-outcome counts."""
    counts = {
        "merged": 0,
        "changes_requested": 0,
        "closed_unmerged": 0,
        "merge_conflicted": 0,
        "timed_out": 0,
        "still_pending": 0,
    }
    async with async_session_factory() as session:
        cards = list(
            (await session.scalars(select(Card).where(Card.pr_number.is_not(None)))).all()
        )
        for card in cards:
            outcome = await _check_one(session, card)
            counts[outcome] += 1
        resolved = sum(v for k, v in counts.items() if k != "still_pending")
        if resolved:
            await session.commit()
            logger.info(
                "pr watcher: merged=%d changes_requested=%d closed_unmerged=%d "
                "merge_conflicted=%d timed_out=%d still_pending=%d",
                counts["merged"],
                counts["changes_requested"],
                counts["closed_unmerged"],
                counts["merge_conflicted"],
                counts["timed_out"],
                counts["still_pending"],
            )
    return counts


async def run_pr_watcher_loop(*, stop_event: asyncio.Event, poll_interval: float) -> None:
    while not stop_event.is_set():
        try:
            await run_pr_watcher_once()
        except Exception:
            # Crash-isolation: one bad pass shouldn't kill the watcher forever —
            # log and try again at the next scheduled wake.
            logger.exception("pr watcher: unhandled error during pass")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
        except TimeoutError:
            pass
