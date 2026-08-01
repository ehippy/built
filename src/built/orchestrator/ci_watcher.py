"""Runs a deterministic sweep on a timer — no LLM involved, same shape as
orchestrator/archiver.py. Polls GitHub's Checks API for every card whose
auto_main deploy pushed successfully but hasn't been confirmed yet
(Card.deploying_commit_sha — see domain/transitions.complete_deployer_visit):
the Deployer's own job, the git-level push, is done, but the card's overall
completion isn't until CI actually confirms it. A red result does NOT attempt to
revert anything on the default branch — an automated revert of a shared branch that
may already have other work built on top of it is a bigger, riskier action than this
watcher should take unattended. Instead it reopens the same card and bounces it back
to Developer to fix forward — not a fresh follow-up card. There's no mystery here
for a new card to reinvestigate: this watcher already knows exactly which card's
merge broke CI and, since it fetches the failing job's actual `##[error]` output (not
just a check name and a link), exactly why. Filing a brand-new "diagnose and fix the
regression" card sounds safe but in practice means every occurrence of the same root
cause (e.g. one broken workflow file that fails every deploy the same way) spawns its
own card, each rediscovering — and often misdiagnosing, with no more evidence than a
check name and a URL — the same problem from scratch. Reopening the original card
instead reuses its existing revision_count safety valve, so a fix that keeps
re-breaking CI eventually BLOCKs for a human rather than growing an unbounded chain
of look-alike cards."""

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
    """Same postmortem hook as agent/loop.py's deploy terminal handlers — this is
    the other place a card can reach DONE/FAILED (an auto_main deploy that had CI
    to wait on), so it needs the same call. mark_ci_wait_timed_out (BLOCKED) never
    reaches here since it isn't terminal — a human/Reviver can still retry it."""
    if card.lifecycle_state not in (LifecycleState.DONE, LifecycleState.FAILED):
        return
    await summarizer.write_card_postmortem(session, project, card, outcome=card.lifecycle_state)


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
        await _maybe_write_postmortem(session, project, card)
        return "confirmed"

    if not check_runs:
        if elapsed < settings.ci_watcher_grace_period_seconds:
            return "still_pending"  # give GitHub a moment to register the push
        await transitions.confirm_ci_passed(session, card, note="no CI checks reported for this commit")
        await _maybe_write_postmortem(session, project, card)
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
        lines = []
        for r in failing:
            lines.append(f"- {r.name}: {r.conclusion} ({r.html_url or 'no URL'})")
            error_text = await deploy_runner.fetch_job_error_lines(project, r.html_url)
            if error_text:
                indented = "\n".join(f"    {line}" for line in error_text.splitlines())
                lines.append(f"  Error output:\n{indented}")
        lines_text = "\n".join(lines)
        commit_sha = card.deploying_commit_sha
        feedback = (
            f"This card's merge to {project.default_branch} deployed as commit {commit_sha[:8]}, but CI "
            f"failed on it:\n{lines_text}\n\n"
            f"Fix the regression on top of the current {project.default_branch} and resubmit — it'll be "
            "redeployed and CI-checked again the same way."
        )
        await transitions.reopen_after_ci_failure(session, card, project, feedback=feedback)
        return "ci_failed"

    await transitions.confirm_ci_passed(session, card, note=f"all {len(check_runs)} check(s) passed")
    await _maybe_write_postmortem(session, project, card)
    return "confirmed"


async def run_ci_watcher_once() -> dict:
    """One pass, in its own session — checks every card currently waiting on CI.
    Returns per-outcome counts."""
    counts = {"confirmed": 0, "ci_failed": 0, "timed_out": 0, "still_pending": 0}
    async with async_session_factory() as session:
        cards = list(
            (await session.scalars(select(Card).where(Card.deploying_commit_sha.is_not(None)))).all()
        )
        for card in cards:
            outcome = await _check_one(session, card)
            counts[outcome] += 1
        if counts["confirmed"] or counts["ci_failed"] or counts["timed_out"]:
            await session.commit()
            logger.info(
                "ci watcher: confirmed=%d ci_failed=%d timed_out=%d still_pending=%d",
                counts["confirmed"],
                counts["ci_failed"],
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
