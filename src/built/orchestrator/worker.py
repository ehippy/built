"""Single-process asyncio worker pool: polls for claimable cards, runs one column
visit each. No Celery/Redis needed at this scale — see the plan for the scaling seam
(swap the claim UPDATE for `SELECT...FOR UPDATE SKIP LOCKED` against Postgres, or a
real queue, if this ever needs multi-process/multi-node scaling)."""

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from built.agent.loop import run_developer_visit, run_pm_visit, run_tester_visit
from built.db.base import async_session_factory
from built.db.models import Card, Project
from built.domain import transitions
from built.domain.enums import Column, LifecycleState
from built.llm.client import FallbackLLMClient
from built.sandbox.container import DockerCommandExecutor
from built.sandbox.worktree import create_card_worktree
from built.services import card_service, endpoint_service
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)

# Deployer isn't implemented yet (Phase 5) — cards that reach it just sit ACTIVE,
# claimed by nobody, rather than being picked up and immediately failing.
IMPLEMENTED_COLUMNS = (Column.PM, Column.DEVELOPER, Column.TESTER)

DEFAULT_LEASE_SECONDS = 600
DEFAULT_POLL_INTERVAL_SECONDS = 1.5
DEFAULT_CONCURRENCY = 4


async def claim_next_card(session: AsyncSession, worker_id: str) -> Card | None:
    """Atomic conditional UPDATE, checked by rowcount — works against SQLite from a
    single process, and is a drop-in seam for a future `SELECT...FOR UPDATE SKIP
    LOCKED` multi-process claim."""
    now = datetime.now(UTC)
    candidate_id = await session.scalar(
        select(Card.id)
        .where(
            Card.lifecycle_state == LifecycleState.ACTIVE,
            Card.column.in_(IMPLEMENTED_COLUMNS),
            or_(Card.claimed_by_worker_id.is_(None), Card.lease_expires_at < now),
        )
        .order_by(Card.updated_at)
        .limit(1)
    )
    if candidate_id is None:
        return None

    result = await session.execute(
        update(Card)
        .where(
            Card.id == candidate_id,
            or_(Card.claimed_by_worker_id.is_(None), Card.lease_expires_at < now),
        )
        .values(
            claimed_by_worker_id=worker_id,
            claimed_at=now,
            lease_expires_at=now + timedelta(seconds=DEFAULT_LEASE_SECONDS),
        )
    )
    await session.commit()
    if result.rowcount == 0:
        return None  # lost the race to another worker

    # session.get() checks the identity map first — an already-loaded Card (e.g. the
    # object create_card() just returned) comes back as-is, without picking up the
    # bulk update above, regardless of synchronize_session settings. populate_existing
    # forces an actual re-read so the returned object reflects the row we just wrote.
    return await session.get(Card, candidate_id, populate_existing=True)


async def release_claim(session: AsyncSession, card: Card) -> None:
    card.claimed_by_worker_id = None
    card.claimed_at = None
    card.lease_expires_at = None
    await session.commit()


async def requeue_stale_claims(session: AsyncSession) -> int:
    """Crash recovery: on boot, free anything still marked claimed with an expired
    lease so a fresh worker can pick it up. An interrupted visit starts over from
    scratch rather than resuming mid-tool-call — v1 simplification, see the plan."""
    now = datetime.now(UTC)
    result = await session.execute(
        update(Card)
        .where(Card.claimed_by_worker_id.is_not(None), Card.lease_expires_at < now)
        .values(claimed_by_worker_id=None, claimed_at=None, lease_expires_at=None)
        .execution_options(synchronize_session="fetch")
    )
    await session.commit()
    return result.rowcount or 0


async def run_one_card(session: AsyncSession, card: Card) -> None:
    """Runs exactly one column visit for an already-claimed card, then releases the
    claim. Setup failures (bad endpoint config, git clone failure) are recorded as a
    blocked run just like an in-loop failure would be — the caller never needs to
    handle an exception from this beyond the crash-isolation safety net in worker_loop."""
    project = await session.get(Project, card.project_id)
    assert project is not None, f"card {card.id} references missing project {card.project_id}"

    try:
        chain = await endpoint_service.get_resolved_chain(session, project_id=project.id, role=card.column)
        llm_client = FallbackLLMClient(chain)
        worktree_path = await create_card_worktree(project, card)
    except Exception as exc:  # noqa: BLE001 — deliberate: setup failures block the card, not the worker
        visit = await transitions.start_visit(session, card)
        await transitions.fail_visit_with_error(session, card, visit, message=f"setup failed: {exc!r}")
        await session.commit()
        await release_claim(session, card)
        return

    if card.worktree_path != str(worktree_path):
        card.worktree_path = str(worktree_path)
        await session.commit()

    executor_kwargs = {"image": project.sandbox_image} if project.sandbox_image else {}
    dispatcher = ToolDispatcher(
        ctx=ToolContext(card_id=card.id, worktree_root=worktree_path),
        executor=DockerCommandExecutor(**executor_kwargs),
    )
    visit = await transitions.start_visit(session, card)
    await session.commit()

    max_iterations = project.max_iterations_per_run
    if card.column == Column.PM:
        await run_pm_visit(
            session,
            project,
            card,
            visit,
            llm_client=llm_client,
            dispatcher=dispatcher,
            max_iterations=max_iterations,
        )
    elif card.column == Column.DEVELOPER:
        await run_developer_visit(
            session,
            project,
            card,
            visit,
            llm_client=llm_client,
            dispatcher=dispatcher,
            max_iterations=max_iterations,
        )
    elif card.column == Column.TESTER:
        developer_summary = await card_service.get_latest_visit_summary(session, card.id, Column.DEVELOPER)
        await run_tester_visit(
            session,
            project,
            card,
            visit,
            llm_client=llm_client,
            dispatcher=dispatcher,
            max_iterations=max_iterations,
            developer_summary=developer_summary,
        )
    else:  # pragma: no cover — guarded by IMPLEMENTED_COLUMNS in claim_next_card
        raise AssertionError(f"unimplemented column: {card.column}")

    await release_claim(session, card)


async def _worker_loop(worker_id: str, *, stop_event: asyncio.Event, poll_interval: float) -> None:
    while not stop_event.is_set():
        async with async_session_factory() as session:
            card = await claim_next_card(session, worker_id)
            if card is not None:
                try:
                    await run_one_card(session, card)
                except Exception:
                    # Crash-isolation safety net: run_one_card already converts
                    # ordinary run failures into a blocked card internally, so
                    # reaching here means something unexpected broke. Don't let one
                    # bad card take down the rest of the pool.
                    logger.exception("worker %s: unhandled error running card %s", worker_id, card.id)
                    await session.rollback()
                    async with async_session_factory() as cleanup_session:
                        fresh_card = await cleanup_session.get(Card, card.id)
                        if fresh_card is not None:
                            await release_claim(cleanup_session, fresh_card)
                continue  # immediately look for more work, no idle wait
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
        except TimeoutError:
            pass


async def run_worker_pool(
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    stop_event: asyncio.Event | None = None,
) -> None:
    stop_event = stop_event or asyncio.Event()

    async with async_session_factory() as session:
        requeued = await requeue_stale_claims(session)
        if requeued:
            logger.info("requeued %d card(s) with expired claims on startup", requeued)

    workers = [
        asyncio.create_task(
            _worker_loop(f"worker-{uuid.uuid4().hex[:8]}", stop_event=stop_event, poll_interval=poll_interval)
        )
        for _ in range(concurrency)
    ]
    await asyncio.gather(*workers)
