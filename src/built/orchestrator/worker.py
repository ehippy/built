"""Single-process asyncio worker pool: polls for claimable cards, runs one column
visit each. No Celery/Redis needed at this scale — see the plan for the scaling seam
(swap the claim UPDATE for `SELECT...FOR UPDATE SKIP LOCKED` against Postgres, or a
real queue, if this ever needs multi-process/multi-node scaling)."""

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import case, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from built.agent.context_window import ContextWindowConfig
from built.agent.discovery import run_pm_discovery
from built.agent.loop import (
    run_deployer_visit,
    run_developer_visit,
    run_pm_visit,
    run_tester_visit,
)
from built.config import settings as builtin_settings
from built.db.base import async_session_factory
from built.db.models import Card, Project
from built.domain import transitions
from built.domain.enums import Column, DeployMode, LifecycleState
from built.llm.client import FallbackLLMClient
from built.sandbox.container import DockerCommandExecutor
from built.sandbox.worktree import (
    create_card_worktree,
    ensure_tool_worktree,
    read_default_branch_file,
)
from built.services import card_service, endpoint_service
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)

IMPLEMENTED_COLUMNS = (Column.PM, Column.DEVELOPER, Column.TESTER, Column.DEPLOYER)

DEFAULT_LEASE_SECONDS = 600
DEFAULT_POLL_INTERVAL_SECONDS = 1.5
DEFAULT_CONCURRENCY = 4

# "Stop starting, start finishing": claim cards closer to done first (Deployer, then
# Tester, then Developer, then PM), so limited concurrency drains work-in-progress
# toward completion instead of spreading thin by starting new cards while older ones
# sit half-finished. updated_at only breaks ties within the same column.
#
# Explicit (Card.column == X, priority) comparisons, not case()'s dict/value= form —
# the dict form binds each key as a bare literal without the column's enum type
# decorator applied, so every WHEN silently fails to match and the whole expression
# evaluates to NULL for every row (confirmed empirically).
_CLAIM_COLUMN_PRIORITY = case(
    *((Card.column == column, idx) for idx, column in enumerate(reversed(IMPLEMENTED_COLUMNS)))
)


async def claim_next_card(session: AsyncSession, worker_id: str) -> Card | None:
    """Atomic conditional UPDATE, checked by rowcount — works against SQLite from a
    single process, and is a drop-in seam for a future `SELECT...FOR UPDATE SKIP
    LOCKED` multi-process claim.

    Also enforces per-project serialization: a card is only claimable if no other
    card in the same project currently holds an active (non-expired) claim. Two
    cards in the same repo never build concurrently and step on each other's
    worktree/branch assumptions with zero coordination between them. The busy-project
    check is repeated in the UPDATE's WHERE clause (not just the candidate SELECT) so
    it's re-evaluated against committed state at claim time, closing the race where
    two workers both pick a different card from the same idle project in the same
    poll cycle.

    A paused project (Project.paused_at set) is excluded the same way — a human
    asked to have this repo left alone, so no new claims start; a claim already in
    flight when pause was clicked still runs to completion. An archived card
    (Card.archived_at set) is excluded outright — a human decided this one isn't
    worth doing."""
    now = datetime.now(UTC)
    busy_project_ids = select(Card.project_id).where(
        Card.claimed_by_worker_id.is_not(None),
        Card.lease_expires_at >= now,
    )
    paused_project_ids = select(Project.id).where(Project.paused_at.is_not(None))
    candidate_id = await session.scalar(
        select(Card.id)
        .where(
            Card.lifecycle_state == LifecycleState.ACTIVE,
            Card.column.in_(IMPLEMENTED_COLUMNS),
            Card.archived_at.is_(None),
            or_(Card.claimed_by_worker_id.is_(None), Card.lease_expires_at < now),
            Card.project_id.not_in(busy_project_ids),
            Card.project_id.not_in(paused_project_ids),
        )
        .order_by(_CLAIM_COLUMN_PRIORITY, Card.updated_at)
        .limit(1)
    )
    if candidate_id is None:
        return None

    result = await session.execute(
        update(Card)
        .where(
            Card.id == candidate_id,
            Card.archived_at.is_(None),
            or_(Card.claimed_by_worker_id.is_(None), Card.lease_expires_at < now),
            Card.project_id.not_in(busy_project_ids),
            Card.project_id.not_in(paused_project_ids),
        )
        .values(
            claimed_by_worker_id=worker_id,
            claimed_at=now,
            lease_expires_at=now + timedelta(seconds=DEFAULT_LEASE_SECONDS),
        )
    )
    await session.commit()
    if result.rowcount == 0:
        return None  # lost the race to another worker, or another card in this project got claimed first

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

    is_deployer_auto_main = (
        card.column == Column.DEPLOYER
        and project.deploy_config is not None
        and project.deploy_config.mode == DeployMode.AUTO_MAIN
    )
    try:
        if card.column == Column.DEPLOYER and project.deploy_config is None:
            raise ValueError(
                "no deploy config — configure one in Project Settings before this card can deploy"
            )
        chain = await endpoint_service.get_resolved_chain(session, project_id=project.id, role=card.column)
        llm_client = FallbackLLMClient(chain)
        # auto_main deploys operate in the Deployer's own dedicated worktree, not the
        # card's — that's where a merge (and any conflict) actually happens, so the
        # agent's file tools need to be scoped there instead of the card's branch.
        if is_deployer_auto_main:
            worktree_path = await ensure_tool_worktree(project, tool="deployer")
        else:
            worktree_path = await create_card_worktree(project, card)
    except Exception as exc:  # noqa: BLE001 — deliberate: setup failures block the card, not the worker
        visit = await transitions.start_visit(session, card)
        await transitions.fail_visit_with_error(session, card, visit, message=f"setup failed: {exc!r}")
        await session.commit()
        await release_claim(session, card)
        return

    # Resolve the context window budget: per-endpoint model size → per-project override → default.
    if project.max_tokens:
        max_tokens = project.max_tokens
    elif any(e.context_window for e in chain):
        max_tokens = max(e.context_window for e in chain if e.context_window)
    else:
        max_tokens = builtin_settings.default_max_tokens

    context_window_config = ContextWindowConfig(
        max_tokens=max_tokens,
        keep_messages=builtin_settings.default_keep_messages,
    )

    if card.worktree_path != str(worktree_path):
        card.worktree_path = str(worktree_path)
        await session.commit()

    executor_kwargs = {"image": project.sandbox_image} if project.sandbox_image else {}
    dispatcher = ToolDispatcher(
        ctx=ToolContext(card_id=card.id, worktree_root=worktree_path, auto_commit=not is_deployer_auto_main),
        executor=DockerCommandExecutor(**executor_kwargs),
    )
    visit = await transitions.start_visit(session, card)
    await session.commit()

    # A retried or bounced-back visit isn't the first attempt at this column — hand it
    # a recap of what the previous attempt already did, so it doesn't start cold.
    retry_recap = await card_service.get_previous_attempt_recap(
        session, card.id, card.column, before_attempt=visit.attempt_number
    )
    # A human-authored retry note is one-shot: surfaced to this one visit, then
    # cleared so it doesn't linger and get replayed on later, unrelated retries.
    retry_note = card.retry_note
    if retry_note is not None:
        card.retry_note = None
        await session.commit()

    # AGENTS.md (maintained by the Tender — agent/tender.py) is read straight from the
    # tip of default_branch, not the card's own long-lived worktree — the card's
    # branch diverged from default_branch at creation time and won't pick up later
    # doc commits on its own.
    agents_doc = await read_default_branch_file(project, "AGENTS.md")

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
            context_window_config=context_window_config,
            retry_recap=retry_recap,
            retry_note=retry_note,
            agents_doc=agents_doc,
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
            context_window_config=context_window_config,
            retry_recap=retry_recap,
            retry_note=retry_note,
            agents_doc=agents_doc,
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
            context_window_config=context_window_config,
            developer_summary=developer_summary,
            retry_recap=retry_recap,
            retry_note=retry_note,
            agents_doc=agents_doc,
        )
    elif card.column == Column.DEPLOYER:
        await run_deployer_visit(
            session,
            project,
            card,
            visit,
            llm_client=llm_client,
            dispatcher=dispatcher,
            max_iterations=max_iterations,
            context_window_config=context_window_config,
            mode=project.deploy_config.mode,
            retry_recap=retry_recap,
            retry_note=retry_note,
            agents_doc=agents_doc,
        )
    else:  # pragma: no cover — guarded by IMPLEMENTED_COLUMNS in claim_next_card
        raise AssertionError(f"unimplemented column: {card.column}")

    await release_claim(session, card)


# Project IDs with a discovery run currently in flight. In-memory only — this is a
# single-process app, and an in-flight run doesn't need to survive a restart since
# the asyncio task itself dies with the process too. is_discovery_running() and the
# add() in run_project_discovery() below are both synchronous with no `await` between
# a check and the corresponding add/discard, so two near-simultaneous triggers can't
# both slip past the guard — asyncio only switches tasks at an await point.
_discovery_in_progress: set[str] = set()


def is_discovery_running(project_id: str) -> bool:
    return project_id in _discovery_in_progress


async def run_project_discovery(project_id: str) -> None:
    """PM's autonomous discovery mode for one project: opens its own session and
    runs to completion, then returns — meant to be fired as a detached background
    task from an API/UI handler (see api/routers/projects.py), not awaited inline in
    a request, since a full discovery pass is an LLM agentic loop and can take a
    while. Never raises — run_pm_discovery already swallows its own failures.

    Skips outright if a discovery run for this project is already in flight — two
    runs racing each other see the same "existing card titles" snapshot and can
    propose near-duplicate cards, since discovery doesn't go through claim_next_card
    and isn't covered by its per-project claim serialization."""
    if project_id in _discovery_in_progress:
        logger.info("discovery already running for project %s — skipping", project_id)
        return
    _discovery_in_progress.add(project_id)
    try:
        async with async_session_factory() as session:
            project = await session.get(Project, project_id)
            if project is None:
                logger.warning("discovery requested for missing project %s", project_id)
                return
            try:
                chain = await endpoint_service.get_resolved_chain(
                    session, project_id=project.id, role=Column.PM
                )
                llm_client = FallbackLLMClient(chain)
                # Own dedicated worktree + branch, not Deployer's — git refuses to
                # check out the same branch in two worktrees at once.
                wt_path = await ensure_tool_worktree(project, tool="discovery")
            except Exception:
                logger.exception("discovery setup failed for project %s", project_id)
                return

            executor_kwargs = {"image": project.sandbox_image} if project.sandbox_image else {}
            dispatcher = ToolDispatcher(
                ctx=ToolContext(card_id=f"discovery-{project.id}", worktree_root=wt_path),
                executor=DockerCommandExecutor(**executor_kwargs),
            )

            if project.max_tokens:
                max_tokens = project.max_tokens
            elif any(e.context_window for e in chain):
                max_tokens = max(e.context_window for e in chain if e.context_window)
            else:
                max_tokens = builtin_settings.default_max_tokens

            agents_doc = await read_default_branch_file(project, "AGENTS.md")

            created = await run_pm_discovery(
                session,
                project,
                llm_client=llm_client,
                dispatcher=dispatcher,
                max_iterations=project.max_iterations_per_run,
                agents_doc=agents_doc,
                context_window_config=ContextWindowConfig(
                    max_tokens=max_tokens,
                    keep_messages=builtin_settings.default_keep_messages,
                ),
            )
            logger.info("discovery for project %s created %d card(s)", project_id, len(created))
    finally:
        _discovery_in_progress.discard(project_id)


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
