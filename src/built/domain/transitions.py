"""The card state machine. This is the one place revision-cap, deploy-cap, and
run-error handling live — the safety valves in an otherwise fully-autonomous pipeline.
Every function here is pure business logic over ORM objects; it doesn't care whether
it's called by an integration test (Phase 2) or the agent loop after a tool call
(Phase 3+)."""

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from built.db.models import Card, CardColumnVisit, Project
from built.domain.enums import Column, EventType, LifecycleState, VisitOutcome
from built.domain.events import append_event


async def start_visit(session: AsyncSession, card: Card) -> CardColumnVisit:
    """Open a new CardColumnVisit for the card's current column, numbering re-entries
    (e.g. a Developer visit after a revision bounce) with an incrementing attempt.
    Queries CardColumnVisit directly rather than touching `card.column_visits` — the
    caller may have fetched `card` without eager-loading that relationship, and async
    SQLAlchemy has no implicit lazy-load to fall back on."""
    prior_attempts = await session.scalar(
        select(func.count())
        .select_from(CardColumnVisit)
        .where(CardColumnVisit.card_id == card.id, CardColumnVisit.column == card.column)
    )
    visit = CardColumnVisit(card_id=card.id, column=card.column, attempt_number=(prior_attempts or 0) + 1)
    session.add(visit)
    await session.flush()
    return visit


async def _close_visit(
    session: AsyncSession,
    visit: CardColumnVisit,
    *,
    outcome: VisitOutcome,
    summary: str,
    endpoint_used: str | None = None,
) -> None:
    visit.ended_at = datetime.now(UTC)
    visit.outcome = outcome
    visit.summary = summary
    visit.endpoint_used = endpoint_used
    await append_event(
        session,
        card_id=visit.card_id,
        column_visit_id=visit.id,
        type=EventType.TRANSITION,
        payload={"column": visit.column.value, "outcome": outcome.value, "summary": summary},
    )


async def complete_pm_visit(
    session: AsyncSession,
    card: Card,
    visit: CardColumnVisit,
    *,
    spec: str,
    acceptance_criteria: list[str],
    summary: str,
    endpoint_used: str | None = None,
) -> Card:
    """PM's submit_spec: record the spec + acceptance criteria, advance to Developer."""
    card.spec = spec
    card.acceptance_criteria = acceptance_criteria
    card.column = Column.DEVELOPER
    await _close_visit(
        session, visit, outcome=VisitOutcome.SUBMITTED, summary=summary, endpoint_used=endpoint_used
    )
    return card


async def complete_developer_visit(
    session: AsyncSession,
    card: Card,
    visit: CardColumnVisit,
    *,
    summary: str,
    endpoint_used: str | None = None,
) -> Card:
    """Developer's submit_for_test: advance to Tester."""
    card.column = Column.TESTER
    await _close_visit(
        session, visit, outcome=VisitOutcome.SUBMITTED, summary=summary, endpoint_used=endpoint_used
    )
    return card


async def complete_tester_visit_approved(
    session: AsyncSession,
    card: Card,
    visit: CardColumnVisit,
    *,
    summary: str,
    endpoint_used: str | None = None,
) -> Card:
    """Tester's approve: advance to Deployer. The caller (Phase 3+ tool dispatcher) is
    responsible for rejecting `approve` server-side unless the latest RunAttempt for
    this visit actually succeeded — this function trusts that check already happened."""
    card.column = Column.DEPLOYER
    await _close_visit(
        session, visit, outcome=VisitOutcome.APPROVED, summary=summary, endpoint_used=endpoint_used
    )
    return card


async def complete_tester_visit_changes_requested(
    session: AsyncSession,
    card: Card,
    visit: CardColumnVisit,
    *,
    feedback: str,
    summary: str,
    endpoint_used: str | None = None,
) -> Card:
    """Tester's request_changes: bounce back to Developer and bump the revision
    counter. Exceeding the project's max_revisions blocks the card for a human —
    the safety valve in the Tester<->Developer loop, since nothing else bounds it."""
    card.revision_count += 1
    card.latest_feedback = feedback
    card.column = Column.DEVELOPER
    # session.get() hits the identity map, not a query, since the project is already
    # loaded in this session — async SQLAlchemy has no implicit lazy-load, so
    # `card.project.max_revisions` would raise MissingGreenlet instead.
    project = await session.get(Project, card.project_id)
    if card.revision_count > project.max_revisions:
        card.lifecycle_state = LifecycleState.BLOCKED
    await _close_visit(
        session, visit, outcome=VisitOutcome.CHANGES_REQUESTED, summary=summary, endpoint_used=endpoint_used
    )
    return card


async def complete_deployer_visit(
    session: AsyncSession,
    card: Card,
    visit: CardColumnVisit,
    *,
    success: bool,
    summary: str,
    deploy_url: str | None = None,
    endpoint_used: str | None = None,
) -> Card:
    """Deployer's run_deploy (auto_main) or open_pull_request (pr_to_operator):
    success is terminal (done) either way. Failure retries up to the project's
    max_deploy_attempts, then is terminal (failed) with no further action — there is
    no human-approval gate to fall back to in this pipeline."""
    if success:
        if deploy_url is not None:
            card.deploy_url = deploy_url
        card.lifecycle_state = LifecycleState.DONE
        await _close_visit(
            session, visit, outcome=VisitOutcome.DONE, summary=summary, endpoint_used=endpoint_used
        )
        return card

    card.deploy_attempt_count += 1
    project = await session.get(Project, card.project_id)
    if card.deploy_attempt_count >= project.max_deploy_attempts:
        card.lifecycle_state = LifecycleState.FAILED
    await _close_visit(
        session, visit, outcome=VisitOutcome.FAILED, summary=summary, endpoint_used=endpoint_used
    )
    return card


async def fail_visit_with_error(
    session: AsyncSession, card: Card, visit: CardColumnVisit, *, message: str
) -> Card:
    """Iteration cap exceeded, endpoint fallback chain exhausted, or an unhandled
    exception mid-run — always blocks the card for a human, regardless of column."""
    card.lifecycle_state = LifecycleState.BLOCKED
    await _close_visit(session, visit, outcome=VisitOutcome.ERROR, summary=message)
    return card


async def mark_visit_interrupted(session: AsyncSession, card: Card, visit: CardColumnVisit) -> Card:
    """The worker process died mid-visit. The card stays ACTIVE — on restart the
    orchestrator starts a fresh attempt of the same column rather than trying to
    resume mid-tool-call (accepted v1 simplification)."""
    await _close_visit(
        session, visit, outcome=VisitOutcome.INTERRUPTED, summary="Worker process interrupted."
    )
    return card


async def retry_card(session: AsyncSession, card: Card, *, note: str | None = None) -> Card:
    """The one human touchpoint that exists *outside* the autonomous pipeline: un-stick
    a blocked or failed card with a clean safety-valve budget. An optional note is
    surfaced to whichever column runs next (see agent/context.py's retry_note
    handling) and cleared after that one visit — see orchestrator/worker.py."""
    if card.lifecycle_state not in (LifecycleState.BLOCKED, LifecycleState.FAILED):
        raise ValueError(f"cannot retry a card in state {card.lifecycle_state.value!r}")
    card.lifecycle_state = LifecycleState.ACTIVE
    card.revision_count = 0
    card.deploy_attempt_count = 0
    card.latest_feedback = None
    card.retry_note = note
    await append_event(
        session, card_id=card.id, type=EventType.SYSTEM_NOTE, payload={"action": "retry", "note": note}
    )
    return card


async def cancel_card(session: AsyncSession, card: Card) -> Card:
    """Also a human-only action. There's no separate CANCELLED state — a cancelled
    card is recorded as FAILED with a system note explaining why."""
    if card.lifecycle_state in (LifecycleState.DONE, LifecycleState.FAILED):
        raise ValueError(f"cannot cancel a card in state {card.lifecycle_state.value!r}")
    card.lifecycle_state = LifecycleState.FAILED
    await append_event(session, card_id=card.id, type=EventType.SYSTEM_NOTE, payload={"action": "cancel"})
    return card
