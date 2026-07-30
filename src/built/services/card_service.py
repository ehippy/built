import logging
import re
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from built.db.models import Card, CardColumnVisit, CardEvent, Project
from built.domain import transitions
from built.domain.enums import Column, EventType, LifecycleState, Priority
from built.domain.events import append_event
from built.services.project_service import NotFoundError, get_project

logger = logging.getLogger(__name__)


def _branch_slug(card_id: str, title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.strip().lower()).strip("-")[:40] or "card"
    return f"card/{card_id[:8]}-{slug}"


def _as_utc(dt: datetime) -> datetime:
    # SQLite doesn't reliably round-trip tzinfo (see Card.is_being_worked, templates._timeago)
    # — a value just read back from the DB can be naive even though it was written as UTC.
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


async def _attach_last_activity(session: AsyncSession, cards: list[Card]) -> None:
    """Sets card.last_activity_at to the more recent of the Card row's own
    updated_at and its most recent transcript event. updated_at only moves on a
    column transition or claim/release — while an agent is mid-run (a tool call
    every few seconds, no Card-row write in between), it reads as stale next to
    what's actually happening, which is what the board tile and card header show."""
    if not cards:
        return
    stmt = (
        select(CardEvent.card_id, func.max(CardEvent.created_at))
        .where(CardEvent.card_id.in_([c.id for c in cards]))
        .group_by(CardEvent.card_id)
    )
    latest_event_at = dict((await session.execute(stmt)).all())
    for card in cards:
        candidates = [_as_utc(card.updated_at)]
        event_at = latest_event_at.get(card.id)
        if event_at is not None:
            candidates.append(_as_utc(event_at))
        card.last_activity_at = max(candidates)


async def create_card(
    session: AsyncSession,
    project_id: str,
    *,
    title: str,
    raw_request: str,
    source: str = "human",
    priority: Priority = Priority.NORMAL,
) -> Card:
    await get_project(session, project_id)  # 404s early if the project doesn't exist
    card = Card(project_id=project_id, title=title, raw_request=raw_request, priority=priority)
    session.add(card)
    await session.flush()
    card.branch_name = _branch_slug(card.id, title)
    await append_event(
        session,
        card_id=card.id,
        type=EventType.SYSTEM_NOTE,
        payload={"action": "created", "title": title, "source": source},
    )
    await session.flush()
    logger.info("card %s (%r) created for project %s, source=%s", card.id, title, project_id, source)
    return card


async def list_recent_card_titles(session: AsyncSession, project_id: str, *, limit: int = 50) -> list[str]:
    """Recent card titles for this project, newest first — context for PM discovery
    so it doesn't propose work that's already queued or done."""
    stmt = (
        select(Card.title).where(Card.project_id == project_id).order_by(Card.created_at.desc()).limit(limit)
    )
    return list((await session.scalars(stmt)).all())


async def get_card(
    session: AsyncSession, card_id: str, *, with_visits: bool = False, with_last_activity: bool = False
) -> Card:
    stmt = select(Card).where(Card.id == card_id)
    if with_visits:
        stmt = stmt.options(selectinload(Card.column_visits))
    card = await session.scalar(stmt)
    if card is None:
        raise NotFoundError(f"no card {card_id!r}")
    if with_last_activity:
        await _attach_last_activity(session, [card])
    return card


async def list_cards(session: AsyncSession, project_id: str, *, include_archived: bool = False) -> list[Card]:
    stmt = select(Card).where(Card.project_id == project_id).order_by(Card.created_at)
    if not include_archived:
        stmt = stmt.where(Card.archived_at.is_(None))
    return list((await session.scalars(stmt)).all())


async def count_column_backlog(session: AsyncSession, project_id: str, column: Column) -> int:
    """How many non-archived cards currently sit in this column for a project —
    what the board's swimlane for that column visually shows. orchestrator/curator.py
    uses this as a WIP-limit gate: no point proposing more PM-column work than the
    (concurrency-capped) orchestrator can realistically work through."""
    stmt = (
        select(func.count())
        .select_from(Card)
        .where(Card.project_id == project_id, Card.column == column, Card.archived_at.is_(None))
    )
    return await session.scalar(stmt) or 0


_PRIORITY_SORT_RANK = {Priority.HIGH: 0, Priority.NORMAL: 1, Priority.LOW: 2}


async def get_board(
    session: AsyncSession, project_id: str, *, include_archived: bool = False
) -> dict[Column, list[Card]]:
    """Cards for a project, grouped by their current column and ordered within each
    column by priority first, then most recent activity — what the dashboard and
    the `/board` API endpoint render. Archived cards are left off by default —
    that's the whole point of archiving something."""
    cards = await list_cards(session, project_id, include_archived=include_archived)
    await _attach_last_activity(session, cards)
    # Two stable sorts: recency first, then priority — the second pass reorders by
    # priority rank while preserving each rank's existing (already recency-sorted)
    # relative order, without needing a combined sort key.
    cards.sort(key=lambda c: c.last_activity_at, reverse=True)
    cards.sort(key=lambda c: _PRIORITY_SORT_RANK[c.priority])
    board: dict[Column, list[Card]] = {column: [] for column in Column}
    for card in cards:
        board[card.column].append(card)
    return board


async def get_project_activity_summary(session: AsyncSession, project_id: str) -> dict:
    """Card counts by lifecycle_state, whether anything's actively being worked right
    now, and the single most recent closed visit — what the Projects list page shows
    as each project's live status, so a project effectively narrates its own progress
    as cards move through it."""
    cards = await list_cards(session, project_id)
    counts = {state.value: 0 for state in LifecycleState}
    is_being_worked = False
    for card in cards:
        counts[card.lifecycle_state.value] += 1
        if card.is_being_worked:
            is_being_worked = True
    latest = await list_recent_visit_outcomes(session, project_id, limit=1)
    return {
        "counts": counts,
        "total": len(cards),
        "is_being_worked": is_being_worked,
        "latest": latest[0] if latest else None,
    }


async def list_stuck_cards(session: AsyncSession, *, limit: int = 50) -> list[Card]:
    """BLOCKED or FAILED cards, longest-stuck first — the Reviver's (agent/reviver.py)
    candidate pool. Ordered by updated_at so cards that have been waiting longest get
    priority attention within a bounded pass. Excludes paused projects — a human asked
    to have that repo left alone, so the Reviver shouldn't retry/revive its cards."""
    stmt = (
        select(Card)
        .join(Project, Project.id == Card.project_id)
        .where(
            Card.lifecycle_state.in_((LifecycleState.BLOCKED, LifecycleState.FAILED)),
            Project.paused_at.is_(None),
        )
        .order_by(Card.updated_at)
        .limit(limit)
    )
    return list((await session.scalars(stmt)).all())


async def list_recent_visit_outcomes(
    session: AsyncSession, project_id: str, *, since=None, limit: int = 30
) -> list[dict]:
    """Closed visits (ended_at IS NOT NULL) for a project, newest first — the
    agents_md curation kind's (orchestrator/curator.py) raw material for deciding
    whether anything from recent activity is worth capturing as a durable practice.
    Optionally scoped to only what closed since a given timestamp, so a pass with
    nothing new to look at can be skipped entirely."""
    stmt = (
        select(CardColumnVisit, Card.title)
        .join(Card, Card.id == CardColumnVisit.card_id)
        .where(Card.project_id == project_id, CardColumnVisit.ended_at.is_not(None))
        .order_by(CardColumnVisit.ended_at.desc())
        .limit(limit)
    )
    if since is not None:
        stmt = stmt.where(CardColumnVisit.ended_at > since)
    rows = (await session.execute(stmt)).all()
    return [
        {
            "card_title": title,
            "column": visit.column.value,
            "outcome": visit.outcome.value if visit.outcome else "unknown",
            "summary": visit.summary or "",
            "ended_at": visit.ended_at,
        }
        for visit, title in rows
    ]


async def list_column_visits(session: AsyncSession, card_id: str) -> list[CardColumnVisit]:
    await get_card(session, card_id)
    stmt = (
        select(CardColumnVisit).where(CardColumnVisit.card_id == card_id).order_by(CardColumnVisit.started_at)
    )
    return list((await session.scalars(stmt)).all())


async def get_latest_visit_summary(session: AsyncSession, card_id: str, column: Column) -> str | None:
    """The prior column's closing summary, handed to the next column as context — e.g.
    Tester reads Developer's summary of what changed."""
    stmt = (
        select(CardColumnVisit.summary)
        .where(CardColumnVisit.card_id == card_id, CardColumnVisit.column == column)
        .order_by(CardColumnVisit.attempt_number.desc())
        .limit(1)
    )
    return await session.scalar(stmt)


_RECAP_EVENT_LIMIT = 12
_RECAP_RESULT_CHARS = 300


def _describe_tool_call_event(payload: dict) -> str:
    name = payload.get("name", "?")
    args = payload.get("arguments") or {}
    descriptor = args.get("command") or args.get("path") or args.get("pattern") or ""
    status = "FAILED" if payload.get("is_error") else "ok"
    result = str(payload.get("result", ""))[:_RECAP_RESULT_CHARS]
    return f"- {name}({descriptor!r}) [{status}]: {result}"


async def get_previous_attempt_recap(
    session: AsyncSession, card_id: str, column: Column, *, before_attempt: int
) -> str | None:
    """A compact recap of the most recent prior attempt at this same column (if any):
    how it ended, plus its last few tool calls. Handed to a retried or bounced-back
    visit so it doesn't start completely cold — re-reading files and re-discovering
    environment quirks (e.g. a sandbox toolchain workaround) it already learned last
    time, burning iterations on rediscovery instead of picking up where it left off."""
    if before_attempt <= 1:
        return None
    prev_visit = await session.scalar(
        select(CardColumnVisit)
        .where(
            CardColumnVisit.card_id == card_id,
            CardColumnVisit.column == column,
            CardColumnVisit.attempt_number < before_attempt,
        )
        .order_by(CardColumnVisit.attempt_number.desc())
        .limit(1)
    )
    if prev_visit is None:
        return None

    events = list(
        (
            await session.scalars(
                select(CardEvent)
                .where(CardEvent.column_visit_id == prev_visit.id, CardEvent.type == EventType.TOOL_CALL)
                .order_by(CardEvent.seq.desc())
                .limit(_RECAP_EVENT_LIMIT)
            )
        ).all()
    )
    events.reverse()

    outcome = prev_visit.outcome.value if prev_visit.outcome else "unknown"
    lines = [
        f"Attempt #{prev_visit.attempt_number} ended: {outcome} — {prev_visit.summary or '(no summary)'}"
    ]
    if events:
        lines.append("Its last actions there, most recent last (avoid repeating work already done):")
        lines.extend(_describe_tool_call_event(e.payload) for e in events)
    return "\n".join(lines)


async def get_latest_attempt_recap(session: AsyncSession, card: Card) -> str | None:
    """Same recap as get_previous_attempt_recap, but for the most recent attempt at
    card's current column rather than the one before a not-yet-started attempt — the
    Reviver (agent/reviver.py) wants "what happened last time," not "what happened
    before whatever's about to happen next."""
    attempt_count = await session.scalar(
        select(func.count())
        .select_from(CardColumnVisit)
        .where(CardColumnVisit.card_id == card.id, CardColumnVisit.column == card.column)
    )
    return await get_previous_attempt_recap(
        session, card.id, card.column, before_attempt=(attempt_count or 0) + 1
    )


async def list_events(
    session: AsyncSession, card_id: str, *, since_seq: int = 0, limit: int = 200
) -> list[CardEvent]:
    """Pages forward from a cursor — for an API consumer accumulating a card's full
    history incrementally (poll with since_seq=<last seq you saw> to get only the
    next batch). NOT what the dashboard wants: since_seq defaults to 0 and nothing
    calls this with an advancing cursor there, calling this with the default on
    every poll would return the same oldest-`limit` window forever on any card
    past `limit` events — see list_recent_events for what the UI actually uses."""
    await get_card(session, card_id)
    stmt = (
        select(CardEvent)
        .where(CardEvent.card_id == card_id, CardEvent.seq > since_seq)
        .order_by(CardEvent.seq)
        .limit(limit)
    )
    return list((await session.scalars(stmt)).all())


async def list_recent_events(session: AsyncSession, card_id: str, *, limit: int = 200) -> list[CardEvent]:
    """The most recent `limit` events, chronological order for display — what the
    card detail page and its live-polling transcript fragment actually render.
    Confirmed in production: a card past 200 total events showed the same ~18-
    minutes-stale window on every 2-second poll forever, because list_events(
    since_seq=0) always returns the *oldest* `limit` events, not the newest."""
    await get_card(session, card_id)
    stmt = (
        select(CardEvent).where(CardEvent.card_id == card_id).order_by(CardEvent.seq.desc()).limit(limit)
    )
    events = list((await session.scalars(stmt)).all())
    events.reverse()
    return events


async def retry_card(session: AsyncSession, card_id: str, *, note: str | None = None) -> Card:
    card = await get_card(session, card_id)
    await transitions.retry_card(session, card, note=note)
    await session.flush()
    logger.info(
        "card %s (%r) retried, back to column=%s%s",
        card.id,
        card.title,
        card.column.value,
        f" — note: {note}" if note else "",
    )
    return card


async def cancel_card(session: AsyncSession, card_id: str) -> Card:
    card = await get_card(session, card_id)
    await transitions.cancel_card(session, card)
    await session.flush()
    logger.info("card %s (%r) cancelled", card.id, card.title)
    return card


async def update_card(session: AsyncSession, card_id: str, *, title: str, raw_request: str) -> Card:
    """Edits the two fields a human actually authors (title, raw_request). Doesn't
    touch spec/acceptance_criteria — those are PM-generated once the card's been
    through that column, and an in-flight visit may already be relying on them."""
    card = await get_card(session, card_id)
    card.title = title
    card.raw_request = raw_request
    await session.flush()
    return card


async def set_priority(session: AsyncSession, card_id: str, priority: Priority) -> Card:
    """The one-click "bless this as important" action — deliberately separate from
    update_card (which edits authored content) since this is a quick, no-context
    action a human should be able to take from anywhere the card appears. Purely a
    manual signal: never touched by any agent or automated pass. See
    orchestrator/worker.py's _CLAIM_PRIORITY_ORDER for how it affects claim order."""
    card = await get_card(session, card_id)
    card.priority = priority
    await session.flush()
    return card


async def archive_card(session: AsyncSession, card_id: str) -> Card:
    """Hides the card from the board and stops the orchestrator from ever claiming
    it again — doesn't touch lifecycle_state or an in-flight claim, so a visit
    already running finishes normally. History/events/visits are untouched; the card
    stays reachable at its own URL, which is where Unarchive lives."""
    card = await get_card(session, card_id)
    card.archived_at = datetime.now(UTC)
    await session.flush()
    return card


async def unarchive_card(session: AsyncSession, card_id: str) -> Card:
    card = await get_card(session, card_id)
    card.archived_at = None
    await session.flush()
    return card
