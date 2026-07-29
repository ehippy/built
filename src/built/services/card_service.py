import re

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from built.db.models import Card, CardColumnVisit, CardEvent
from built.domain import transitions
from built.domain.enums import Column, EventType, LifecycleState
from built.domain.events import append_event
from built.services.project_service import NotFoundError, get_project


def _branch_slug(card_id: str, title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.strip().lower()).strip("-")[:40] or "card"
    return f"card/{card_id[:8]}-{slug}"


async def create_card(
    session: AsyncSession, project_id: str, *, title: str, raw_request: str, source: str = "human"
) -> Card:
    await get_project(session, project_id)  # 404s early if the project doesn't exist
    card = Card(project_id=project_id, title=title, raw_request=raw_request)
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
    return card


async def list_recent_card_titles(session: AsyncSession, project_id: str, *, limit: int = 50) -> list[str]:
    """Recent card titles for this project, newest first — context for PM discovery
    so it doesn't propose work that's already queued or done."""
    stmt = (
        select(Card.title).where(Card.project_id == project_id).order_by(Card.created_at.desc()).limit(limit)
    )
    return list((await session.scalars(stmt)).all())


async def get_card(session: AsyncSession, card_id: str, *, with_visits: bool = False) -> Card:
    stmt = select(Card).where(Card.id == card_id)
    if with_visits:
        stmt = stmt.options(selectinload(Card.column_visits))
    card = await session.scalar(stmt)
    if card is None:
        raise NotFoundError(f"no card {card_id!r}")
    return card


async def list_cards(session: AsyncSession, project_id: str) -> list[Card]:
    stmt = select(Card).where(Card.project_id == project_id).order_by(Card.created_at)
    return list((await session.scalars(stmt)).all())


async def get_board(session: AsyncSession, project_id: str) -> dict[Column, list[Card]]:
    """Cards for a project, grouped by their current column — what the dashboard and
    the `/board` API endpoint render."""
    cards = await list_cards(session, project_id)
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
    priority attention within a bounded pass."""
    stmt = (
        select(Card)
        .where(Card.lifecycle_state.in_((LifecycleState.BLOCKED, LifecycleState.FAILED)))
        .order_by(Card.updated_at)
        .limit(limit)
    )
    return list((await session.scalars(stmt)).all())


async def list_recent_visit_outcomes(
    session: AsyncSession, project_id: str, *, since=None, limit: int = 30
) -> list[dict]:
    """Closed visits (ended_at IS NOT NULL) for a project, newest first — the
    Tender's (agent/tender.py) raw material for deciding whether anything from
    recent activity is worth capturing as a durable practice. Optionally scoped to
    only what closed since a given timestamp, so a pass with nothing new to look at
    can be skipped entirely."""
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
    await get_card(session, card_id)
    stmt = (
        select(CardEvent)
        .where(CardEvent.card_id == card_id, CardEvent.seq > since_seq)
        .order_by(CardEvent.seq)
        .limit(limit)
    )
    return list((await session.scalars(stmt)).all())


async def retry_card(session: AsyncSession, card_id: str, *, note: str | None = None) -> Card:
    card = await get_card(session, card_id)
    await transitions.retry_card(session, card, note=note)
    await session.flush()
    return card


async def cancel_card(session: AsyncSession, card_id: str) -> Card:
    card = await get_card(session, card_id)
    await transitions.cancel_card(session, card)
    await session.flush()
    return card
