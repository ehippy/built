import logging
import re
from datetime import UTC, datetime

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from built.config import settings
from built.db.models import Card, CardColumnVisit, CardDependency, CardEvent, EpicLink, Project
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


async def list_open_cards(session: AsyncSession, project_id: str, *, limit: int = 50) -> list[Card]:
    """Genuinely open cards — not archived and not in a terminal lifecycle_state —
    newest first. Unlike list_recent_card_titles (last-N by creation date,
    unfiltered by status, used by chat.py), this is what agent/curation.py's
    duplicate-detection needs: a window that can't mix done/failed noise into
    what's actually still live on the board. Returns full Card rows, not just
    titles, since the dedup check needs raw_request too."""
    stmt = (
        select(Card)
        .where(
            Card.project_id == project_id,
            Card.archived_at.is_(None),
            Card.lifecycle_state.not_in([LifecycleState.DONE, LifecycleState.FAILED]),
        )
        .order_by(Card.created_at.desc())
        .limit(limit)
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


async def search_cards(
    session: AsyncSession,
    project_id: str,
    *,
    query: str = "",
    include_archived: bool = False,
    limit: int = 20,
) -> list[Card]:
    """Case-insensitive substring match over title and raw_request, newest first —
    agent/chat.py's search_tickets tool, so the chat LLM can find a card a human
    references by description rather than needing its id up front. Empty query
    lists the most recent cards instead of searching, same idea as
    list_recent_card_titles but returning full rows (with ids) capped much lower,
    since chat renders each match as a line rather than just a title."""
    stmt = select(Card).where(Card.project_id == project_id)
    if not include_archived:
        stmt = stmt.where(Card.archived_at.is_(None))
    if query:
        pattern = f"%{query}%"
        stmt = stmt.where(or_(Card.title.ilike(pattern), Card.raw_request.ilike(pattern)))
    stmt = stmt.order_by(Card.created_at.desc()).limit(limit)
    return list((await session.scalars(stmt)).all())


async def count_column_backlog(session: AsyncSession, project_id: str, column: Column) -> int:
    """How many non-archived cards currently sit in this column for a project —
    what the board's swimlane for that column visually shows. orchestrator/curator.py
    uses this as a WIP-limit gate: no point proposing more PM-column work than the
    (concurrency-capped) orchestrator can realistically work through.

    Excludes epic-parent cards: they're parked at column=PM forever (see
    services/card_service.py's link_epic_child), so without this exclusion every
    surviving epic would inflate the WIP gate permanently, one unit per epic,
    throttling curation more than intended as epics accumulate."""
    stmt = (
        select(func.count())
        .select_from(Card)
        .where(
            Card.project_id == project_id,
            Card.column == column,
            Card.archived_at.is_(None),
            Card.id.not_in(select(EpicLink.parent_card_id)),
        )
    )
    return await session.scalar(stmt) or 0


_PRIORITY_SORT_RANK = {Priority.HIGH: 0, Priority.NORMAL: 1, Priority.LOW: 2}


async def get_board(
    session: AsyncSession, project_id: str, *, include_archived: bool = False
) -> dict[Column, list[Card]]:
    """Cards for a project, grouped by their current column and ordered within each
    column by priority first, then most recent activity — what the dashboard and
    the `/board` API endpoint render. Archived cards are left off by default —
    that's the whole point of archiving something.

    Epic-parent cards are excluded from the ordinary 5-column swimlane entirely —
    they get their own panel (see list_epics) instead of sitting in the PM column
    forever looking like stuck, unscoped work."""
    cards = await list_cards(session, project_id, include_archived=include_archived)
    epic_parent_ids = set(await session.scalars(select(EpicLink.parent_card_id)))
    cards = [c for c in cards if c.id not in epic_parent_ids]
    await _attach_last_activity(session, cards)
    await _attach_dependency_status(session, cards)
    await _attach_epic_parent_title(session, cards)
    await _attach_progress(session, cards, project_id)
    # Two stable sorts: recency first, then priority — the second pass reorders by
    # priority rank while preserving each rank's existing (already recency-sorted)
    # relative order, without needing a combined sort key.
    cards.sort(key=lambda c: c.last_activity_at, reverse=True)
    cards.sort(key=lambda c: _PRIORITY_SORT_RANK[c.priority])
    board: dict[Column, list[Card]] = {column: [] for column in Column}
    for card in cards:
        board[card.column].append(card)
    return board


# --- Epics: a Card that stays alive tracking child cards, created via the PM's --
# --- define_epic (agent/loop.py); "is_epic" is EXISTS(epic_links), not a flag ---


async def link_epic_child(session: AsyncSession, *, parent_card_id: str, child_card_id: str) -> EpicLink:
    """Called once per child right after create_card, by the PM's define_epic
    handler. No nesting: a card can't be both an epic parent and someone's child,
    and a child can't belong to two epics."""
    if await is_epic(session, child_card_id):
        raise ValueError(f"card {child_card_id!r} is already an epic parent — can't also be a child")
    if await get_epic_parent(session, parent_card_id) is not None:
        raise ValueError(f"card {parent_card_id!r} is already an epic child — can't also be a parent")
    existing_parent = await get_epic_parent(session, child_card_id)
    if existing_parent is not None:
        raise ValueError(f"card {child_card_id!r} already belongs to epic {existing_parent.id!r}")
    link = EpicLink(card_id=child_card_id, parent_card_id=parent_card_id)
    session.add(link)
    await session.flush()
    return link


async def is_epic(session: AsyncSession, card_id: str) -> bool:
    stmt = select(EpicLink.card_id).where(EpicLink.parent_card_id == card_id).limit(1)
    return await session.scalar(stmt) is not None


async def get_epic_parent(session: AsyncSession, child_card_id: str) -> Card | None:
    stmt = (
        select(Card)
        .join(EpicLink, EpicLink.parent_card_id == Card.id)
        .where(EpicLink.card_id == child_card_id)
    )
    return await session.scalar(stmt)


async def list_epic_children(
    session: AsyncSession, parent_card_id: str, *, include_archived: bool = False
) -> list[Card]:
    stmt = (
        select(Card)
        .join(EpicLink, EpicLink.card_id == Card.id)
        .where(EpicLink.parent_card_id == parent_card_id)
        .order_by(Card.created_at)
    )
    if not include_archived:
        stmt = stmt.where(Card.archived_at.is_(None))
    children = list((await session.scalars(stmt)).all())
    # macros.html.j2's card_tile (the board's epic panel, and any future caller)
    # reads card.last_activity_at unconditionally — unlike epic_parent_title/
    # unmet_dependency_count, which templates only access inside {% if %} and so
    # tolerate being unset, last_activity_at goes through the |timeago filter,
    # a real Python function that raises on Jinja's Undefined rather than treating
    # it as falsy. Attach it here so every caller gets a renderable Card for free.
    await _attach_last_activity(session, children)
    return children


async def list_epics(session: AsyncSession, project_id: str, *, include_archived: bool = False) -> list[Card]:
    """Every epic-parent card for this project, each annotated with non-persisted
    children/child_total/child_done — one list_epic_children call per epic, an
    unoptimized N+1 done deliberately: epic counts per project are expected to be
    single digits, unlike card counts, so this doesn't need _attach_last_activity's
    bulk-query treatment."""
    stmt = (
        select(Card)
        .join(EpicLink, EpicLink.parent_card_id == Card.id)
        .where(Card.project_id == project_id)
        .distinct()
        .order_by(Card.created_at)
    )
    if not include_archived:
        stmt = stmt.where(Card.archived_at.is_(None))
    epics = list((await session.scalars(stmt)).all())
    for epic in epics:
        children = await list_epic_children(session, epic.id, include_archived=True)
        epic.children = children
        epic.child_total = len(children)
        epic.child_done = sum(1 for c in children if c.lifecycle_state == LifecycleState.DONE)
    return epics


# --- Dependencies: card_id can't be claimed until depends_on_card_id is DONE ----


async def add_dependency(session: AsyncSession, *, card_id: str, depends_on_card_id: str) -> CardDependency:
    """Idempotent (returns the existing row rather than raising on the unique
    constraint) and guarded: no self-dependency, both cards must share a project,
    and no cycle through existing edges. define_epic's own edges never hit the
    cycle guard (depends_on there is restricted to earlier-sibling indices, acyclic
    by construction) — this is the general-purpose entry point other future
    callers (a manual UI control, a later Developer-facing tool) will use."""
    if card_id == depends_on_card_id:
        raise ValueError("a card cannot depend on itself")
    card = await get_card(session, card_id)
    depends_on = await get_card(session, depends_on_card_id)
    if card.project_id != depends_on.project_id:
        raise ValueError("dependencies must be within the same project")

    existing = await session.scalar(
        select(CardDependency).where(
            CardDependency.card_id == card_id, CardDependency.depends_on_card_id == depends_on_card_id
        )
    )
    if existing is not None:
        return existing

    # BFS forward from card_id along "X depends on current" edges (i.e. current
    # is itself a dependency of X): if depends_on_card_id is reachable this way,
    # it already transitively depends on card_id, so adding "card_id depends on
    # depends_on_card_id" would close a cycle back on itself.
    frontier = [card_id]
    seen = {card_id}
    while frontier:
        next_ids = (
            await session.scalars(
                select(CardDependency.card_id).where(CardDependency.depends_on_card_id.in_(frontier))
            )
        ).all()
        frontier = []
        for next_id in next_ids:
            if next_id == depends_on_card_id:
                raise ValueError("would create a dependency cycle")
            if next_id not in seen:
                seen.add(next_id)
                frontier.append(next_id)

    dependency = CardDependency(card_id=card_id, depends_on_card_id=depends_on_card_id)
    session.add(dependency)
    await session.flush()
    return dependency


async def remove_dependency(session: AsyncSession, *, card_id: str, depends_on_card_id: str) -> None:
    dependency = await session.scalar(
        select(CardDependency).where(
            CardDependency.card_id == card_id, CardDependency.depends_on_card_id == depends_on_card_id
        )
    )
    if dependency is not None:
        await session.delete(dependency)
        await session.flush()


async def list_dependencies(session: AsyncSession, card_id: str) -> list[Card]:
    """Cards this one depends on — card-detail's Dependencies section."""
    stmt = (
        select(Card)
        .join(CardDependency, CardDependency.depends_on_card_id == Card.id)
        .where(CardDependency.card_id == card_id)
        .order_by(Card.created_at)
    )
    return list((await session.scalars(stmt)).all())


async def list_dependents(session: AsyncSession, card_id: str) -> list[Card]:
    """Cards blocked on this one — card-detail's "blocks" section."""
    stmt = (
        select(Card)
        .join(CardDependency, CardDependency.card_id == Card.id)
        .where(CardDependency.depends_on_card_id == card_id)
        .order_by(Card.created_at)
    )
    return list((await session.scalars(stmt)).all())


async def _attach_dependency_status(session: AsyncSession, cards: list[Card]) -> None:
    """Sets card.unmet_dependency_count on every card via one grouped query. A
    dependency that's archived-but-not-done is treated as resolved — archiving
    already means "drop out of consideration" everywhere else in this app, and
    without this a human abandoning one piece of a chain would otherwise block its
    dependent forever with no escape hatch. A FAILED (not archived) dependency
    deliberately still blocks — that's a real, unresolved problem to notice, not
    something to silently route around."""
    if not cards:
        return
    stmt = (
        select(CardDependency.card_id, func.count())
        .join(Card, Card.id == CardDependency.depends_on_card_id)
        .where(
            CardDependency.card_id.in_([c.id for c in cards]),
            Card.lifecycle_state != LifecycleState.DONE,
            Card.archived_at.is_(None),
        )
        .group_by(CardDependency.card_id)
    )
    counts = dict((await session.execute(stmt)).all())
    for card in cards:
        card.unmet_dependency_count = counts.get(card.id, 0)


async def _attach_epic_parent_title(session: AsyncSession, cards: list[Card]) -> None:
    """Sets card.epic_parent_title for child cards — the board tile's "· epic: X"
    trailer."""
    if not cards:
        return
    stmt = (
        select(EpicLink.card_id, Card.title)
        .join(Card, Card.id == EpicLink.parent_card_id)
        .where(EpicLink.card_id.in_([c.id for c in cards]))
    )
    titles = dict((await session.execute(stmt)).all())
    for card in cards:
        card.epic_parent_title = titles.get(card.id)


def _describe_llm_response(payload: dict) -> str:
    """One line describing an `llm_response` CardEvent for the board tile's live
    preview — the model's own text if it said anything, else what it's about to do
    with a tool, else a bare "thinking…". Same shape as
    ui/routers/board.py's _describe_curation_event, but that one reads a
    CurationEvent's payload; this reads an llm_response CardEvent's payload
    directly (agent/loop.py's shape: {"iteration", "content", "tool_calls": [name,
    ...]} — just names, no arguments, since arguments live on the separate
    tool_call events emitted afterward)."""
    content = (payload.get("content") or "").strip()
    if content:
        return content
    tool_calls = payload.get("tool_calls") or []
    if tool_calls:
        return f"calling {', '.join(tool_calls)}…"
    return "thinking…"


async def _attach_progress(session: AsyncSession, cards: list[Card], project_id: str) -> None:
    """Sets card.current_iteration/max_iterations/typical_iterations/activity_preview
    on every card currently being worked — the board tile's progress bar plus a
    truncated peek at what the agent is actually saying/doing right now. Scoped to
    each card's currently-open CardColumnVisit specifically (not every llm_response
    event the card has ever produced): a card that's been through, say, three
    Developer revisions has three closed visits plus at most one open one, and
    only the open one's iteration count means anything.

    typical_iterations (not max_iterations) is what the bar is actually drawn
    against: max_iterations_per_run is a rarely-hit safety ceiling (often set an
    order of magnitude above what a normal run takes), so a bar denominated
    against it sits permanently near-empty and tells you nothing at a glance.
    "How many turns does this column *usually* take, for this project" is the
    useful number — the average LLM_RESPONSE count over this project's own closed
    visits in the same column. Falls back to max_iterations when there's no
    history yet (a column this project has never finished a visit in)."""
    in_flight = [c for c in cards if c.is_being_worked]
    if not in_flight:
        return
    project = await session.get(Project, project_id)
    max_iterations = project.max_iterations_per_run if project else settings.default_max_iterations_per_run

    per_visit_counts = (
        select(CardEvent.column_visit_id.label("visit_id"), func.count().label("cnt"))
        .where(CardEvent.type == EventType.LLM_RESPONSE)
        .group_by(CardEvent.column_visit_id)
        .subquery()
    )
    typical_stmt = (
        select(CardColumnVisit.column, func.avg(func.coalesce(per_visit_counts.c.cnt, 0)))
        .join(Card, Card.id == CardColumnVisit.card_id)
        .outerjoin(per_visit_counts, per_visit_counts.c.visit_id == CardColumnVisit.id)
        .where(Card.project_id == project_id, CardColumnVisit.ended_at.is_not(None))
        .group_by(CardColumnVisit.column)
    )
    typical_by_column = dict((await session.execute(typical_stmt)).all())

    visit_stmt = select(CardColumnVisit.card_id, CardColumnVisit.id).where(
        CardColumnVisit.card_id.in_([c.id for c in in_flight]),
        CardColumnVisit.ended_at.is_(None),
    )
    visit_id_by_card = dict((await session.execute(visit_stmt)).all())
    if not visit_id_by_card:
        return
    events_stmt = (
        select(CardEvent.column_visit_id, CardEvent.payload)
        .where(
            CardEvent.column_visit_id.in_(visit_id_by_card.values()),
            CardEvent.type == EventType.LLM_RESPONSE,
        )
        .order_by(CardEvent.seq)
    )
    responses_by_visit: dict[str, list[dict]] = {}
    for visit_id, payload in (await session.execute(events_stmt)).all():
        responses_by_visit.setdefault(visit_id, []).append(payload)
    for card in in_flight:
        responses = responses_by_visit.get(visit_id_by_card.get(card.id), [])
        card.current_iteration = len(responses)
        card.max_iterations = max_iterations
        typical = typical_by_column.get(card.column)
        card.typical_iterations = round(typical) if typical else max_iterations
        card.activity_preview = _describe_llm_response(responses[-1]) if responses else None


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
    # `summary` above is deliberately terse ("a one-line summary for the audit log") —
    # the actual substance of a request_changes/reject is in the transition event's
    # `feedback`, which otherwise only ever reaches the Developer via card.latest_feedback
    # (a single shared field the next reviewer down the line can overwrite). Without this,
    # a Tester/Reviewer re-invoked after its own rejection has no way to check whether its
    # own specific findings were actually addressed — it can only start a generically fresh
    # review and hope it re-derives the same list.
    transition_payload = await session.scalar(
        select(CardEvent.payload)
        .where(CardEvent.column_visit_id == prev_visit.id, CardEvent.type == EventType.TRANSITION)
        .limit(1)
    )
    feedback_text = (transition_payload or {}).get("feedback")
    if feedback_text:
        lines.append(
            f"Your own detailed feedback from that attempt (verify each item was addressed):\n"
            f"{feedback_text}"
        )
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


async def add_nudge(session: AsyncSession, card_id: str, *, note: str) -> Card:
    """Drop a note in for the agent regardless of what it's doing right now —
    unlike retry_card, this needs no blocked/failed state and doesn't touch
    lifecycle_state, column, or any safety-valve counter. A single slot, not a
    queue: a second nudge before the first is seen overwrites it, same as
    retry_note. See agent/loop.py's run_column_visit for where it's consumed."""
    card = await get_card(session, card_id)
    card.pending_nudge = note
    await session.flush()
    logger.info("card %s (%r) nudged: %r", card.id, card.title, note)
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


async def archive_all_in_column(session: AsyncSession, project_id: str, column: Column) -> int:
    """The board's per-column "archive all" action — same per-card semantics as
    archive_card (doesn't touch lifecycle_state or an in-flight claim), just applied
    to every not-yet-archived card in one column at once. Excludes epic-parent cards
    for the same reason count_column_backlog does: they're parked at column=PM
    forever by design, not stuck work waiting to be cleared out. Returns how many
    cards were archived."""
    stmt = select(Card).where(
        Card.project_id == project_id,
        Card.column == column,
        Card.archived_at.is_(None),
        Card.id.not_in(select(EpicLink.parent_card_id)),
    )
    cards = list((await session.scalars(stmt)).all())
    now = datetime.now(UTC)
    for card in cards:
        card.archived_at = now
    await session.flush()
    return len(cards)
