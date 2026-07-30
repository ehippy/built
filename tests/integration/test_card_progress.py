"""The board tile's per-card progress bar and activity preview —
services/card_service.py's _attach_progress, wired into get_board. The bar is
denominated against typical_iterations (this project's own historical average
turns for that column), not the rarely-hit max_iterations_per_run hard cap."""

from datetime import UTC, datetime, timedelta

from built.db.models import CardColumnVisit
from built.domain.enums import Column, EventType
from built.domain.events import append_event
from built.services import card_service, project_service


async def _make_project(session, **overrides):
    defaults = {
        "name": f"progress-{overrides.pop('_n', 'x')}",
        "overarching_goal": "goal",
        "repo_remote_url": "https://example.invalid/repo.git",
        "max_iterations_per_run": 10,
    }
    defaults.update(overrides)
    return await project_service.create_project(session, **defaults)


async def _claim(card):
    card.claimed_by_worker_id = "worker-a"
    card.lease_expires_at = datetime.now(UTC) + timedelta(minutes=5)


async def _open_visit(session, card, *, column=Column.DEVELOPER) -> CardColumnVisit:
    card.column = column  # keep the card's own column in sync with the visit it's in
    visit = CardColumnVisit(card_id=card.id, column=column)
    session.add(visit)
    await session.flush()
    return visit


async def test_current_iteration_counts_llm_responses_on_the_open_visit(db_session):
    project = await _make_project(db_session, _n="1")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await _claim(card)
    visit = await _open_visit(db_session, card)
    for i in range(3):
        await append_event(
            db_session,
            card_id=card.id,
            column_visit_id=visit.id,
            type=EventType.LLM_RESPONSE,
            payload={"iteration": i + 1, "content": f"turn {i + 1}", "tool_calls": []},
        )

    board = await card_service.get_board(db_session, project.id)

    [tile] = board[Column.DEVELOPER]
    assert tile.current_iteration == 3
    assert tile.max_iterations == 10
    assert tile.typical_iterations == 10  # no closed visits yet — falls back to the hard cap
    assert tile.activity_preview == "turn 3"


async def test_activity_preview_falls_back_to_tool_calls_then_thinking(db_session):
    project = await _make_project(db_session, _n="2")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await _claim(card)
    visit = await _open_visit(db_session, card)
    await append_event(
        db_session,
        card_id=card.id,
        column_visit_id=visit.id,
        type=EventType.LLM_RESPONSE,
        payload={"iteration": 1, "content": "", "tool_calls": ["read_file"]},
    )

    board = await card_service.get_board(db_session, project.id)

    [tile] = board[Column.DEVELOPER]
    assert tile.activity_preview == "calling read_file…"


async def test_no_llm_response_yet_gives_zero_iterations_and_no_preview(db_session):
    project = await _make_project(db_session, _n="3")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await _claim(card)
    await _open_visit(db_session, card)

    board = await card_service.get_board(db_session, project.id)

    [tile] = board[Column.DEVELOPER]
    assert tile.current_iteration == 0
    assert tile.max_iterations == 10
    assert tile.typical_iterations == 10
    assert tile.activity_preview is None


async def test_progress_ignores_closed_visits_from_earlier_revisions(db_session):
    """A card on its second Developer visit (after a Tester bounce-back) must not
    have its iteration count inflated by the first, already-closed visit's events."""
    project = await _make_project(db_session, _n="4")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await _claim(card)
    closed_visit = await _open_visit(db_session, card)
    for i in range(5):
        await append_event(
            db_session,
            card_id=card.id,
            column_visit_id=closed_visit.id,
            type=EventType.LLM_RESPONSE,
            payload={"iteration": i + 1, "content": "old attempt", "tool_calls": []},
        )
    closed_visit.ended_at = datetime.now(UTC)
    open_visit = await _open_visit(db_session, card)
    await append_event(
        db_session,
        card_id=card.id,
        column_visit_id=open_visit.id,
        type=EventType.LLM_RESPONSE,
        payload={"iteration": 1, "content": "new attempt", "tool_calls": []},
    )

    board = await card_service.get_board(db_session, project.id)

    [tile] = board[Column.DEVELOPER]
    assert tile.current_iteration == 1
    assert tile.activity_preview == "new attempt"
    # The bar's denominator, in contrast, *should* reflect that one closed visit —
    # "how long does Developer usually take" is exactly what a closed visit tells you.
    assert tile.typical_iterations == 5


async def test_typical_iterations_averages_this_projects_closed_visits_in_the_column(db_session):
    project = await _make_project(db_session, _n="6")
    finished_card_a = await card_service.create_card(db_session, project.id, title="a", raw_request="r")
    visit_a = await _open_visit(db_session, finished_card_a)
    for i in range(4):
        await append_event(
            db_session,
            card_id=finished_card_a.id,
            column_visit_id=visit_a.id,
            type=EventType.LLM_RESPONSE,
            payload={"iteration": i + 1, "content": "x", "tool_calls": []},
        )
    visit_a.ended_at = datetime.now(UTC)

    finished_card_b = await card_service.create_card(db_session, project.id, title="b", raw_request="r")
    visit_b = await _open_visit(db_session, finished_card_b)
    for i in range(6):
        await append_event(
            db_session,
            card_id=finished_card_b.id,
            column_visit_id=visit_b.id,
            type=EventType.LLM_RESPONSE,
            payload={"iteration": i + 1, "content": "x", "tool_calls": []},
        )
    visit_b.ended_at = datetime.now(UTC)

    in_flight_card = await card_service.create_card(db_session, project.id, title="c", raw_request="r")
    await _claim(in_flight_card)
    await _open_visit(db_session, in_flight_card)

    board = await card_service.get_board(db_session, project.id)

    [tile] = [c for c in board[Column.DEVELOPER] if c.id == in_flight_card.id]
    assert tile.typical_iterations == 5  # average of 4 and 6


async def test_typical_iterations_is_scoped_to_column(db_session):
    """A column that historically takes a lot of turns (say, Developer) must not
    inflate the typical-turns estimate for a column that's normally much quicker
    (say, Reviewer) — averages are computed per column, not project-wide."""
    project = await _make_project(db_session, _n="7")
    long_running_card = await card_service.create_card(db_session, project.id, title="a", raw_request="r")
    long_visit = await _open_visit(db_session, long_running_card, column=Column.DEVELOPER)
    for i in range(50):
        await append_event(
            db_session,
            card_id=long_running_card.id,
            column_visit_id=long_visit.id,
            type=EventType.LLM_RESPONSE,
            payload={"iteration": i + 1, "content": "x", "tool_calls": []},
        )
    long_visit.ended_at = datetime.now(UTC)

    in_flight_card = await card_service.create_card(db_session, project.id, title="b", raw_request="r")
    await _claim(in_flight_card)
    await _open_visit(db_session, in_flight_card, column=Column.REVIEWER)

    board = await card_service.get_board(db_session, project.id)

    [tile] = board[Column.REVIEWER]
    assert tile.typical_iterations == 10  # falls back to the hard cap, unaffected by Developer's 50


async def test_progress_not_attached_when_card_is_not_being_worked(db_session):
    project = await _make_project(db_session, _n="5")
    await card_service.create_card(db_session, project.id, title="t", raw_request="r")

    board = await card_service.get_board(db_session, project.id)

    [tile] = board[Column.PM]
    assert getattr(tile, "current_iteration", None) is None
