"""last_activity_at — card_service._attach_last_activity keeps the board tile and
the card detail header showing genuinely recent activity, not just the last time
the Card row itself was written. That row only changes on a column transition or
claim/release; an agent can run tool calls for minutes with no Card-row write in
between, which is exactly the "9m ago on the board, 'just now' inside the card"
staleness this was added to fix."""

from datetime import UTC, datetime, timedelta

from built.db.models import CardEvent
from built.domain.enums import EventType
from built.services import card_service, project_service


async def _make_card(session):
    project = await project_service.create_project(
        session,
        name=f"last-activity-{id(session)}",
        overarching_goal="goal",
        repo_remote_url="https://example.invalid/repo.git",
    )
    card = await card_service.create_card(session, project.id, title="t", raw_request="r")
    await session.commit()
    return project, card


async def test_get_card_without_the_flag_does_not_attach_last_activity(db_session):
    _, card = await _make_card(db_session)
    fetched = await card_service.get_card(db_session, card.id)
    assert not hasattr(fetched, "last_activity_at")


async def test_get_card_reflects_a_recent_event_even_though_updated_at_is_stale(db_session):
    """The exact scenario reported: a card sits mid-run in an active column, its row
    hasn't been touched in minutes, but a tool_call event landed moments ago."""
    _, card = await _make_card(db_session)
    stale_updated_at = card.updated_at
    recent = datetime.now(UTC) + timedelta(minutes=5)
    db_session.add(
        CardEvent(
            card_id=card.id,
            seq=999,
            type=EventType.TOOL_CALL,
            payload={"name": "bash", "arguments": {}, "commit_sha": None},
            created_at=recent,
        )
    )
    await db_session.commit()

    fetched = await card_service.get_card(db_session, card.id, with_last_activity=True)
    assert fetched.last_activity_at == recent
    assert fetched.last_activity_at > stale_updated_at


async def test_get_card_falls_back_to_updated_at_when_it_is_the_more_recent_change(db_session):
    _, card = await _make_card(db_session)
    card.title = "renamed"
    await db_session.commit()

    fetched = await card_service.get_card(db_session, card.id, with_last_activity=True)
    assert fetched.last_activity_at == fetched.updated_at


async def test_get_board_attaches_last_activity_to_every_card(db_session):
    project, _ = await _make_card(db_session)
    board = await card_service.get_board(db_session, project.id)
    all_cards = [c for cards in board.values() for c in cards]
    assert len(all_cards) == 1
    assert hasattr(all_cards[0], "last_activity_at")
