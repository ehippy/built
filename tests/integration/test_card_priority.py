"""A human's manual "bless this as important" signal — orthogonal to column and
lifecycle_state. See orchestrator/worker.py's _CLAIM_PRIORITY_ORDER for how it
affects which card the orchestrator claims next; this file covers the CRUD/display
side (defaults, setting it, and board ordering)."""

from httpx import ASGITransport, AsyncClient

from built.domain.enums import Priority
from built.main import app
from built.services import card_service, project_service

AUTH = {"X-API-Key": "test-api-key"}


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _make_project(session, **overrides):
    defaults = {
        "name": f"priority-{overrides.pop('_n', 'x')}",
        "overarching_goal": "goal",
        "repo_remote_url": "https://example.invalid/repo.git",
    }
    defaults.update(overrides)
    return await project_service.create_project(session, **defaults)


async def test_new_card_defaults_to_normal_priority(db_session):
    project = await _make_project(db_session, _n="1")

    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")

    assert card.priority == Priority.NORMAL


async def test_create_card_with_explicit_priority(db_session):
    project = await _make_project(db_session, _n="2")

    card = await card_service.create_card(
        db_session, project.id, title="t", raw_request="r", priority=Priority.HIGH
    )

    assert card.priority == Priority.HIGH


async def test_set_priority_updates_the_card(db_session):
    project = await _make_project(db_session, _n="3")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    assert card.priority == Priority.NORMAL

    updated = await card_service.set_priority(db_session, card.id, Priority.LOW)

    assert updated.priority == Priority.LOW


async def test_board_sorts_high_priority_first_regardless_of_recency(db_session):
    """last_activity_at would otherwise put "older" last, since get_board's
    recency pass runs newest-first — priority must win over that ordering."""
    project = await _make_project(db_session, _n="4")
    older_high = await card_service.create_card(
        db_session, project.id, title="older, high", raw_request="r", priority=Priority.HIGH
    )
    newer_normal = await card_service.create_card(
        db_session, project.id, title="newer, normal", raw_request="r"
    )
    assert older_high.created_at <= newer_normal.created_at

    board = await card_service.get_board(db_session, project.id)

    assert [c.id for c in board[older_high.column]] == [older_high.id, newer_normal.id]


async def test_low_priority_sorts_last_even_when_most_recently_active(db_session):
    project = await _make_project(db_session, _n="5")
    normal_card = await card_service.create_card(db_session, project.id, title="normal", raw_request="r")
    low_card = await card_service.create_card(
        db_session, project.id, title="low, more recent", raw_request="r", priority=Priority.LOW
    )
    assert normal_card.created_at <= low_card.created_at

    board = await card_service.get_board(db_session, project.id)

    assert [c.id for c in board[normal_card.column]] == [normal_card.id, low_card.id]


async def test_ui_priority_round_trip(db_session):
    project = await _make_project(db_session, _n="6")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await db_session.commit()

    async with _client() as client:
        detail = await client.get(f"/ui/cards/{card.id}")
        assert "high priority" not in detail.text

        resp = await client.post(
            f"/ui/cards/{card.id}/priority", data={"priority": "high"}, follow_redirects=False
        )
        assert resp.status_code == 303

        detail_after = await client.get(f"/ui/cards/{card.id}")
        assert "high priority" in detail_after.text

        board = await client.get(f"/ui/projects/{project.id}/board")
        assert "high" in board.text
