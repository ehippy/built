"""The human-in-the-loop touchpoint that doesn't need the card blocked/failed
first: a nudge can be dropped in on any active card, regardless of whether a
visit is running right now. See agent/loop.py's run_column_visit for where a
running visit actually picks it up (covered separately in
test_visit_nudge.py) — this file covers the CRUD/display side, same split as
test_card_priority.py."""

from httpx import ASGITransport, AsyncClient

from built.main import app
from built.services import card_service, project_service

AUTH = {"X-API-Key": "test-api-key"}


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _make_project(session, **overrides):
    defaults = {
        "name": f"nudge-{overrides.pop('_n', 'x')}",
        "overarching_goal": "goal",
        "repo_remote_url": "https://example.invalid/repo.git",
    }
    defaults.update(overrides)
    return await project_service.create_project(session, **defaults)


async def test_new_card_has_no_pending_nudge(db_session):
    project = await _make_project(db_session, _n="1")

    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")

    assert card.pending_nudge is None


async def test_add_nudge_sets_the_note(db_session):
    project = await _make_project(db_session, _n="2")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")

    updated = await card_service.add_nudge(db_session, card.id, note="watch out for the rate limiter")

    assert updated.pending_nudge == "watch out for the rate limiter"


async def test_second_nudge_overwrites_the_first_unread_one(db_session):
    project = await _make_project(db_session, _n="3")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")

    await card_service.add_nudge(db_session, card.id, note="first")
    updated = await card_service.add_nudge(db_session, card.id, note="second")

    assert updated.pending_nudge == "second"


async def test_ui_nudge_round_trip(db_session):
    project = await _make_project(db_session, _n="4")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await db_session.commit()

    async with _client() as client:
        detail = await client.get(f"/ui/cards/{card.id}")
        assert "Pending nudge" not in detail.text

        resp = await client.post(
            f"/ui/cards/{card.id}/nudge", data={"note": "double-check the migration"}, follow_redirects=False
        )
        assert resp.status_code == 303

        detail_after = await client.get(f"/ui/cards/{card.id}")
        assert "Pending nudge" in detail_after.text
        assert "double-check the migration" in detail_after.text


async def test_api_nudge_requires_auth(db_session):
    project = await _make_project(db_session, _n="5")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await db_session.commit()

    async with _client() as client:
        unauthed = await client.post(f"/api/v1/cards/{card.id}/nudge", json={"note": "x"})
        assert unauthed.status_code == 401

        resp = await client.post(f"/api/v1/cards/{card.id}/nudge", json={"note": "x"}, headers=AUTH)
        assert resp.status_code == 200
        assert resp.json()["pending_nudge"] == "x"


async def test_api_nudge_missing_card_is_404(db_session):
    async with _client() as client:
        resp = await client.post("/api/v1/cards/does-not-exist/nudge", json={"note": "x"}, headers=AUTH)
        assert resp.status_code == 404
