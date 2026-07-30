"""Basic task-editing tools: editing a card's title/request, and archiving/
unarchiving to get it off the board without deleting its history."""

from httpx import ASGITransport, AsyncClient

from built.domain.enums import Column
from built.main import app
from built.services import card_service, project_service


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _make_project(session, **overrides):
    defaults = {
        "name": f"archive-{overrides.pop('_n', 'x')}",
        "overarching_goal": "goal",
        "repo_remote_url": "https://example.invalid/repo.git",
    }
    defaults.update(overrides)
    return await project_service.create_project(session, **defaults)


async def test_update_card_edits_title_and_request(db_session):
    project = await _make_project(db_session, _n="1")
    card = await card_service.create_card(db_session, project.id, title="old", raw_request="old request")

    updated = await card_service.update_card(db_session, card.id, title="new", raw_request="new request")

    assert updated.title == "new"
    assert updated.raw_request == "new request"


async def test_archive_card_sets_archived_at(db_session):
    project = await _make_project(db_session, _n="2")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")

    archived = await card_service.archive_card(db_session, card.id)

    assert archived.archived_at is not None


async def test_unarchive_card_clears_archived_at(db_session):
    project = await _make_project(db_session, _n="3")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await card_service.archive_card(db_session, card.id)

    unarchived = await card_service.unarchive_card(db_session, card.id)

    assert unarchived.archived_at is None


async def test_archived_card_excluded_from_list_and_board_by_default(db_session):
    project = await _make_project(db_session, _n="4")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await card_service.archive_card(db_session, card.id)

    cards = await card_service.list_cards(db_session, project.id)
    board = await card_service.get_board(db_session, project.id)

    assert cards == []
    assert all(c.id != card.id for column_cards in board.values() for c in column_cards)


async def test_archived_card_included_when_requested(db_session):
    project = await _make_project(db_session, _n="5")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await card_service.archive_card(db_session, card.id)

    cards = await card_service.list_cards(db_session, project.id, include_archived=True)
    board = await card_service.get_board(db_session, project.id, include_archived=True)

    assert [c.id for c in cards] == [card.id]
    assert any(c.id == card.id for column_cards in board.values() for c in column_cards)


async def test_ui_edit_archive_unarchive_round_trip(db_session):
    project = await _make_project(db_session, _n="6")
    card = await card_service.create_card(db_session, project.id, title="old", raw_request="old request")
    await db_session.commit()

    async with _client() as client:
        edit_resp = await client.post(
            f"/ui/cards/{card.id}/edit",
            data={"title": "new title", "raw_request": "new request"},
            follow_redirects=False,
        )
        assert edit_resp.status_code == 303

        detail = await client.get(f"/ui/cards/{card.id}")
        assert "new title" in detail.text

        archive_resp = await client.post(f"/ui/cards/{card.id}/archive", follow_redirects=False)
        assert archive_resp.status_code == 303

        board = await client.get(f"/ui/projects/{project.id}/board")
        assert "new title" not in board.text

        board_with_archived = await client.get(f"/ui/projects/{project.id}/board?show_archived=true")
        assert "new title" in board_with_archived.text

        unarchive_resp = await client.post(f"/ui/cards/{card.id}/unarchive", follow_redirects=False)
        assert unarchive_resp.status_code == 303

        board_after = await client.get(f"/ui/projects/{project.id}/board")
        assert "new title" in board_after.text


async def test_archive_all_in_column_only_archives_that_columns_non_archived_cards(db_session):
    project = await _make_project(db_session, _n="7")
    pm1 = await card_service.create_card(db_session, project.id, title="pm1", raw_request="r")
    pm2 = await card_service.create_card(db_session, project.id, title="pm2", raw_request="r")
    already_archived = await card_service.create_card(db_session, project.id, title="pm3", raw_request="r")
    await card_service.archive_card(db_session, already_archived.id)
    other_column = await card_service.create_card(db_session, project.id, title="dev1", raw_request="r")
    other_column.column = Column.DEVELOPER
    await db_session.flush()

    count = await card_service.archive_all_in_column(db_session, project.id, Column.PM)

    assert count == 2  # already_archived wasn't touched again, other_column is a different column
    board = await card_service.get_board(db_session, project.id, include_archived=True)
    archived_ids = {c.id for c in board[Column.PM] if c.archived_at is not None}
    assert archived_ids == {pm1.id, pm2.id, already_archived.id}
    assert other_column.archived_at is None


async def test_archive_all_in_column_excludes_epic_parents(db_session):
    project = await _make_project(db_session, _n="8")
    epic = await card_service.create_card(db_session, project.id, title="epic", raw_request="r")
    child = await card_service.create_card(db_session, project.id, title="child", raw_request="r")
    await card_service.link_epic_child(db_session, parent_card_id=epic.id, child_card_id=child.id)

    count = await card_service.archive_all_in_column(db_session, project.id, Column.PM)

    assert count == 1  # only the child, not the epic parent
    assert child.archived_at is not None
    assert epic.archived_at is None


async def test_ui_archive_all_in_column(db_session):
    project = await _make_project(db_session, _n="9")
    card = await card_service.create_card(db_session, project.id, title="to be archived", raw_request="r")
    await db_session.commit()

    async with _client() as client:
        resp = await client.post(
            f"/ui/projects/{project.id}/board/columns/pm/archive-all", follow_redirects=False
        )
        assert resp.status_code == 303

        board = await client.get(f"/ui/projects/{project.id}/board")
        assert "to be archived" not in board.text

        board_with_archived = await client.get(f"/ui/projects/{project.id}/board?show_archived=true")
        assert "to be archived" in board_with_archived.text
