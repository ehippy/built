"""Board view groups DONE cards into a collapsed <details> section per column,
separate from cards still actively moving through the pipeline — see
_board_fragment.html.j2. Added alongside the archiver so a card that just finished
is still visible (collapsed) before the archiver eventually sweeps it away."""

from httpx import ASGITransport, AsyncClient

from built.domain.enums import Column, LifecycleState
from built.main import app
from built.services import card_service, project_service


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _make_project(session, name):
    return await project_service.create_project(
        session, name=name, overarching_goal="goal", repo_remote_url="https://example.invalid/repo.git"
    )


async def test_done_cards_are_collapsed_separately_from_in_flight_cards(db_session):
    project = await _make_project(db_session, "collapse-mixed")
    in_flight = await card_service.create_card(
        db_session, project.id, title="Still deploying", raw_request="r"
    )
    in_flight.column = Column.DEPLOYER
    done = await card_service.create_card(db_session, project.id, title="Already shipped", raw_request="r")
    done.column = Column.DEPLOYER
    done.lifecycle_state = LifecycleState.DONE
    await db_session.commit()

    async with _client() as client:
        page = await client.get(f"/ui/projects/{project.id}/board")

    assert page.status_code == 200
    assert "1 done" in page.text
    assert "Still deploying" in page.text
    assert "Already shipped" in page.text
    assert 'data-column="deployer"' in page.text
    # The done card's tile sits after the "1 done" summary, inside the <details>.
    assert page.text.index("1 done") < page.text.index("Already shipped")


async def test_empty_column_still_shows_no_cards_message(db_session):
    project = await _make_project(db_session, "collapse-empty")
    await db_session.commit()

    async with _client() as client:
        page = await client.get(f"/ui/projects/{project.id}/board")

    assert page.status_code == 200
    assert page.text.count("No cards") == 5  # every column genuinely empty


async def test_column_with_only_done_cards_has_no_empty_message(db_session):
    project = await _make_project(db_session, "collapse-only-done")
    done = await card_service.create_card(db_session, project.id, title="Finished thing", raw_request="r")
    done.column = Column.DEPLOYER
    done.lifecycle_state = LifecycleState.DONE
    await db_session.commit()

    async with _client() as client:
        page = await client.get(f"/ui/projects/{project.id}/board")

    assert page.status_code == 200
    assert "1 done" in page.text
    # pm/developer/tester/reviewer are genuinely empty; deployer isn't (one collapsed
    # done card), so "No cards" shows for exactly the other 4 columns.
    assert page.text.count("No cards") == 4
