"""Pause/resume: the project_service functions, and the UI form endpoints that
drive them from the dashboard."""

from httpx import ASGITransport, AsyncClient

from built.main import app
from built.services import project_service


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _make_project(session, **overrides):
    defaults = {
        "name": f"pause-{overrides.pop('_n', 'x')}",
        "overarching_goal": "goal",
        "repo_remote_url": "https://example.invalid/repo.git",
    }
    defaults.update(overrides)
    return await project_service.create_project(session, **defaults)


async def test_pause_project_sets_paused_at(db_session):
    project = await _make_project(db_session, _n="1")
    assert project.paused_at is None

    paused = await project_service.pause_project(db_session, project.id)

    assert paused.paused_at is not None


async def test_resume_project_clears_paused_at(db_session):
    project = await _make_project(db_session, _n="2")
    await project_service.pause_project(db_session, project.id)

    resumed = await project_service.resume_project(db_session, project.id)

    assert resumed.paused_at is None


async def test_ui_pause_and_resume_round_trip(db_session):
    project = await _make_project(db_session, _n="3")
    await db_session.commit()

    async with _client() as client:
        pause_resp = await client.post(f"/ui/projects/{project.id}/pause", follow_redirects=False)
        assert pause_resp.status_code == 303

        list_resp = await client.get("/ui/projects/fragment")
        assert "paused" in list_resp.text
        assert f"/ui/projects/{project.id}/resume" in list_resp.text

        resume_resp = await client.post(f"/ui/projects/{project.id}/resume", follow_redirects=False)
        assert resume_resp.status_code == 303

        list_resp_after = await client.get("/ui/projects/fragment")
        assert f"/ui/projects/{project.id}/pause" in list_resp_after.text


async def test_board_page_reflects_paused_state(db_session):
    """The pause/resume control lives on the board page, not settings — see
    board.html.j2's header. Settings dropped its own copy of this control
    entirely rather than duplicating it."""
    project = await _make_project(db_session, _n="4")
    await db_session.commit()

    async with _client() as client:
        before = await client.get(f"/ui/projects/{project.id}/board")
        assert f'action="/ui/projects/{project.id}/pause"' in before.text
        assert f'action="/ui/projects/{project.id}/resume"' not in before.text

        await client.post(f"/ui/projects/{project.id}/pause")

        after = await client.get(f"/ui/projects/{project.id}/board")
        assert f'action="/ui/projects/{project.id}/resume"' in after.text
        assert f'action="/ui/projects/{project.id}/pause"' not in after.text

        settings = await client.get(f"/ui/projects/{project.id}/settings")
        assert f'action="/ui/projects/{project.id}/pause"' not in settings.text
        assert f'action="/ui/projects/{project.id}/resume"' not in settings.text
