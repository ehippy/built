import asyncio

from httpx import ASGITransport, AsyncClient

from built.api.routers import projects as projects_router
from built.main import app

AUTH = {"X-API-Key": "test-api-key"}


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_create_project_requires_api_key():
    async with _client() as client:
        response = await client.post(
            "/api/v1/projects",
            json={
                "name": "No Auth",
                "overarching_goal": "goal",
                "repo_remote_url": "https://example.invalid/repo.git",
            },
        )
    assert response.status_code == 401


async def test_project_and_card_lifecycle_via_api():
    async with _client() as client:
        create_resp = await client.post(
            "/api/v1/projects",
            json={
                "name": "API Project",
                "overarching_goal": "Build a thing via the API.",
                "repo_remote_url": "https://example.invalid/api-project.git",
            },
            headers=AUTH,
        )
        assert create_resp.status_code == 201, create_resp.text
        project = create_resp.json()
        assert project["slug"] == "api-project"
        assert project["deploy_config"] is None

        get_resp = await client.get(f"/api/v1/projects/{project['id']}")
        assert get_resp.status_code == 200
        assert get_resp.json()["name"] == "API Project"

        deploy_resp = await client.put(
            f"/api/v1/projects/{project['id']}/deploy-config",
            json={"kind": "command", "command": "true", "timeout_seconds": 30},
            headers=AUTH,
        )
        assert deploy_resp.status_code == 200, deploy_resp.text
        assert deploy_resp.json()["kind"] == "command"

        reread_resp = await client.get(f"/api/v1/projects/{project['id']}")
        assert reread_resp.json()["deploy_config"]["command"] == "true"

        card_resp = await client.post(
            f"/api/v1/projects/{project['id']}/cards",
            json={"title": "Add a health check", "raw_request": "add /healthz"},
            headers=AUTH,
        )
        assert card_resp.status_code == 201, card_resp.text
        card = card_resp.json()
        assert card["column"] == "pm"
        assert card["lifecycle_state"] == "active"
        assert card["priority"] == "normal"
        assert card["branch_name"] == f"card/{card['id'][:8]}-add-a-health-check"

        priority_resp = await client.post(
            f"/api/v1/cards/{card['id']}/priority", headers=AUTH, json={"priority": "high"}
        )
        assert priority_resp.status_code == 200, priority_resp.text
        assert priority_resp.json()["priority"] == "high"

        board_resp = await client.get(f"/api/v1/projects/{project['id']}/board")
        assert board_resp.status_code == 200
        board = board_resp.json()
        assert [c["id"] for c in board["pm"]] == [card["id"]]
        assert board["developer"] == []

        events_resp = await client.get(f"/api/v1/cards/{card['id']}/events")
        assert events_resp.status_code == 200
        events = events_resp.json()
        assert any(e["payload"].get("action") == "created" for e in events)

        # Cancel is allowed on an ACTIVE card...
        cancel_resp = await client.post(f"/api/v1/cards/{card['id']}/cancel", headers=AUTH)
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["lifecycle_state"] == "failed"

        # ...retry un-sticks it...
        retry_resp = await client.post(f"/api/v1/cards/{card['id']}/retry", headers=AUTH)
        assert retry_resp.status_code == 200
        assert retry_resp.json()["lifecycle_state"] == "active"

        # ...but retrying an already-ACTIVE card is rejected, not silently accepted.
        second_retry_resp = await client.post(f"/api/v1/cards/{card['id']}/retry", headers=AUTH)
        assert second_retry_resp.status_code == 409

        # A note attached to a retry is stored on the card for the next visit to see.
        await client.post(f"/api/v1/cards/{card['id']}/cancel", headers=AUTH)
        noted_retry_resp = await client.post(
            f"/api/v1/cards/{card['id']}/retry", headers=AUTH, json={"note": "rebase onto main first"}
        )
        assert noted_retry_resp.status_code == 200
        assert noted_retry_resp.json()["retry_note"] == "rebase onto main first"

        # Basic task editing: fix the title/request, then archive it off the board.
        edit_resp = await client.patch(
            f"/api/v1/cards/{card['id']}",
            headers=AUTH,
            json={"title": "Add a health check (v2)", "raw_request": "add /healthz, return 200"},
        )
        assert edit_resp.status_code == 200
        assert edit_resp.json()["title"] == "Add a health check (v2)"

        archive_resp = await client.post(f"/api/v1/cards/{card['id']}/archive", headers=AUTH)
        assert archive_resp.status_code == 200
        assert archive_resp.json()["archived_at"] is not None

        board_after_archive = await client.get(f"/api/v1/projects/{project['id']}/board")
        assert board_after_archive.json()["pm"] == []

        board_with_archived = await client.get(
            f"/api/v1/projects/{project['id']}/board", params={"include_archived": True}
        )
        assert [c["id"] for c in board_with_archived.json()["pm"]] == [card["id"]]

        unarchive_resp = await client.post(f"/api/v1/cards/{card['id']}/unarchive", headers=AUTH)
        assert unarchive_resp.status_code == 200
        assert unarchive_resp.json()["archived_at"] is None


async def test_curate_endpoint_requires_auth_and_starts_in_background(monkeypatch):
    started = asyncio.Event()

    async def fake_curation_activity(*_args, **_kwargs):
        started.set()

    monkeypatch.setattr(projects_router, "run_curation_activity", fake_curation_activity)

    async with _client() as client:
        create_resp = await client.post(
            "/api/v1/projects",
            json={
                "name": "Curation API Project",
                "overarching_goal": "goal",
                "repo_remote_url": "https://example.invalid/curation-project.git",
            },
            headers=AUTH,
        )
        project = create_resp.json()

        no_auth_resp = await client.post(f"/api/v1/projects/{project['id']}/curate/bug_sweep")
        assert no_auth_resp.status_code == 401

        started_resp = await client.post(f"/api/v1/projects/{project['id']}/curate/bug_sweep", headers=AUTH)
        assert started_resp.status_code == 202
        assert started_resp.json() == {"status": "started"}
        async with asyncio.timeout(5):
            await started.wait()

        invalid_kind_resp = await client.post(
            f"/api/v1/projects/{project['id']}/curate/not-a-real-kind", headers=AUTH
        )
        assert invalid_kind_resp.status_code == 422

        missing_resp = await client.post("/api/v1/projects/does-not-exist/curate/bug_sweep", headers=AUTH)
        assert missing_resp.status_code == 404


async def test_pause_and_resume_project_via_api():
    async with _client() as client:
        create_resp = await client.post(
            "/api/v1/projects",
            json={
                "name": "Pause API Project",
                "overarching_goal": "goal",
                "repo_remote_url": "https://example.invalid/pause-project.git",
            },
            headers=AUTH,
        )
        project = create_resp.json()
        assert project["paused_at"] is None

        no_auth_resp = await client.post(f"/api/v1/projects/{project['id']}/pause")
        assert no_auth_resp.status_code == 401

        pause_resp = await client.post(f"/api/v1/projects/{project['id']}/pause", headers=AUTH)
        assert pause_resp.status_code == 200
        assert pause_resp.json()["paused_at"] is not None

        reread_resp = await client.get(f"/api/v1/projects/{project['id']}")
        assert reread_resp.json()["paused_at"] is not None

        resume_resp = await client.post(f"/api/v1/projects/{project['id']}/resume", headers=AUTH)
        assert resume_resp.status_code == 200
        assert resume_resp.json()["paused_at"] is None

        missing_resp = await client.post("/api/v1/projects/does-not-exist/pause", headers=AUTH)
        assert missing_resp.status_code == 404


async def test_project_role_guidance_round_trips_via_api():
    async with _client() as client:
        create_resp = await client.post(
            "/api/v1/projects",
            json={
                "name": "Guidance Project",
                "overarching_goal": "goal",
                "repo_remote_url": "https://example.invalid/guidance-project.git",
                "pm_guidance": "File tickets in small batches.",
            },
            headers=AUTH,
        )
        assert create_resp.status_code == 201, create_resp.text
        project = create_resp.json()
        assert project["pm_guidance"] == "File tickets in small batches."
        assert project["developer_guidance"] is None

        update_resp = await client.patch(
            f"/api/v1/projects/{project['id']}",
            json={
                "developer_guidance": "Never touch the legacy billing module.",
                "reviewer_guidance": "Reject anything that adds a new npm dependency.",
            },
            headers=AUTH,
        )
        assert update_resp.status_code == 200, update_resp.text
        updated = update_resp.json()
        assert updated["pm_guidance"] == "File tickets in small batches."
        assert updated["developer_guidance"] == "Never touch the legacy billing module."
        assert updated["reviewer_guidance"] == "Reject anything that adds a new npm dependency."

        reread_resp = await client.get(f"/api/v1/projects/{project['id']}")
        assert reread_resp.json()["developer_guidance"] == "Never touch the legacy billing module."


async def test_endpoint_config_fallback_chain_resolution():
    async with _client() as client:
        project_resp = await client.post(
            "/api/v1/projects",
            json={
                "name": "Chain Project",
                "overarching_goal": "goal",
                "repo_remote_url": "https://example.invalid/chain.git",
            },
            headers=AUTH,
        )
        project_id = project_resp.json()["id"]

        # Global default, applies to every role.
        await client.post(
            "/api/v1/endpoint-configs",
            json={"base_url": "https://global.invalid/v1", "model": "global-model", "priority": 0},
            headers=AUTH,
        )
        # Project-specific override for the developer role: a two-entry fallback chain.
        await client.post(
            "/api/v1/endpoint-configs",
            json={
                "project_id": project_id,
                "role": "developer",
                "base_url": "https://primary.invalid/v1",
                "model": "strong-model",
                "priority": 0,
            },
            headers=AUTH,
        )
        await client.post(
            "/api/v1/endpoint-configs",
            json={
                "project_id": project_id,
                "role": "developer",
                "base_url": "https://fallback.invalid/v1",
                "model": "backup-model",
                "priority": 1,
            },
            headers=AUTH,
        )

        dev_chain_resp = await client.get(f"/api/v1/projects/{project_id}/endpoint-chain/developer")
        dev_chain = dev_chain_resp.json()
        assert [e["base_url"] for e in dev_chain] == [
            "https://primary.invalid/v1",
            "https://fallback.invalid/v1",
        ]

        # No project+role or project-wide config for "tester" -> falls through to the global default.
        tester_chain_resp = await client.get(f"/api/v1/projects/{project_id}/endpoint-chain/tester")
        tester_chain = tester_chain_resp.json()
        assert [e["base_url"] for e in tester_chain] == ["https://global.invalid/v1"]
