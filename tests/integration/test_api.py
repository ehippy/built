from httpx import ASGITransport, AsyncClient

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
        assert card["branch_name"] == f"card/{card['id'][:8]}-add-a-health-check"

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
