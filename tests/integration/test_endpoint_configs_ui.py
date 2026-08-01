from httpx import ASGITransport, AsyncClient

import built.llm.client as llm_client_module
from built.main import app

AUTH = {"X-API-Key": "test-api-key"}


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _create_via_api(client: AsyncClient, **overrides) -> str:
    payload = {
        "base_url": "https://original.invalid/v1",
        "model": "original-model",
        "role": "developer",
        "priority": 1,
        "api_key_ref": "SOME_KEY",
        "max_concurrency": 2,
        "context_window": 32000,
        "supports_tool_calling": True,
    }
    payload.update(overrides)
    resp = await client.post("/api/v1/endpoint-configs", json=payload, headers=AUTH)
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


async def _reread(client: AsyncClient, endpoint_id: str) -> dict:
    resp = await client.get("/api/v1/endpoint-configs", headers=AUTH)
    assert resp.status_code == 200
    (row,) = [e for e in resp.json() if e["id"] == endpoint_id]
    return row


async def test_ui_edit_route_can_change_and_clear_every_field():
    """Regression: services/endpoint_service.update_endpoint_config skips any field
    whose incoming value is None, by design (the JSON PATCH API relies on that to
    treat an omitted body field as "leave unchanged"). The UI's dedicated edit route
    must not inherit that limitation — clearing a field in the edit modal (role back
    to "all roles", api_key_ref/context_window back to unset) has to actually stick."""
    async with _client() as client:
        endpoint_id = await _create_via_api(client)

        edit_resp = await client.post(
            f"/ui/endpoint-configs/{endpoint_id}/edit",
            data={
                "base_url": "https://edited.invalid/v1",
                "model": "edited-model",
                "role": "",
                "priority": "5",
                "api_key_ref": "",
                "max_concurrency": "3",
                "context_window": "",
                # supports_tool_calling omitted entirely, as an unchecked box would be.
            },
        )
        assert edit_resp.status_code == 303

        row = await _reread(client, endpoint_id)
        assert row["base_url"] == "https://edited.invalid/v1"
        assert row["model"] == "edited-model"
        assert row["role"] is None
        assert row["priority"] == 5
        assert row["api_key_ref"] is None
        assert row["max_concurrency"] == 3
        assert row["context_window"] is None
        assert row["supports_tool_calling"] is False


async def test_ui_health_route_skips_probing_disabled_endpoints(monkeypatch):
    calls = []

    async def _tracking_acompletion(**kwargs):
        calls.append(kwargs)
        raise AssertionError("should not be called for a disabled endpoint")

    monkeypatch.setattr(llm_client_module.litellm, "acompletion", _tracking_acompletion)

    async with _client() as client:
        endpoint_id = await _create_via_api(client, enabled=False)
        resp = await client.get(f"/ui/endpoint-configs/{endpoint_id}/health")

        assert resp.status_code == 200
        assert "bg-secondary" in resp.text
        assert calls == []


async def test_ui_health_route_reports_ok_and_error_states(monkeypatch):
    from types import SimpleNamespace

    async def _ok_acompletion(**kwargs):
        message = SimpleNamespace(content="pong", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    async with _client() as client:
        ok_id = await _create_via_api(client, base_url="https://ok.invalid/v1")
        monkeypatch.setattr(llm_client_module.litellm, "acompletion", _ok_acompletion)
        ok_resp = await client.get(f"/ui/endpoint-configs/{ok_id}/health")
        assert "bg-success" in ok_resp.text

        async def _failing_acompletion(**kwargs):
            raise RuntimeError("connection refused")

        down_id = await _create_via_api(client, base_url="https://down.invalid/v1")
        monkeypatch.setattr(llm_client_module.litellm, "acompletion", _failing_acompletion)
        down_resp = await client.get(f"/ui/endpoint-configs/{down_id}/health")
        assert "bg-danger" in down_resp.text
