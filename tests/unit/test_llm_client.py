import asyncio
from types import SimpleNamespace

import pytest

import built.llm.client as llm_client_module
from built.db.models import EndpointConfig
from built.llm.client import FallbackLLMClient


def _endpoint(**overrides) -> EndpointConfig:
    defaults = {
        "base_url": "https://example.invalid/v1",
        "model": "some-model",
        "supports_tool_calling": True,
    }
    defaults.update(overrides)
    return EndpointConfig(**defaults)


def test_rejects_empty_chain():
    with pytest.raises(ValueError, match="no usable"):
        FallbackLLMClient([])


def test_drops_endpoints_that_dont_support_tool_calling():
    non_tool_calling = _endpoint(supports_tool_calling=False)
    with pytest.raises(ValueError, match="no usable"):
        FallbackLLMClient([non_tool_calling])


def test_keeps_only_tool_calling_endpoints_in_priority_order():
    good_1 = _endpoint(base_url="https://a.invalid", priority=0)
    bad = _endpoint(base_url="https://b.invalid", priority=1, supports_tool_calling=False)
    good_2 = _endpoint(base_url="https://c.invalid", priority=2)

    client = FallbackLLMClient([good_1, bad, good_2])

    assert [e.base_url for e in client._chain] == ["https://a.invalid", "https://c.invalid"]


def _fake_response() -> SimpleNamespace:
    message = SimpleNamespace(content="ok", tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def _tracking_acompletion(peak: list[int], in_flight: list[int], *, hold_seconds: float = 0.05):
    async def _acompletion(**kwargs):
        in_flight[0] += 1
        peak[0] = max(peak[0], in_flight[0])
        try:
            await asyncio.sleep(hold_seconds)
            return _fake_response()
        finally:
            in_flight[0] -= 1

    return _acompletion


async def test_concurrent_calls_are_capped_at_max_concurrency(monkeypatch):
    endpoint = _endpoint(base_url="https://cap-one.invalid", model="m", max_concurrency=1)
    peak, in_flight = [0], [0]
    monkeypatch.setattr(llm_client_module.litellm, "acompletion", _tracking_acompletion(peak, in_flight))
    client = FallbackLLMClient([endpoint])

    await asyncio.gather(*[client.complete(messages=[], tools=[]) for _ in range(3)])

    assert peak[0] == 1


async def test_concurrent_calls_allow_up_to_max_concurrency(monkeypatch):
    endpoint = _endpoint(base_url="https://cap-two.invalid", model="m", max_concurrency=2)
    peak, in_flight = [0], [0]
    monkeypatch.setattr(llm_client_module.litellm, "acompletion", _tracking_acompletion(peak, in_flight))
    client = FallbackLLMClient([endpoint])

    await asyncio.gather(*[client.complete(messages=[], tools=[]) for _ in range(3)])

    assert peak[0] == 2


async def test_concurrency_cap_is_shared_across_endpointconfig_rows_with_the_same_backend(monkeypatch):
    """A project-specific row and a global row that happen to point at the same
    physical (base_url, model) backend must share one cap — otherwise the two rows
    would each get their own semaphore and the backend could still see double the
    intended load."""
    project_scoped = _endpoint(
        base_url="https://shared-backend.invalid", model="m", project_id="proj-1", max_concurrency=1
    )
    global_scoped = _endpoint(
        base_url="https://shared-backend.invalid", model="m", project_id=None, max_concurrency=1
    )
    peak, in_flight = [0], [0]
    monkeypatch.setattr(llm_client_module.litellm, "acompletion", _tracking_acompletion(peak, in_flight))
    client_a = FallbackLLMClient([project_scoped])
    client_b = FallbackLLMClient([global_scoped])

    await asyncio.gather(
        *[client_a.complete(messages=[], tools=[]) for _ in range(2)],
        *[client_b.complete(messages=[], tools=[]) for _ in range(2)],
    )

    assert peak[0] == 1
