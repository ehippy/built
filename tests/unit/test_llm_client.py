import pytest

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
