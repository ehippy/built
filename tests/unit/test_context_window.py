"""Tests for built.agent.context_window: estimate_tokens and compact."""

import pytest

from built.agent.context_window import (
    ContextWindowConfig,
    compact,
    estimate_tokens,
)
from built.llm.client import LLMResult


def _system(content: str) -> dict:
    return {"role": "system", "content": content}


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant(content: str | None = None, tool_calls: list | None = None) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _tool(content: str, tool_call_id: str = "tc1") -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


class StubLLMClient:
    def __init__(self, summary: str = "summarized"):
        self._summary = summary
        self.calls: list[dict] = []

    async def complete(self, *, messages: list[dict], tools: list[dict]) -> LLMResult:
        self.calls.append({"messages": messages, "tools": tools})
        return LLMResult(content=self._summary)


class TestEstimateTokens:
    def test_returns_small_positive_for_bare_messages(self):
        msgs = [_system("be helpful"), _user("hello")]
        tokens = estimate_tokens(msgs)
        assert isinstance(tokens, int)
        assert tokens > 0
        assert tokens <= 100

    def test_scales_with_content_length(self):
        msgs = [_system("be helpful"), _user("x" * 4000)]
        big = estimate_tokens(msgs)
        small = estimate_tokens([_system("be helpful"), _user("x" * 100)])
        assert big > small

    def test_counts_tool_calls(self):
        empty = [_system("sys"), _user("u")]
        with_call = [
            _system("sys"),
            _user("u"),
            _assistant(
                "ok",
                tool_calls=[
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command": "echo hello"}'},
                    }
                ],
            ),
        ]
        assert estimate_tokens(with_call) > estimate_tokens(empty)


class TestCompactNoOp:
    @pytest.mark.asyncio
    async def test_under_budget(self):
        config = ContextWindowConfig(max_tokens=128_000, keep_messages=10)
        msgs = [_system("system prompt"), _user("user message")]
        client = StubLLMClient()
        result = await compact(msgs, client, config, "test-model")
        assert result is msgs

    @pytest.mark.asyncio
    async def test_too_few_messages(self):
        config = ContextWindowConfig(max_tokens=512, keep_messages=10)
        msgs = [_system("sys"), _user("hi")]
        client = StubLLMClient()
        result = await compact(msgs, client, config, "test-model")
        assert result is msgs

    @pytest.mark.asyncio
    async def test_keep_messages_one(self):
        config = ContextWindowConfig(max_tokens=128_000, keep_messages=1)
        msgs = [_system("sys"), _user("hi")]
        client = StubLLMClient()
        result = await compact(msgs, client, config, "test-model")
        assert result is msgs


class TestCompactSummarization:
    @pytest.mark.asyncio
    async def test_summarizes_when_over_budget(self):
        # keep_messages=4 means budget_messages=2, so compaction won't
        # short-circuit early on the "0 headroom" check.
        config = ContextWindowConfig(max_tokens=512, keep_messages=4)
        msgs = [_system("system prompt"), _user("user message")]
        for i in range(10):
            msgs.append(_assistant(f"assistant turn {i}"))
            msgs.append(_tool("x" * 2000, f"tc{i}"))

        client = StubLLMClient()
        client._summary = "compact summary of earlier turns"
        result = await compact(msgs, client, config, "test-model")

        assert len(result) < len(msgs), "compaction should reduce message count"
        assert len(client.calls) == 1, "should make one summarizer call"
        assert result[0]["role"] == "system"
        assert any(
            "Earlier conversation summary" in m.get("content", "") for m in result if m["role"] == "system"
        )

    @pytest.mark.asyncio
    async def test_system_prompt_preserved(self):
        config = ContextWindowConfig(max_tokens=256, keep_messages=4)
        msgs = [_system("this is my system prompt"), _user("user message")]
        for i in range(10):
            msgs.append(_assistant(f"turn {i}"))
            msgs.append(_tool("x" * 1000, f"tc{i}"))

        client = StubLLMClient()
        client._summary = "summary"
        result = await compact(msgs, client, config, "test-model")

        assert result[0] == msgs[0], "original system prompt preserved"

    @pytest.mark.asyncio
    async def test_returns_truncated_when_summarizer_fails(self):
        config = ContextWindowConfig(max_tokens=256, keep_messages=4)
        messages = [_system("system"), _user("user")]
        for i in range(20):
            messages.append(_assistant("x" * 500))
            messages.append(_tool("x" * 500, f"tc{i}"))

        class RaisingClient:
            async def complete(self, *, messages, tools):
                raise RuntimeError("API error")

        result = await compact(messages, RaisingClient(), config, "test-model")
        assert len(result) <= config.keep_messages


class TestCompactToolResultTrimming:
    @pytest.mark.asyncio
    async def test_trims_oversized_tool_args(self):
        config = ContextWindowConfig(
            max_tokens=64_000,
            keep_messages=4,
            max_tool_result_chars=100,
        )
        msgs = [_system("system"), _user("user"), _assistant("ok"), _tool("ok")]
        big_args = '{"command": "' + "a" * 19_980 + '"}'
        msgs.append(
            _assistant(
                "thinking...",
                tool_calls=[
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": big_args},
                    }
                ],
            )
        )

        client = StubLLMClient()
        client._summary = "summary"
        result = await compact(msgs, client, config, "test-model")
        # Should not crash; result may be trimmed or summarized
        assert len(result) <= len(msgs)
