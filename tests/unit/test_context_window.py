"""Tests for built.agent.context_window: estimate_tokens and compact."""

import pytest

from built.agent.context_window import (
    CompactionEvent,
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


def _tool_call(name: str, arguments: str, tool_call_id: str = "tc1") -> dict:
    return {"id": tool_call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


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
    """When nothing needs to change, compact() must also report no CompactionEvent
    — that's what agent/loop.py uses to decide whether to log a CardEvent at all."""

    @pytest.mark.asyncio
    async def test_under_budget(self):
        config = ContextWindowConfig(max_tokens=128_000, keep_messages=10)
        msgs = [_system("system prompt"), _user("user message")]
        client = StubLLMClient()
        result, event = await compact(msgs, client, config, "test-model")
        assert result is msgs
        assert event is None

    @pytest.mark.asyncio
    async def test_too_few_messages(self):
        config = ContextWindowConfig(max_tokens=512, keep_messages=10)
        msgs = [_system("sys"), _user("hi")]
        client = StubLLMClient()
        result, event = await compact(msgs, client, config, "test-model")
        assert result is msgs
        assert event is None

    @pytest.mark.asyncio
    async def test_keep_messages_one(self):
        config = ContextWindowConfig(max_tokens=128_000, keep_messages=1)
        msgs = [_system("sys"), _user("hi")]
        client = StubLLMClient()
        result, event = await compact(msgs, client, config, "test-model")
        assert result is msgs
        assert event is None


class TestCompactSummarization:
    @pytest.mark.asyncio
    async def test_summarizes_when_over_budget(self):
        config = ContextWindowConfig(max_tokens=512, keep_messages=4)
        msgs = [_system("system prompt"), _user("user message")]
        for i in range(10):
            msgs.append(_assistant(f"assistant turn {i}"))
            msgs.append(_tool("x" * 2000, f"tc{i}"))

        client = StubLLMClient()
        client._summary = "compact summary of earlier turns"
        result, event = await compact(msgs, client, config, "test-model")

        assert len(result) < len(msgs), "compaction should reduce message count"
        assert len(client.calls) == 1, "should make one summarizer call"
        assert result[0]["role"] == "system"
        assert any(
            "Earlier conversation summary" in m.get("content", "") for m in result if m["role"] == "system"
        )

    @pytest.mark.asyncio
    async def test_summarization_reports_a_compaction_event(self):
        """agent/loop.py logs this as a CardEvent — the whole point of this
        conversation is that compaction used to be completely invisible."""
        config = ContextWindowConfig(max_tokens=512, keep_messages=4)
        msgs = [_system("system prompt"), _user("user message")]
        for i in range(10):
            msgs.append(_assistant(f"assistant turn {i}"))
            msgs.append(_tool("x" * 2000, f"tc{i}"))

        client = StubLLMClient()
        client._summary = "compact summary of earlier turns"
        result, event = await compact(msgs, client, config, "test-model")

        assert isinstance(event, CompactionEvent)
        assert event.messages_before == len(msgs)
        assert event.messages_after == len(result)
        assert event.tokens_before > event.tokens_after
        assert event.summary is not None
        assert "compact summary of earlier turns" in event.summary

    @pytest.mark.asyncio
    async def test_system_prompt_preserved(self):
        config = ContextWindowConfig(max_tokens=256, keep_messages=4)
        msgs = [_system("this is my system prompt"), _user("user message")]
        for i in range(10):
            msgs.append(_assistant(f"turn {i}"))
            msgs.append(_tool("x" * 1000, f"tc{i}"))

        client = StubLLMClient()
        client._summary = "summary"
        result, _event = await compact(msgs, client, config, "test-model")

        assert result[0] == msgs[0], "original system prompt preserved"

    @pytest.mark.asyncio
    async def test_original_task_prompt_survives_compaction(self):
        """Regression: confirmed in production — an agent visit that ran long
        enough to trigger compaction lost the card's spec and acceptance
        criteria entirely (messages[1], the original task-defining user
        message) with no trace, leaving the model able to see recent tool-call
        history but no idea what it was actually supposed to be building. Every
        compaction path must preserve messages[1] verbatim, not just messages[0]."""
        config = ContextWindowConfig(max_tokens=512, keep_messages=4)
        task_prompt = _user("Card: build the thing\n\nAcceptance criteria:\n- it works")
        msgs = [_system("system prompt"), task_prompt]
        for i in range(10):
            msgs.append(_assistant(f"assistant turn {i}"))
            msgs.append(_tool("x" * 2000, f"tc{i}"))

        client = StubLLMClient()
        client._summary = "compact summary of earlier turns"
        result, _event = await compact(msgs, client, config, "test-model")

        assert task_prompt in result
        assert result[1] == task_prompt

    @pytest.mark.asyncio
    async def test_returns_truncated_when_summarizer_fails(self):
        config = ContextWindowConfig(max_tokens=256, keep_messages=4)
        task_prompt = _user("user")
        messages = [_system("system"), task_prompt]
        for i in range(20):
            messages.append(_assistant("x" * 500))
            messages.append(_tool("x" * 500, f"tc{i}"))

        class RaisingClient:
            async def complete(self, *, messages, tools):
                raise RuntimeError("API error")

        result, event = await compact(messages, RaisingClient(), config, "test-model")
        # Pinned prefix (system + task prompt) plus the recent tail — a blind
        # last-`keep_messages` slice of the whole list would drop the pinned
        # prefix, which is exactly the bug this fallback must not reintroduce.
        assert len(result) <= config.keep_messages + 2
        assert result[0] == messages[0]
        assert result[1] == task_prompt
        # Still reported as a compaction event (messages really were dropped),
        # just with no summary text — the UI shows a "dropped without a
        # summary" note in this case rather than silently saying nothing at all.
        assert isinstance(event, CompactionEvent)
        assert event.summary is None
        assert event.messages_after < event.messages_before


class TestCompactToolResultTrimming:
    @pytest.mark.asyncio
    async def test_trims_an_oversized_tool_result_without_summarizing(self):
        """Regression: this trim pass used to only ever inspect assistant
        tool_calls' *arguments* (what a model sent as input to a tool, e.g.
        review_diff's `{}` — essentially never large) instead of `tool`-role
        messages' *content* (an actual tool result — a diff, a file read, bash
        output — which is where real bloat lives). Confirmed in production: a
        ~13k-token request was rejected by the model server as needing 8.4M
        tokens, and this no-op trim step was silently never engaging on the
        20KB diff result that was the only large thing in the conversation."""
        config = ContextWindowConfig(
            max_tokens=64_000,
            keep_messages=1,
            max_tool_result_chars=100,
        )
        big_result = "x" * 250_000  # ~62.5k estimated tokens on its own
        msgs = [
            _system("system"),
            _user("user"),
            _assistant("thinking...", tool_calls=[_tool_call("bash", "{}")]),
            _tool(big_result, tool_call_id="tc1"),
            _assistant("ok"),
        ]

        client = StubLLMClient()
        result, event = await compact(msgs, client, config, "test-model")

        # Trimmed cheaply — the summarizer (an extra real LLM call) was never
        # needed, and never invoked.
        assert client.calls == []
        assert event is not None
        assert event.summary is None
        trimmed_tool_msg = next(m for m in result if m.get("role") == "tool")
        assert len(trimmed_tool_msg["content"]) < len(big_result)
        assert "truncated from 250000 chars" in trimmed_tool_msg["content"]
        # Pinned prefix and the tail both survive untouched.
        assert result[0] == msgs[0]
        assert result[1] == msgs[1]
        assert result[-1] == msgs[-1]

    @pytest.mark.asyncio
    async def test_does_not_trim_large_tool_call_arguments(self):
        """The inverse of the regression above: large assistant tool_call
        *arguments* are deliberately left alone by this trim pass — they're
        rarely the source of real bloat, and conflating them with tool results
        was the original bug. A conversation whose only bloat is oversized
        arguments should fall through to full summarization instead."""
        config = ContextWindowConfig(
            max_tokens=64_000,
            keep_messages=1,
            max_tool_result_chars=100,
        )
        big_args = '{"content": "' + "a" * 250_000 + '"}'
        msgs = [
            _system("system"),
            _user("user"),
            _assistant("thinking...", tool_calls=[_tool_call("write_file", big_args)]),
            _tool("ok", tool_call_id="tc1"),
            _assistant("ok"),
        ]

        client = StubLLMClient(summary="summary")
        result, event = await compact(msgs, client, config, "test-model")

        # Not trimmed (arguments are never touched) — fell through to the
        # summarizer instead, which is why this call actually happened.
        assert len(client.calls) == 1
        assert event is not None
        assert event.summary == "summary"
        assert len(result) < len(msgs)
