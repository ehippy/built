"""Token-aware context window management for agent loops.

Provides estimate_tokens(), ContextWindowConfig, and compact() which
summarizes older conversation turns when the message list threatens
to exceed the model's context window.
"""

import logging
import math
import re
import textwrap
from dataclasses import dataclass

from built.llm.client import LLMClient

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = textwrap.dedent(
    """\
Summarize the following conversation segment, which was used earlier
in this session. Focus on:

- What the agent discovered by reading files or running commands
- What actions the agent took (write/edit files, bash commands)
- Any decisions or conclusions reached
- Context the agent can assume from here without re-deriving

Ignore filler and repetitions. Do NOT include the exact file paths
or full file contents — instead describe what each file contained in
a sentence or two. Keep it concise enough to fit in a single system
message block (preferably under 1500 words).

---
{conversation}
---

Return only the summary. No introductory text, no conclusion, no
meta-commentary — just the summary in prose form.
"""
)


@dataclass
class CompactionEvent:
    """Details of one compaction pass, for the caller to log as a CardEvent —
    this module has no session/card_id of its own by design (it's a pure
    transform over a message list), so it hands the details back rather than
    writing anything itself. `summary` is None when messages were dropped
    without ever producing one (trim-only, or a fallback path — see compact())."""

    messages_before: int
    messages_after: int
    tokens_before: int
    tokens_after: int
    summary: str | None = None


@dataclass
class ContextWindowConfig:
    max_tokens: int
    keep_messages: int = 10
    summary_max_tokens: int = 4096
    max_tool_result_chars: int = 4000

    @property
    def keep_tokens(self) -> int:
        """Conservative estimate for system + system-level messages. These
        are never compacted away, so we subtract this from the budget."""
        return math.ceil(self.max_tokens * 0.05)


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token count for a list of OpenAI-format messages.

    Uses a simple heuristic tuned on the OpenAI tokenizer family:
    each character is about 0.25 tokens on average. This is within
    ~10-15% of the actual tiktoken count and avoids importing yet
    another dependency (tiktoken is pulled in by litellm already,
    but this stays self-contained).

    System prompt and role annotations add a small overhead.
    """
    tokens = 0
    for msg in messages:
        content = msg.get("content") or ""
        # Each message has overhead: role tokens, delimiters
        tokens += 3
        # Content is roughly 0.25 tokens per character
        tokens += len(content) // 4
        # Tool call metadata adds some tokens
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            tokens += len(fn.get("name", "")) // 4
            tokens += len(fn.get("arguments", "")) // 4
        # Tool result responses
        if msg.get("role") == "tool":
            tokens += 3  # role prefix + delimiter
            tokens += len(content or "") // 4
    return tokens


async def summarize_segment(llm_client: LLMClient, *, conversation_text: str, model_name: str) -> str:
    """Ask the LLM to summarize a conversation segment into compact prose."""
    prompt = SUMMARY_PROMPT.format(conversation=conversation_text)
    summary_messages = [
        {"role": "user", "content": prompt},
    ]
    result = await llm_client.complete(
        messages=summary_messages,
        tools=[],
    )
    summary = (result.content or "").strip()
    # Remove any "Here's the summary" style preambles if the model
    # adds them despite instructions
    summary = re.sub(
        r"^here'?s (a )?(the )?summary[:.]?\s*",
        "",
        summary,
        flags=re.IGNORECASE | re.MULTILINE,
    ).strip()
    logger.info(
        "compact summary of %d chars from %s",
        len(summary),
        model_name,
    )
    return summary


def _build_conversation_text(messages: list[dict], start_idx: int) -> str:
    """Render the message segment from start_idx onward as prose for summarization."""
    lines = []
    for msg in messages[start_idx:]:
        role = msg["role"]
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []
        if role == "tool":
            tool_id = msg.get("tool_call_id", "?")
            lines.append(f"[tool call {tool_id} result]: {content[:500]}")
        elif role == "assistant":
            if content:
                lines.append(f"[assistant]: {content}")
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                lines.append(f"[assistant tool call]: {fn.get('name', '?')}({args[:200]})")
        else:
            lines.append(f"[{role}]: {content[:500]}")
    return "\n".join(lines)


def _format_summary_block(summary: str) -> dict:
    """Wrap a compact summary in a system message the LLM will carry."""
    return {
        "role": "system",
        "content": (f"Earlier conversation summary (kept for context, not shown in full):\n{summary}"),
    }


async def compact(
    messages: list[dict],
    llm_client: LLMClient,
    config: ContextWindowConfig,
    model_name: str,
) -> tuple[list[dict], CompactionEvent | None]:
    """Condense the message list when token count threatens the context window.

    messages[0] (the system prompt) and messages[1] (the original task-defining
    user prompt — a card's spec and acceptance criteria) are pinned: every path
    below preserves both verbatim, never summarized and never dropped. Confirmed
    in production: an earlier version of this function only ever kept messages[0]
    in its final reconstruction, silently discarding messages[1:keep_messages] —
    including the task prompt — the moment a visit actually ran long enough to
    compact. The agent could still see recent tool-call history afterward, but
    had no way to recover what it was actually supposed to be building (it isn't
    restated anywhere else — the system prompt is generic role instructions, not
    this card's spec).

    Strategy:
    1. Estimate current token count; return as-is if under budget.
    2. Otherwise, keep the pinned prefix and the most recent `keep_messages`
       verbatim, and summarize whatever falls in between.
    3. If there's nothing meaningful in between, or summarization fails, fall
       back to pinned + tail — never a blind last-N slice of the whole list,
       which is exactly what could drop the pinned prefix.

    Returns (messages, CompactionEvent | None) — the second element is None only
    when nothing actually changed (still under budget, or nothing old enough to
    touch). Confirmed in production that compaction itself was otherwise
    completely invisible: no CardEvent, no card-specific trace anywhere, not even
    the extra LLM call this makes to do the summarizing. The caller (agent/loop.py,
    which has session/card_id) is responsible for actually logging this.
    """
    token_count = estimate_tokens(messages)
    budget = config.max_tokens - config.keep_tokens

    if token_count <= budget * 0.85:
        return messages, None

    pinned = messages[:2] if len(messages) > 1 else messages[:1]
    rest = messages[len(pinned) :]

    def _event(result: list[dict], *, summary: str | None) -> CompactionEvent:
        return CompactionEvent(
            messages_before=len(messages),
            messages_after=len(result),
            tokens_before=token_count,
            tokens_after=estimate_tokens(result),
            summary=summary,
        )

    # If any tool RESULT since the pinned prefix is very large, try trimming it first —
    # cheaper than a full LLM summarization pass below, and often enough on its own.
    # Targets role == "tool" messages' `content`, which is where actual tool output lives
    # (a read_file/bash/review_diff result) — confirmed this previously targeted assistant
    # tool_calls' *arguments* instead (what a model sent as *input* to a tool, e.g.
    # review_diff's `{}` — essentially never large), making this trim step a no-op for the
    # actual common case of a large tool result.
    #
    # Deliberately runs over the whole of `rest`, and before the "nothing old enough to
    # summarize" bailout below rather than after it: an oversized result can blow the
    # budget on the very first tool call of a run — e.g. Reviewer's first call being
    # review_diff — when `rest` is still shorter than `keep_messages` and there's no
    # "middle" segment to summarize at all. The old ordering treated "too few messages to
    # summarize" as "nothing to do" and returned the oversized message completely
    # untouched no matter how far over budget it pushed things; confirmed in production
    # against an 8.4M-token rejection where the whole conversation was just
    # [system, user, assistant(tool_call), tool(huge diff)].
    trimmed = False
    for msg in rest:
        if msg["role"] == "tool":
            content = msg.get("content") or ""
            if len(content) > config.max_tool_result_chars:
                msg["content"] = (
                    content[: config.max_tool_result_chars] + f"... [truncated from {len(content)} chars]"
                )
                trimmed = True
    if trimmed:
        trimmed_token_count = estimate_tokens(pinned + rest)
        if trimmed_token_count <= budget * 0.85 + 1000:
            result = pinned + rest
            return result, _event(result, summary=None)

    if len(rest) <= config.keep_messages:
        # Nothing old enough to summarize beyond the pinned prefix + recent tail. If the
        # trim pass above already changed something but wasn't enough on its own to clear
        # budget, that's still the best this function can do without dropping the pinned
        # prefix or recent turns — report it; otherwise truly nothing changed.
        if not trimmed:
            return messages, None
        result = pinned + rest
        return result, _event(result, summary=None)

    tail = rest[-config.keep_messages :] if config.keep_messages else []
    middle = rest[: len(rest) - len(tail)]

    # Build conversation text and summarize just the middle segment.
    conv_text = _build_conversation_text(middle, 0)
    if len(conv_text) < 200:
        # Not enough content to meaningfully summarize — drop the middle outright.
        result = pinned + tail
        return result, _event(result, summary=None)

    try:
        summary = await summarize_segment(llm_client, conversation_text=conv_text, model_name=model_name)
    except Exception:  # noqa: BLE001 — summarizer failure is not fatal; fall back to truncation
        logger.exception("summarizer failed, falling back to hard truncation")
        result = pinned + tail
        return result, _event(result, summary=None)

    if not summary:
        result = pinned + tail
        return result, _event(result, summary=None)

    summary_block = _format_summary_block(summary)
    compacted = pinned + [summary_block] + tail
    new_token_count = estimate_tokens(compacted)

    # If still over budget, hard truncate the summary block's content.
    if new_token_count > budget * 0.95:
        # Rough char-to-token ratio of 4
        target_chars = max(200, math.ceil((budget - 50) * 0.08))  # ~50 tokens for the wrapper
        summary = summary[:target_chars] + "... [summary truncated]"
        compacted = pinned + [_format_summary_block(summary)] + tail

    logger.info(
        "compacted %d -> %d messages (%d -> %d est tokens)",
        len(messages),
        len(compacted),
        token_count,
        estimate_tokens(compacted),
    )
    return compacted, _event(compacted, summary=summary)
