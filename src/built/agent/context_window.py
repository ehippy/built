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
class ContextWindowConfig:
    max_tokens: int
    keep_messages: int = 10
    summary_max_tokens: int = 4096
    max_tool_result_chars: int = 4000

    @property
    def budget_messages(self) -> int:
        """Messages beyond this index are fair game for summarization."""
        return max(0, self.keep_messages - 2)

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


async def summarize_segment(
    llm_client: LLMClient, *, conversation_text: str, model_name: str
) -> str:
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
                lines.append(
                    f"[assistant tool call]: {fn.get('name', '?')}({args[:200]})"
                )
        else:
            lines.append(f"[{role}]: {content[:500]}")
    return "\n".join(lines)


def _format_summary_block(summary: str) -> dict:
    """Wrap a compact summary in a system message the LLM will carry."""
    return {
        "role": "system",
        "content": (
            "Earlier conversation summary (kept for context, not shown in full):\n"
            f"{summary}"
        ),
    }


async def compact(
    messages: list[dict],
    llm_client: LLMClient,
    config: ContextWindowConfig,
    model_name: str,
) -> list[dict]:
    """Condense the message list when token count threatens the context window.

    Strategy:
    1. Estimate current token count.
    2. If under budget (with margin), return messages as-is.
    3. Otherwise, take the oldest non-system messages, send them to the LLM
       for summarization, and replace them with a single summary block.
    4. If the summary call failed or the block is still too large, fall back
       to hard truncation.

    Always preserves the first K messages (config.keep_messages) as a minimum.
    Returns the compacted message list.
    """
    token_count = estimate_tokens(messages)
    budget = config.max_tokens - config.keep_tokens

    if token_count <= budget * 0.85:
        return messages

    budget_messages = config.budget_messages
    if budget_messages <= 0:
        # Not enough headroom — just drop to keep_messages from the end.
        return messages[-config.keep_messages :]

    # Collect messages to summarize: everything after the safe zone.
    summary_candidates = list(messages[budget_messages:])

    # If tool results in these messages are very large, try trimming them first.
    trimmed = False
    if len(summary_candidates) > 2:
        for msg in summary_candidates:
            if msg["role"] == "assistant":
                for tc in msg.get("tool_calls") or []:
                    args = tc.get("function", {}).get("arguments", "{}")
                    if len(args) > config.max_tool_result_chars:
                        tc["function"]["arguments"] = (
                            args[: config.max_tool_result_chars]
                            + f"... [truncated from {len(args)} chars]"
                        )
                        trimmed = True
        if trimmed:
            token_count = estimate_tokens(messages[:budget_messages] + summary_candidates)
            if token_count <= budget * 0.85 + 1000:
                return messages[:budget_messages] + summary_candidates

    # Build conversation text and summarize.
    conv_text = _build_conversation_text(messages, budget_messages)
    if len(conv_text) < 200:
        # Not enough content to meaningfully summarize — just truncate.
        return messages[-config.keep_messages :]

    try:
        summary = await summarize_segment(llm_client, conversation_text=conv_text, model_name=model_name)
    except Exception:  # noqa: BLE001 — summarizer failure is not fatal; fall back to truncation
        logger.exception("summarizer failed, falling back to hard truncation")
        return messages[-config.keep_messages :]

    if not summary:
        return messages[-config.keep_messages :]

    summary_block = _format_summary_block(summary)
    compacted = [messages[0], summary_block] + list(messages[min(config.keep_messages, len(messages)):])
    new_token_count = estimate_tokens(compacted)

    # If still over budget, hard truncate the summary block's content.
    if new_token_count > budget * 0.95:
        # Rough char-to-token ratio of 4
        target_chars = max(200, math.ceil((budget - 50) * 0.08))  # ~50 tokens for the wrapper
        summary = summary[:target_chars] + "... [summary truncated]"
        compacted = [messages[0], _format_summary_block(summary)] + list(
            messages[min(config.keep_messages, len(messages)) : config.keep_messages + 5]
        )

    logger.info(
        "compacted %d -> %d messages (%d -> %d est tokens)",
        len(messages),
        len(compacted),
        token_count,
        estimate_tokens(compacted),
    )
    return compacted
