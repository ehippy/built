import re

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from built.db.models import CardEvent
from built.domain.enums import EventType

MAX_PAYLOAD_CHARS = 20_000

# Coarse allowlist of common secret shapes (OpenAI/Anthropic-style keys, AWS access
# keys, GitHub tokens, ...). Not exhaustive — defense in depth, not the only control.
_SECRET_PATTERN = re.compile(r"(sk-[A-Za-z0-9_-]{10,}|AKIA[A-Z0-9]{12,}|gh[pousr]_[A-Za-z0-9]{20,})")


async def append_event(
    session: AsyncSession,
    *,
    card_id: str,
    type: EventType,
    payload: dict,
    column_visit_id: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    latency_ms: int | None = None,
    cost_estimate: float | None = None,
) -> CardEvent:
    """Append a transcript row for the dashboard to tail. `seq` is monotonic per card,
    computed here rather than via DB autoincrement. Safe under concurrency because only
    one orchestrator worker ever holds a given card's claim/lease at a time (Phase 4)."""
    current_max = await session.scalar(select(func.max(CardEvent.seq)).where(CardEvent.card_id == card_id))
    event = CardEvent(
        card_id=card_id,
        column_visit_id=column_visit_id,
        seq=(current_max or 0) + 1,
        type=type,
        payload=_scrub_and_cap(payload),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
        cost_estimate=cost_estimate,
    )
    session.add(event)
    await session.flush()
    return event


def _scrub_and_cap(value):
    """Redact anything that looks like a secret and cap oversized strings before a
    payload is ever persisted — CardEvent payloads are rendered verbatim in the
    dashboard, and stdout from a build/deploy script can echo an env var."""
    if isinstance(value, str):
        value = _SECRET_PATTERN.sub("[REDACTED]", value)
        if len(value) > MAX_PAYLOAD_CHARS:
            value = value[:MAX_PAYLOAD_CHARS] + f"... [truncated, {len(value)} chars total]"
        return value
    if isinstance(value, dict):
        return {k: _scrub_and_cap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_and_cap(v) for v in value]
    return value
