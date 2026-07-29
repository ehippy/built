"""The Reviver: an autonomous background pass over blocked/failed cards. Wakes on its
own on a timer (see orchestrator/reviver.py) — no manual trigger, no card or project
of its own. Decides, per stuck card, whether to retry it (optionally with a
diagnostic note) or leave it for a human, using its own judgment from the card's
history rather than hardcoded error-string matching.

Structurally different from agent/loop.py's run_column_visit: there's no single
terminal tool ending the run. revive_card and leave_blocked are consequential but
non-terminal — the loop keeps going after each one, working through as many stuck
cards as it judges worth reviewing in one pass, until it calls done_for_now (or the
iteration budget runs out)."""

import json

from sqlalchemy.ext.asyncio import AsyncSession

from built.config import settings
from built.domain.enums import EventType, LifecycleState
from built.domain.events import append_event
from built.llm.client import LLMClient
from built.llm.tool_schemas import REVIVER_TERMINAL_TOOL, REVIVER_TOOLS
from built.services import card_service

REVIVER_SYSTEM_PROMPT = (
    "You are the Reviver agent in an autonomous software factory. Cards move through "
    "PM -> Developer -> Tester -> Deployer on their own; when something goes wrong, a card "
    "gets blocked or failed and normally just waits for a human to notice and retry it. Your "
    "job is to check on stuck cards yourself and decide what to do with each one:\n\n"
    "- A transient infrastructure failure (a timeout, a connection error, a Docker daemon "
    "hiccup) is usually worth a plain retry — no note needed, the problem wasn't the work "
    "itself.\n"
    "- A genuine issue you can diagnose from its history (e.g. a merge conflict against work "
    "that already merged, a clear misunderstanding of the task) is worth retrying WITH a "
    "specific, actionable note for whichever column runs next.\n"
    "- Anything that needs a human decision — missing configuration, missing credentials, an "
    "ambiguous product decision, or a card that keeps failing the same way even after you've "
    "already retried it — should be left alone. Don't guess at things you can't actually fix.\n\n"
    "Use list_stuck_cards to see what needs attention, read_card_history on anything whose "
    "cause isn't already obvious from the list, then revive_card or leave_blocked for each one "
    "you've made a decision about. Call done_for_now once you've reviewed everything worth "
    "reviewing this pass (or there was nothing stuck at all) — nobody is watching this run "
    "interactively."
)


async def run_reviver_pass(session: AsyncSession, *, llm_client: LLMClient, max_iterations: int) -> dict:
    """Runs one pass and returns a small summary dict (revived/left_blocked/errors
    counts). Never raises — a bad pass just does nothing further; the next scheduled
    wake tries again from scratch."""
    messages: list[dict] = [
        {"role": "system", "content": REVIVER_SYSTEM_PROMPT},
        {"role": "user", "content": "Check on stuck cards and decide what to do with each."},
    ]
    counts = {"revived": 0, "left_blocked": 0, "errors": 0}

    try:
        for _ in range(max_iterations):
            result = await llm_client.complete(messages=messages, tools=REVIVER_TOOLS)

            if not result.tool_calls:
                messages.append({"role": "assistant", "content": result.content or ""})
                messages.append(
                    {
                        "role": "user",
                        "content": "Call a tool: list_stuck_cards to see what's stuck, or "
                        "done_for_now if you're finished.",
                    }
                )
                continue

            messages.append(
                {
                    "role": "assistant",
                    "content": result.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in result.tool_calls
                    ],
                }
            )

            done = False
            for tool_call in result.tool_calls:
                if tool_call.name == REVIVER_TERMINAL_TOOL:
                    done = True
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": "Acknowledged."}
                    )
                    continue
                output = await _dispatch(session, tool_call.name, tool_call.arguments, counts)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})

            if done:
                return counts

        return counts
    except Exception:  # noqa: BLE001 — deliberate: a bad pass does nothing further, not a crash
        return counts


async def _dispatch(session: AsyncSession, name: str, arguments: dict, counts: dict) -> str:
    if name == "list_stuck_cards":
        return await _list_stuck_cards(session)
    if name == "read_card_history":
        return await _read_card_history(session, str(arguments.get("card_id", "")))
    if name == "revive_card":
        return await _revive_card(session, str(arguments.get("card_id", "")), arguments.get("note"), counts)
    if name == "leave_blocked":
        return await _leave_blocked(
            session, str(arguments.get("card_id", "")), str(arguments.get("reason", "")), counts
        )
    return f"unknown tool: {name!r}"


async def _list_stuck_cards(session: AsyncSession) -> str:
    cards = await card_service.list_stuck_cards(session, limit=settings.reviver_max_cards_per_pass)
    if not cards:
        return "No stuck cards right now."
    lines = []
    for card in cards:
        recap = await card_service.get_latest_attempt_recap(session, card)
        reason = (recap or "(no recap available)").splitlines()[0]
        lines.append(
            f"- {card.id} | {card.title!r} | project {card.project_id} | column {card.column.value} | "
            f"lifecycle {card.lifecycle_state.value} | auto-revives used: "
            f"{card.auto_revive_count}/{settings.reviver_max_auto_revives} | {reason}"
        )
    return "\n".join(lines)


async def _read_card_history(session: AsyncSession, card_id: str) -> str:
    try:
        card = await card_service.get_card(session, card_id)
    except card_service.NotFoundError:
        return f"no such card: {card_id!r}"
    recap = await card_service.get_latest_attempt_recap(session, card)
    lines = [
        f"Card: {card.title}",
        f"Column: {card.column.value} | Lifecycle: {card.lifecycle_state.value}",
        f"Spec:\n{card.spec or '(no spec)'}",
    ]
    if recap:
        lines.append(recap)
    return "\n".join(lines)


async def _revive_card(session: AsyncSession, card_id: str, note: object, counts: dict) -> str:
    try:
        card = await card_service.get_card(session, card_id)
    except card_service.NotFoundError:
        counts["errors"] += 1
        return f"no such card: {card_id!r}"
    if card.lifecycle_state not in (LifecycleState.BLOCKED, LifecycleState.FAILED):
        counts["errors"] += 1
        return (
            f"card {card_id} isn't blocked or failed (currently {card.lifecycle_state.value}) — "
            "nothing to revive."
        )
    if card.auto_revive_count >= settings.reviver_max_auto_revives:
        counts["errors"] += 1
        return (
            f"card {card_id} has already used its automatic-retry budget "
            f"({card.auto_revive_count}/{settings.reviver_max_auto_revives}) — leave it for a human instead."
        )
    note_str = str(note) if note else None
    await card_service.retry_card(session, card_id, note=note_str)
    card.auto_revive_count += 1
    await append_event(
        session,
        card_id=card.id,
        type=EventType.SYSTEM_NOTE,
        payload={"action": "auto_revived", "note": note_str, "attempt": card.auto_revive_count},
    )
    await session.commit()
    counts["revived"] += 1
    return f"revived card {card_id}."


async def _leave_blocked(session: AsyncSession, card_id: str, reason: str, counts: dict) -> str:
    try:
        card = await card_service.get_card(session, card_id)
    except card_service.NotFoundError:
        counts["errors"] += 1
        return f"no such card: {card_id!r}"
    await append_event(
        session,
        card_id=card.id,
        type=EventType.SYSTEM_NOTE,
        payload={"action": "reviver_left_blocked", "reason": reason},
    )
    await session.commit()
    counts["left_blocked"] += 1
    return f"left card {card_id} blocked: {reason}"
