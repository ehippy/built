"""ActivityKind.PM_TRIAGE: the one background pass with authority to act on cards
already sitting in the PM column, rather than propose new ones. Scheduled and
toggled exactly like every other curation kind (orchestrator/curator.py), and
just as able to browse the repo read-only via the same explore tools and
dispatcher — judging whether a card is stale, already done, or a genuine
duplicate often needs an actual look at the code, not just titles. Its one
terminal tool, groom_backlog, either reprioritizes a card or merges duplicates by
archiving all but one of each group. Every card_id it references is re-validated
against a fresh read of the PM column right before anything is applied — never
trusted from the model's own say-so, the same "server-verified, not the model's
say-so" pattern as domain/run_attempts.py and agent/curation.py's
_dedupe_against_live_cards. Archiving is the only destructive-looking action, and
it isn't really: fully reversible via the existing unarchive path, same as a
human's own "archive" button."""

import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from built.agent.context import build_pm_triage_prompt
from built.agent.context_window import ContextWindowConfig, compact, estimate_tokens
from built.config import settings as builtin_settings
from built.db.models import Card, Project
from built.domain.enums import ActivityKind, EventType, Priority
from built.domain.events import append_curation_event, append_event
from built.llm.client import LLMClient
from built.llm.tool_schemas import MAX_GROOM_ACTIONS, PM_TRIAGE_TERMINAL_TOOL, PM_TRIAGE_TOOLS
from built.services import card_service
from built.tools.dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)

_KIND = ActivityKind.PM_TRIAGE


def _format_pm_backlog(cards: list[Card]) -> str:
    lines = []
    for c in cards:
        request = (c.raw_request or "").strip().replace("\n", " ")[:300]
        lines.append(f"- id={c.id} priority={c.priority.value} title={c.title!r}: {request}")
    return "\n".join(lines) or "(nothing here)"


async def run_pm_triage_pass(
    session: AsyncSession,
    project: Project,
    *,
    llm_client: LLMClient,
    dispatcher: ToolDispatcher,
    max_iterations: int,
    run_id: str,
    context_window_config: ContextWindowConfig | None = None,
    agents_doc: str | None = None,
    extra_context: str | None = None,
) -> str:
    """Runs one PM-triage pass and returns a short summary of what changed (for
    project_service.finish_curation_run — the same summary the board's Curation
    panel displays for every other kind). Never raises — mirrors
    run_curation_pass's never-crash-the-worker contract. extra_context is recent
    board activity (visit outcomes + postmortems — orchestrator/curator.py's
    _needs_run), not the PM backlog itself: that's fetched fresh here, the same
    live read _apply_groom_backlog re-checks everything against later. run_id (the
    CurationRun orchestrator/curator.py already opened before calling this) tags
    every event the same way run_curation_pass does."""
    pm_cards = await card_service.list_pm_backlog(session, project.id)
    current_epic = await session.get(Card, project.current_epic_id) if project.current_epic_id else None
    system, user = build_pm_triage_prompt(
        project,
        _format_pm_backlog(pm_cards),
        extra_context or "(nothing recent)",
        agents_doc=agents_doc,
        current_epic=current_epic,
    )
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    config = context_window_config or ContextWindowConfig(
        max_tokens=builtin_settings.default_max_tokens,
        keep_messages=builtin_settings.default_keep_messages,
    )

    try:
        for iteration in range(1, max_iterations + 1):
            token_count = estimate_tokens(messages)
            budget = config.max_tokens - config.keep_tokens
            if token_count > budget * 0.85:
                messages = await compact(messages, llm_client, config, model_name=f"curation-{_KIND.value}")

            result = await llm_client.complete(messages=messages, tools=PM_TRIAGE_TOOLS)
            await append_curation_event(
                session,
                project_id=project.id,
                kind=_KIND,
                run_id=run_id,
                type=EventType.LLM_RESPONSE,
                payload={
                    "iteration": iteration,
                    "content": result.content,
                    "tool_calls": [tc.name for tc in result.tool_calls],
                },
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                latency_ms=result.latency_ms,
            )
            await session.commit()

            if not result.tool_calls:
                messages.append({"role": "assistant", "content": result.content or ""})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Call groom_backlog when you're done reviewing — empty lists are fine if "
                            "nothing here needs changing — or use a read tool to keep exploring."
                        ),
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

            terminal_call = None
            for tool_call in result.tool_calls:
                if tool_call.name == PM_TRIAGE_TERMINAL_TOOL:
                    terminal_call = tool_call
                    continue
                outcome = await dispatcher.dispatch(tool_call.name, tool_call.arguments)
                await append_curation_event(
                    session,
                    project_id=project.id,
                    kind=_KIND,
                    run_id=run_id,
                    type=EventType.TOOL_CALL,
                    payload={
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                        "result": outcome.result.output,
                        "is_error": outcome.result.is_error,
                    },
                )
                await session.commit()
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": outcome.result.output}
                )

            if terminal_call is None:
                continue

            summary = await _apply_groom_backlog(session, project, terminal_call.arguments)
            await append_curation_event(
                session,
                project_id=project.id,
                kind=_KIND,
                run_id=run_id,
                type=EventType.TOOL_CALL,
                payload={"name": terminal_call.name, "arguments": terminal_call.arguments, "result": summary},
            )
            await session.commit()
            return summary

        return "no changes — gave up without calling groom_backlog"
    except Exception as exc:  # noqa: BLE001 — deliberate: a bad pass yields no changes, not a crash
        # Logged (not just persisted) so it shows up on the Logs page in real time,
        # not only after someone thinks to open this run's history.
        logger.exception("pm_triage failed for project %s", project.id)
        await append_curation_event(
            session,
            project_id=project.id,
            kind=_KIND,
            run_id=run_id,
            type=EventType.ERROR,
            payload={"error": str(exc)},
        )
        await session.commit()
        return "no changes — pass errored"


async def _apply_groom_backlog(session: AsyncSession, project: Project, arguments: dict) -> str:
    """Validates groom_backlog's arguments against a *fresh* read of the live PM
    backlog and applies whatever survives. Every card_id must still be open,
    non-archived, and in Column.PM at this exact moment — a card the model saw at
    prompt-build time may have moved on since. Silently drops (never raises on)
    anything that doesn't check out; this is enforcement, not a place to trust the
    model's arguments."""
    live_by_id = {c.id: c for c in await card_service.list_pm_backlog(session, project.id)}
    claimed: set[str] = set()

    reprioritized = 0
    for entry in _as_list(arguments.get("reprioritizations"))[:MAX_GROOM_ACTIONS]:
        if not isinstance(entry, dict):
            continue
        card_id = str(entry.get("card_id", ""))
        reason = str(entry.get("reason", "")).strip()
        card = live_by_id.get(card_id)
        if card is None or not reason:
            continue
        try:
            priority = Priority(str(entry.get("priority", "")).strip().lower())
        except ValueError:
            continue
        if card.priority == priority:
            continue
        old_priority = card.priority
        await card_service.set_priority(session, card_id, priority)
        await append_event(
            session,
            card_id=card_id,
            type=EventType.SYSTEM_NOTE,
            payload={
                "action": "pm_triage_reprioritized",
                "old_priority": old_priority.value,
                "new_priority": priority.value,
                "reason": reason,
            },
        )
        reprioritized += 1

    archived = 0
    for group in _as_list(arguments.get("duplicate_groups"))[:MAX_GROOM_ACTIONS]:
        if not isinstance(group, dict):
            continue
        keep_id = str(group.get("keep_card_id", ""))
        reason = str(group.get("reason", "")).strip()
        if keep_id not in live_by_id or keep_id in claimed or not reason:
            continue
        dup_ids = [
            str(d)
            for d in _as_list(group.get("duplicate_card_ids"))
            if str(d) in live_by_id and str(d) != keep_id and str(d) not in claimed
        ]
        if not dup_ids:
            continue
        claimed.add(keep_id)
        claimed.update(dup_ids)
        for dup_id in dup_ids:
            await card_service.archive_card(session, dup_id)
            await append_event(
                session,
                card_id=dup_id,
                type=EventType.SYSTEM_NOTE,
                payload={"action": "pm_triage_archived_duplicate", "duplicate_of": keep_id, "reason": reason},
            )
        await append_event(
            session,
            card_id=keep_id,
            type=EventType.SYSTEM_NOTE,
            payload={"action": "pm_triage_merge_kept", "archived_duplicates": dup_ids, "reason": reason},
        )
        archived += len(dup_ids)

    if not reprioritized and not archived:
        return "no changes needed"
    parts = []
    if reprioritized:
        parts.append(f"reprioritized {reprioritized} card(s)")
    if archived:
        parts.append(f"archived {archived} duplicate(s)")
    return ", ".join(parts)


def _as_list(value: object) -> list:
    return value if isinstance(value, list) else []
