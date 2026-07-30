"""Curation: a family of read-only passes over a project that never edit the repo —
the only thing any of them can do is propose new cards, exactly like a human PM
filing a ticket. Structurally similar to agent/loop.py's run_column_visit, but
without CardColumnVisit/CardEvent bookkeeping — there's no card to attach a
transcript to until propose_tasks actually creates one.

Several kinds (ActivityKind), one mechanism: explore with read-only tools, decide,
call propose_tasks. See orchestrator/curator.py for scheduling and cadence."""

import json

from sqlalchemy.ext.asyncio import AsyncSession

from built.agent.context import build_curation_prompt
from built.agent.context_window import (
    ContextWindowConfig,
    compact,
    estimate_tokens,
)
from built.config import settings as builtin_settings
from built.db.models import Card, Project
from built.domain.enums import ActivityKind, EventType
from built.domain.events import append_curation_event
from built.llm.client import LLMClient
from built.llm.tool_schemas import CURATION_TERMINAL_TOOL, CURATION_TOOLS, MAX_PROPOSED_TASKS
from built.services import card_service
from built.tools.dispatcher import ToolDispatcher


async def run_curation_pass(
    session: AsyncSession,
    project: Project,
    kind: ActivityKind,
    *,
    llm_client: LLMClient,
    dispatcher: ToolDispatcher,
    max_iterations: int,
    context_window_config: ContextWindowConfig | None = None,
    agents_doc: str | None = None,
    extra_context: str | None = None,
) -> list[Card]:
    """Runs one curation pass and returns whatever new cards it created — possibly
    none, if the model never called propose_tasks within the iteration budget or an
    endpoint/tool failure ended the run early. Never raises for ordinary failures,
    mirroring run_column_visit's never-crash-the-worker contract."""
    existing_titles = await card_service.list_recent_card_titles(session, project.id)
    system, user = build_curation_prompt(
        project, kind, existing_titles, agents_doc=agents_doc, extra_context=extra_context
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
            # Compact if the message list approaches the context window.
            token_count = estimate_tokens(messages)
            budget = config.max_tokens - config.keep_tokens
            if token_count > budget * 0.85:
                messages = await compact(
                    messages,
                    llm_client,
                    config,
                    model_name=f"curation-{kind.value}",
                )

            result = await llm_client.complete(messages=messages, tools=CURATION_TOOLS)
            await append_curation_event(
                session,
                project_id=project.id,
                kind=kind,
                type=EventType.LLM_RESPONSE,
                payload={
                    "iteration": iteration,
                    "content": result.content,
                    "tool_calls": [tc.name for tc in result.tool_calls],
                },
            )
            await session.commit()

            if not result.tool_calls:
                messages.append({"role": "assistant", "content": result.content or ""})
                messages.append(
                    {
                        "role": "user",
                        "content": "Call propose_tasks when ready, or a read tool to keep exploring.",
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
                if tool_call.name == CURATION_TERMINAL_TOOL:
                    terminal_call = tool_call
                    continue
                outcome = await dispatcher.dispatch(tool_call.name, tool_call.arguments)
                await append_curation_event(
                    session,
                    project_id=project.id,
                    kind=kind,
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

            tasks = terminal_call.arguments.get("tasks")
            created = await _create_proposed_cards(session, project, kind, tasks)
            await append_curation_event(
                session,
                project_id=project.id,
                kind=kind,
                type=EventType.TOOL_CALL,
                payload={
                    "name": terminal_call.name,
                    "arguments": terminal_call.arguments,
                    "result": f"created {len(created)} card(s)" if created else "rejected: no valid tasks",
                    "is_error": not created,
                },
            )
            await session.commit()
            if created:
                return created
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": terminal_call.id,
                    "content": "tasks must be a non-empty list of {title, raw_request} objects.",
                }
            )

        return []
    except Exception as exc:  # noqa: BLE001 — deliberate: a bad curation run yields no cards, not a crash
        # Still record what happened — otherwise a broken endpoint just looks like
        # the pass never ran at all, with no clue why in the status panel.
        await append_curation_event(
            session, project_id=project.id, kind=kind, type=EventType.ERROR, payload={"error": str(exc)}
        )
        await session.commit()
        return []


async def _create_proposed_cards(
    session: AsyncSession, project: Project, kind: ActivityKind, tasks: object
) -> list[Card]:
    if not isinstance(tasks, list) or not tasks:
        return []
    created: list[Card] = []
    for task in tasks[:MAX_PROPOSED_TASKS]:
        if not isinstance(task, dict):
            continue
        title = str(task.get("title", "")).strip()
        raw_request = str(task.get("raw_request", "")).strip()
        if not title or not raw_request:
            continue
        card = await card_service.create_card(
            session, project.id, title=title, raw_request=raw_request, source=f"curation:{kind.value}"
        )
        created.append(card)
    if created:
        await session.commit()
    return created
