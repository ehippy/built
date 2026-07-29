"""PM's autonomous discovery mode: not tied to any card. Explores the repository on
its own initiative and proposes new work as new cards, rather than refining a request
a human already wrote. Structurally similar to agent/loop.py's run_column_visit, but
without CardColumnVisit/CardEvent bookkeeping — there's no card to attach a transcript
to until propose_tasks actually creates one."""

import json

from sqlalchemy.ext.asyncio import AsyncSession

from built.agent.context import build_discovery_prompt
from built.agent.context_window import (
    ContextWindowConfig,
    compact,
    estimate_tokens,
)
from built.config import settings as builtin_settings
from built.db.models import Card, Project
from built.llm.client import LLMClient
from built.llm.tool_schemas import DISCOVERY_TERMINAL_TOOL, DISCOVERY_TOOLS, MAX_PROPOSED_TASKS
from built.services import card_service
from built.tools.dispatcher import ToolDispatcher


async def run_pm_discovery(
    session: AsyncSession,
    project: Project,
    *,
    llm_client: LLMClient,
    dispatcher: ToolDispatcher,
    max_iterations: int,
    context_window_config: ContextWindowConfig | None = None,
) -> list[Card]:
    """Runs one discovery pass and returns whatever new cards it created — possibly
    none, if the model never called propose_tasks within the iteration budget or an
    endpoint/tool failure ended the run early. Never raises for ordinary failures,
    mirroring run_column_visit's never-crash-the-worker contract."""
    existing_titles = await card_service.list_recent_card_titles(session, project.id)
    system, user = build_discovery_prompt(project, existing_titles)
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    config = context_window_config or ContextWindowConfig(
        max_tokens=builtin_settings.default_max_tokens,
        keep_messages=builtin_settings.default_keep_messages,
    )

    try:
        for _ in range(max_iterations):
            # Compact if the message list approaches the context window.
            token_count = estimate_tokens(messages)
            budget = config.max_tokens - config.keep_tokens
            if token_count > budget * 0.85:
                messages = await compact(
                    messages,
                    llm_client,
                    config,
                    model_name="discovery",
                )

            result = await llm_client.complete(messages=messages, tools=DISCOVERY_TOOLS)

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
                if tool_call.name == DISCOVERY_TERMINAL_TOOL:
                    terminal_call = tool_call
                    continue
                outcome = await dispatcher.dispatch(tool_call.name, tool_call.arguments)
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": outcome.result.output}
                )

            if terminal_call is None:
                continue

            tasks = terminal_call.arguments.get("tasks")
            created = await _create_proposed_cards(session, project, tasks)
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
    except Exception:  # noqa: BLE001 — deliberate: a bad discovery run yields no cards, not a crash
        return []


async def _create_proposed_cards(session: AsyncSession, project: Project, tasks: object) -> list[Card]:
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
            session, project.id, title=title, raw_request=raw_request, source="pm_discovery"
        )
        created.append(card)
    if created:
        await session.commit()
    return created
