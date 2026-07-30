"""Project-level chat: a human and an LLM brainstorming and workshopping ticket
ideas together, with read-only repo-browsing tools plus create_ticket/update_ticket
to file or edit cards mid-conversation. The read-only-explore-plus-file-tickets
sibling of agent/curation.py, but step-wise rather than run-to-completion: unlike
every other agent/* module, a "run" here is one bounded reply to whatever's new, not
a loop that only ends via a terminal tool. Persisted history
(services/chat_service.py) is reloaded and replayed on every call — there is no
long-lived in-memory conversation state anywhere in this process.

Holds both the pure step function and its setup/lock wrapper, deviating from the
curator/reviver split of agent/ (mechanism) vs. orchestrator/ (scheduler loop): chat
has no scheduler loop at all, it's invoked synchronously per human message, so
there's nothing for an orchestrator/chat.py to own."""

import asyncio
import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from built.agent.context import build_chat_prompt
from built.config import settings
from built.db.base import async_session_factory
from built.db.models import Card, ChatMessage, Project
from built.domain.enums import ChatRole, Column
from built.llm.client import FallbackLLMClient, LLMClient
from built.llm.tool_schemas import CHAT_TOOLS, MAX_TICKETS_PER_CHAT_TURN
from built.sandbox.container import DockerCommandExecutor
from built.sandbox.worktree import ensure_tool_worktree, read_default_branch_file
from built.services import card_service, chat_service, endpoint_service
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)

# Per-project locks, not a skip-set like curator's _curation_in_progress: chat wants
# a queued turn to wait and then run (a second human message must still get
# answered), not be dropped like a duplicate curation trigger would be.
_chat_locks: dict[str, asyncio.Lock] = {}
_chat_turn_running: set[str] = set()


def lock_for(project_id: str) -> asyncio.Lock:
    lock = _chat_locks.get(project_id)
    if lock is None:
        lock = asyncio.Lock()
        _chat_locks[project_id] = lock
    return lock


def is_chat_turn_running(project_id: str) -> bool:
    return project_id in _chat_turn_running


async def run_chat_turn(
    session: AsyncSession,
    project: Project,
    *,
    llm_client: LLMClient,
    dispatcher: ToolDispatcher,
    max_tool_iterations: int = 6,
    agents_doc: str | None = None,
) -> list[ChatMessage]:
    """Answers whatever unanswered human message sits at the tail of this project's
    (un-cleared) chat transcript, bounded to max_tool_iterations rounds of tool-call
    chaining, and returns the new rows appended. A no-op (returns []) if the
    transcript doesn't currently end on a user message — the guard against a
    redundant trailing reply when several human messages queue up behind one
    in-flight run_chat_turn_activity call (see its docstring). Never raises: an
    endpoint/tool failure ends the turn with a visible error reply instead of
    leaving the human staring at a stuck spinner."""
    history = await chat_service.list_recent_messages(
        session, project.id, cleared_before_seq=project.chat_cleared_before_seq or 0
    )
    if not history or history[-1].role != ChatRole.USER:
        return []

    try:
        existing_titles = await card_service.list_recent_card_titles(session, project.id)
        current_epic = await session.get(Card, project.current_epic_id) if project.current_epic_id else None
        system = build_chat_prompt(project, existing_titles, agents_doc=agents_doc, current_epic=current_epic)
        messages: list[dict] = [
            {"role": "system", "content": system},
            *chat_service.to_openai_messages(history),
        ]

        appended: list[ChatMessage] = []
        tickets_created = 0
        for _ in range(max_tool_iterations):
            result = await llm_client.complete(messages=messages, tools=CHAT_TOOLS)

            if not result.tool_calls:
                row = await chat_service.append_assistant_message(
                    session,
                    project.id,
                    content=result.content or "",
                    tokens_in=result.tokens_in,
                    tokens_out=result.tokens_out,
                    latency_ms=result.latency_ms,
                )
                await session.commit()
                appended.append(row)
                return appended

            stored_calls = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in result.tool_calls
            ]
            assistant_row = await chat_service.append_assistant_message(
                session,
                project.id,
                content=result.content,
                tool_calls=stored_calls,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                latency_ms=result.latency_ms,
            )
            await session.commit()
            appended.append(assistant_row)
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

            for tool_call in result.tool_calls:
                output, is_error, card_id = await _dispatch(
                    session, dispatcher, project, tool_call.name, tool_call.arguments,
                    tickets_created=tickets_created,
                )
                if tool_call.name == "create_ticket" and not is_error:
                    tickets_created += 1
                tool_row = await chat_service.append_tool_message(
                    session,
                    project.id,
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    content=output,
                    is_error=is_error,
                    card_id=card_id,
                )
                await session.commit()
                appended.append(tool_row)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})

        limit_notice = (
            "(Reached this turn's tool-call limit — say \"continue\" if you'd like me to keep going.)"
        )
        row = await chat_service.append_assistant_message(session, project.id, content=limit_notice)
        await session.commit()
        appended.append(row)
        return appended
    except Exception as exc:  # noqa: BLE001 — a bad turn ends visibly, not silently
        logger.exception("chat turn failed for project %s", project.id)
        row = await chat_service.append_assistant_message(
            session, project.id, content=f"Something went wrong on that turn: {exc}", is_error=True
        )
        await session.commit()
        return [row]


async def _dispatch(
    session: AsyncSession,
    dispatcher: ToolDispatcher,
    project: Project,
    name: str,
    arguments: dict,
    *,
    tickets_created: int,
) -> tuple[str, bool, str | None]:
    """Read-only browsing tools go through the shared, DB-agnostic ToolDispatcher
    (same as every other column); create_ticket/update_ticket are dispatched here
    instead, inline, because they need `session` and card_service — the first case
    in this codebase of one tool-call batch needing both a worktree-scoped
    dispatcher AND direct DB access."""
    if name == "create_ticket":
        if tickets_created >= MAX_TICKETS_PER_CHAT_TURN:
            return f"reached this turn's limit of {MAX_TICKETS_PER_CHAT_TURN} new tickets.", True, None
        title = str(arguments.get("title", "")).strip()
        raw_request = str(arguments.get("raw_request", "")).strip()
        if not title or not raw_request:
            return "title and raw_request are both required.", True, None
        card = await card_service.create_card(
            session, project.id, title=title, raw_request=raw_request, source="chat"
        )
        await session.commit()
        return f"Created card {card.id}: {card.title!r} (now in the PM column).", False, card.id
    if name == "update_ticket":
        card_id = str(arguments.get("card_id", ""))
        title = str(arguments.get("title", "")).strip()
        raw_request = str(arguments.get("raw_request", "")).strip()
        if not card_id or not title or not raw_request:
            return "card_id, title, and raw_request are all required.", True, None
        try:
            card = await card_service.update_card(session, card_id, title=title, raw_request=raw_request)
        except card_service.NotFoundError:
            return f"no such card: {card_id!r}", True, None
        await session.commit()
        return f"Updated card {card.id}: {card.title!r}.", False, card.id
    outcome = await dispatcher.dispatch(name, arguments)
    return outcome.result.output, outcome.result.is_error, None


async def run_chat_turn_activity(project_id: str) -> None:
    """Setup wrapper mirroring orchestrator/curator.py's run_curation_activity —
    minus any scheduler, since chat has none. Meant to be fired as a detached
    background task per human message (ui/routers/chat.py). Serialized per project
    via lock_for: two turns for the same project must never race the shared
    tool="chat" worktree or interleave replies. Once a queued call finally gets the
    lock, run_chat_turn's own no-op guard prevents an extra, redundant reply if an
    earlier call already answered everything new."""
    async with lock_for(project_id):
        _chat_turn_running.add(project_id)
        try:
            async with async_session_factory() as session:
                project = await session.get(Project, project_id)
                if project is None:
                    logger.warning("chat turn requested for missing project %s", project_id)
                    return
                try:
                    chain = await endpoint_service.get_resolved_chain(
                        session, project_id=project.id, role=Column.PM
                    )
                    llm_client = FallbackLLMClient(chain)
                    wt_path = await ensure_tool_worktree(project, tool="chat")
                except Exception:
                    logger.exception("chat turn setup failed for project %s", project_id)
                    return
                executor_kwargs = {"image": project.sandbox_image} if project.sandbox_image else {}
                dispatcher = ToolDispatcher(
                    ctx=ToolContext(card_id=f"chat-{project.id}", worktree_root=wt_path),
                    executor=DockerCommandExecutor(**executor_kwargs),
                )
                agents_doc = await read_default_branch_file(project, "AGENTS.md")
                await run_chat_turn(
                    session,
                    project,
                    llm_client=llm_client,
                    dispatcher=dispatcher,
                    max_tool_iterations=settings.chat_max_tool_iterations,
                    agents_doc=agents_doc,
                )
        finally:
            _chat_turn_running.discard(project_id)
