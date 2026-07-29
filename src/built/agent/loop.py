"""The core agentic loop, shared by every column: build context -> call the LLM
(fallback chain) -> dispatch tool calls -> repeat until a terminal tool is called or a
cap is hit. Terminal transitions and the iteration/error safety valves live in
domain/transitions.py — this module drives the loop and hands off to that state
machine, it doesn't reimplement it.

Commits happen after each LLM response and each tool call (not just once at the end)
so a card's progress is visible to anything polling CardEvent mid-run — the dashboard
(Phase 6) reads this table live."""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from built.agent.context import (
    build_deployer_prompt,
    build_developer_prompt,
    build_pm_prompt,
    build_tester_prompt,
)
from built.agent.context_window import (
    ContextWindowConfig,
    compact,
    estimate_tokens,
)
from built.db.models import Card, CardColumnVisit, Project
from built.domain import run_attempts, transitions
from built.domain.enums import DeployMode, EventType, LifecycleState
from built.domain.events import append_event
from built.llm.client import LLMClient, ToolCallRequest
from built.llm.tool_schemas import (
    DEPLOYER_ABANDON_TERMINAL_TOOL,
    DEPLOYER_AUTO_MAIN_TERMINAL_TOOL,
    DEPLOYER_PR_TERMINAL_TOOL,
    DEVELOPER_TERMINAL_TOOL,
    DEVELOPER_TOOLS,
    PM_TERMINAL_TOOL,
    PM_TOOLS,
    TESTER_TOOLS,
    deployer_tools,
)
from built.sandbox import deploy_runner
from built.tools.dispatcher import DispatchOutcome, ToolDispatcher


async def _maybe_compact(
    messages: list[dict],
    llm_client: LLMClient,
    config: ContextWindowConfig,
    iteration: int,
) -> list[dict]:
    """Compact messages if they approach the context window. Skips silently
    when config.max_tokens is very large, and only does a first-time compact
    to avoid repeated summarization overhead."""
    token_count = estimate_tokens(messages)
    budget = config.max_tokens - config.keep_tokens
    if token_count <= budget * 0.85:
        return messages
    # Compact the message list — the summarizer uses the first endpoint in
    # the chain (same model family is typical for a fallback chain).
    compacted = await compact(
        messages,
        llm_client,
        config,
        model_name=f"iteration-{iteration}",
    )
    return compacted


@dataclass
class TerminalHandlerResult:
    handled: bool
    feedback: str | None = None


TerminalHandler = Callable[
    [AsyncSession, Card, CardColumnVisit, ToolCallRequest, str], Awaitable[TerminalHandlerResult]
]
OnToolResult = Callable[
    [AsyncSession, Card, CardColumnVisit, ToolCallRequest, DispatchOutcome, int], Awaitable[None]
]


async def run_column_visit(
    session: AsyncSession,
    card: Card,
    visit: CardColumnVisit,
    *,
    llm_client: LLMClient,
    dispatcher: ToolDispatcher,
    max_iterations: int,
    system_prompt: str,
    user_prompt: str,
    tools: list[dict],
    terminal_handlers: dict[str, TerminalHandler],
    context_window_config: ContextWindowConfig | None = None,
    on_tool_result: OnToolResult | None = None,
) -> Card:
    """Runs one column visit to completion and applies the resulting domain
    transition. Never raises for ordinary run failures — those come back as a
    BLOCKED card (via fail_visit_with_error), not an exception the caller must handle."""
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    config = context_window_config or ContextWindowConfig(
        max_tokens=128_000,
    )

    try:
        for iteration in range(1, max_iterations + 1):
            # A human can cancel a card at any moment via the UI/API, independent of
            # whatever this loop is doing — cancel_card only flips lifecycle_state,
            # it doesn't (can't, from a different request) reach into a running
            # loop. Without this check the loop has no way to find out and just
            # keeps going to its natural conclusion — with
            # orchestrator_concurrency's default of 1, that means the single worker
            # stays monopolized on now-unwanted work for as long as the run takes,
            # unable to pick up anything else in the meantime (confirmed in
            # production: a cancelled card kept running for 10+ minutes after
            # cancellation before a restart finally stopped it).
            await session.refresh(card)
            if card.lifecycle_state != LifecycleState.ACTIVE:
                await transitions.abandon_visit_for_lifecycle_change(session, card, visit)
                await session.commit()
                return card

            # Compact if the message list approaches the context window.
            messages = await _maybe_compact(messages, llm_client, config, iteration)

            result = await llm_client.complete(messages=messages, tools=tools)
            await append_event(
                session,
                card_id=card.id,
                column_visit_id=visit.id,
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
                # No tool call at all: nudge, don't silently treat plain text as done.
                messages.append({"role": "assistant", "content": result.content or ""})
                messages.append(
                    {
                        "role": "user",
                        "content": "You must call a tool to make progress or finish. "
                        "Plain text responses are not saved.",
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

            terminal_call: ToolCallRequest | None = None
            for tool_call in result.tool_calls:
                if tool_call.name in terminal_handlers:
                    terminal_call = tool_call
                    continue  # handled once all non-terminal calls in this turn ran
                outcome = await dispatcher.dispatch(tool_call.name, tool_call.arguments)
                event = await append_event(
                    session,
                    card_id=card.id,
                    column_visit_id=visit.id,
                    type=EventType.TOOL_CALL,
                    payload={
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                        "result": outcome.result.output,
                        "is_error": outcome.result.is_error,
                        "commit_sha": outcome.commit_sha,
                    },
                )
                if on_tool_result is not None:
                    await on_tool_result(session, card, visit, tool_call, outcome, event.seq)
                await session.commit()
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": outcome.result.output}
                )

            if terminal_call is not None:
                handler = terminal_handlers[terminal_call.name]
                handler_result = await handler(session, card, visit, terminal_call, result.endpoint_used)
                if handler_result.handled:
                    await session.commit()
                    return card
                # Rejected server-side (e.g. Tester's approve without a passing run) —
                # tell the model why and keep going, rather than ending the visit.
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": terminal_call.id,
                        "content": handler_result.feedback or "Not accepted yet — keep working.",
                    }
                )

        await transitions.fail_visit_with_error(
            session, card, visit, message=f"exceeded max_iterations_per_run ({max_iterations})"
        )
        await session.commit()
        return card
    except Exception as exc:  # noqa: BLE001 — deliberate: any unhandled failure blocks the card, not the process
        await transitions.fail_visit_with_error(session, card, visit, message=f"unhandled error: {exc!r}")
        await session.commit()
        return card


# --- Terminal handlers -------------------------------------------------------------


async def _pm_submit_spec_handler(
    session: AsyncSession, card: Card, visit: CardColumnVisit, tool_call: ToolCallRequest, endpoint_used: str
) -> TerminalHandlerResult:
    criteria = tool_call.arguments.get("acceptance_criteria")
    if not isinstance(criteria, list) or not all(isinstance(c, str) for c in criteria) or not criteria:
        return TerminalHandlerResult(
            handled=False, feedback="acceptance_criteria must be a non-empty list of strings."
        )
    await transitions.complete_pm_visit(
        session,
        card,
        visit,
        spec=str(tool_call.arguments.get("spec", "")),
        acceptance_criteria=criteria,
        summary=str(tool_call.arguments.get("summary", "")),
        endpoint_used=endpoint_used,
    )
    return TerminalHandlerResult(handled=True)


async def _require_passing_test_run(
    session: AsyncSession, card: Card, visit: CardColumnVisit, *, verb: str
) -> str | None:
    """Shared gate for Developer's submit_for_test and Tester's approve. Returns
    feedback to send back to the model if the handoff isn't earned yet, or None if
    it's clear to proceed."""
    project = await session.get(Project, card.project_id)
    if not project.test_command:
        return (
            "This project has no test command configured, so there's no way to verify work "
            f"server-side — {verb} is blocked until an operator sets one in project settings."
        )
    if not await run_attempts.has_passing_run_since_last_change(session, visit.id, project.test_command):
        return (
            f"No passing run of the project's test command is recorded since your last change. "
            f"Run exactly `{project.test_command}` via bash, confirm exit code 0, and don't touch "
            f"any file afterward (write_file/edit_file/bash) before calling {verb} — this is "
            "checked server-side against your most recent matching run."
        )
    return None


async def _developer_submit_for_test_handler(
    session: AsyncSession, card: Card, visit: CardColumnVisit, tool_call: ToolCallRequest, endpoint_used: str
) -> TerminalHandlerResult:
    feedback = await _require_passing_test_run(session, card, visit, verb="submit_for_test")
    if feedback is not None:
        return TerminalHandlerResult(handled=False, feedback=feedback)
    await transitions.complete_developer_visit(
        session, card, visit, summary=str(tool_call.arguments.get("summary", "")), endpoint_used=endpoint_used
    )
    return TerminalHandlerResult(handled=True)


async def _tester_approve_handler(
    session: AsyncSession, card: Card, visit: CardColumnVisit, tool_call: ToolCallRequest, endpoint_used: str
) -> TerminalHandlerResult:
    feedback = await _require_passing_test_run(session, card, visit, verb="approve")
    if feedback is not None:
        return TerminalHandlerResult(handled=False, feedback=feedback)
    await transitions.complete_tester_visit_approved(
        session, card, visit, summary=str(tool_call.arguments.get("notes", "")), endpoint_used=endpoint_used
    )
    return TerminalHandlerResult(handled=True)


async def _tester_request_changes_handler(
    session: AsyncSession, card: Card, visit: CardColumnVisit, tool_call: ToolCallRequest, endpoint_used: str
) -> TerminalHandlerResult:
    feedback = str(tool_call.arguments.get("feedback", ""))
    summary = str(tool_call.arguments.get("summary") or feedback[:120])
    await transitions.complete_tester_visit_changes_requested(
        session, card, visit, feedback=feedback, summary=summary, endpoint_used=endpoint_used
    )
    return TerminalHandlerResult(handled=True)


async def _deployer_abandon_handler(
    session: AsyncSession, card: Card, visit: CardColumnVisit, tool_call: ToolCallRequest, endpoint_used: str
) -> TerminalHandlerResult:
    reason = str(tool_call.arguments.get("reason", "")) or "abandoned without a stated reason"
    await transitions.complete_deployer_visit(
        session, card, visit, success=False, summary=f"abandoned: {reason}", endpoint_used=endpoint_used
    )
    return TerminalHandlerResult(handled=True)


async def _deployer_open_pr_handler(
    session: AsyncSession, card: Card, visit: CardColumnVisit, tool_call: ToolCallRequest, endpoint_used: str
) -> TerminalHandlerResult:
    project = await session.get(Project, card.project_id)
    result = await deploy_runner.open_pull_request(
        project, card, summary=str(tool_call.arguments.get("summary", ""))
    )
    await transitions.complete_deployer_visit(
        session,
        card,
        visit,
        success=result.success,
        summary=result.message,
        deploy_url=result.url,
        endpoint_used=endpoint_used,
    )
    return TerminalHandlerResult(handled=True)


async def _record_bash_run_attempt(
    session: AsyncSession,
    card: Card,
    visit: CardColumnVisit,
    tool_call: ToolCallRequest,
    outcome: DispatchOutcome,
    event_seq: int,
) -> None:
    if tool_call.name == "bash" and outcome.command_result is not None:
        await run_attempts.record_run_attempt(
            session,
            card_id=card.id,
            column_visit_id=visit.id,
            command=str(tool_call.arguments.get("command", "")),
            exit_code=outcome.command_result.exit_code,
            stdout=outcome.command_result.stdout,
            stderr=outcome.command_result.stderr,
            card_event_seq=event_seq,
        )


# --- Per-role entry points -----------------------------------------------------------


async def run_pm_visit(
    session: AsyncSession,
    project: Project,
    card: Card,
    visit: CardColumnVisit,
    *,
    llm_client: LLMClient,
    dispatcher: ToolDispatcher,
    max_iterations: int,
    context_window_config: ContextWindowConfig | None = None,
    retry_recap: str | None = None,
    retry_note: str | None = None,
    agents_doc: str | None = None,
) -> Card:
    system, user = build_pm_prompt(
        project, card, retry_recap=retry_recap, retry_note=retry_note, agents_doc=agents_doc
    )
    return await run_column_visit(
        session,
        card,
        visit,
        llm_client=llm_client,
        dispatcher=dispatcher,
        max_iterations=max_iterations,
        system_prompt=system,
        user_prompt=user,
        tools=PM_TOOLS,
        terminal_handlers={PM_TERMINAL_TOOL: _pm_submit_spec_handler},
        context_window_config=context_window_config,
    )


async def run_developer_visit(
    session: AsyncSession,
    project: Project,
    card: Card,
    visit: CardColumnVisit,
    *,
    llm_client: LLMClient,
    dispatcher: ToolDispatcher,
    max_iterations: int,
    context_window_config: ContextWindowConfig | None = None,
    retry_recap: str | None = None,
    retry_note: str | None = None,
    agents_doc: str | None = None,
) -> Card:
    system, user = build_developer_prompt(
        project, card, retry_recap=retry_recap, retry_note=retry_note, agents_doc=agents_doc
    )
    return await run_column_visit(
        session,
        card,
        visit,
        llm_client=llm_client,
        dispatcher=dispatcher,
        max_iterations=max_iterations,
        system_prompt=system,
        user_prompt=user,
        tools=DEVELOPER_TOOLS,
        terminal_handlers={DEVELOPER_TERMINAL_TOOL: _developer_submit_for_test_handler},
        context_window_config=context_window_config,
        on_tool_result=_record_bash_run_attempt,
    )


async def run_tester_visit(
    session: AsyncSession,
    project: Project,
    card: Card,
    visit: CardColumnVisit,
    *,
    llm_client: LLMClient,
    dispatcher: ToolDispatcher,
    max_iterations: int,
    context_window_config: ContextWindowConfig | None = None,
    developer_summary: str | None = None,
    retry_recap: str | None = None,
    retry_note: str | None = None,
    agents_doc: str | None = None,
) -> Card:
    system, user = build_tester_prompt(
        project,
        card,
        developer_summary=developer_summary,
        retry_recap=retry_recap,
        retry_note=retry_note,
        agents_doc=agents_doc,
    )
    return await run_column_visit(
        session,
        card,
        visit,
        llm_client=llm_client,
        dispatcher=dispatcher,
        max_iterations=max_iterations,
        system_prompt=system,
        user_prompt=user,
        tools=TESTER_TOOLS,
        terminal_handlers={
            "approve": _tester_approve_handler,
            "request_changes": _tester_request_changes_handler,
        },
        on_tool_result=_record_bash_run_attempt,
        context_window_config=context_window_config,
    )


async def run_deployer_visit(
    session: AsyncSession,
    project: Project,
    card: Card,
    visit: CardColumnVisit,
    *,
    llm_client: LLMClient,
    dispatcher: ToolDispatcher,
    max_iterations: int,
    context_window_config: ContextWindowConfig | None = None,
    mode: DeployMode,
    retry_recap: str | None = None,
    retry_note: str | None = None,
    agents_doc: str | None = None,
) -> Card:
    system, user = build_deployer_prompt(
        project, card, mode=mode, retry_recap=retry_recap, retry_note=retry_note, agents_doc=agents_doc
    )
    if mode == DeployMode.AUTO_MAIN:

        async def _run_deploy_handler(
            session: AsyncSession,
            card: Card,
            visit: CardColumnVisit,
            tool_call: ToolCallRequest,
            endpoint_used: str,
        ) -> TerminalHandlerResult:
            # Closure, not a free function: run_auto_main_deploy needs the exact
            # worktree the dispatcher's file tools are scoped to, so a conflict fix
            # made via write_file/edit_file in an earlier turn is the same worktree
            # this looks at when the agent calls run_deploy() again.
            result = await deploy_runner.run_auto_main_deploy(project, card, dispatcher.ctx.worktree_root)
            if result.conflict:
                return TerminalHandlerResult(handled=False, feedback=result.message)
            await transitions.complete_deployer_visit(
                session,
                card,
                visit,
                success=result.success,
                summary=result.message,
                pending_ci_commit_sha=result.commit_sha,
                endpoint_used=endpoint_used,
            )
            return TerminalHandlerResult(handled=True)

        terminal_handlers = {
            DEPLOYER_AUTO_MAIN_TERMINAL_TOOL: _run_deploy_handler,
            DEPLOYER_ABANDON_TERMINAL_TOOL: _deployer_abandon_handler,
        }
    else:
        terminal_handlers = {DEPLOYER_PR_TERMINAL_TOOL: _deployer_open_pr_handler}
    return await run_column_visit(
        session,
        card,
        visit,
        llm_client=llm_client,
        dispatcher=dispatcher,
        max_iterations=max_iterations,
        system_prompt=system,
        user_prompt=user,
        tools=deployer_tools(mode),
        terminal_handlers=terminal_handlers,
        context_window_config=context_window_config,
    )
