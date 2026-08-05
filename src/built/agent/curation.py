"""Curation: a family of read-only passes over a project that never edit the repo —
the only thing any of them can do is propose new cards, exactly like a human PM
filing a ticket. Structurally similar to agent/loop.py's run_column_visit, but
without CardColumnVisit/CardEvent bookkeeping — there's no card to attach a
transcript to until propose_tasks actually creates one.

Several kinds (ActivityKind), one mechanism: explore with read-only tools, decide,
call propose_tasks. See orchestrator/curator.py for scheduling and cadence."""

import json
import logging
from dataclasses import dataclass
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from built.agent.context import build_curation_prompt
from built.agent.context_window import (
    ContextWindowConfig,
    compact,
    estimate_tokens,
)
from built.config import settings as builtin_settings
from built.db.models import Card, Project
from built.domain.enums import ActivityKind, EventType, Severity
from built.domain.events import append_curation_event
from built.llm.client import LLMClient
from built.llm.tool_schemas import (
    CURATION_DEDUP_TERMINAL_TOOL,
    CURATION_DEDUP_TOOLS,
    CURATION_TERMINAL_TOOL,
    CURATION_TOOLS,
    MAX_PROPOSED_TASKS,
    OVERSEER_PROMPT_ASSESSMENT_TERMINAL_TOOL,
    OVERSEER_PROMPT_ASSESSMENT_TOOLS,
    OVERSEER_TOOLS,
)
from built.services import card_service
from built.tools.dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)

FilePolicy = Literal["file_all_above_bar", "file_best_only"]

# Server-side enforcement of how many of a pass's proposed candidates actually
# become cards — never trusts the model's own restraint about volume, mirroring
# this codebase's "server-verified, not taken on the model's say-so" pattern
# elsewhere (domain/run_attempts.py for Developer/Tester/Deployer transitions).
# file_all_above_bar: file every candidate rated critical/high, drop medium/low.
# file_best_only: file just the single highest-severity candidate from the batch.
# agents_md and retro have no entry — _whittle_tasks skips severity filtering for
# any kind not listed here (each of their own prompts already caps it at one card).
#
# OVERSEER uses file_all_above_bar: the old "cap most kinds to a single candidate"
# policy existed to stop several independent, uncontrolled lenses (bug_sweep,
# opportunity_brainstorm, polish_review, ...) from flooding the backlog
# simultaneously — that reasoning doesn't transfer to one operator-scoped kind,
# where the operator's own overseer_prompt already controls scope/depth. Severity
# self-rating is the real backstop instead of an artificial one-card cap; revisit
# if OVERSEER turns out to over-file in practice.
_CURATION_FILE_POLICY: dict[ActivityKind, FilePolicy] = {
    ActivityKind.OVERSEER: "file_all_above_bar",
}

_SEVERITY_RANK: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
}


async def run_curation_pass(
    session: AsyncSession,
    project: Project,
    kind: ActivityKind,
    *,
    llm_client: LLMClient,
    dispatcher: ToolDispatcher,
    max_iterations: int,
    run_id: str,
    context_window_config: ContextWindowConfig | None = None,
    agents_doc: str | None = None,
    extra_context: str | None = None,
) -> list[Card]:
    """Runs one curation pass and returns whatever new cards it created — possibly
    none, if the model never called propose_tasks within the iteration budget or an
    endpoint/tool failure ended the run early. Never raises for ordinary failures,
    mirroring run_column_visit's never-crash-the-worker contract. run_id (the
    CurationRun orchestrator/curator.py already opened before calling this) tags
    every event so project_service.finish_curation_run can later pull just this
    invocation's transcript back out of the (project, kind) stream to sum its token
    usage and detect whether it errored."""
    existing_titles = [c.title for c in await card_service.list_open_cards(session, project.id)]
    current_epic = await session.get(Card, project.current_epic_id) if project.current_epic_id else None
    system, user = build_curation_prompt(
        project,
        kind,
        existing_titles,
        agents_doc=agents_doc,
        extra_context=extra_context,
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

            tools = OVERSEER_TOOLS if kind == ActivityKind.OVERSEER else CURATION_TOOLS
            result = await llm_client.complete(messages=messages, tools=tools)
            await append_curation_event(
                session,
                project_id=project.id,
                kind=kind,
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

            tasks = terminal_call.arguments.get("tasks")
            tasks = await _whittle_tasks(session, project, kind, tasks, llm_client=llm_client)
            created = await _create_proposed_cards(session, project, kind, tasks)
            await append_curation_event(
                session,
                project_id=project.id,
                kind=kind,
                run_id=run_id,
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
        # the pass never ran at all, with no clue why in the status panel. Also
        # logged (not just persisted) so it shows up on the Logs page in real
        # time, not only after someone thinks to open this run's history.
        logger.exception("curation %s failed for project %s", kind.value, project.id)
        await append_curation_event(
            session,
            project_id=project.id,
            kind=kind,
            run_id=run_id,
            type=EventType.ERROR,
            payload={"error": str(exc)},
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


def _parsed_severity(task: dict) -> Severity | None:
    try:
        return Severity(str(task.get("severity", "")).strip().lower())
    except ValueError:
        return None  # missing/non-compliant model output — untrusted, never treated as "low"


async def _apply_file_policy(
    session: AsyncSession, project: Project, kind: ActivityKind, tasks: list[dict], policy: FilePolicy
) -> list[dict]:
    """Server-side enforcement of _CURATION_FILE_POLICY — never trusts the model's
    own restraint about how many candidates are actually worth filing. A missing or
    non-enum severity is treated as not clearing the bar, not silently passed
    through: a model that skips rating shouldn't get a free pass over one that
    rated honestly and lost."""
    rated = [(task, _parsed_severity(task)) for task in tasks]
    if policy == "file_all_above_bar":
        kept = [task for task, severity in rated if severity in (Severity.CRITICAL, Severity.HIGH)]
    else:  # file_best_only — ties broken by original order via enumerate index
        _, (best_task, _) = min(
            enumerate(rated), key=lambda pair: (_SEVERITY_RANK.get(pair[1][1], len(_SEVERITY_RANK)), pair[0])
        )
        kept = [best_task]
    dropped = [task for task in tasks if task not in kept]
    if dropped:
        await append_curation_event(
            session,
            project_id=project.id,
            kind=kind,
            type=EventType.SYSTEM_NOTE,
            payload={
                "stage": "severity_filter",
                "policy": policy,
                # Raw severity string preserved (not just the parsed enum) so an
                # invalid/missing rating is distinguishable from a genuine "low".
                "dropped": [{"title": t.get("title"), "severity": t.get("severity")} for t in dropped],
                "kept": [t.get("title") for t in kept],
            },
        )
        await session.commit()
    return kept


async def _dedupe_against_live_cards(
    session: AsyncSession,
    project: Project,
    kind: ActivityKind,
    candidates: list[dict],
    *,
    llm_client: LLMClient,
) -> list[dict]:
    """The enforcement checkpoint for duplicate detection — re-fetches the live
    open-card set fresh, right before cards would be created, rather than trusting
    the "existing titles" hint baked into the prompt at the start of the pass
    (run_curation_pass). That hint is steering, not enforcement: it can be stale by
    the time propose_tasks is actually called (a long multi-iteration exploration
    turn), and two different kinds can run concurrently for the same project (see
    orchestrator/curator.py's _curation_in_progress), so a pass never sees another
    in-flight pass's about-to-be-created cards unless it checks again right here."""
    if not candidates:
        return candidates
    live = await card_service.list_open_cards(session, project.id)
    if not live:
        return candidates  # nothing to compare against — skip the LLM call entirely

    numbered = "\n".join(
        f"{i}. {c.get('title', '')}: {str(c.get('raw_request', ''))[:200]}" for i, c in enumerate(candidates)
    )
    open_desc = "\n".join(f"- id={c.id} title={c.title!r}: {(c.raw_request or '')[:200]}" for c in live)
    messages = [
        {
            "role": "system",
            "content": (
                "You check proposed backlog candidates against a project's already-open cards for "
                "genuine duplicates — the same underlying request, not just a similar topic. Call "
                "report_duplicate_candidates exactly once."
            ),
        },
        {"role": "user", "content": f"Proposed candidates:\n{numbered}\n\nExisting open cards:\n{open_desc}"},
    ]
    try:
        result = await llm_client.complete(messages=messages, tools=CURATION_DEDUP_TOOLS)
    except Exception as exc:  # noqa: BLE001 — fail open: a broken dedup call shouldn't lose real work
        await append_curation_event(
            session,
            project_id=project.id,
            kind=kind,
            type=EventType.ERROR,
            payload={"stage": "dedup", "error": str(exc)},
        )
        await session.commit()
        return candidates

    dup_indices: set[int] = set()
    for tool_call in result.tool_calls:
        if tool_call.name != CURATION_DEDUP_TERMINAL_TOOL:
            continue
        for dup in tool_call.arguments.get("duplicates", []):
            idx = dup.get("candidate_index")
            if isinstance(idx, int) and 0 <= idx < len(candidates):
                dup_indices.add(idx)

    survivors = [c for i, c in enumerate(candidates) if i not in dup_indices]
    dropped = [c for i, c in enumerate(candidates) if i in dup_indices]
    if dropped:
        await append_curation_event(
            session,
            project_id=project.id,
            kind=kind,
            type=EventType.SYSTEM_NOTE,
            payload={"stage": "dedup", "dropped": [c.get("title") for c in dropped]},
        )
        await session.commit()
    return survivors


async def _whittle_tasks(
    session: AsyncSession, project: Project, kind: ActivityKind, tasks: object, *, llm_client: LLMClient
) -> list[dict]:
    """Severity-policy filtering, then live-DB dedup — the one place any curation
    kind's raw propose_tasks output gets narrowed down before _create_proposed_cards
    ever inserts a row. Called once from run_curation_pass, right after the
    terminal call, before _create_proposed_cards."""
    if not isinstance(tasks, list):
        return []
    valid = [
        t
        for t in tasks
        if isinstance(t, dict) and str(t.get("title", "")).strip() and str(t.get("raw_request", "")).strip()
    ]
    if not valid:
        return []
    policy = _CURATION_FILE_POLICY.get(kind)
    survivors = await _apply_file_policy(session, project, kind, valid, policy) if policy else valid
    if not survivors:
        return []
    return await _dedupe_against_live_cards(session, project, kind, survivors, llm_client=llm_client)


@dataclass
class OverseerPromptAssessment:
    comprehensive: bool
    issues: list[str]


async def assess_overseer_prompt(
    prompt: str, project: Project, *, llm_client: LLMClient
) -> OverseerPromptAssessment:
    """Single-shot judgment call (not the explore-then-propose loop) run whenever an
    operator saves/changes Project.overseer_prompt via project settings — judges
    whether the prompt is comprehensive enough to drive a real investigation, with
    itemized, concrete feedback if not, mirroring the specificity discipline
    Tester/Reviewer already demand toward Developer (never a vague "could be more
    detailed"). Structurally the same shape as _dedupe_against_live_cards above:
    a plain two-message list, one forced-schema tool call, no ToolDispatcher.

    Deliberately does NOT fail open like that function does: raises on any
    LLM/parsing failure instead of silently returning "comprehensive". A broken
    dedup call just risks one duplicate card slipping through; a judge call that
    silently passes everything would defeat the entire point of this gate. Callers
    (ui/routers/projects.py, api/routers/projects.py) turn any exception here into
    the same soft-block UX as an honest "not comprehensive" verdict — the operator
    can always explicitly override either outcome, but never gets an unreviewed
    save presented as a reviewed one."""
    messages = [
        {
            "role": "system",
            "content": (
                "You judge whether an operator-authored investigation mandate is comprehensive "
                "enough to drive a real background investigation of a software project, or too "
                "generic to do so. This mandate becomes the primary system prompt for an "
                "autonomous agent that explores a real repository (read-only file tools plus "
                "bash — it can run tests, coverage, linters, curl a live endpoint, git log/blame, "
                "etc.) and files backlog cards for whatever it finds. Comprehensive means it names "
                "a specific area, kind of problem, or investigation to actually run — not generic "
                "filler like 'look for bugs' or 'improve code quality' that would read identically "
                "for any project. Narrow and specific beats broad and vague — a mandate scoped to "
                "one concrete thing (e.g. 'audit the payment webhook handler for idempotency "
                "bugs') can absolutely be comprehensive. If it's not comprehensive, give concrete, "
                "itemized feedback: one item per gap, naming specifically what's missing or too "
                "vague and what a stronger mandate would say instead — never a vague note like "
                "'could be more detailed'. Call report_overseer_prompt_assessment exactly once."
            ),
        },
        {
            "role": "user",
            "content": f"Project goal: {project.overarching_goal}\n\nProposed overseer mandate:\n{prompt}",
        },
    ]
    result = await llm_client.complete(messages=messages, tools=OVERSEER_PROMPT_ASSESSMENT_TOOLS)
    for tool_call in result.tool_calls:
        if tool_call.name != OVERSEER_PROMPT_ASSESSMENT_TERMINAL_TOOL:
            continue
        issues = [str(i) for i in tool_call.arguments.get("issues", []) if str(i).strip()]
        comprehensive = bool(tool_call.arguments.get("comprehensive"))
        return OverseerPromptAssessment(comprehensive=comprehensive, issues=issues)
    raise ValueError("judge model did not call report_overseer_prompt_assessment")
