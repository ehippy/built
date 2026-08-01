"""The Summarizer: an LLM role, not a Column and not an ActivityKind, that writes
a short postmortem for a card the moment it reaches a terminal lifecycle_state
(DONE or FAILED). Called from agent/loop.py's deploy terminal handlers and
orchestrator/ci_watcher.py's CI-confirmation paths — never from
domain/transitions.py itself, which stays pure/no-I/O.

Unlike run_column_visit or run_curation_pass, this has no explore tools over the
repo — its material is the card's own CardColumnVisit history, handed to it
directly in the prompt as a compact map (one line per visit: column, attempt,
outcome, and the summary the agent working that step wrote, right after they
finished it, about what happened *and how it went* — see llm/tool_schemas.py's
SUBMIT_FOR_TEST/APPROVE/REVIEWER_APPROVE). get_visit_detail is the one narrow
tool available, for when a visit's one-line summary hints at real trouble and
the model wants the full feedback text and tool-call history behind it — most
visits are routine and never need it, so this stays a short, bounded loop, not a
full agentic pass. Called before the caller's own commit of the terminal
transition, so the card is only actually marked DONE/FAILED once its postmortem
has been written — see each call site.

ActivityKind.RETRO (agent/curation.py, orchestrator/curator.py) is the other half
of this: a periodic pass that reads a batch of these postmortems looking for the
same struggle recurring across cards, and files a targeted card to fix it."""

import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from built.db.models import Card, CardColumnVisit, CardPostmortem, Project
from built.domain.enums import EventType, LifecycleState
from built.domain.events import append_event
from built.llm.client import FallbackLLMClient, LLMClient, ToolCallRequest
from built.llm.tool_schemas import SUMMARIZER_DETAIL_TOOL, SUMMARIZER_TERMINAL_TOOL, SUMMARIZER_TOOLS
from built.services import card_service, endpoint_service

logger = logging.getLogger(__name__)

# get_visit_detail, then submit_postmortem is the common case (2); a couple of
# extra look-ups for an unusually tangled card is the point of this being a loop
# at all rather than one call — this just bounds how far that can run.
_MAX_ITERATIONS = 4


def _describe_visit(visit: CardColumnVisit) -> str:
    outcome = visit.outcome.value if visit.outcome else "unresolved"
    summary = visit.summary or "(no summary recorded)"
    return f"[{visit.column.value} attempt {visit.attempt_number}] {outcome}: {summary}"


def _format_visits(visits: list[CardColumnVisit]) -> str:
    """The card's whole journey, one line per column visit, in order — leans on
    data that already exists rather than replaying raw CardEvents: attempt_number
    alone shows revision-loop friction structurally (a Developer visit 2 means
    Tester or Reviewer sent it back once already), and each visit's own summary
    is the account of that step written by the agent who was actually there.
    Far denser signal per token than reconstructing the same story from scratch
    out of tool-call-level events — and get_visit_detail is there for the rare
    case this one line isn't enough."""
    return "\n".join(_describe_visit(v) for v in visits)


def _build_prompt(card: Card, outcome: LifecycleState, visit_map: str) -> tuple[str, str]:
    system = (
        "You are the Summarizer agent in an autonomous software factory. A card just reached a "
        "terminal outcome — below is the map of every column it visited, in order, each with the "
        "outcome and the summary the agent working that step wrote when they finished it. Read it "
        "and write a short, honest postmortem for the whole card. Two things: what went well (if "
        "anything specific stands out), and what was a genuine struggle — retries, back-and-forth "
        "between columns, dead ends, confusing feedback. A card that bounced back and forth or took "
        "several attempts is a stronger signal than any one visit's summary alone; look at the shape "
        "of the whole journey, not just the last step.\n\n"
        "If one of those one-line summaries hints at real trouble but doesn't say enough to write an "
        "honest, specific postmortem, call get_visit_detail for that visit — it returns the full "
        "feedback text and what tools were actually called there. Don't call it for visits that were "
        "routine; most are, and the map alone is enough for them.\n\n"
        "Focus on things a future agent working on a similar card could actually learn from, not a "
        "restatement of what the card did. 'Nothing notable' is a completely fine answer for either "
        "field — don't invent texture that wasn't there. Call submit_postmortem exactly once, when "
        "you're done (with or without looking up any detail first)."
    )
    user = (
        f"Card: {card.title}\n"
        f"Outcome: {outcome.value}\n"
        f"Revisions: {card.revision_count}, deploy attempts: {card.deploy_attempt_count}\n\n"
        f"Column visits:\n{visit_map or '(no column visits recorded)'}"
    )
    return system, user


async def _visit_detail(
    session: AsyncSession, visits_by_key: dict[tuple[str, int], CardColumnVisit], arguments: dict
) -> str:
    column = str(arguments.get("column", ""))
    attempt_number = arguments.get("attempt_number")
    visit = visits_by_key.get((column, attempt_number)) if isinstance(attempt_number, int) else None
    if visit is None:
        return f"No such visit: {column!r} attempt {attempt_number!r}. Check the column-visit map above."

    feedback_text, events = await card_service.get_visit_activity(session, visit.id)
    lines = [_describe_visit(visit)]
    if feedback_text:
        lines.append(f"Feedback given at that step:\n{feedback_text}")
    if events:
        lines.append("Tool calls during that visit, oldest first:")
        lines.extend(
            f"- {e.payload.get('name', '?')}" + (" (error)" if e.payload.get("is_error") else "")
            for e in events
        )
    if not feedback_text and not events:
        lines.append("(no further detail recorded for this visit)")
    return "\n".join(lines)


def _assistant_tool_call_message(content: str | None, tool_calls: list[ToolCallRequest]) -> dict:
    return {
        "role": "assistant",
        "content": content or None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in tool_calls
        ],
    }


async def write_card_postmortem(
    session: AsyncSession,
    project: Project,
    card: Card,
    *,
    outcome: LifecycleState,
    llm_client: LLMClient | None = None,
) -> CardPostmortem | None:
    """Best-effort, mirroring run_curation_pass's never-crash contract: a broken
    endpoint, an exhausted iteration budget, or a malformed tool call loses the
    postmortem, but must never block the card it's attached to from actually
    closing. Returns None on failure.

    Does not commit — the caller decides when, since this is meant to run inside
    the same still-open transaction as the terminal transition it's paired with
    (the whole point being that DONE/FAILED and the postmortem land together).

    llm_client is normally left unset — resolved here via the project's role-less
    endpoint chain (agent/summarizer.py isn't a Column, see endpoint_resolution.py)
    — and only ever passed explicitly by tests, the same seam
    tests/unit/fakes.py's ScriptedLLMClient exists for elsewhere in this codebase."""
    visits = await card_service.list_column_visits(session, card.id)
    visit_map = _format_visits(visits)
    visits_by_key = {(v.column.value, v.attempt_number): v for v in visits}
    system, user = _build_prompt(card, outcome, visit_map)
    messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    call: ToolCallRequest | None = None
    try:
        if llm_client is None:
            chain = await endpoint_service.get_resolved_chain(session, project_id=project.id, role=None)
            llm_client = FallbackLLMClient(chain)

        for _ in range(_MAX_ITERATIONS):
            result = await llm_client.complete(messages=messages, tools=SUMMARIZER_TOOLS)
            if not result.tool_calls:
                break
            terminal_call = next(
                (tc for tc in result.tool_calls if tc.name == SUMMARIZER_TERMINAL_TOOL), None
            )
            detail_calls = [tc for tc in result.tool_calls if tc.name == SUMMARIZER_DETAIL_TOOL]

            messages.append(_assistant_tool_call_message(result.content, result.tool_calls))
            for tc in detail_calls:
                detail_text = await _visit_detail(session, visits_by_key, tc.arguments)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": detail_text})

            if terminal_call is not None:
                call = terminal_call
                break
            if not detail_calls:
                break  # unrecognized tool call(s) only — bail rather than loop forever
    except Exception:
        logger.exception("summarizer: couldn't write postmortem for card %s", card.id)
        return None

    if call is None:
        logger.warning("summarizer: no submit_postmortem call for card %s", card.id)
        return None

    postmortem = CardPostmortem(
        card_id=card.id,
        project_id=project.id,
        outcome=outcome,
        went_well=str(call.arguments.get("went_well", "")).strip(),
        struggles=str(call.arguments.get("struggles", "")).strip(),
        revision_count=card.revision_count,
        deploy_attempt_count=card.deploy_attempt_count,
    )
    session.add(postmortem)
    await append_event(
        session,
        card_id=card.id,
        type=EventType.SYSTEM_NOTE,
        payload={"action": "postmortem_written", "struggles": postmortem.struggles[:200]},
    )
    await session.flush()
    return postmortem
