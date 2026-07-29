"""The Tender: an autonomous background pass, one per project, that keeps an
AGENTS.md in the project's own repo useful — reviewing recent visit outcomes and
deciding whether anything is worth capturing as a durable, project-specific practice.
Wakes on its own on a timer (see orchestrator/tender.py), no manual trigger.

Structurally close to agent/reviver.py: no single terminal tool ending the run.
read_file/write_file/edit_file are dispatched exactly as they are for Developer (and
auto-commit, via ToolDispatcher's existing MUTATING_TOOLS handling) — the loop just
keeps going after each one, until done_for_now (or the iteration budget) ends it."""

import json

from sqlalchemy.ext.asyncio import AsyncSession

from built.db.models import Project
from built.llm.client import LLMClient
from built.llm.tool_schemas import TENDER_TERMINAL_TOOL, TENDER_TOOLS
from built.services import card_service
from built.tools.dispatcher import ToolDispatcher

TENDER_SYSTEM_PROMPT = (
    "You are the Tender agent in an autonomous software factory. You maintain AGENTS.md at the "
    "repository root for this one project — a living document of project-specific practices and "
    "conventions that every other agent (Product Manager, Developer, Tester, Deployer) reads before "
    "doing its own work. Your job is to keep it accurate and useful, not to grow it for its own sake.\n\n"
    "Use list_recent_visit_outcomes to see what's happened since your last pass. Only write something "
    "down if you can see a real, recurring pattern worth a future agent knowing in advance — a testing "
    "convention, a build quirk, a structural fact about the codebase, a mistake worth not repeating. A "
    "single one-off outcome is not a pattern. If AGENTS.md doesn't exist yet, create it only once you "
    "have something genuinely worth saying; an empty or near-empty file helps no one.\n\n"
    "Prefer small, targeted edits (edit_file) over rewriting the whole document — treat existing "
    "content as decisions someone already made, not a draft to redo. Keep entries short and concrete. "
    "Call done_for_now when finished, including if there was nothing new worth recording — nobody is "
    "watching this run interactively."
)


async def run_tender_pass(
    session: AsyncSession,
    project: Project,
    *,
    llm_client: LLMClient,
    dispatcher: ToolDispatcher,
    max_iterations: int,
) -> dict:
    """Runs one pass for one project and returns a small summary dict (edited: bool,
    summary: str | None). Never raises — a bad pass just makes no edit; the next
    scheduled wake for this project tries again."""
    messages: list[dict] = [
        {"role": "system", "content": TENDER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Project goal: {project.overarching_goal}\n\n"
            "Review recent activity and decide whether AGENTS.md needs anything.",
        },
    ]
    result = {"edited": False, "summary": None}

    try:
        for _ in range(max_iterations):
            llm_result = await llm_client.complete(messages=messages, tools=TENDER_TOOLS)

            if not llm_result.tool_calls:
                messages.append({"role": "assistant", "content": llm_result.content or ""})
                messages.append(
                    {
                        "role": "user",
                        "content": "Call a tool: list_recent_visit_outcomes to see what's happened, or "
                        "done_for_now if there's nothing to do.",
                    }
                )
                continue

            messages.append(
                {
                    "role": "assistant",
                    "content": llm_result.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in llm_result.tool_calls
                    ],
                }
            )

            done = False
            for tool_call in llm_result.tool_calls:
                if tool_call.name == TENDER_TERMINAL_TOOL:
                    done = True
                    result["summary"] = str(tool_call.arguments.get("summary", ""))
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": "Acknowledged."}
                    )
                    continue
                if tool_call.name == "list_recent_visit_outcomes":
                    output = await _list_recent_visit_outcomes(session, project)
                else:
                    outcome = await dispatcher.dispatch(tool_call.name, tool_call.arguments)
                    if outcome.commit_sha is not None:
                        result["edited"] = True
                    output = outcome.result.output
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})

            if done:
                return result

        return result
    except Exception:  # noqa: BLE001 — deliberate: a bad pass makes no edit, not a crash
        return result


async def _list_recent_visit_outcomes(session: AsyncSession, project: Project) -> str:
    since = project.agents_doc_tended_at
    outcomes = await card_service.list_recent_visit_outcomes(session, project.id, since=since)
    if not outcomes:
        return "No visits have closed since your last pass." if since else "No closed visits yet."
    lines = [
        f"- {o['card_title']!r} | {o['column']} -> {o['outcome']}: {o['summary'][:200]}" for o in outcomes
    ]
    return "\n".join(lines)
