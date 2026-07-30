"""Compaction (agent/context_window.compact()) used to be completely invisible:
no CardEvent, no card-specific trace anywhere, not even the extra LLM call it makes
to do the summarizing. This covers the actual wiring in agent/loop.py that logs a
CardEvent when compaction fires — built.agent.loop._maybe_compact is monkeypatched
to return a controlled CompactionEvent rather than trying to orchestrate real token
thresholds through a scripted conversation, which would be exact-token-count fragile
for no real benefit; context_window.compact() itself already has direct unit
coverage in tests/unit/test_context_window.py."""

import built.agent.loop as agent_loop
from built.agent.context_window import CompactionEvent
from built.agent.loop import run_pm_visit
from built.domain import transitions
from built.domain.enums import EventType
from built.llm.client import LLMResult, ToolCallRequest
from built.sandbox import worktree
from built.sandbox.container import CommandResult
from built.services import card_service, project_service
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher
from tests.unit.fakes import FakeCommandExecutor, ScriptedLLMClient


async def _make_pm_card(db_session, toy_repo_remote):
    project = await project_service.create_project(
        db_session,
        name="compaction-logging",
        overarching_goal="goal",
        repo_remote_url=str(toy_repo_remote),
    )
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    wt_path = await worktree.create_card_worktree(project, card)
    card.worktree_path = str(wt_path)
    await db_session.flush()
    return project, card, wt_path


async def test_compaction_event_is_logged_when_compact_runs(db_session, toy_repo_remote, monkeypatch):
    project, card, wt_path = await _make_pm_card(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card)

    event = CompactionEvent(
        messages_before=42,
        messages_after=7,
        tokens_before=9000,
        tokens_after=1500,
        summary="The agent explored the repo and found app.py defines greet().",
    )

    async def _fake_maybe_compact(messages, llm_client, config, iteration):
        return messages, event

    monkeypatch.setattr(agent_loop, "_maybe_compact", _fake_maybe_compact)

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="submit_spec",
                        arguments={"spec": "s", "acceptance_criteria": ["a"], "summary": "ok"},
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )
    dispatcher = ToolDispatcher(
        ctx=ToolContext(card_id=card.id, worktree_root=wt_path),
        executor=FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr="")),
    )

    await run_pm_visit(
        db_session, project, card, visit, llm_client=llm, dispatcher=dispatcher, max_iterations=5
    )

    events = await card_service.list_events(db_session, card.id)
    compaction_events = [e for e in events if e.type == EventType.COMPACTION]
    assert len(compaction_events) == 1
    payload = compaction_events[0].payload
    assert payload["messages_before"] == 42
    assert payload["messages_after"] == 7
    assert payload["tokens_before"] == 9000
    assert payload["tokens_after"] == 1500
    assert payload["summary"] == "The agent explored the repo and found app.py defines greet()."


async def test_no_compaction_event_when_compact_is_a_no_op(db_session, toy_repo_remote, monkeypatch):
    project, card, wt_path = await _make_pm_card(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card)

    async def _fake_maybe_compact(messages, llm_client, config, iteration):
        return messages, None

    monkeypatch.setattr(agent_loop, "_maybe_compact", _fake_maybe_compact)

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="submit_spec",
                        arguments={"spec": "s", "acceptance_criteria": ["a"], "summary": "ok"},
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )
    dispatcher = ToolDispatcher(
        ctx=ToolContext(card_id=card.id, worktree_root=wt_path),
        executor=FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr="")),
    )

    await run_pm_visit(
        db_session, project, card, visit, llm_client=llm, dispatcher=dispatcher, max_iterations=5
    )

    events = await card_service.list_events(db_session, card.id)
    assert not any(e.type == EventType.COMPACTION for e in events)
