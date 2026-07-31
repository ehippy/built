"""agent/loop.py's per-iteration pending_nudge check — a human note reaches a
running visit within one iteration instead of waiting for the next column,
since it piggybacks on the same session.refresh(card) the cancellation check
(test_visit_cancellation.py) already does every iteration. Exercised via
run_developer_visit — the check lives in the shared run_column_visit, so any
column's loop would do."""

from built.agent.loop import run_developer_visit
from built.domain import transitions
from built.domain.enums import Column, EventType
from built.llm.client import LLMResult, ToolCallRequest
from built.sandbox import worktree
from built.sandbox.container import CommandResult
from built.services import card_service, project_service
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher
from tests.unit.fakes import FakeCommandExecutor, ScriptedLLMClient


async def _make_developer_card(db_session, toy_repo_remote):
    project = await project_service.create_project(
        db_session,
        name=f"nudge-mid-run-{id(db_session)}",
        overarching_goal="goal",
        repo_remote_url=str(toy_repo_remote),
        test_command="pytest -q",
    )
    card = await card_service.create_card(db_session, project.id, title="Add subtract()", raw_request="r")
    card.spec = "Add a subtract(a, b) function to app.py."
    card.acceptance_criteria = ["subtract(2, 1) == 1"]
    card.column = Column.DEVELOPER
    await db_session.flush()
    wt_path = await worktree.create_card_worktree(project, card)
    card.worktree_path = str(wt_path)
    await db_session.flush()
    return project, card, wt_path


class _NudgesCardAfterNCalls:
    """Behaves like ScriptedLLMClient, but externally sets the card's
    pending_nudge (and commits, on the same session the loop itself uses) right
    after a chosen call — simulating a human leaving a nudge via a completely
    separate request while this visit is still mid-run."""

    def __init__(self, session, card, responses, *, nudge_after_call, note):
        self._session = session
        self._card = card
        self._responses = list(responses)
        self._nudge_after_call = nudge_after_call
        self._note = note
        self.calls: list[dict] = []

    async def complete(self, *, messages: list[dict], tools: list[dict]) -> LLMResult:
        self.calls.append({"messages": messages, "tools": tools})
        result = self._responses.pop(0)
        if len(self.calls) == self._nudge_after_call:
            self._card.pending_nudge = self._note
            await self._session.commit()
        return result


async def test_pending_nudge_reaches_the_model_within_one_iteration(db_session, toy_repo_remote):
    project, card, wt_path = await _make_developer_card(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card)

    llm = _NudgesCardAfterNCalls(
        db_session,
        card,
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="read_file", arguments={"path": "app.py"})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_2", name="submit_for_test", arguments={"summary": "done"})
                ],
                endpoint_used="fake::model",
            ),
        ],
        nudge_after_call=1,
        note="don't touch the payments module, that's handled elsewhere",
    )
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)

    # max_iterations=2, not some larger number: this test only cares whether the
    # nudge reaches the second call's messages, not whether submit_for_test is
    # ultimately accepted (it won't be — nothing here satisfies the "actually
    # changed files and ran the tests" gate) — capping at 2 means the loop ends
    # right after that second call instead of needing a third scripted response.
    await run_developer_visit(
        db_session, project, card, visit, llm_client=llm, dispatcher=dispatcher, max_iterations=2
    )

    # The second call's messages include the nudge — injected before that
    # iteration's LLM request, not just recorded somewhere for later.
    assert len(llm.calls) == 2
    second_call_contents = [m.get("content", "") for m in llm.calls[1]["messages"]]
    assert any("don't touch the payments module" in (c or "") for c in second_call_contents)

    # Single slot, consumed once seen — not left standing for a future visit.
    await db_session.refresh(card)
    assert card.pending_nudge is None


async def test_pending_nudge_is_recorded_as_a_system_note_event(db_session, toy_repo_remote):
    project, card, wt_path = await _make_developer_card(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card)

    llm = _NudgesCardAfterNCalls(
        db_session,
        card,
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="read_file", arguments={"path": "app.py"})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_2", name="submit_for_test", arguments={"summary": "done"})
                ],
                endpoint_used="fake::model",
            ),
        ],
        nudge_after_call=1,
        note="use the staging DB for this one",
    )
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)

    await run_developer_visit(
        db_session, project, card, visit, llm_client=llm, dispatcher=dispatcher, max_iterations=10
    )

    events = await card_service.list_recent_events(db_session, card.id)
    # Filtered to action="nudge" — create_card itself already emits a system_note
    # (action="created"), so a raw SYSTEM_NOTE count would include that too.
    nudge_notes = [
        e for e in events if e.type == EventType.SYSTEM_NOTE and e.payload.get("action") == "nudge"
    ]
    assert len(nudge_notes) == 1
    assert nudge_notes[0].payload["note"] == "use the staging DB for this one"


async def test_no_nudge_means_no_system_note_event(db_session, toy_repo_remote):
    project, card, wt_path = await _make_developer_card(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card)

    # A single plain-text, no-tool-call turn is enough: with max_iterations=1 the
    # loop ends (over the iteration cap) right after processing it, so exactly one
    # scripted response is needed regardless of whether the visit itself succeeds
    # — this test only cares that no nudge means no system_note event.
    llm = ScriptedLLMClient([LLMResult(content="thinking...", tool_calls=[])])
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)

    await run_developer_visit(
        db_session, project, card, visit, llm_client=llm, dispatcher=dispatcher, max_iterations=1
    )

    events = await card_service.list_recent_events(db_session, card.id)
    # create_card's own system_note (action="created") is expected and fine —
    # only a nudge-actioned one would mean this test's premise broke.
    nudge_notes = [
        e for e in events if e.type == EventType.SYSTEM_NOTE and e.payload.get("action") == "nudge"
    ]
    assert nudge_notes == []
