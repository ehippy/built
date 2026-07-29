"""agent/loop.py's per-iteration lifecycle_state check — regression test for a
real production incident: a human cancelled a card while its Tester visit was
running, cancel_card only flipped lifecycle_state (nothing else previously
noticed a running loop should stop), and with orchestrator_concurrency's default
of 1, the single worker stayed monopolized on the now-unwanted visit for another
10+ minutes until a server restart finally interrupted it. Exercised via
run_developer_visit — the check lives in the shared run_column_visit, so any
column's loop would do."""

from built.agent.loop import run_developer_visit
from built.domain import transitions
from built.domain.enums import Column, LifecycleState, VisitOutcome
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
        name=f"cancel-mid-run-{id(db_session)}",
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


class _CancelsCardAfterNCalls:
    """Behaves like ScriptedLLMClient, but externally flips the card's
    lifecycle_state (and commits, on the same session the loop itself uses) right
    after a chosen call — simulating a human cancelling the card via a completely
    separate request while this visit is still mid-run."""

    def __init__(self, session, card, responses, *, cancel_after_call):
        self._session = session
        self._card = card
        self._responses = list(responses)
        self._cancel_after_call = cancel_after_call
        self.calls = 0

    async def complete(self, *, messages: list[dict], tools: list[dict]) -> LLMResult:
        self.calls += 1
        result = self._responses.pop(0)
        if self.calls == self._cancel_after_call:
            self._card.lifecycle_state = LifecycleState.FAILED
            await self._session.commit()
        return result


async def test_loop_stops_promptly_when_the_card_is_cancelled_mid_run(db_session, toy_repo_remote):
    project, card, wt_path = await _make_developer_card(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card)

    llm = _CancelsCardAfterNCalls(
        db_session,
        card,
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="read_file", arguments={"path": "app.py"})],
                endpoint_used="fake::model",
            ),
            # Never actually reached if the fix works — the loop's next-iteration
            # check should catch the cancellation before requesting this.
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_2", name="submit_for_test", arguments={"summary": "done"})
                ],
                endpoint_used="fake::model",
            ),
        ],
        cancel_after_call=1,
    )
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)

    result = await run_developer_visit(
        db_session, project, card, visit, llm_client=llm, dispatcher=dispatcher, max_iterations=10
    )

    # Stopped after the one call already in flight — never asked the LLM for a
    # second turn once the cancellation was visible.
    assert llm.calls == 1
    assert visit.outcome == VisitOutcome.CANCELLED
    assert visit.ended_at is not None
    # The state cancel_card already set is left alone, not overwritten.
    assert result.lifecycle_state == LifecycleState.FAILED
    assert result.column == Column.DEVELOPER  # never advanced to Tester


async def test_loop_stops_immediately_if_already_cancelled_before_it_starts(db_session, toy_repo_remote):
    project, card, wt_path = await _make_developer_card(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card)
    card.lifecycle_state = LifecycleState.FAILED
    await db_session.commit()

    # Zero scripted responses: if the pre-iteration check didn't fire, the very
    # first llm_client.complete() call would raise (ScriptedLLMClient's own
    # "ran out of responses" guard), which the loop's outer handler would catch
    # and record as outcome=ERROR — a different, distinguishable failure mode
    # from the CANCELLED outcome this test actually asserts on.
    llm = ScriptedLLMClient([])
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)

    result = await run_developer_visit(
        db_session, project, card, visit, llm_client=llm, dispatcher=dispatcher, max_iterations=10
    )

    assert llm.calls == []
    assert visit.outcome == VisitOutcome.CANCELLED
    assert result.lifecycle_state == LifecycleState.FAILED
