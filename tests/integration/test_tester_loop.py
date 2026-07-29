"""Tester agent loop against a real toy git repo — LLM and Docker faked, everything
else real, including the server-side `approve` gate against a real RunAttempt row."""

from built.agent.loop import run_tester_visit
from built.domain import transitions
from built.domain.enums import Column, LifecycleState, VisitOutcome
from built.llm.client import LLMResult, ToolCallRequest
from built.sandbox import worktree
from built.sandbox.container import CommandResult
from built.services import card_service, project_service
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher
from tests.unit.fakes import FakeCommandExecutor, ScriptedLLMClient


async def _make_tester_card(db_session, toy_repo_remote):
    project = await project_service.create_project(
        db_session,
        name="tester-loop",
        overarching_goal="Add basic arithmetic helpers to app.py.",
        repo_remote_url=str(toy_repo_remote),
    )
    card = await card_service.create_card(
        db_session, project.id, title="Add subtract()", raw_request="Add a subtract(a, b) function to app.py"
    )
    card.spec = "Add subtract(a, b) returning a - b."
    card.acceptance_criteria = ["subtract(2, 1) == 1"]
    card.column = Column.TESTER
    await db_session.flush()
    wt_path = await worktree.create_card_worktree(project, card)
    card.worktree_path = str(wt_path)
    await db_session.flush()
    return project, card, wt_path


async def test_tester_approve_is_rejected_without_a_passing_run_then_succeeds(db_session, toy_repo_remote):
    project, card, wt_path = await _make_tester_card(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card)

    llm = ScriptedLLMClient(
        [
            # Claims success without running anything — must be rejected server-side.
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="approve", arguments={"notes": "looks fine"})],
                endpoint_used="fake::model",
            ),
            # Now actually runs the tests...
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_2", name="bash", arguments={"command": "pytest -q"})],
                endpoint_used="fake::model",
            ),
            # ...and approves for real.
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_3", name="approve", arguments={"notes": "tests pass"})],
                endpoint_used="fake::model",
            ),
        ]
    )
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="1 passed", stderr=""))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)

    result = await run_tester_visit(
        db_session,
        project,
        card,
        visit,
        llm_client=llm,
        dispatcher=dispatcher,
        max_iterations=10,
        developer_summary="Added subtract()",
    )

    assert result.column == Column.DEPLOYER
    assert result.lifecycle_state == LifecycleState.ACTIVE
    assert visit.outcome == VisitOutcome.APPROVED
    # The rejected first attempt shows up in the transcript as a normal tool round-trip,
    # not a crash — the loop just kept going.
    assert llm.calls[2] is not None  # a third LLM call happened, i.e. the loop continued


async def test_tester_approve_stays_rejected_if_the_run_failed(db_session, toy_repo_remote):
    project, card, wt_path = await _make_tester_card(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card)

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="bash", arguments={"command": "pytest -q"})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_2", name="approve", arguments={"notes": "ship it"})],
                endpoint_used="fake::model",
            ),
        ]
    )
    executor = FakeCommandExecutor(CommandResult(exit_code=1, stdout="", stderr="1 failed"))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)

    result = await run_tester_visit(
        db_session, project, card, visit, llm_client=llm, dispatcher=dispatcher, max_iterations=2
    )

    # Ran out of iterations because approve kept getting rejected (the failing test was
    # never fixed) — blocked for a human, not silently advanced to Deployer.
    assert result.column == Column.TESTER
    assert result.lifecycle_state == LifecycleState.BLOCKED
    assert visit.outcome == VisitOutcome.ERROR


async def test_tester_request_changes_bounces_back_to_developer(db_session, toy_repo_remote):
    project, card, wt_path = await _make_tester_card(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card)

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="request_changes",
                        arguments={
                            "feedback": "subtract() is missing entirely",
                            "summary": "not implemented",
                        },
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)

    result = await run_tester_visit(
        db_session, project, card, visit, llm_client=llm, dispatcher=dispatcher, max_iterations=10
    )

    assert result.column == Column.DEVELOPER
    assert result.lifecycle_state == LifecycleState.ACTIVE
    assert result.revision_count == 1
    assert result.latest_feedback == "subtract() is missing entirely"
    assert visit.outcome == VisitOutcome.CHANGES_REQUESTED
