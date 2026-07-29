"""PM agent loop against a real toy git repo — LLM faked, everything else real."""

from datetime import UTC, datetime, timedelta

from built.agent.loop import run_pm_visit
from built.config import settings
from built.domain import transitions
from built.domain.enums import Column, LifecycleState, VisitOutcome
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
        name="pm-loop",
        overarching_goal="Add basic arithmetic helpers to app.py.",
        repo_remote_url=str(toy_repo_remote),
    )
    card = await card_service.create_card(
        db_session, project.id, title="Add subtract()", raw_request="Add a subtract(a, b) function to app.py"
    )
    wt_path = await worktree.create_card_worktree(project, card)
    card.worktree_path = str(wt_path)
    await db_session.flush()
    return project, card, wt_path


async def test_pm_loop_explores_then_submits_a_spec(db_session, toy_repo_remote):
    project, card, wt_path = await _make_pm_card(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card)

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="list_files", arguments={"path": "."})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_2", name="read_file", arguments={"path": "app.py"})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_3",
                        name="submit_spec",
                        arguments={
                            "spec": "Add subtract(a, b) returning a - b.",
                            "acceptance_criteria": ["app.py defines subtract(a, b)", "subtract(2, 1) == 1"],
                            "summary": "Spec ready",
                        },
                    )
                ],
                endpoint_used="fake::model",
            ),
        ]
    )
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)

    result = await run_pm_visit(
        db_session, project, card, visit, llm_client=llm, dispatcher=dispatcher, max_iterations=10
    )

    assert result.column == Column.DEVELOPER
    assert result.lifecycle_state == LifecycleState.ACTIVE
    assert result.spec == "Add subtract(a, b) returning a - b."
    assert result.acceptance_criteria == ["app.py defines subtract(a, b)", "subtract(2, 1) == 1"]
    assert visit.outcome == VisitOutcome.SUBMITTED


async def test_lease_is_renewed_each_iteration_so_a_long_visit_never_looks_abandoned(
    db_session, toy_repo_remote
):
    """Card.is_being_worked (the dashboard spinner) and claim_next_card's staleness
    check both key off lease_expires_at. A multi-iteration visit must keep renewing
    it every iteration — otherwise a visit that simply runs longer than
    claim_lease_seconds looks abandoned even though it's still making real
    progress: the spinner goes dark, and at concurrency > 1 another worker could
    claim the same card out from under the one still running it."""
    project, card, wt_path = await _make_pm_card(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card)
    # Simulate exactly what a long-running visit produces in production: a claim
    # whose lease has already gone stale by the time a later iteration runs.
    card.claimed_by_worker_id = "worker-a"
    card.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="list_files", arguments={"path": "."})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_2",
                        name="submit_spec",
                        arguments={
                            "spec": "Add subtract(a, b) returning a - b.",
                            "acceptance_criteria": ["subtract(2, 1) == 1"],
                            "summary": "Spec ready",
                        },
                    )
                ],
                endpoint_used="fake::model",
            ),
        ]
    )
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)

    before = datetime.now(UTC)
    await run_pm_visit(
        db_session, project, card, visit, llm_client=llm, dispatcher=dispatcher, max_iterations=10
    )

    assert card.lease_expires_at > before + timedelta(seconds=settings.claim_lease_seconds - 5)


async def test_pm_loop_rejects_malformed_acceptance_criteria_and_recovers(db_session, toy_repo_remote):
    project, card, wt_path = await _make_pm_card(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card)

    llm = ScriptedLLMClient(
        [
            # Malformed: acceptance_criteria is a string, not a list — rejected server-side.
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="submit_spec",
                        arguments={"spec": "x", "acceptance_criteria": "not a list", "summary": "bad"},
                    )
                ],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_2",
                        name="submit_spec",
                        arguments={
                            "spec": "Add subtract(a, b).",
                            "acceptance_criteria": ["subtract(2, 1) == 1"],
                            "summary": "fixed",
                        },
                    )
                ],
                endpoint_used="fake::model",
            ),
        ]
    )
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)

    result = await run_pm_visit(
        db_session, project, card, visit, llm_client=llm, dispatcher=dispatcher, max_iterations=10
    )

    assert result.column == Column.DEVELOPER
    assert result.acceptance_criteria == ["subtract(2, 1) == 1"]
