"""Reviewer agent loop against a real toy git repo — LLM and Docker faked, everything
else real, including review_diff's actual `git diff main...HEAD` and the fact that
Reviewer has no write/edit/bash tools at all (see llm/tool_schemas.REVIEWER_TOOLS)."""

from built.agent.loop import run_reviewer_visit
from built.domain import transitions
from built.domain.enums import Column, LifecycleState, VisitOutcome
from built.llm.client import LLMResult, ToolCallRequest
from built.llm.tool_schemas import REVIEWER_TOOLS
from built.sandbox import worktree
from built.sandbox.container import CommandResult
from built.services import card_service, project_service
from built.tools import git_tools
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher
from tests.unit.fakes import FakeCommandExecutor, ScriptedLLMClient

# Reviewer never has a bash tool, so this executor should never actually be invoked —
# it's only here because ToolDispatcher requires one.
_UNUSED_EXECUTOR = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))


async def _make_reviewer_card(db_session, toy_repo_remote):
    project = await project_service.create_project(
        db_session,
        name="reviewer-loop",
        overarching_goal="Add basic arithmetic helpers to app.py.",
        repo_remote_url=str(toy_repo_remote),
        test_command="pytest -q",
    )
    card = await card_service.create_card(
        db_session, project.id, title="Add subtract()", raw_request="Add a subtract(a, b) function to app.py"
    )
    card.spec = "Add subtract(a, b) returning a - b."
    card.acceptance_criteria = ["subtract(2, 1) == 1"]
    card.column = Column.REVIEWER
    await db_session.flush()
    wt_path = await worktree.create_card_worktree(project, card)
    card.worktree_path = str(wt_path)
    await db_session.flush()

    # Simulate the Developer's committed change, so there's an actual diff between
    # the card's branch and the project's default branch for Reviewer to look at.
    (wt_path / "app.py").write_text(
        "def greet():\n    return 'hi'\n\n\ndef subtract(a, b):\n    return a - b\n"
    )
    await git_tools.commit_all(wt_path, message="Add subtract()")

    return project, card, wt_path


def test_reviewer_has_no_write_edit_or_bash_tools():
    """Reviewer can inspect the diff and surrounding code but cannot change anything
    itself — that separation is what makes it a genuine second opinion on Developer's
    (and Tester's) work rather than the same actor wearing a different hat."""
    tool_names = {t["function"]["name"] for t in REVIEWER_TOOLS}
    assert "write_file" not in tool_names
    assert "edit_file" not in tool_names
    assert "bash" not in tool_names
    assert {"review_diff", "approve", "request_changes"} <= tool_names


async def test_reviewer_approve_advances_to_deployer(db_session, toy_repo_remote):
    project, card, wt_path = await _make_reviewer_card(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card)

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="review_diff", arguments={})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_2", name="approve", arguments={"notes": "clean, matches spec"})
                ],
                endpoint_used="fake::model",
            ),
        ]
    )
    dispatcher = ToolDispatcher(
        ctx=ToolContext(card_id=card.id, worktree_root=wt_path, base_ref=project.default_branch),
        executor=_UNUSED_EXECUTOR,
    )

    result = await run_reviewer_visit(
        db_session,
        project,
        card,
        visit,
        llm_client=llm,
        dispatcher=dispatcher,
        max_iterations=10,
        tester_summary="Tests pass for subtract()",
    )

    assert result.column == Column.DEPLOYER
    assert result.lifecycle_state == LifecycleState.ACTIVE
    assert visit.outcome == VisitOutcome.APPROVED
    # The review_diff call actually saw the real diff, not an empty/placeholder one.
    diff_event = next(e for e in llm.calls[1]["messages"] if e.get("role") == "tool")
    assert "def subtract(a, b):" in diff_event["content"]


async def test_reviewer_request_changes_bounces_to_developer(db_session, toy_repo_remote):
    project, card, wt_path = await _make_reviewer_card(db_session, toy_repo_remote)
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
                            "feedback": "subtract() has no input validation for non-numeric args",
                            "summary": "missing validation",
                        },
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )
    dispatcher = ToolDispatcher(
        ctx=ToolContext(card_id=card.id, worktree_root=wt_path, base_ref=project.default_branch),
        executor=_UNUSED_EXECUTOR,
    )

    result = await run_reviewer_visit(
        db_session,
        project,
        card,
        visit,
        llm_client=llm,
        dispatcher=dispatcher,
        max_iterations=10,
        tester_summary="Tests pass for subtract()",
    )

    assert result.column == Column.DEVELOPER
    assert result.lifecycle_state == LifecycleState.ACTIVE
    assert result.revision_count == 1
    assert result.latest_feedback == "subtract() has no input validation for non-numeric args"
    assert visit.outcome == VisitOutcome.CHANGES_REQUESTED
