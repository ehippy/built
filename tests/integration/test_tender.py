"""The Tender's pass against a real toy git repo — LLM faked, everything else real:
creating AGENTS.md, amending it, and doing nothing when there's nothing new."""

from built.agent.tender import run_tender_pass
from built.domain import transitions
from built.llm.client import LLMResult, ToolCallRequest
from built.orchestrator.tender import run_tender_once
from built.sandbox import worktree
from built.sandbox.container import CommandResult
from built.services import card_service, project_service
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher
from tests.unit.fakes import FakeCommandExecutor, ScriptedLLMClient


async def _make_project(db_session, toy_repo_remote, **overrides):
    defaults = {
        "name": f"tender-{overrides.pop('_n', 'x')}",
        "overarching_goal": "goal",
        "repo_remote_url": str(toy_repo_remote),
    }
    defaults.update(overrides)
    return await project_service.create_project(db_session, **defaults)


def _dispatcher(wt_path) -> ToolDispatcher:
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    return ToolDispatcher(ctx=ToolContext(card_id="tender-x", worktree_root=wt_path), executor=executor)


async def test_tender_creates_agents_md_when_something_is_worth_recording(db_session, toy_repo_remote):
    project = await _make_project(db_session, toy_repo_remote, _n="1")
    wt_path = await worktree.ensure_tool_worktree(project, tool="tender")

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="list_recent_visit_outcomes", arguments={})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_2",
                        name="write_file",
                        arguments={
                            "path": "AGENTS.md",
                            "content": "# Practices\n\n- Tests use pytest, run via `pytest -q`.\n",
                        },
                    )
                ],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_3", name="done_for_now", arguments={"summary": "documented"})
                ],
                endpoint_used="fake::model",
            ),
        ]
    )

    result = await run_tender_pass(
        db_session, project, llm_client=llm, dispatcher=_dispatcher(wt_path), max_iterations=10
    )

    assert result == {"edited": True, "summary": "documented"}
    assert "pytest" in (wt_path / "AGENTS.md").read_text()


async def test_tender_does_nothing_when_there_is_nothing_worth_recording(db_session, toy_repo_remote):
    project = await _make_project(db_session, toy_repo_remote, _n="2")
    wt_path = await worktree.ensure_tool_worktree(project, tool="tender")

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="list_recent_visit_outcomes", arguments={})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_2", name="done_for_now", arguments={"summary": "nothing new"})
                ],
                endpoint_used="fake::model",
            ),
        ]
    )

    result = await run_tender_pass(
        db_session, project, llm_client=llm, dispatcher=_dispatcher(wt_path), max_iterations=10
    )

    assert result == {"edited": False, "summary": "nothing new"}
    assert not (wt_path / "AGENTS.md").exists()


async def test_tender_amends_an_existing_doc_with_edit_file(db_session, toy_repo_remote):
    project = await _make_project(db_session, toy_repo_remote, _n="3")
    wt_path = await worktree.ensure_tool_worktree(project, tool="tender")
    (wt_path / "AGENTS.md").write_text("# Practices\n\n- Tests use pytest.\n")
    from built.tools import git_tools

    await git_tools.commit_all(wt_path, message="seed AGENTS.md")

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="read_file", arguments={"path": "AGENTS.md"})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_2",
                        name="edit_file",
                        arguments={
                            "path": "AGENTS.md",
                            "old_str": "- Tests use pytest.\n",
                            "new_str": "- Tests use pytest.\n- Sandbox needs HOME=/tmp for npm.\n",
                        },
                    )
                ],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_3", name="done_for_now", arguments={"summary": "amended"})
                ],
                endpoint_used="fake::model",
            ),
        ]
    )

    result = await run_tender_pass(
        db_session, project, llm_client=llm, dispatcher=_dispatcher(wt_path), max_iterations=10
    )

    assert result["edited"] is True
    assert "HOME=/tmp for npm" in (wt_path / "AGENTS.md").read_text()


async def test_run_tender_once_skips_a_paused_project(db_session, toy_repo_remote):
    """A paused project has visits worth tending (_needs_tending would say yes), but
    run_tender_once's per-project loop should never even get that far — proven here
    by the fact that agents_doc_tended_at stays untouched, which only happens on the
    early-continue path since _tend_one_project always sets it."""
    project = await _make_project(db_session, toy_repo_remote, _n="4")
    card = await card_service.create_card(db_session, project.id, title="c", raw_request="r")
    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_pm_visit(
        db_session, card, visit, spec="s", acceptance_criteria=["x"], summary="s"
    )
    await project_service.pause_project(db_session, project.id)
    await db_session.commit()

    await run_tender_once()

    await db_session.refresh(project)
    assert project.paused_at is not None
    assert project.agents_doc_tended_at is None
