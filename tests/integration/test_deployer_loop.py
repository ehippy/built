"""Deployer agent loop against a real toy git repo — LLM and GitHub faked, everything
else real, including the actual merge/push (auto_main) and push+PR (pr_to_operator)."""

import subprocess

import httpx

from built.agent.loop import run_deployer_visit
from built.domain import transitions
from built.domain.enums import Column, DeployKind, DeployMode, LifecycleState, VisitOutcome
from built.llm.client import LLMResult, ToolCallRequest
from built.sandbox import deploy_runner, worktree
from built.sandbox.container import CommandResult
from built.services import card_service, project_service
from built.tools import git_tools
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher
from tests.unit.fakes import FakeCommandExecutor, ScriptedLLMClient

_RealAsyncClient = httpx.AsyncClient


def _mock_github(monkeypatch, handler) -> None:
    """See test_deploy_runner.py's _mock_github for why this must close over the real
    AsyncClient captured at import time rather than referencing httpx.AsyncClient
    again inside the lambda (deploy_runner.httpx is the same module object)."""
    monkeypatch.setattr(
        deploy_runner.httpx,
        "AsyncClient",
        lambda **kwargs: _RealAsyncClient(transport=httpx.MockTransport(handler)),
    )


async def _make_deployer_card(db_session, repo_remote, *, mode: DeployMode, command: str = "true"):
    project = await project_service.create_project(
        db_session,
        name=f"deployer-loop-{mode.value}",
        overarching_goal="goal",
        repo_remote_url=str(repo_remote),
    )
    await project_service.set_deploy_config(
        db_session,
        project.id,
        kind=DeployKind.COMMAND,
        mode=mode,
        command=command,
        github_token_ref="TEST_GH_TOKEN",
    )
    project = await project_service.get_project(db_session, project.id)

    card = await card_service.create_card(db_session, project.id, title="Add farewell", raw_request="r")
    card.column = Column.DEPLOYER
    await db_session.flush()
    wt_path = await worktree.create_card_worktree(project, card)
    card.worktree_path = str(wt_path)
    (wt_path / "app.py").write_text("def greet():\n    return 'hi'\n\n\ndef farewell():\n    return 'bye'\n")
    await git_tools.commit_all(wt_path, message="add farewell")
    await db_session.flush()
    return project, card, wt_path


def _allow_push_to_checked_out_branch(repo_remote) -> None:
    subprocess.run(
        ["git", "config", "receive.denyCurrentBranch", "updateInstead"],
        cwd=repo_remote,
        check=True,
        capture_output=True,
    )


def _dispatcher(card_id: str, wt_path) -> ToolDispatcher:
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    return ToolDispatcher(ctx=ToolContext(card_id=card_id, worktree_root=wt_path), executor=executor)


async def test_deployer_loop_auto_main_happy_path_merges_and_deploys(db_session, toy_repo_remote):
    _allow_push_to_checked_out_branch(toy_repo_remote)
    project, card, wt_path = await _make_deployer_card(db_session, toy_repo_remote, mode=DeployMode.AUTO_MAIN)
    visit = await transitions.start_visit(db_session, card)

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="run_deploy", arguments={})],
                endpoint_used="fake::model",
            )
        ]
    )

    result = await run_deployer_visit(
        db_session,
        project,
        card,
        visit,
        llm_client=llm,
        dispatcher=_dispatcher(card.id, wt_path),
        max_iterations=5,
        mode=DeployMode.AUTO_MAIN,
    )

    assert result.lifecycle_state == LifecycleState.DONE
    assert visit.outcome == VisitOutcome.DONE
    log = subprocess.run(
        ["git", "log", "--oneline", "main"], cwd=toy_repo_remote, check=True, capture_output=True, text=True
    ).stdout
    assert "Merge" in log


async def test_deployer_loop_pr_to_operator_happy_path_opens_pr(db_session, toy_repo_remote, monkeypatch):
    _allow_push_to_checked_out_branch(toy_repo_remote)
    project, card, wt_path = await _make_deployer_card(
        db_session, toy_repo_remote, mode=DeployMode.PR_TO_OPERATOR
    )
    project.repo_remote_url = "https://github.com/owner/repo.git"
    visit = await transitions.start_visit(db_session, card)
    monkeypatch.setenv("TEST_GH_TOKEN", "fake-token")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={"html_url": "https://github.com/owner/repo/pull/7"})

    _mock_github(monkeypatch, handler)

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1", name="open_pull_request", arguments={"summary": "adds farewell()"}
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )

    result = await run_deployer_visit(
        db_session,
        project,
        card,
        visit,
        llm_client=llm,
        dispatcher=_dispatcher(card.id, wt_path),
        max_iterations=5,
        mode=DeployMode.PR_TO_OPERATOR,
    )

    assert result.lifecycle_state == LifecycleState.DONE
    assert result.deploy_url == "https://github.com/owner/repo/pull/7"
    assert visit.outcome == VisitOutcome.DONE


async def test_deployer_loop_auto_main_deploy_failure_stays_active_under_cap(db_session, toy_repo_remote):
    _allow_push_to_checked_out_branch(toy_repo_remote)
    project, card, wt_path = await _make_deployer_card(
        db_session, toy_repo_remote, mode=DeployMode.AUTO_MAIN, command="exit 1"
    )
    visit = await transitions.start_visit(db_session, card)

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="run_deploy", arguments={})],
                endpoint_used="fake::model",
            )
        ]
    )

    result = await run_deployer_visit(
        db_session,
        project,
        card,
        visit,
        llm_client=llm,
        dispatcher=_dispatcher(card.id, wt_path),
        max_iterations=5,
        mode=DeployMode.AUTO_MAIN,
    )

    # One failed attempt, under the default max_deploy_attempts=2 — stays ACTIVE in
    # Deployer for the orchestrator to retry, not yet blocked. (Cap-exhaustion itself
    # is already covered at the transitions layer in test_transitions.py.)
    assert result.column == Column.DEPLOYER
    assert result.lifecycle_state == LifecycleState.ACTIVE
    assert result.deploy_attempt_count == 1
    assert visit.outcome == VisitOutcome.FAILED
