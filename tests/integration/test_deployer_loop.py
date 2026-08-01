"""Deployer agent loop against a real toy git repo — LLM and GitHub faked, everything
else real, including the actual merge/push (auto_main) and push+PR (pr_to_operator)."""

import subprocess

import httpx
from sqlalchemy import select

from built.agent import summarizer
from built.agent.loop import run_deployer_visit
from built.db.models import CardPostmortem
from built.domain import transitions
from built.domain.enums import Column, DeployKind, DeployMode, LifecycleState, VisitOutcome
from built.llm.client import LLMResult, ToolCallRequest
from built.orchestrator.ci_watcher import run_ci_watcher_once
from built.orchestrator.pr_watcher import run_pr_watcher_once
from built.sandbox import deploy_runner, worktree
from built.sandbox.container import CommandResult
from built.services import card_service, project_service
from built.tools import git_tools
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher
from tests.unit.fakes import FakeCommandExecutor, ScriptedLLMClient


def _stub_postmortem_llm(monkeypatch, went_well: str = "n/a", struggles: str = "n/a") -> ScriptedLLMClient:
    """agent/summarizer.py builds its own FallbackLLMClient internally when no
    llm_client is injected (the production path from agent/loop.py and
    orchestrator/ci_watcher.py never injects one) — patching the class it
    constructs is the seam that lets these loop-level tests fake that call too,
    without needing a real EndpointConfig row."""
    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="pm_1",
                        name="submit_postmortem",
                        arguments={"went_well": went_well, "struggles": struggles},
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )
    monkeypatch.setattr(summarizer, "FallbackLLMClient", lambda chain: llm)
    return llm


async def _postmortems_for(db_session, card_id: str) -> list[CardPostmortem]:
    return list(
        (await db_session.scalars(select(CardPostmortem).where(CardPostmortem.card_id == card_id))).all()
    )


async def _async_result(value):
    return value


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


async def _make_deployer_card(
    db_session, repo_remote, *, mode: DeployMode, command: str = "true", content: str | None = None
):
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
    card_wt_path = await worktree.create_card_worktree(project, card)
    card.worktree_path = str(card_wt_path)
    (card_wt_path / "app.py").write_text(
        content or "def greet():\n    return 'hi'\n\n\ndef farewell():\n    return 'bye'\n"
    )
    await git_tools.commit_all(card_wt_path, message="add farewell")
    await db_session.flush()

    # auto_main's dispatcher worktree is the Deployer's own dedicated worktree, not
    # the card's — matching production (orchestrator/worker.py): that's where a
    # merge (and any conflict) actually happens. pr_to_operator never merges, so it
    # stays on the card's own branch, matching production there too.
    if mode == DeployMode.AUTO_MAIN:
        wt_path = await worktree.ensure_tool_worktree(project, tool="deployer")
    else:
        wt_path = card_wt_path
    return project, card, wt_path


def _allow_push_to_checked_out_branch(repo_remote) -> None:
    subprocess.run(
        ["git", "config", "receive.denyCurrentBranch", "updateInstead"],
        cwd=repo_remote,
        check=True,
        capture_output=True,
    )


def _dispatcher(card_id: str, wt_path, *, auto_commit: bool = True) -> ToolDispatcher:
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    return ToolDispatcher(
        ctx=ToolContext(card_id=card_id, worktree_root=wt_path, auto_commit=auto_commit), executor=executor
    )


async def _make_conflict_setup(db_session, repo_remote):
    """A project with two DEPLOYER-column cards whose branches both fork off the
    *same* original main before either one merges — genuinely diverged, so merging
    the second into the already-merged first produces a real 3-way conflict, not a
    trivial fast-forward. (Branching card_b off main *after* card_a already merged
    would just inherit card_a's change as card_b's own base — create_card_worktree
    re-fetches first, so 'created later' silently means 'based on later main' too.)
    Deploys card_a immediately, landing it cleanly; card_b is left ready for a
    Deployer visit to hit the conflict."""
    project = await project_service.create_project(
        db_session, name="deployer-loop-conflict", overarching_goal="goal", repo_remote_url=str(repo_remote)
    )
    await project_service.set_deploy_config(
        db_session, project.id, kind=DeployKind.COMMAND, mode=DeployMode.AUTO_MAIN, command="true"
    )
    project = await project_service.get_project(db_session, project.id)

    card_a = await card_service.create_card(db_session, project.id, title="change a", raw_request="r")
    card_a.column = Column.DEPLOYER
    await db_session.flush()
    wt_a = await worktree.create_card_worktree(project, card_a)
    (wt_a / "app.py").write_text("def greet():\n    return 'hello from a'\n")
    await git_tools.commit_all(wt_a, message="change a")
    await db_session.flush()

    card_b = await card_service.create_card(db_session, project.id, title="change b", raw_request="r")
    card_b.column = Column.DEPLOYER
    await db_session.flush()
    wt_b = await worktree.create_card_worktree(project, card_b)
    card_b.worktree_path = str(wt_b)
    (wt_b / "app.py").write_text("def greet():\n    return 'hello from b'\n")
    await git_tools.commit_all(wt_b, message="change b")
    await db_session.flush()

    wt_path = await worktree.ensure_tool_worktree(project, tool="deployer")
    await deploy_runner.run_auto_main_deploy(project, card_a, wt_path)
    return project, card_b, wt_path


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

    # The Deployer's own job — the git-level push — is done, but the card isn't
    # DONE yet: repo_remote_url here is a local path, not github.com, so there's
    # no CI to ever wait on, but that's still only resolved by the watcher's own
    # pass, not synchronously inside run_deployer_visit.
    assert result.lifecycle_state == LifecycleState.ACTIVE
    assert result.deploying_commit_sha is not None
    assert visit.outcome == VisitOutcome.DEPLOYED_PENDING_CI
    log = subprocess.run(
        ["git", "log", "--oneline", "main"], cwd=toy_repo_remote, check=True, capture_output=True, text=True
    ).stdout
    assert "Merge" in log

    counts = await run_ci_watcher_once()
    assert counts["confirmed"] == 1
    await db_session.refresh(card)
    assert card.lifecycle_state == LifecycleState.DONE
    assert card.deploying_commit_sha is None


async def test_deployer_loop_auto_main_conflict_bounces_card_back_to_developer(db_session, toy_repo_remote):
    """The actual point of this feature: run_deploy() hitting a real merge conflict
    is no longer something the Deployer agent resolves itself (it has no bash/test
    tools to re-verify a fix). One run_deploy call is enough to end the visit —
    the card goes straight back to Developer with feedback explaining why, and the
    conflicted merge worktree is left clean, not sitting there waiting for a second
    run_deploy call that will never come."""
    _allow_push_to_checked_out_branch(toy_repo_remote)
    project, card_b, wt_path = await _make_conflict_setup(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card_b)

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="run_deploy", arguments={})],
                endpoint_used="fake::model",
            ),
        ]
    )

    result = await run_deployer_visit(
        db_session,
        project,
        card_b,
        visit,
        llm_client=llm,
        dispatcher=_dispatcher(card_b.id, wt_path),
        max_iterations=10,
        mode=DeployMode.AUTO_MAIN,
    )

    assert result.column == Column.DEVELOPER
    assert result.lifecycle_state == LifecycleState.ACTIVE
    assert result.revision_count == 1
    assert result.deploy_attempt_count == 0
    assert visit.outcome == VisitOutcome.DEPLOY_CONFLICT
    assert result.latest_feedback is not None
    assert "app.py" in result.latest_feedback
    assert not await git_tools.merge_in_progress(wt_path)
    assert await git_tools.status(wt_path) == ""


async def test_deployer_loop_auto_main_abandon_deploy(db_session, toy_repo_remote):
    _allow_push_to_checked_out_branch(toy_repo_remote)
    project, card, wt_path = await _make_deployer_card(db_session, toy_repo_remote, mode=DeployMode.AUTO_MAIN)
    visit = await transitions.start_visit(db_session, card)

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="abandon_deploy",
                        arguments={"reason": "deploy config looks wrong, needs a human"},
                    )
                ],
                endpoint_used="fake::model",
            ),
        ]
    )

    result = await run_deployer_visit(
        db_session,
        project,
        card,
        visit,
        llm_client=llm,
        dispatcher=_dispatcher(card.id, wt_path),
        max_iterations=10,
        mode=DeployMode.AUTO_MAIN,
    )

    # One failed (abandoned) attempt, under the default max_deploy_attempts=2 —
    # stays ACTIVE for a human or the Reviver to look at, not yet BLOCKED.
    assert result.column == Column.DEPLOYER
    assert result.lifecycle_state == LifecycleState.ACTIVE
    assert result.deploy_attempt_count == 1
    assert visit.outcome == VisitOutcome.FAILED
    assert "abandoned: deploy config looks wrong" in visit.summary


async def test_deployer_loop_pr_to_operator_happy_path_opens_pr(db_session, toy_repo_remote, monkeypatch):
    _allow_push_to_checked_out_branch(toy_repo_remote)
    project, card, wt_path = await _make_deployer_card(
        db_session, toy_repo_remote, mode=DeployMode.PR_TO_OPERATOR
    )
    project.repo_remote_url = "https://github.com/owner/repo.git"
    visit = await transitions.start_visit(db_session, card)
    monkeypatch.setenv("TEST_GH_TOKEN", "fake-token")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            # existing-PR lookup: no open PR for this branch yet
            return httpx.Response(200, json=[])
        return httpx.Response(
            201,
            json={"html_url": "https://github.com/owner/repo/pull/7", "number": 7},
        )

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

    # Opening the PR is the Deployer's job, but the card's completion isn't — it
    # stays ACTIVE (excluded from claiming via pr_number) for the pr_watcher to
    # confirm the PR is reviewed and merged, exactly like auto_main + CI.
    assert result.lifecycle_state == LifecycleState.ACTIVE
    assert result.deploy_url == "https://github.com/owner/repo/pull/7"
    assert result.pr_number == 7
    assert visit.outcome == VisitOutcome.DEPLOYED_PENDING_PR


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


# --- Postmortem hook (agent/summarizer.py) -----------------------------------------


async def test_deployer_pr_to_operator_defers_postmortem_until_merged(
    db_session, toy_repo_remote, monkeypatch
):
    """pr_to_operator's success path lands in DEPLOYED_PENDING_PR, not DONE — a
    human/pr_watcher review still has to happen — so the postmortem must not fire
    yet; it only fires once orchestrator/pr_watcher.py confirms the PR merged,
    matching where DONE itself happens (domain.transitions.confirm_pr_merged)."""
    _allow_push_to_checked_out_branch(toy_repo_remote)
    project, card, wt_path = await _make_deployer_card(
        db_session, toy_repo_remote, mode=DeployMode.PR_TO_OPERATOR
    )
    project.repo_remote_url = "https://github.com/owner/repo.git"
    visit = await transitions.start_visit(db_session, card)
    monkeypatch.setenv("TEST_GH_TOKEN", "fake-token")
    _stub_postmortem_llm(monkeypatch, went_well="clean PR", struggles="none")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            # existing-PR lookup: no open PR for this branch yet
            return httpx.Response(200, json=[])
        return httpx.Response(201, json={"html_url": "https://github.com/owner/repo/pull/7", "number": 7})

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

    # Opening the PR is the Deployer's job; the card's completion isn't — it stays
    # ACTIVE (excluded from claiming via pr_number), so no postmortem yet.
    assert result.lifecycle_state == LifecycleState.ACTIVE
    assert result.pr_number == 7
    assert await _postmortems_for(db_session, card.id) == []

    monkeypatch.setattr(
        deploy_runner,
        "fetch_pr_status",
        lambda project, card: _async_result(deploy_runner.PullRequestStatus(merged=True, state="closed")),
    )

    # The PR merges (by a human, or the watcher after an approving review).
    async def _merged(_project, _card):
        return deploy_runner.PullRequestStatus(merged=True, state="closed")

    monkeypatch.setattr(deploy_runner, "fetch_pr_status", _merged)
    await run_pr_watcher_once()
    await db_session.refresh(card)

    assert card.lifecycle_state == LifecycleState.DONE
    postmortems = await _postmortems_for(db_session, card.id)
    assert len(postmortems) == 1
    assert postmortems[0].outcome == LifecycleState.DONE
    assert postmortems[0].went_well == "clean PR"


async def test_deployer_auto_main_pending_ci_defers_postmortem_until_confirmed(
    db_session, toy_repo_remote, monkeypatch
):
    """auto_main's success path lands in DEPLOYED_PENDING_CI, not DONE — the
    postmortem must not fire yet; it only fires once orchestrator/ci_watcher.py
    actually confirms the card, matching where DONE itself happens."""
    _allow_push_to_checked_out_branch(toy_repo_remote)
    project, card, wt_path = await _make_deployer_card(db_session, toy_repo_remote, mode=DeployMode.AUTO_MAIN)
    visit = await transitions.start_visit(db_session, card)
    _stub_postmortem_llm(monkeypatch, went_well="merged cleanly", struggles="none")

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

    assert result.lifecycle_state == LifecycleState.ACTIVE
    assert await _postmortems_for(db_session, card.id) == []

    await run_ci_watcher_once()
    await db_session.refresh(card)

    assert card.lifecycle_state == LifecycleState.DONE
    postmortems = await _postmortems_for(db_session, card.id)
    assert len(postmortems) == 1
    assert postmortems[0].went_well == "merged cleanly"


async def test_deployer_failed_over_cap_writes_a_postmortem(db_session, toy_repo_remote, monkeypatch):
    """Under the cap (first abandon), no postmortem yet — the card can still
    retry. Only once deploy_attempt_count reaches max_deploy_attempts and
    lifecycle_state actually flips to FAILED does one get written."""
    _allow_push_to_checked_out_branch(toy_repo_remote)
    project, card, wt_path = await _make_deployer_card(db_session, toy_repo_remote, mode=DeployMode.AUTO_MAIN)
    _stub_postmortem_llm(monkeypatch, went_well="n/a", struggles="deploy config was never fixed")

    def _abandon_llm() -> ScriptedLLMClient:
        return ScriptedLLMClient(
            [
                LLMResult(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(id="call_1", name="abandon_deploy", arguments={"reason": "broken"})
                    ],
                    endpoint_used="fake::model",
                )
            ]
        )

    visit_1 = await transitions.start_visit(db_session, card)
    result = await run_deployer_visit(
        db_session,
        project,
        card,
        visit_1,
        llm_client=_abandon_llm(),
        dispatcher=_dispatcher(card.id, wt_path),
        max_iterations=5,
        mode=DeployMode.AUTO_MAIN,
    )
    assert result.lifecycle_state == LifecycleState.ACTIVE
    assert result.deploy_attempt_count == 1
    assert await _postmortems_for(db_session, card.id) == []

    visit_2 = await transitions.start_visit(db_session, card)
    result = await run_deployer_visit(
        db_session,
        project,
        card,
        visit_2,
        llm_client=_abandon_llm(),
        dispatcher=_dispatcher(card.id, wt_path),
        max_iterations=5,
        mode=DeployMode.AUTO_MAIN,
    )

    assert result.lifecycle_state == LifecycleState.FAILED
    assert result.deploy_attempt_count == 2
    postmortems = await _postmortems_for(db_session, card.id)
    assert len(postmortems) == 1
    assert postmortems[0].outcome == LifecycleState.FAILED
