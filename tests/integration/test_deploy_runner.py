"""Deployer's trusted execution path against a real toy git repo (real git, real
filesystem) — neither a live GitHub API nor a real deploy target is available in this
dev environment, so httpx is monkeypatched for the PR-opening tests. See
tests/unit/test_deploy_runner.py for the pure/no-IO cases (URL parsing, missing
token, non-GitHub remote)."""

import subprocess

import httpx
import pytest

from built.domain.enums import DeployKind, DeployMode
from built.sandbox import deploy_runner
from built.sandbox.worktree import create_card_worktree, ensure_tool_worktree
from built.services import card_service, project_service
from built.tools import git_tools

_RealAsyncClient = httpx.AsyncClient


def _mock_github(monkeypatch, handler) -> None:
    """Patches deploy_runner's httpx.AsyncClient to route through a MockTransport.
    Must close over the real AsyncClient captured at import time — deploy_runner.httpx
    is the same module object as this file's `httpx`, so a lambda that calls
    `httpx.AsyncClient(...)` after patching would recurse into itself."""
    monkeypatch.setattr(
        deploy_runner.httpx,
        "AsyncClient",
        lambda **kwargs: _RealAsyncClient(transport=httpx.MockTransport(handler)),
    )


@pytest.fixture
def deployable_repo_remote(toy_repo_remote):
    """toy_repo_remote is a plain (non-bare) working directory with `main` checked
    out — git refuses a push to the currently checked-out branch of a non-bare repo
    by default. That restriction doesn't exist for a real GitHub remote (this app's
    actual repo_remote_url in production); it's purely a local-fixture artifact, so
    it's worked around here rather than in the shared fixture."""
    subprocess.run(
        ["git", "config", "receive.denyCurrentBranch", "updateInstead"],
        cwd=toy_repo_remote,
        check=True,
        capture_output=True,
    )
    return toy_repo_remote


async def _make_project(
    session, repo_remote, *, mode=DeployMode.AUTO_MAIN, kind=DeployKind.COMMAND, command="true"
):
    project = await project_service.create_project(
        session,
        name=f"deploy-{mode.value}-{kind.value}-{command!r}",
        overarching_goal="goal",
        repo_remote_url=str(repo_remote),
    )
    await project_service.set_deploy_config(
        session,
        project.id,
        kind=kind,
        mode=mode,
        command=command,
        github_token_ref="TEST_GH_TOKEN",
    )
    return await project_service.get_project(session, project.id)


async def _make_card_with_change(session, project, title, *, content):
    card = await card_service.create_card(session, project.id, title=title, raw_request="r")
    wt_path = await create_card_worktree(project, card)
    card.worktree_path = str(wt_path)
    (wt_path / "app.py").write_text(content)
    await git_tools.commit_all(wt_path, message=f"update app.py for {title}")
    await session.commit()
    return card


async def test_run_auto_main_deploy_clean_merge_and_deploy_success(db_session, deployable_repo_remote):
    project = await _make_project(db_session, deployable_repo_remote, command="true")
    card = await _make_card_with_change(
        db_session,
        project,
        "add farewell",
        content="def greet():\n    return 'hi'\n\n\ndef farewell():\n    return 'bye'\n",
    )

    wt_path = await ensure_tool_worktree(project, tool="deployer")
    result = await deploy_runner.run_auto_main_deploy(project, card, wt_path)

    assert result.success is True
    assert "deploy command succeeded" in result.message

    # The merge actually landed on the remote's default branch.
    log = subprocess.run(
        ["git", "log", "--oneline", "main"],
        cwd=deployable_repo_remote,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "Merge" in log
    assert (deployable_repo_remote / "app.py").read_text().count("farewell") == 1


async def test_run_auto_main_deploy_discards_stray_untracked_files_before_a_fresh_merge(
    db_session, deployable_repo_remote
):
    """Regression: observed in production — a Deployer agent preemptively wrote a
    file into the worktree before ever calling run_deploy (misreading "the file
    isn't here yet" as something it needed to create). That stray untracked file
    then blocked the real merge with an unrelated git error ("untracked working
    tree file would be overwritten"), not a real conflict — and the agent has no
    tool that can clean that up itself. A fresh merge attempt must discard
    anything sitting in the worktree first, precisely so a model's mistake here
    can never wedge the deploy."""
    project = await _make_project(db_session, deployable_repo_remote, command="true")
    card = await _make_card_with_change(
        db_session,
        project,
        "add farewell",
        content="def greet():\n    return 'hi'\n\n\ndef farewell():\n    return 'bye'\n",
    )

    wt_path = await ensure_tool_worktree(project, tool="deployer")
    (wt_path / "app.py").write_text("this is NOT what the card actually implemented\n")

    result = await deploy_runner.run_auto_main_deploy(project, card, wt_path)

    assert result.success is True, result.message
    assert result.conflict is False
    assert "farewell" in (deployable_repo_remote / "app.py").read_text()
    assert "NOT what the card" not in (deployable_repo_remote / "app.py").read_text()


async def test_run_auto_main_deploy_with_no_deploy_step_still_merges_and_succeeds(
    db_session, deployable_repo_remote
):
    """kind=NONE: merging to the default branch and pushing is the whole job — a
    project with no real deploy target shouldn't be forced to configure a fake
    command just to let auto-main mode reach DONE."""
    project = await _make_project(db_session, deployable_repo_remote, kind=DeployKind.NONE)
    card = await _make_card_with_change(
        db_session,
        project,
        "add farewell",
        content="def greet():\n    return 'hi'\n\n\ndef farewell():\n    return 'bye'\n",
    )

    wt_path = await ensure_tool_worktree(project, tool="deployer")
    result = await deploy_runner.run_auto_main_deploy(project, card, wt_path)

    assert result.success is True
    assert "no deploy step configured" in result.message
    log = subprocess.run(
        ["git", "log", "--oneline", "main"],
        cwd=deployable_repo_remote,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "Merge" in log


async def test_run_auto_main_deploy_deploy_command_failure(db_session, deployable_repo_remote):
    project = await _make_project(db_session, deployable_repo_remote, command="exit 1")
    card = await _make_card_with_change(
        db_session, project, "bad change", content="def greet():\n    return 'hi'\n\n\nx = 1\n"
    )

    wt_path = await ensure_tool_worktree(project, tool="deployer")
    result = await deploy_runner.run_auto_main_deploy(project, card, wt_path)

    assert result.success is False
    assert "exited 1" in result.message


async def test_run_auto_main_deploy_merge_conflict_reports_conflicted_paths(
    db_session, deployable_repo_remote
):
    project = await _make_project(db_session, deployable_repo_remote)
    card_a = await _make_card_with_change(
        db_session, project, "change a", content="def greet():\n    return 'hello from a'\n"
    )
    card_b = await _make_card_with_change(
        db_session, project, "change b", content="def greet():\n    return 'hello from b'\n"
    )
    wt_path = await ensure_tool_worktree(project, tool="deployer")
    first = await deploy_runner.run_auto_main_deploy(project, card_a, wt_path)
    assert first.success is True

    second = await deploy_runner.run_auto_main_deploy(project, card_b, wt_path)

    assert second.success is False
    assert second.conflict is True
    assert second.conflicted_paths == ["app.py"]
    assert "app.py" in second.message
    assert "Merge conflict in" in second.message


async def test_run_auto_main_deploy_aborts_conflict_and_leaves_worktree_clean(
    db_session, deployable_repo_remote
):
    """Unlike the old behavior, a conflict is no longer left for the Deployer agent
    to resolve in place (it has no bash/test tools to re-verify a fix, and doing so
    here would ship straight to production with no further Tester/Reviewer pass —
    see domain/transitions.complete_deployer_visit_conflict, which bounces the card
    back to Developer instead). The merge is aborted immediately, so the worktree
    is left exactly as clean as if run_deploy had never been attempted."""
    project = await _make_project(db_session, deployable_repo_remote)
    card_a = await _make_card_with_change(
        db_session, project, "change a", content="def greet():\n    return 'hello from a'\n"
    )
    card_b = await _make_card_with_change(
        db_session, project, "change b", content="def greet():\n    return 'hello from b'\n"
    )
    wt_path = await ensure_tool_worktree(project, tool="deployer")
    await deploy_runner.run_auto_main_deploy(project, card_a, wt_path)

    result = await deploy_runner.run_auto_main_deploy(project, card_b, wt_path)

    assert result.conflict is True
    assert not await git_tools.merge_in_progress(wt_path)
    assert await git_tools.status(wt_path) == ""
    assert "hello from a" in (wt_path / "app.py").read_text()


async def test_open_pull_request_success(db_session, deployable_repo_remote, monkeypatch):
    project = await _make_project(db_session, deployable_repo_remote, mode=DeployMode.PR_TO_OPERATOR)
    card = await _make_card_with_change(
        db_session,
        project,
        "add farewell",
        content="def greet():\n    return 'hi'\n\n\ndef farewell():\n    return 'bye'\n",
    )
    monkeypatch.setenv("TEST_GH_TOKEN", "fake-token")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer fake-token"
        return httpx.Response(201, json={"html_url": "https://github.com/owner/repo/pull/42"})

    _mock_github(monkeypatch, handler)
    project.repo_remote_url = "https://github.com/owner/repo.git"

    result = await deploy_runner.open_pull_request(project, card, summary="did a thing")

    assert result.success is True
    assert result.url == "https://github.com/owner/repo/pull/42"

    # The card's branch actually landed on the remote, unmerged.
    branches = subprocess.run(
        ["git", "branch", "--list", card.branch_name],
        cwd=deployable_repo_remote,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert card.branch_name in branches


async def test_open_pull_request_github_api_error(db_session, deployable_repo_remote, monkeypatch):
    project = await _make_project(db_session, deployable_repo_remote, mode=DeployMode.PR_TO_OPERATOR)
    card = await _make_card_with_change(
        db_session, project, "t", content="def greet():\n    return 'hi'\nx = 1\n"
    )
    monkeypatch.setenv("TEST_GH_TOKEN", "fake-token")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"message": "Validation Failed"})

    _mock_github(monkeypatch, handler)
    project.repo_remote_url = "https://github.com/owner/repo.git"

    result = await deploy_runner.open_pull_request(project, card, summary="did a thing")

    assert result.success is False
    assert "422" in result.message
