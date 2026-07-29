"""Pure/no-IO paths of Deployer's trusted execution path — URL parsing and the
early-exit validation branches of open_pull_request. See
tests/integration/test_deploy_runner.py for the full push/merge/HTTP flows against a
real toy git repo."""

from built.db.models import Card, DeployConfig, Project
from built.domain.enums import DeployKind, DeployMode
from built.sandbox.deploy_runner import _parse_github_owner_repo, open_pull_request


def test_parse_github_owner_repo_https():
    assert _parse_github_owner_repo("https://github.com/octocat/hello-world.git") == (
        "octocat",
        "hello-world",
    )


def test_parse_github_owner_repo_https_no_dot_git():
    assert _parse_github_owner_repo("https://github.com/octocat/hello-world") == ("octocat", "hello-world")


def test_parse_github_owner_repo_ssh():
    assert _parse_github_owner_repo("git@github.com:octocat/hello-world.git") == ("octocat", "hello-world")


def test_parse_github_owner_repo_non_github_returns_none():
    assert _parse_github_owner_repo("https://gitlab.com/octocat/hello-world.git") is None


def _project(**overrides) -> Project:
    defaults = {
        "name": "p",
        "slug": "p",
        "overarching_goal": "goal",
        "repo_remote_url": "https://github.com/octocat/hello-world.git",
    }
    defaults.update(overrides)
    project = Project(**defaults)
    project.deploy_config = None
    return project


def _card() -> Card:
    return Card(project_id="p", title="t", raw_request="r", branch_name="card/x-test")


async def test_open_pull_request_fails_cleanly_on_non_github_remote():
    project = _project(repo_remote_url="https://gitlab.com/octocat/hello-world.git")

    result = await open_pull_request(project, _card(), summary="s")

    assert result.success is False
    assert "not a github.com URL" in result.message


async def test_open_pull_request_fails_cleanly_with_no_github_token_ref():
    project = _project()
    project.deploy_config = DeployConfig(
        project_id="p", kind=DeployKind.COMMAND, mode=DeployMode.PR_TO_OPERATOR, github_token_ref=None
    )

    result = await open_pull_request(project, _card(), summary="s")

    assert result.success is False
    assert "no GitHub token configured" in result.message


async def test_open_pull_request_fails_cleanly_when_token_env_var_is_unset(monkeypatch):
    monkeypatch.delenv("SOME_UNSET_TOKEN_VAR", raising=False)
    project = _project()
    project.deploy_config = DeployConfig(
        project_id="p",
        kind=DeployKind.COMMAND,
        mode=DeployMode.PR_TO_OPERATOR,
        github_token_ref="SOME_UNSET_TOKEN_VAR",
    )

    result = await open_pull_request(project, _card(), summary="s")

    assert result.success is False
    assert "no GitHub token configured" in result.message
