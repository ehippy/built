"""Exercises sandbox/worktree.py and tools/git_tools.py against a real local git
repo (no network) — this is real `git` shelling out, not a mock, since getting the
bare-repo ref layout wrong would silently break every Developer run."""

import subprocess

from built.db.models import Card, Project
from built.sandbox import worktree
from built.tools import git_tools


def _project(remote_path) -> Project:
    return Project(id="proj-wt", repo_remote_url=str(remote_path), default_branch="main")


def _card(project: Project) -> Card:
    return Card(id="card-wt", project_id=project.id, branch_name="card/card-wt-add-feature")


async def test_ensure_managed_clone_creates_bare_repo(toy_repo_remote):
    project = _project(toy_repo_remote)
    bare_path = await worktree.ensure_managed_clone(project)
    assert bare_path.exists()
    assert bare_path.name.endswith(".git")

    # Idempotent: calling again (a fetch, not a re-clone) doesn't error.
    bare_path_again = await worktree.ensure_managed_clone(project)
    assert bare_path_again == bare_path


async def test_create_card_worktree_and_commit_flow(toy_repo_remote):
    project = _project(toy_repo_remote)
    card = _card(project)

    wt_path = await worktree.create_card_worktree(project, card)
    assert (wt_path / "app.py").is_file()
    assert (wt_path / ".git").exists()

    # Idempotent: a second call for the same card reuses the existing worktree.
    wt_path_again = await worktree.create_card_worktree(project, card)
    assert wt_path_again == wt_path

    (wt_path / "app.py").write_text("def greet():\n    return 'hello'\n")
    sha = await git_tools.commit_all(wt_path, message="Developer: update greet()")
    assert sha and len(sha) == 40

    # Nothing changed since the last commit -> no-op, not an empty commit.
    no_op_sha = await git_tools.commit_all(wt_path, message="nothing to see here")
    assert no_op_sha is None

    log = await git_tools.run_git("log", "--oneline", "-1", cwd=wt_path)
    assert "update greet" in log

    diff_output = await git_tools.diff(wt_path, staged=False)
    assert diff_output == ""  # committed, so nothing left unstaged/uncommitted


async def test_read_default_branch_file_returns_none_when_missing(toy_repo_remote):
    project = Project(id="proj-wt-read-1", repo_remote_url=str(toy_repo_remote), default_branch="main")
    content = await worktree.read_default_branch_file(project, "AGENTS.md")
    assert content is None


async def test_read_default_branch_file_reads_committed_content(toy_repo_remote):
    project = Project(id="proj-wt-read-2", repo_remote_url=str(toy_repo_remote), default_branch="main")
    wt_path = await worktree.ensure_tool_worktree(project, tool="read_test")
    (wt_path / "AGENTS.md").write_text("# Practices\n\n- Use pytest.\n")
    await git_tools.commit_all(wt_path, message="add AGENTS.md")
    # Committing lands on the tool's own dedicated branch, not on default_branch
    # itself — it only reaches "main" once pushed there, same as any real usage
    # (deploy_runner.py's auto-main push, or the Tender's push after an edit).
    # toy_repo_remote is a plain (non-bare) checkout with "main" checked out — git
    # refuses a push to a non-bare repo's currently checked-out branch by default;
    # that restriction doesn't exist for a real GitHub remote, so it's worked around
    # here rather than in the shared fixture.
    subprocess.run(
        ["git", "config", "receive.denyCurrentBranch", "updateInstead"],
        cwd=toy_repo_remote,
        check=True,
        capture_output=True,
    )
    await git_tools.run_git("push", "origin", f"HEAD:{project.default_branch}", cwd=wt_path)

    content = await worktree.read_default_branch_file(project, "AGENTS.md")

    assert content is not None
    assert "Use pytest" in content


async def test_tool_worktrees_are_independent_and_coexist(toy_repo_remote):
    """Each tool gets its own dedicated branch, not the literal default branch — git
    refuses to check out the same branch in two worktrees at once, so this would
    fail outright if deployer and tender both tried to use `main` directly."""
    project = Project(id="proj-wt-tools", repo_remote_url=str(toy_repo_remote), default_branch="main")

    deployer_path = await worktree.ensure_tool_worktree(project, tool="deployer")
    tender_path = await worktree.ensure_tool_worktree(project, tool="tender")

    assert deployer_path != tender_path
    assert deployer_path.exists()
    assert tender_path.exists()

    # Reusing an existing tool worktree resets it to the tip of default_branch rather
    # than erroring or re-creating it.
    deployer_path_again = await worktree.ensure_tool_worktree(project, tool="deployer")
    assert deployer_path_again == deployer_path
