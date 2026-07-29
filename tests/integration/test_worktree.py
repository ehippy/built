"""Exercises sandbox/worktree.py and tools/git_tools.py against a real local git
repo (no network) — this is real `git` shelling out, not a mock, since getting the
bare-repo ref layout wrong would silently break every Developer run."""

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
