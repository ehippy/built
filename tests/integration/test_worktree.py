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


async def test_sync_card_branch_with_default_picks_up_new_commits_cleanly(toy_repo_remote):
    """The scenario this exists for: another card merges to main while this card's
    worktree, branched off main at creation, has no way to see it on its own."""
    # Distinct project/card ids from every other test in this file — data_dir is a
    # single shared tmp dir for the whole test session (tests/conftest.py), keyed
    # by these ids, so reusing _project()/_card()'s fixed ids here would silently
    # fetch from a stale, unrelated toy_repo_remote left over from another test.
    project = Project(id="proj-wt-sync-clean", repo_remote_url=str(toy_repo_remote), default_branch="main")
    card = Card(id="card-wt-sync-clean", project_id=project.id, branch_name="card/sync-clean")
    wt_path = await worktree.create_card_worktree(project, card)

    # Someone else's card lands a change directly on the remote's main, unrelated
    # to anything this card touches.
    (toy_repo_remote / "other.py").write_text("value = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=toy_repo_remote, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "add other.py"], cwd=toy_repo_remote, check=True, capture_output=True
    )
    await worktree.ensure_managed_clone(project)

    conflicted = await worktree.sync_card_branch_with_default(project, wt_path)

    assert conflicted == []
    assert (wt_path / "other.py").read_text() == "value = 1\n"
    assert not await git_tools.merge_in_progress(wt_path)


async def test_sync_card_branch_with_default_leaves_conflict_markers_for_developer(toy_repo_remote):
    """A real conflict is left unresolved (MERGE_HEAD set, markers in the file) —
    unlike Deployer's own merge attempt, nothing here aborts it: this is for a
    caller (Developer, via orchestrator/worker.py) with its own read/write/bash
    tools to actually fix it."""
    project = Project(id="proj-wt-sync-conflict", repo_remote_url=str(toy_repo_remote), default_branch="main")
    card = Card(id="card-wt-sync-conflict", project_id=project.id, branch_name="card/sync-conflict")
    wt_path = await worktree.create_card_worktree(project, card)

    # This card's own work changes app.py...
    (wt_path / "app.py").write_text("def greet():\n    return 'from the card'\n")
    await git_tools.commit_all(wt_path, message="card change")

    # ...but another card already landed a conflicting change to the same file on
    # main in the meantime.
    (toy_repo_remote / "app.py").write_text("def greet():\n    return 'from main'\n")
    subprocess.run(
        ["git", "commit", "-aqm", "main change"], cwd=toy_repo_remote, check=True, capture_output=True
    )
    await worktree.ensure_managed_clone(project)

    conflicted = await worktree.sync_card_branch_with_default(project, wt_path)

    assert conflicted == ["app.py"]
    assert await git_tools.merge_in_progress(wt_path)
    assert "<<<<<<<" in (wt_path / "app.py").read_text()


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


async def test_ensure_tool_worktree_self_heals_after_abandoned_merge_conflict(toy_repo_remote):
    """Regression: deploy_runner.py deliberately leaves a real merge conflict
    unresolved in the tool worktree so the Deployer agent can fix it across tool
    calls within one visit — but if that visit ends without finishing (a human
    cancelling the card, a crash), nothing used to clean it up. Every subsequent
    ensure_tool_worktree call for the project then failed at `git checkout`
    ("you need to resolve your current index first"), wedging every card that
    reached Deployer behind that one abandoned conflict. ensure_tool_worktree must
    recover on its own, with no manual git intervention."""
    project = Project(id="proj-wt-heal", repo_remote_url=str(toy_repo_remote), default_branch="main")

    wt_path = await worktree.ensure_tool_worktree(project, tool="deployer")
    bare_path = await worktree.ensure_managed_clone(project)

    # A branch that conflicts with app.py's content on main.
    conflict_wt = wt_path.parent / "conflict-source"
    await git_tools.run_git(
        "worktree", "add", "-b", "conflicting-branch", str(conflict_wt), "main", cwd=bare_path
    )
    (conflict_wt / "app.py").write_text("def greet():\n    return 'conflicting'\n")
    await git_tools.commit_all(conflict_wt, message="conflicting change")

    # Simulate the abandoned auto_main merge: start a merge that genuinely
    # conflicts, and leave it exactly as a cancelled visit would — no abort, no
    # resolution, nothing committed.
    (wt_path / "app.py").write_text("def greet():\n    return 'unresolved local change'\n")
    await git_tools.commit_all(wt_path, message="local change on tool branch")
    try:
        # -c user.name/user.email, matching deploy_runner.py's own real merge call —
        # --no-ff needs a committer identity to even attempt the merge, and without
        # this the command fails on "Committer identity unknown" before ever
        # reaching conflict detection on a machine/CI runner with no global git
        # config (confirmed empirically: passed locally where ~/.gitconfig already
        # has an identity, failed in CI where nothing does).
        await git_tools.run_git(
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "merge",
            "--no-ff",
            "-m",
            "merge attempt",
            "conflicting-branch",
            cwd=wt_path,
        )
        raise AssertionError("expected this merge to conflict")
    except git_tools.GitCommandError:
        pass
    assert await git_tools.merge_in_progress(wt_path)

    # The next visit's setup call must not raise, and must leave the worktree clean
    # and reset to the tip of default_branch — not still holding the conflict.
    healed_path = await worktree.ensure_tool_worktree(project, tool="deployer")

    assert healed_path == wt_path
    assert not await git_tools.merge_in_progress(wt_path)
    assert await git_tools.status(wt_path) == ""
    assert (wt_path / "app.py").read_text() == "def greet():\n    return 'hi'\n"
