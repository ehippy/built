"""run_check (llm/tool_schemas.RUN_CHECK) exists so roles with no write/edit tools
(Reviewer, Deployer in pr_to_operator mode — see llm/tool_schemas.REVIEWER_TOOLS/
deployer_tools) can still run a real verification command — the project's own
build/E2E/lint command — without that becoming a backdoor way to fix things. The
dispatcher enforces that by discarding whatever the command did to the worktree the
moment it returns, win or lose (see tools/dispatcher.py's "run_check" handling and
tools/git_tools.discard_uncommitted_changes). Real git, real filesystem — the
FakeCommandExecutor never touches disk itself (see tests/unit/fakes.py), so these
tests simulate "the command already ran and left a mess" by mutating the worktree
directly before dispatching, then assert the discard step cleans it up regardless
of the (separately faked) exit code."""

from built.db.models import Card, Project
from built.sandbox import worktree
from built.sandbox.container import CommandResult
from built.tools import git_tools
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher
from tests.unit.fakes import FakeCommandExecutor


async def _make_card_worktree(toy_repo_remote, suffix: str):
    project = Project(
        id=f"proj-run-check-{suffix}", repo_remote_url=str(toy_repo_remote), default_branch="main"
    )
    card = Card(id=f"card-run-check-{suffix}", project_id=project.id, branch_name=f"card/run-check-{suffix}")
    wt_path = await worktree.create_card_worktree(project, card)
    return card, wt_path


async def test_run_check_never_commits_even_on_a_tracked_file_change(toy_repo_remote):
    card, wt_path = await _make_card_worktree(toy_repo_remote, "commit")
    # Stands in for what a real check command running inside the sandbox would have
    # just done to the worktree — FakeCommandExecutor itself never touches disk.
    (wt_path / "app.py").write_text("def greet():\n    return 'MUTATED BY CHECK'\n")

    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="ok", stderr=""))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)

    outcome = await dispatcher.dispatch("run_check", {"command": "npm run test:e2e"})

    assert outcome.commit_sha is None
    assert await git_tools.status(wt_path) == ""
    assert (wt_path / "app.py").read_text() == "def greet():\n    return 'hi'\n"


async def test_run_check_reports_the_real_failure_but_still_discards_it(toy_repo_remote):
    """The verification signal itself must survive (a failing check must still read
    as failing) even though its side effects don't."""
    card, wt_path = await _make_card_worktree(toy_repo_remote, "failure")
    (wt_path / "app.py").write_text("def greet():\n    return 'MUTATED BY CHECK'\n")
    (wt_path / "stray_output.log").write_text("some build output\n")

    executor = FakeCommandExecutor(CommandResult(exit_code=1, stdout="1 failed", stderr="boom"))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)

    outcome = await dispatcher.dispatch("run_check", {"command": "npm run test:e2e"})

    assert outcome.result.is_error
    assert "1 failed" in outcome.result.output
    assert outcome.commit_sha is None
    assert await git_tools.status(wt_path) == ""
    assert (wt_path / "app.py").read_text() == "def greet():\n    return 'hi'\n"
    assert not (wt_path / "stray_output.log").exists()


async def test_run_check_preserves_gitignored_artifacts_for_reuse(toy_repo_remote):
    """Deliberately not `git clean -fdx`: a dependency install (node_modules,
    browser binaries) a check just did is gitignored, so it's already excluded from
    git add -A on its own — leaving it in place saves a later run_check call in the
    same visit from redoing that setup, with no risk of it leaking into a commit."""
    card, wt_path = await _make_card_worktree(toy_repo_remote, "gitignore")
    (wt_path / ".gitignore").write_text("node_modules/\n")
    await git_tools.commit_all(wt_path, message="add gitignore")

    (wt_path / "node_modules").mkdir()
    (wt_path / "node_modules" / "some_dep.js").write_text("// installed by the check\n")

    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)

    await dispatcher.dispatch("run_check", {"command": "npm install"})

    assert (wt_path / "node_modules" / "some_dep.js").exists()
