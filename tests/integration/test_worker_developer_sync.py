"""orchestrator/worker.py's wiring for the Developer-visit branch sync
(sandbox/worktree.sync_card_branch_with_default) — run_one_card must call it during
worktree setup for a DEVELOPER-column card, before the LLM is ever called. That
ordering is what this test actually exercises: pointing the project at an
unreachable LLM endpoint makes the visit fail fast (see
test_worker_lifecycle_logging.py for that same trick), but the sync step already
ran as part of worktree setup by the time that failure happens, so its effect on
the worktree is still observable."""

import subprocess

from built.domain.enums import Column, LifecycleState
from built.orchestrator.worker import claim_next_card, run_one_card
from built.sandbox import worktree
from built.services import card_service, endpoint_service, project_service
from built.tools import git_tools


async def test_run_one_card_syncs_developer_branch_before_the_llm_call(db_session, toy_repo_remote):
    project = await project_service.create_project(
        db_session,
        name="worker-sync-conflict",
        overarching_goal="goal",
        repo_remote_url=str(toy_repo_remote),
    )
    await endpoint_service.create_endpoint_config(
        db_session,
        base_url="http://127.0.0.1:1",  # unreachable — connection refused, fails fast
        model="fake-model",
        project_id=project.id,
        role=Column.DEVELOPER,
    )
    card = await card_service.create_card(db_session, project.id, title="Add a widget", raw_request="r")
    card.column = Column.DEVELOPER
    await db_session.flush()
    wt_path = await worktree.create_card_worktree(project, card)
    (wt_path / "app.py").write_text("def greet():\n    return 'from the card'\n")
    await git_tools.commit_all(wt_path, message="card change")
    await db_session.commit()

    # Another card lands a conflicting change directly on main while this card's
    # worktree — branched off main once, at creation — has no way to see it.
    (toy_repo_remote / "app.py").write_text("def greet():\n    return 'from main'\n")
    subprocess.run(
        ["git", "commit", "-aqm", "main change"], cwd=toy_repo_remote, check=True, capture_output=True
    )

    claimed = await claim_next_card(db_session, "worker-test-sync")
    assert claimed is not None
    await db_session.commit()

    await run_one_card(db_session, claimed)

    # The sync happened even though the visit itself never got past setup.
    assert await git_tools.merge_in_progress(wt_path)
    assert "<<<<<<<" in (wt_path / "app.py").read_text()
    assert claimed.lifecycle_state == LifecycleState.BLOCKED


async def test_run_one_card_does_not_sync_for_non_developer_columns(db_session, toy_repo_remote):
    """The sync is specific to Developer visits — a Tester/Reviewer/PM worktree
    should not suddenly grow an in-progress merge underneath a column with no
    tools to resolve one."""
    project = await project_service.create_project(
        db_session,
        name="worker-sync-tester",
        overarching_goal="goal",
        repo_remote_url=str(toy_repo_remote),
    )
    await endpoint_service.create_endpoint_config(
        db_session,
        base_url="http://127.0.0.1:1",
        model="fake-model",
        project_id=project.id,
        role=Column.TESTER,
    )
    card = await card_service.create_card(db_session, project.id, title="Add a widget", raw_request="r")
    card.column = Column.TESTER
    await db_session.flush()
    wt_path = await worktree.create_card_worktree(project, card)
    (wt_path / "app.py").write_text("def greet():\n    return 'from the card'\n")
    await git_tools.commit_all(wt_path, message="card change")
    await db_session.commit()

    (toy_repo_remote / "app.py").write_text("def greet():\n    return 'from main'\n")
    subprocess.run(
        ["git", "commit", "-aqm", "main change"], cwd=toy_repo_remote, check=True, capture_output=True
    )

    claimed = await claim_next_card(db_session, "worker-test-no-sync")
    assert claimed is not None
    await db_session.commit()

    await run_one_card(db_session, claimed)

    assert not await git_tools.merge_in_progress(wt_path)
    assert "from the card" in (wt_path / "app.py").read_text()
