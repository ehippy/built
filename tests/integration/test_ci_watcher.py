"""orchestrator/ci_watcher.py — polls GitHub's Checks API for cards whose
auto_main deploy pushed successfully but hasn't been confirmed by CI yet
(Card.deploying_commit_sha). fetch_check_runs itself is monkeypatched throughout
— it's a thin GitHub API wrapper with its own URL-parsing tests in
tests/unit/test_deploy_runner.py; what matters here is the watcher's own
decision logic given whatever it returns."""

from datetime import UTC, datetime, timedelta

from built.config import settings
from built.domain.enums import Column, DeployKind, DeployMode, LifecycleState
from built.orchestrator.ci_watcher import run_ci_watcher_once
from built.sandbox import deploy_runner
from built.sandbox.deploy_runner import CheckRun, CIStatusUnavailableError
from built.services import card_service, project_service


async def _make_pending_card(session, *, deploying_since=None, max_revisions=3):
    project = await project_service.create_project(
        session,
        name=f"ci-watcher-{id(session)}-{id(deploying_since)}",
        overarching_goal="goal",
        repo_remote_url="https://github.com/octocat/hello-world.git",
        max_revisions=max_revisions,
    )
    await project_service.set_deploy_config(
        session,
        project.id,
        kind=DeployKind.NONE,
        mode=DeployMode.AUTO_MAIN,
        github_token_ref="TEST_GH_TOKEN",
    )
    card = await card_service.create_card(session, project.id, title="Ship it", raw_request="r")
    card.column = Column.DEPLOYER
    card.deploying_commit_sha = "abc123def456"
    card.deploying_since = deploying_since or datetime.now(UTC)
    await session.commit()
    return card


async def test_all_checks_green_confirms_done(db_session, monkeypatch):
    card = await _make_pending_card(db_session)
    monkeypatch.setattr(
        deploy_runner,
        "fetch_check_runs",
        lambda project, sha: _async_result(
            [CheckRun(name="build", status="completed", conclusion="success")]
        ),
    )

    counts = await run_ci_watcher_once()

    assert counts == {"confirmed": 1, "bounced": 0, "timed_out": 0, "still_pending": 0}
    await db_session.refresh(card)
    assert card.lifecycle_state == LifecycleState.DONE
    assert card.deploying_commit_sha is None
    assert card.deploying_since is None


async def test_a_failing_check_bounces_back_to_developer(db_session, monkeypatch):
    card = await _make_pending_card(db_session)
    monkeypatch.setattr(
        deploy_runner,
        "fetch_check_runs",
        lambda project, sha: _async_result(
            [
                CheckRun(name="build", status="completed", conclusion="success"),
                CheckRun(name="e2e", status="completed", conclusion="failure", html_url="https://x/1"),
            ]
        ),
    )

    counts = await run_ci_watcher_once()

    assert counts == {"confirmed": 0, "bounced": 1, "timed_out": 0, "still_pending": 0}
    await db_session.refresh(card)
    assert card.column == Column.DEVELOPER
    assert card.lifecycle_state == LifecycleState.ACTIVE
    assert card.revision_count == 1
    assert "e2e" in card.latest_feedback
    assert "failure" in card.latest_feedback
    assert card.deploying_commit_sha is None


async def test_bounce_past_max_revisions_blocks_instead(db_session, monkeypatch):
    card = await _make_pending_card(db_session, max_revisions=0)
    monkeypatch.setattr(
        deploy_runner,
        "fetch_check_runs",
        lambda project, sha: _async_result(
            [CheckRun(name="e2e", status="completed", conclusion="failure")]
        ),
    )

    await run_ci_watcher_once()

    await db_session.refresh(card)
    assert card.lifecycle_state == LifecycleState.BLOCKED


async def test_still_running_checks_are_left_alone_within_the_timeout(db_session, monkeypatch):
    card = await _make_pending_card(db_session)
    monkeypatch.setattr(
        deploy_runner,
        "fetch_check_runs",
        lambda project, sha: _async_result([CheckRun(name="build", status="in_progress", conclusion=None)]),
    )

    counts = await run_ci_watcher_once()

    assert counts == {"confirmed": 0, "bounced": 0, "timed_out": 0, "still_pending": 1}
    await db_session.refresh(card)
    assert card.lifecycle_state == LifecycleState.ACTIVE
    assert card.deploying_commit_sha is not None


async def test_still_running_checks_past_the_timeout_block_for_a_human(db_session, monkeypatch):
    long_ago = datetime.now(UTC) - timedelta(seconds=settings.ci_watcher_timeout_seconds + 60)
    card = await _make_pending_card(db_session, deploying_since=long_ago)
    monkeypatch.setattr(
        deploy_runner,
        "fetch_check_runs",
        lambda project, sha: _async_result([CheckRun(name="build", status="in_progress", conclusion=None)]),
    )

    counts = await run_ci_watcher_once()

    assert counts == {"confirmed": 0, "bounced": 0, "timed_out": 1, "still_pending": 0}
    await db_session.refresh(card)
    assert card.lifecycle_state == LifecycleState.BLOCKED
    assert card.deploying_commit_sha is None


async def test_no_checks_reported_yet_waits_out_the_grace_period(db_session, monkeypatch):
    card = await _make_pending_card(db_session)  # deploying_since = now
    monkeypatch.setattr(deploy_runner, "fetch_check_runs", lambda project, sha: _async_result([]))

    counts = await run_ci_watcher_once()

    assert counts == {"confirmed": 0, "bounced": 0, "timed_out": 0, "still_pending": 1}
    await db_session.refresh(card)
    assert card.deploying_commit_sha is not None


async def test_no_checks_reported_past_the_grace_period_confirms_done(db_session, monkeypatch):
    past_grace = datetime.now(UTC) - timedelta(seconds=settings.ci_watcher_grace_period_seconds + 30)
    card = await _make_pending_card(db_session, deploying_since=past_grace)
    monkeypatch.setattr(deploy_runner, "fetch_check_runs", lambda project, sha: _async_result([]))

    counts = await run_ci_watcher_once()

    assert counts == {"confirmed": 1, "bounced": 0, "timed_out": 0, "still_pending": 0}
    await db_session.refresh(card)
    assert card.lifecycle_state == LifecycleState.DONE


async def test_no_ci_possible_confirms_immediately_regardless_of_elapsed_time(db_session, monkeypatch):
    """fetch_check_runs returning None (no token, or not a github.com remote) is
    permanent — there's nothing to ever wait for, unlike an empty check-run list,
    which might just mean CI hasn't registered the push yet."""
    card = await _make_pending_card(db_session)  # deploying_since = now, well within grace
    monkeypatch.setattr(deploy_runner, "fetch_check_runs", lambda project, sha: _async_result(None))

    counts = await run_ci_watcher_once()

    assert counts == {"confirmed": 1, "bounced": 0, "timed_out": 0, "still_pending": 0}
    await db_session.refresh(card)
    assert card.lifecycle_state == LifecycleState.DONE


async def test_transient_fetch_failure_does_not_prematurely_confirm(db_session, monkeypatch):
    card = await _make_pending_card(db_session)

    def _raise(project, sha):
        raise CIStatusUnavailableError("GitHub API returned 502")

    monkeypatch.setattr(deploy_runner, "fetch_check_runs", _raise)

    counts = await run_ci_watcher_once()

    assert counts == {"confirmed": 0, "bounced": 0, "timed_out": 0, "still_pending": 1}
    await db_session.refresh(card)
    assert card.lifecycle_state == LifecycleState.ACTIVE
    assert card.deploying_commit_sha is not None


async def test_ignores_cards_that_are_not_awaiting_ci(db_session):
    project = await project_service.create_project(
        db_session,
        name="ci-watcher-unrelated",
        overarching_goal="goal",
        repo_remote_url="https://example.invalid/repo.git",
    )
    await card_service.create_card(db_session, project.id, title="unrelated", raw_request="r")
    await db_session.commit()

    counts = await run_ci_watcher_once()

    assert counts == {"confirmed": 0, "bounced": 0, "timed_out": 0, "still_pending": 0}


async def _async_result(value):
    return value
