"""orchestrator/pr_watcher.py — polls GitHub for cards whose pr_to_operator deploy
opened a PR but hasn't had it reviewed/merged yet (Card.pr_number). fetch_pr_status
and merge_pull_request are monkeypatched throughout — they're thin GitHub API
wrappers with their own URL-parsing tests in tests/integration/test_deploy_runner.py;
what matters here is the watcher's own decision logic given whatever they return."""

from datetime import UTC, datetime, timedelta

from built.agent import summarizer
from built.config import settings
from built.domain.enums import Column, DeployKind, DeployMode, LifecycleState
from built.orchestrator.pr_watcher import run_pr_watcher_once
from built.sandbox import deploy_runner
from built.sandbox.deploy_runner import PrStatusUnavailableError, PullRequestStatus
from built.services import card_service, project_service


async def _make_pending_pr_card(session, *, pr_waiting_since=None, pr_number=7):
    project = await project_service.create_project(
        session,
        name=f"pr-watcher-{id(session)}-{pr_number}",
        overarching_goal="goal",
        repo_remote_url="https://github.com/octocat/hello-world.git",
    )
    await project_service.set_deploy_config(
        session,
        project.id,
        kind=DeployKind.NONE,
        mode=DeployMode.PR_TO_OPERATOR,
        github_token_ref="TEST_GH_TOKEN",
    )
    card = await card_service.create_card(session, project.id, title="Ship it", raw_request="r")
    card.column = Column.DEPLOYER
    card.deploy_url = f"https://github.com/octocat/hello-world/pull/{pr_number}"
    card.pr_number = pr_number
    card.pr_waiting_since = pr_waiting_since or datetime.now(UTC)
    await session.commit()
    return card


def _status(*, merged=False, state="open", review_decision=None, feedback=None):
    return PullRequestStatus(
        merged=merged, state=state, review_decision=review_decision, feedback=feedback
    )


async def _async_result(value):
    return value


async def test_merged_pr_confirms_done(db_session, monkeypatch):
    card = await _make_pending_pr_card(db_session)
    monkeypatch.setattr(
        deploy_runner,
        "fetch_pr_status",
        lambda project, card: _async_result(_status(merged=True, state="closed")),
    )

    counts = await run_pr_watcher_once()

    assert counts == {
        "merged": 1,
        "changes_requested": 0,
        "closed_unmerged": 0,
        "merge_conflicted": 0,
        "timed_out": 0,
        "still_pending": 0,
    }
    await db_session.refresh(card)
    assert card.lifecycle_state == LifecycleState.DONE
    assert card.pr_number is None
    assert card.pr_waiting_since is None


async def test_merged_pr_writes_a_postmortem(db_session, monkeypatch):
    """DONE is the pr_watcher's only terminal (card-closing) outcome, so the
    postmortem hook — the same one ci_watcher has for auto_main — must fire
    exactly there, and only there."""
    await _make_pending_pr_card(db_session)
    monkeypatch.setattr(
        deploy_runner,
        "fetch_pr_status",
        lambda project, card: _async_result(_status(merged=True, state="closed")),
    )

    calls = []

    async def _fake_postmortem(session, project, card, *, outcome):
        calls.append(outcome)
        return None

    monkeypatch.setattr(summarizer, "write_card_postmortem", _fake_postmortem)

    counts = await run_pr_watcher_once()

    assert counts["merged"] == 1
    assert calls == [LifecycleState.DONE]


async def test_approved_pr_is_merged_and_confirms_done(db_session, monkeypatch):
    card = await _make_pending_pr_card(db_session)
    monkeypatch.setattr(
        deploy_runner,
        "fetch_pr_status",
        lambda project, card: _async_result(_status(review_decision="approved")),
    )
    merged = []

    async def _merge(project, card):
        merged.append(card.pr_number)
        return deploy_runner.DeployRunResult(success=True, message="PR #7 merged")

    monkeypatch.setattr(deploy_runner, "merge_pull_request", _merge)

    counts = await run_pr_watcher_once()

    assert counts["merged"] == 1
    assert merged == [7]
    await db_session.refresh(card)
    assert card.lifecycle_state == LifecycleState.DONE
    assert card.pr_number is None


async def test_approved_but_not_mergeable_blocks_for_a_human(db_session, monkeypatch):
    """default_branch advanced into a real conflict — an autonomous rebase of shared
    history is out of scope, so the card blocks for a human rather than looping."""
    card = await _make_pending_pr_card(db_session)
    monkeypatch.setattr(
        deploy_runner,
        "fetch_pr_status",
        lambda project, card: _async_result(_status(review_decision="approved")),
    )
    monkeypatch.setattr(
        deploy_runner,
        "merge_pull_request",
        lambda project, card: _async_result(
            deploy_runner.DeployRunResult(
                success=False, message="PR #7 is not mergeable (conflicts with main): ..."
            )
        ),
    )

    counts = await run_pr_watcher_once()

    assert counts == {
        "merged": 0,
        "changes_requested": 0,
        "closed_unmerged": 0,
        "merge_conflicted": 1,
        "timed_out": 0,
        "still_pending": 0,
    }
    await db_session.refresh(card)
    assert card.lifecycle_state == LifecycleState.BLOCKED
    assert card.pr_number is None


async def test_transient_merge_failure_is_not_treated_as_a_verdict(db_session, monkeypatch):
    card = await _make_pending_pr_card(db_session)
    monkeypatch.setattr(
        deploy_runner,
        "fetch_pr_status",
        lambda project, card: _async_result(_status(review_decision="approved")),
    )
    monkeypatch.setattr(
        deploy_runner,
        "merge_pull_request",
        lambda project, card: _async_result(
            deploy_runner.DeployRunResult(success=False, message="GitHub API returned 502: ...")
        ),
    )

    counts = await run_pr_watcher_once()

    assert counts["still_pending"] == 1
    await db_session.refresh(card)
    assert card.lifecycle_state == LifecycleState.ACTIVE
    assert card.pr_number == 7


async def test_changes_requested_bounces_back_to_developer_with_feedback(db_session, monkeypatch):
    card = await _make_pending_pr_card(db_session)
    monkeypatch.setattr(
        deploy_runner,
        "fetch_pr_status",
        lambda project, card: _async_result(
            _status(review_decision="changes_requested", feedback="rename it")
        ),
    )

    counts = await run_pr_watcher_once()

    assert counts["changes_requested"] == 1
    await db_session.refresh(card)
    assert card.lifecycle_state == LifecycleState.ACTIVE
    assert card.column == Column.DEVELOPER
    assert card.pr_number is None
    assert card.pr_waiting_since is None
    assert card.latest_feedback == "rename it"
    assert card.revision_count == 1


async def test_unreviewed_pr_is_left_alone_within_the_timeout(db_session, monkeypatch):
    card = await _make_pending_pr_card(db_session)
    monkeypatch.setattr(
        deploy_runner,
        "fetch_pr_status",
        lambda project, card: _async_result(_status()),
    )

    counts = await run_pr_watcher_once()

    assert counts == {
        "merged": 0,
        "changes_requested": 0,
        "closed_unmerged": 0,
        "merge_conflicted": 0,
        "timed_out": 0,
        "still_pending": 1,
    }
    await db_session.refresh(card)
    assert card.lifecycle_state == LifecycleState.ACTIVE
    assert card.pr_number is not None


async def test_unreviewed_pr_past_the_timeout_blocks_for_a_human(db_session, monkeypatch):
    long_ago = datetime.now(UTC) - timedelta(seconds=settings.pr_watcher_timeout_seconds + 60)
    card = await _make_pending_pr_card(db_session, pr_waiting_since=long_ago)
    monkeypatch.setattr(
        deploy_runner,
        "fetch_pr_status",
        lambda project, card: _async_result(_status()),
    )

    counts = await run_pr_watcher_once()

    assert counts["timed_out"] == 1
    await db_session.refresh(card)
    assert card.lifecycle_state == LifecycleState.BLOCKED
    assert card.pr_number is None


async def test_closed_unmerged_pr_blocks_for_a_human(db_session, monkeypatch):
    card = await _make_pending_pr_card(db_session)
    monkeypatch.setattr(
        deploy_runner,
        "fetch_pr_status",
        lambda project, card: _async_result(_status(state="closed")),
    )

    counts = await run_pr_watcher_once()

    assert counts["closed_unmerged"] == 1
    await db_session.refresh(card)
    assert card.lifecycle_state == LifecycleState.BLOCKED
    assert card.pr_number is None


async def test_transient_fetch_failure_does_not_prematurely_resolve(db_session, monkeypatch):
    card = await _make_pending_pr_card(db_session)

    def _raise(project, card):
        raise PrStatusUnavailableError("GitHub API returned 502")

    monkeypatch.setattr(deploy_runner, "fetch_pr_status", _raise)

    counts = await run_pr_watcher_once()

    assert counts["still_pending"] == 1
    await db_session.refresh(card)
    assert card.lifecycle_state == LifecycleState.ACTIVE
    assert card.pr_number is not None


async def test_ignores_cards_that_are_not_awaiting_a_pr(db_session):
    project = await project_service.create_project(
        db_session,
        name="pr-watcher-unrelated",
        overarching_goal="goal",
        repo_remote_url="https://example.invalid/repo.git",
    )
    await card_service.create_card(db_session, project.id, title="unrelated", raw_request="r")
    await db_session.commit()

    counts = await run_pr_watcher_once()

    assert counts == {
        "merged": 0,
        "changes_requested": 0,
        "closed_unmerged": 0,
        "merge_conflicted": 0,
        "timed_out": 0,
        "still_pending": 0,
    }
