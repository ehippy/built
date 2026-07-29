"""get_project_activity_summary: the Projects list page's per-project stats, live
indicator, and latest-visit commentary line."""

from datetime import UTC, datetime, timedelta

from built.domain import transitions
from built.domain.enums import LifecycleState
from built.services import card_service, project_service


async def _make_project(session, **overrides):
    defaults = {
        "name": f"activity-{overrides.pop('_n', 'x')}",
        "overarching_goal": "goal",
        "repo_remote_url": "https://example.invalid/repo.git",
    }
    defaults.update(overrides)
    return await project_service.create_project(session, **defaults)


async def test_empty_project_has_zeroed_summary(db_session):
    project = await _make_project(db_session, _n="1")

    summary = await card_service.get_project_activity_summary(db_session, project.id)

    assert summary["total"] == 0
    assert summary["is_being_worked"] is False
    assert summary["latest"] is None
    assert all(count == 0 for count in summary["counts"].values())


async def test_counts_reflect_lifecycle_state_mix(db_session):
    project = await _make_project(db_session, _n="2")
    active_card = await card_service.create_card(db_session, project.id, title="a", raw_request="r")
    blocked_card = await card_service.create_card(db_session, project.id, title="b", raw_request="r")
    blocked_card.lifecycle_state = LifecycleState.BLOCKED
    done_card = await card_service.create_card(db_session, project.id, title="c", raw_request="r")
    done_card.lifecycle_state = LifecycleState.DONE
    await db_session.commit()

    summary = await card_service.get_project_activity_summary(db_session, project.id)

    assert summary["total"] == 3
    assert summary["counts"]["active"] == 1
    assert summary["counts"]["blocked"] == 1
    assert summary["counts"]["done"] == 1
    assert summary["counts"]["failed"] == 0
    assert active_card.id  # sanity: fixture actually created


async def test_is_being_worked_reflects_a_fresh_claim(db_session):
    project = await _make_project(db_session, _n="3")
    card = await card_service.create_card(db_session, project.id, title="a", raw_request="r")
    card.claimed_by_worker_id = "worker-a"
    card.lease_expires_at = datetime.now(UTC) + timedelta(minutes=5)
    await db_session.commit()

    summary = await card_service.get_project_activity_summary(db_session, project.id)

    assert summary["is_being_worked"] is True


async def test_latest_reflects_the_most_recently_closed_visit(db_session):
    project = await _make_project(db_session, _n="4")
    card = await card_service.create_card(db_session, project.id, title="the card", raw_request="r")

    older_visit = await transitions.start_visit(db_session, card)
    await transitions.complete_pm_visit(
        db_session, card, older_visit, spec="s", acceptance_criteria=["x"], summary="older summary"
    )
    newer_visit = await transitions.start_visit(db_session, card)
    await transitions.complete_developer_visit(db_session, card, newer_visit, summary="newer summary")
    await db_session.commit()

    summary = await card_service.get_project_activity_summary(db_session, project.id)

    assert summary["latest"] is not None
    assert summary["latest"]["summary"] == "newer summary"
    assert summary["latest"]["card_title"] == "the card"
    assert summary["latest"]["column"] == "developer"
