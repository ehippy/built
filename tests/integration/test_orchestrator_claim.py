"""The claim/lease mechanics that make concurrent multi-card execution safe — the
part of the orchestrator that doesn't need a live LLM or Docker to verify."""

from datetime import UTC, datetime, timedelta

from built.domain import transitions
from built.domain.enums import Column, LifecycleState, Priority, VisitOutcome
from built.orchestrator.worker import (
    claim_next_card,
    close_dangling_visits,
    release_claim,
    requeue_stale_claims,
)
from built.services import card_service, project_service


async def _project(db_session, **overrides):
    defaults = {
        "name": f"claim-{overrides.pop('_n', 'x')}",
        "overarching_goal": "goal",
        "repo_remote_url": "https://example.invalid/repo.git",
    }
    defaults.update(overrides)
    return await project_service.create_project(db_session, **defaults)


async def test_claims_an_active_card_in_an_implemented_column(db_session):
    project = await _project(db_session, _n="1")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")

    claimed = await claim_next_card(db_session, "worker-a")

    assert claimed is not None
    assert claimed.id == card.id
    assert claimed.claimed_by_worker_id == "worker-a"
    assert claimed.lease_expires_at is not None


async def test_returns_none_when_nothing_is_claimable(db_session):
    project = await _project(db_session, _n="2")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    card.lifecycle_state = LifecycleState.BLOCKED
    await db_session.commit()

    assert await claim_next_card(db_session, "worker-a") is None


async def test_claims_deployer_column_cards(db_session):
    project = await _project(db_session, _n="3")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    card.column = Column.DEPLOYER
    await db_session.commit()

    claimed = await claim_next_card(db_session, "worker-a")

    assert claimed is not None
    assert claimed.id == card.id


async def test_does_not_claim_a_card_already_held_by_another_worker(db_session):
    project = await _project(db_session, _n="4")
    await card_service.create_card(db_session, project.id, title="t", raw_request="r")

    first = await claim_next_card(db_session, "worker-a")
    assert first is not None

    second = await claim_next_card(db_session, "worker-b")
    assert second is None


async def test_does_not_claim_a_card_waiting_on_its_pr_review(db_session):
    """A pr_to_operator card that opened its PR (pr_number set) stays ACTIVE in
    Deployer but isn't claimable — its Deployer work is done; orchestrator/
    pr_watcher.py owns it until the PR merges or it bounces back to Developer."""
    project = await _project(db_session, _n="pr-pending")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    card.column = Column.DEPLOYER
    card.pr_number = 7
    card.pr_waiting_since = datetime.now(UTC)
    await db_session.commit()

    assert await claim_next_card(db_session, "worker-a") is None

    # A second, unrelated claimable card is still picked up — only the PR-waiting
    # one is excluded.
    await card_service.create_card(db_session, project.id, title="other", raw_request="r")
    await db_session.commit()
    claimed = await claim_next_card(db_session, "worker-a")
    assert claimed is not None
    assert claimed.title == "other"


async def test_claims_a_card_with_an_expired_lease(db_session):
    project = await _project(db_session, _n="5")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    card.claimed_by_worker_id = "worker-dead"
    card.claimed_at = datetime.now(UTC) - timedelta(hours=1)
    card.lease_expires_at = datetime.now(UTC) - timedelta(minutes=1)
    await db_session.commit()

    claimed = await claim_next_card(db_session, "worker-b")

    assert claimed is not None
    assert claimed.claimed_by_worker_id == "worker-b"


async def test_release_claim_clears_claim_fields(db_session):
    project = await _project(db_session, _n="6")
    await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    claimed = await claim_next_card(db_session, "worker-a")
    assert claimed is not None

    await release_claim(db_session, claimed)

    assert claimed.claimed_by_worker_id is None
    assert claimed.claimed_at is None
    assert claimed.lease_expires_at is None


async def test_does_not_claim_a_second_card_from_a_project_already_claimed(db_session):
    project = await _project(db_session, _n="8")
    await card_service.create_card(db_session, project.id, title="first", raw_request="r")
    await card_service.create_card(db_session, project.id, title="second", raw_request="r")

    first = await claim_next_card(db_session, "worker-a")
    assert first is not None

    second = await claim_next_card(db_session, "worker-b")
    assert second is None


async def test_claims_cards_from_different_projects_concurrently(db_session):
    project_a = await _project(db_session, _n="9a")
    project_b = await _project(db_session, _n="9b")
    await card_service.create_card(db_session, project_a.id, title="a", raw_request="r")
    await card_service.create_card(db_session, project_b.id, title="b", raw_request="r")

    first = await claim_next_card(db_session, "worker-a")
    second = await claim_next_card(db_session, "worker-b")

    assert first is not None
    assert second is not None
    assert first.project_id != second.project_id


async def test_frees_up_the_project_once_the_held_card_is_released(db_session):
    project = await _project(db_session, _n="10")
    await card_service.create_card(db_session, project.id, title="first", raw_request="r")
    await card_service.create_card(db_session, project.id, title="second", raw_request="r")

    first = await claim_next_card(db_session, "worker-a")
    assert first is not None
    await release_claim(db_session, first)

    second = await claim_next_card(db_session, "worker-b")
    assert second is not None


async def test_does_not_claim_cards_from_a_paused_project(db_session):
    project = await _project(db_session, _n="11")
    await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await project_service.pause_project(db_session, project.id)

    assert await claim_next_card(db_session, "worker-a") is None


async def test_resuming_a_project_makes_its_cards_claimable_again(db_session):
    project = await _project(db_session, _n="12")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await project_service.pause_project(db_session, project.id)
    await project_service.resume_project(db_session, project.id)

    claimed = await claim_next_card(db_session, "worker-a")

    assert claimed is not None
    assert claimed.id == card.id


async def test_pausing_one_project_does_not_block_claims_in_another(db_session):
    paused_project = await _project(db_session, _n="13a")
    active_project = await _project(db_session, _n="13b")
    await card_service.create_card(db_session, paused_project.id, title="p", raw_request="r")
    active_card = await card_service.create_card(db_session, active_project.id, title="a", raw_request="r")
    await project_service.pause_project(db_session, paused_project.id)

    claimed = await claim_next_card(db_session, "worker-a")

    assert claimed is not None
    assert claimed.id == active_card.id


async def test_claims_cards_closer_to_done_first(db_session):
    """'Stop starting, start finishing': with several projects each ready to claim,
    a card sitting in Deployer should be picked before a brand-new PM card, even
    though the PM card is older — draining in-flight work takes priority over
    starting fresh work."""
    pm_project = await _project(db_session, _n="14a")
    deployer_project = await _project(db_session, _n="14b")
    pm_card = await card_service.create_card(db_session, pm_project.id, title="pm", raw_request="r")
    deployer_card = await card_service.create_card(
        db_session, deployer_project.id, title="deployer", raw_request="r"
    )
    deployer_card.column = Column.DEPLOYER
    await db_session.commit()
    assert pm_card.updated_at <= deployer_card.updated_at  # pm card is not younger

    claimed = await claim_next_card(db_session, "worker-a")

    assert claimed is not None
    assert claimed.id == deployer_card.id


async def test_high_priority_card_is_claimed_before_a_card_closer_to_done(db_session):
    """Priority is the first sort key, ahead of "stop starting, start finishing":
    a human blessing a brand-new PM card as high priority should get it claimed
    before an ordinary-priority card already sitting in Deployer, even though
    column-depth would otherwise put the Deployer card first."""
    pm_project = await _project(db_session, _n="17a")
    deployer_project = await _project(db_session, _n="17b")
    high_priority_pm_card = await card_service.create_card(
        db_session, pm_project.id, title="urgent", raw_request="r", priority=Priority.HIGH
    )
    deployer_card = await card_service.create_card(
        db_session, deployer_project.id, title="deployer", raw_request="r"
    )
    deployer_card.column = Column.DEPLOYER
    await db_session.commit()

    claimed = await claim_next_card(db_session, "worker-a")

    assert claimed is not None
    assert claimed.id == high_priority_pm_card.id


async def test_low_priority_card_is_claimed_after_a_normal_priority_card_further_from_done(db_session):
    """Same check from the other direction: a low-priority card fresh in PM must
    lose to a normal-priority card, even though normal is itself further from
    done than PM in this pairing would otherwise suggest — priority alone decides
    it here since column-depth actually favors the low-priority card's column."""
    low_project = await _project(db_session, _n="18a")
    normal_project = await _project(db_session, _n="18b")
    low_priority_card = await card_service.create_card(
        db_session, low_project.id, title="low", raw_request="r", priority=Priority.LOW
    )
    low_priority_card.column = Column.DEPLOYER
    normal_card = await card_service.create_card(
        db_session, normal_project.id, title="normal", raw_request="r"
    )
    await db_session.commit()

    claimed = await claim_next_card(db_session, "worker-a")

    assert claimed is not None
    assert claimed.id == normal_card.id


async def test_does_not_claim_an_archived_card(db_session):
    project = await _project(db_session, _n="15")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await card_service.archive_card(db_session, card.id)

    assert await claim_next_card(db_session, "worker-a") is None


async def test_unarchiving_a_card_makes_it_claimable_again(db_session):
    project = await _project(db_session, _n="16")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await card_service.archive_card(db_session, card.id)
    await card_service.unarchive_card(db_session, card.id)

    claimed = await claim_next_card(db_session, "worker-a")

    assert claimed is not None
    assert claimed.id == card.id


async def test_requeue_stale_claims_frees_every_claim_regardless_of_lease(db_session):
    """This only ever runs once, at process startup, before this process's own
    workers have claimed anything — so a claim whose lease hasn't technically
    expired yet is still just as stale as one that has: no worker in *this*
    process could have created it. Waiting out the rest of an unexpired lease
    would just leave a perfectly good card idle for no reason."""
    project = await _project(db_session, _n="7")
    expired = await card_service.create_card(db_session, project.id, title="expired", raw_request="r")
    unexpired = await card_service.create_card(db_session, project.id, title="unexpired", raw_request="r")

    expired.claimed_by_worker_id = "worker-dead-1"
    expired.lease_expires_at = datetime.now(UTC) - timedelta(minutes=1)
    unexpired.claimed_by_worker_id = "worker-dead-2"
    unexpired.lease_expires_at = datetime.now(UTC) + timedelta(minutes=10)
    await db_session.commit()

    requeued_count = await requeue_stale_claims(db_session)

    assert requeued_count == 2
    assert expired.claimed_by_worker_id is None
    assert unexpired.claimed_by_worker_id is None


async def test_close_dangling_visits_marks_them_interrupted_not_resumed(db_session):
    """A worker process died mid-visit last time around — the visit row was left
    with no ended_at, which would otherwise show as "in progress" forever on the
    card detail page even though nothing is actually working on it anymore."""
    project = await _project(db_session, _n="10a")
    dangling_card = await card_service.create_card(db_session, project.id, title="d", raw_request="r")
    dangling_visit = await transitions.start_visit(db_session, dangling_card)
    closed_card = await card_service.create_card(db_session, project.id, title="c", raw_request="r")
    closed_visit = await transitions.start_visit(db_session, closed_card)
    await transitions.complete_pm_visit(
        db_session, closed_card, closed_visit, spec="s", acceptance_criteria=["x"], summary="s"
    )
    await db_session.commit()
    assert dangling_visit.ended_at is None

    closed_count = await close_dangling_visits(db_session)

    assert closed_count == 1
    assert dangling_visit.ended_at is not None
    assert dangling_visit.outcome == VisitOutcome.INTERRUPTED
    # The card itself stays ACTIVE — freed to be claimed again and start a fresh
    # attempt, not resumed mid-tool-call.
    assert dangling_card.lifecycle_state == LifecycleState.ACTIVE
    # A visit that already closed normally is left completely alone.
    assert closed_visit.outcome == VisitOutcome.SUBMITTED


async def test_does_not_claim_a_card_with_an_unresolved_dependency(db_session):
    project = await _project(db_session, _n="19")
    prereq = await card_service.create_card(db_session, project.id, title="prereq", raw_request="r")
    dependent = await card_service.create_card(db_session, project.id, title="dependent", raw_request="r")
    await card_service.add_dependency(db_session, card_id=dependent.id, depends_on_card_id=prereq.id)

    claimed = await claim_next_card(db_session, "worker-a")
    assert claimed is not None
    assert claimed.id == prereq.id  # only the prerequisite is claimable

    await release_claim(db_session, claimed)
    visit = await transitions.start_visit(db_session, prereq)
    await transitions.complete_deployer_visit(db_session, prereq, visit, success=True, summary="s")
    await db_session.commit()

    claimed_again = await claim_next_card(db_session, "worker-a")
    assert claimed_again is not None
    assert claimed_again.id == dependent.id  # unblocked now that the prerequisite is DONE


async def test_an_archived_unresolved_dependency_does_not_block_forever(db_session):
    project = await _project(db_session, _n="20")
    prereq = await card_service.create_card(db_session, project.id, title="prereq", raw_request="r")
    dependent = await card_service.create_card(db_session, project.id, title="dependent", raw_request="r")
    await card_service.add_dependency(db_session, card_id=dependent.id, depends_on_card_id=prereq.id)
    await card_service.archive_card(db_session, prereq.id)

    claimed = await claim_next_card(db_session, "worker-a")

    assert claimed is not None
    assert claimed.id == dependent.id


async def test_does_not_claim_an_epic_parent_card(db_session):
    project = await _project(db_session, _n="21")
    epic = await card_service.create_card(db_session, project.id, title="epic", raw_request="r")
    child_a = await card_service.create_card(db_session, project.id, title="a", raw_request="r")
    child_b = await card_service.create_card(db_session, project.id, title="b", raw_request="r")
    await card_service.link_epic_child(db_session, parent_card_id=epic.id, child_card_id=child_a.id)
    await card_service.link_epic_child(db_session, parent_card_id=epic.id, child_card_id=child_b.id)
    await db_session.commit()
    assert epic.lifecycle_state == LifecycleState.ACTIVE
    assert epic.column == Column.PM

    seen_ids = set()
    for _ in range(5):
        claimed = await claim_next_card(db_session, "worker-a")
        if claimed is None:
            break
        seen_ids.add(claimed.id)
        await release_claim(db_session, claimed)
        await db_session.commit()

    assert epic.id not in seen_ids
    assert seen_ids == {child_a.id, child_b.id}
