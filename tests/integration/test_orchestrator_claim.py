"""The claim/lease mechanics that make concurrent multi-card execution safe — the
part of the orchestrator that doesn't need a live LLM or Docker to verify."""

from datetime import UTC, datetime, timedelta

from built.domain.enums import Column, LifecycleState
from built.orchestrator.worker import claim_next_card, release_claim, requeue_stale_claims
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


async def test_requeue_stale_claims_frees_expired_but_not_fresh_claims(db_session):
    project = await _project(db_session, _n="7")
    stale = await card_service.create_card(db_session, project.id, title="stale", raw_request="r")
    fresh = await card_service.create_card(db_session, project.id, title="fresh", raw_request="r")

    stale.claimed_by_worker_id = "worker-dead"
    stale.lease_expires_at = datetime.now(UTC) - timedelta(minutes=1)
    fresh.claimed_by_worker_id = "worker-alive"
    fresh.lease_expires_at = datetime.now(UTC) + timedelta(minutes=10)
    await db_session.commit()

    requeued_count = await requeue_stale_claims(db_session)

    assert requeued_count == 1
    assert stale.claimed_by_worker_id is None
    assert fresh.claimed_by_worker_id == "worker-alive"
