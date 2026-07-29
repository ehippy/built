"""The claim/lease mechanics that make concurrent multi-card execution safe — the
part of the orchestrator that doesn't need a live LLM or Docker to verify."""

from datetime import UTC, datetime, timedelta

from built.domain.enums import Column, LifecycleState
from built.orchestrator import worker
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


async def test_discovery_skips_outright_if_already_marked_in_progress(db_session):
    """Discovery doesn't go through claim_next_card, so it isn't covered by
    per-project claim serialization — a separate in-memory guard prevents two runs
    for the same project racing each other and proposing near-duplicate cards."""
    project = await _project(db_session, _n="8")

    worker._discovery_in_progress.add(project.id)
    try:
        assert worker.is_discovery_running(project.id) is True
        await worker.run_project_discovery(project.id)  # should no-op immediately
    finally:
        worker._discovery_in_progress.discard(project.id)

    all_cards = await card_service.list_cards(db_session, project.id)
    assert all_cards == []


async def test_discovery_releases_the_guard_even_on_setup_failure(db_session):
    project = await _project(db_session, _n="9", repo_remote_url="/nonexistent/path/repo.git")

    await worker.run_project_discovery(project.id)

    assert worker.is_discovery_running(project.id) is False
