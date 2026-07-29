"""Card.is_being_worked — the "an agent is actively working on this" signal the
dashboard spinner is driven by. Deliberately distinct from lifecycle_state == ACTIVE,
which also covers a card sitting idle in the queue waiting for a free worker."""

from datetime import UTC, datetime, timedelta

from built.db.models import Card


def _card(**overrides) -> Card:
    return Card(project_id="p", title="t", raw_request="r", **overrides)


def test_unclaimed_card_is_not_being_worked():
    assert _card().is_being_worked is False


def test_claimed_card_with_a_fresh_lease_is_being_worked():
    card = _card(
        claimed_by_worker_id="worker-a",
        lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    assert card.is_being_worked is True


def test_claimed_card_with_an_expired_lease_is_not_being_worked():
    card = _card(
        claimed_by_worker_id="worker-a",
        lease_expires_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    assert card.is_being_worked is False


def test_naive_lease_expires_at_is_treated_as_utc():
    """SQLite doesn't reliably round-trip tzinfo — a Card read back from the DB can
    have a naive lease_expires_at even though it was written as UTC."""
    card = _card(
        claimed_by_worker_id="worker-a",
        lease_expires_at=datetime.now(UTC).replace(tzinfo=None) + timedelta(minutes=5),
    )
    assert card.is_being_worked is True
