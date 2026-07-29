"""Task-lifecycle log lines on card_service's create/retry/cancel — the unified
"built" logger's coverage of "a new task showed up" / "a human retried or
cancelled one", alongside worker.py's claim/finish coverage of "a task got picked
up" / "a task's stage finished". See logging_config.py."""

import logging

from built.domain import transitions
from built.logging_config import get_logs
from built.services import card_service, project_service


async def _make_project(session, name):
    return await project_service.create_project(
        session, name=name, overarching_goal="goal", repo_remote_url="https://example.invalid/repo.git"
    )


def _cutoff() -> int:
    """The current most-recent seq in the shared, session-global ring buffer, so
    each test can look at only what it itself logs — get_logs()'s default limit
    caps the returned window, so a plain len()-before/after comparison silently
    breaks once other tests (e.g. the ring-buffer-bounding test) have pushed the
    total past that cap."""
    logs = get_logs()
    return logs[-1].seq if logs else 0


async def test_create_card_logs_the_new_task(db_session):
    project = await _make_project(db_session, "lifecycle-create")
    cutoff = _cutoff()

    card = await card_service.create_card(db_session, project.id, title="Add a widget", raw_request="r")

    new_logs = get_logs(since_seq=cutoff)
    assert len(new_logs) == 1
    assert new_logs[0].level == "INFO"
    assert card.id in new_logs[0].message
    assert "Add a widget" in new_logs[0].message


async def test_retry_card_logs_the_retry_and_any_note(db_session):
    project = await _make_project(db_session, "lifecycle-retry")
    card = await card_service.create_card(db_session, project.id, title="Flaky thing", raw_request="r")
    visit = await transitions.start_visit(db_session, card)
    await transitions.fail_visit_with_error(db_session, card, visit, message="boom")
    await db_session.commit()

    cutoff = _cutoff()
    await card_service.retry_card(db_session, card.id, note="rebase onto main first")

    new_logs = get_logs(since_seq=cutoff)
    assert len(new_logs) == 1
    assert card.id in new_logs[0].message
    assert "rebase onto main first" in new_logs[0].message


async def test_cancel_card_logs_the_cancellation(db_session):
    project = await _make_project(db_session, "lifecycle-cancel")
    card = await card_service.create_card(db_session, project.id, title="Not needed anymore", raw_request="r")

    cutoff = _cutoff()
    await card_service.cancel_card(db_session, card.id)

    new_logs = get_logs(since_seq=cutoff)
    assert len(new_logs) == 1
    assert card.id in new_logs[0].message
    assert "Not needed anymore" in new_logs[0].message


def test_card_service_logger_is_under_the_built_namespace():
    """Confirms card_service's log lines actually propagate into the unified
    ring buffer (see logging_config.configure_logging) rather than silently going
    nowhere — logging.getLogger(__name__) must resolve to "built.services.card_service"."""
    assert logging.getLogger(card_service.__name__).name == "built.services.card_service"
