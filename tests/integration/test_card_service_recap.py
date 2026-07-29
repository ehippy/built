"""get_previous_attempt_recap: the context a retried or bounced-back visit gets about
its own prior attempt at the same column, so it doesn't start completely cold."""

from built.db.models import Project
from built.domain import transitions
from built.domain.enums import Column, EventType
from built.domain.events import append_event
from built.services import card_service, project_service


async def _make_project(session, **overrides) -> Project:
    defaults = {
        "name": f"recap-{overrides.get('name', 'x')}",
        "overarching_goal": "Ship a thing.",
        "repo_remote_url": "https://example.invalid/repo.git",
    }
    defaults.update(overrides)
    return await project_service.create_project(session, **defaults)


async def test_no_recap_for_a_first_attempt(db_session):
    project = await _make_project(db_session, name="first")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    visit = await transitions.start_visit(db_session, card)

    recap = await card_service.get_previous_attempt_recap(
        db_session, card.id, card.column, before_attempt=visit.attempt_number
    )

    assert recap is None


async def test_recap_includes_prior_outcome_and_recent_tool_calls(db_session):
    project = await _make_project(db_session, name="second")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")

    first_visit = await transitions.start_visit(db_session, card)
    await append_event(
        db_session,
        card_id=card.id,
        column_visit_id=first_visit.id,
        type=EventType.TOOL_CALL,
        payload={
            "name": "bash",
            "arguments": {"command": "npm install"},
            "result": "added 400 packages",
            "is_error": False,
        },
    )
    await append_event(
        db_session,
        card_id=card.id,
        column_visit_id=first_visit.id,
        type=EventType.TOOL_CALL,
        payload={
            "name": "bash",
            "arguments": {"command": "npm test"},
            "result": "108 passed",
            "is_error": False,
        },
    )
    await transitions.fail_visit_with_error(
        db_session, card, first_visit, message="exceeded max_iterations_per_run (25)"
    )
    await db_session.commit()

    await transitions.retry_card(db_session, card)
    second_visit = await transitions.start_visit(db_session, card)
    await db_session.commit()

    recap = await card_service.get_previous_attempt_recap(
        db_session, card.id, card.column, before_attempt=second_visit.attempt_number
    )

    assert recap is not None
    assert "Attempt #1 ended: error" in recap
    assert "exceeded max_iterations_per_run (25)" in recap
    assert "npm install" in recap
    assert "npm test" in recap
    assert "108 passed" in recap


async def test_recap_only_looks_at_the_same_column(db_session):
    project = await _make_project(db_session, name="cross-column")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")

    pm_visit = await transitions.start_visit(db_session, card)
    await transitions.complete_pm_visit(
        db_session, card, pm_visit, spec="spec", acceptance_criteria=["x"], summary="PM done"
    )
    assert card.column == Column.DEVELOPER

    dev_visit = await transitions.start_visit(db_session, card)

    recap = await card_service.get_previous_attempt_recap(
        db_session, card.id, card.column, before_attempt=dev_visit.attempt_number
    )

    # Developer's first attempt has no prior Developer attempt — PM's visit doesn't count.
    assert recap is None
