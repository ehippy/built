"""Drives built.domain.transitions directly (no HTTP, no agents) — this is the
Phase 2 verification called for in the plan: a card pushed through all five columns,
a revision loop that trips the cap, a deploy that exhausts its retries, and a run
error, asserting CardColumnVisit/CardEvent bookkeeping at each step."""

from built.db.models import Project
from built.domain import transitions
from built.domain.enums import Column, LifecycleState, VisitOutcome
from built.services import card_service, project_service


async def _make_project(session, **overrides) -> Project:
    defaults = {
        "name": f"proj-{id(overrides)}-{overrides.get('name', 'x')}",
        "overarching_goal": "Ship a thing.",
        "repo_remote_url": "https://example.invalid/repo.git",
    }
    defaults.update(overrides)
    return await project_service.create_project(session, **defaults)


async def test_full_pipeline_to_done(db_session):
    project = await _make_project(db_session, name="full-pipeline")
    card = await card_service.create_card(
        db_session, project.id, title="Add feature", raw_request="please add the feature"
    )
    assert card.column == Column.PM
    assert card.lifecycle_state == LifecycleState.ACTIVE

    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_pm_visit(
        db_session, card, visit, spec="Implement X", acceptance_criteria=["X works"], summary="Spec written"
    )
    assert card.column == Column.DEVELOPER
    assert card.spec == "Implement X"

    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_developer_visit(db_session, card, visit, summary="Implemented X")
    assert card.column == Column.TESTER

    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_tester_visit_approved(db_session, card, visit, summary="Tests pass")
    assert card.column == Column.REVIEWER

    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_reviewer_visit_approved(db_session, card, visit, summary="LGTM")
    assert card.column == Column.DEPLOYER

    # Deployer fails once (retryable — under the project's max_deploy_attempts=2)...
    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_deployer_visit(
        db_session, card, visit, success=False, summary="Deploy failed: timeout"
    )
    assert card.lifecycle_state == LifecycleState.ACTIVE
    assert card.deploy_attempt_count == 1

    # ...then succeeds on retry.
    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_deployer_visit(db_session, card, visit, success=True, summary="Deployed")
    assert card.lifecycle_state == LifecycleState.DONE

    visits = await card_service.list_column_visits(db_session, card.id)
    assert [v.column for v in visits] == [
        Column.PM,
        Column.DEVELOPER,
        Column.TESTER,
        Column.REVIEWER,
        Column.DEPLOYER,
        Column.DEPLOYER,
    ]
    assert [v.attempt_number for v in visits] == [1, 1, 1, 1, 1, 2]
    assert visits[-2].outcome == VisitOutcome.FAILED
    assert visits[-1].outcome == VisitOutcome.DONE

    events = await card_service.list_events(db_session, card.id)
    assert any(e.payload.get("action") == "created" for e in events)
    assert sum(1 for e in events if e.type.value == "transition") == 6


async def test_revision_loop_blocks_after_cap_then_retry_unsticks_it(db_session):
    project = await _make_project(db_session, name="revision-loop", max_revisions=1)
    card = await card_service.create_card(db_session, project.id, title="Flaky feature", raw_request="do it")

    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_pm_visit(
        db_session, card, visit, spec="spec", acceptance_criteria=["a"], summary="ok"
    )
    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_developer_visit(db_session, card, visit, summary="ok")
    assert card.column == Column.TESTER

    # First rejection: within the cap (max_revisions=1) — bounces back, stays active.
    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_tester_visit_changes_requested(
        db_session, card, visit, feedback="fix bug 1", summary="failed test A"
    )
    assert card.column == Column.DEVELOPER
    assert card.lifecycle_state == LifecycleState.ACTIVE
    assert card.revision_count == 1

    visit = await transitions.start_visit(db_session, card)
    assert visit.attempt_number == 2  # second Developer visit for this card
    await transitions.complete_developer_visit(db_session, card, visit, summary="ok again")

    # Second rejection: exceeds max_revisions=1 — blocks the card for a human.
    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_tester_visit_changes_requested(
        db_session, card, visit, feedback="fix bug 2", summary="failed test A again"
    )
    assert card.lifecycle_state == LifecycleState.BLOCKED
    assert card.revision_count == 2
    assert card.latest_feedback == "fix bug 2"

    # The one human touchpoint outside the pipeline: retry gives it a clean budget.
    await card_service.retry_card(db_session, card.id)
    assert card.lifecycle_state == LifecycleState.ACTIVE
    assert card.revision_count == 0
    assert card.latest_feedback is None


async def test_reviewer_request_changes_bounces_to_developer_and_shares_revision_cap(db_session):
    project = await _make_project(db_session, name="reviewer-loop", max_revisions=1)
    card = await card_service.create_card(db_session, project.id, title="Risky change", raw_request="do it")

    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_pm_visit(
        db_session, card, visit, spec="spec", acceptance_criteria=["a"], summary="ok"
    )
    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_developer_visit(db_session, card, visit, summary="ok")
    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_tester_visit_approved(db_session, card, visit, summary="tests pass")
    assert card.column == Column.REVIEWER

    # Reviewer finds a real problem tests didn't catch — bounces to Developer,
    # consuming the same revision_count budget as Tester's request_changes.
    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_reviewer_visit_changes_requested(
        db_session, card, visit, feedback="SQL built via string formatting", summary="security issue"
    )
    assert card.column == Column.DEVELOPER
    assert card.lifecycle_state == LifecycleState.ACTIVE
    assert card.revision_count == 1
    assert card.latest_feedback == "SQL built via string formatting"

    # Developer fixes it, Tester re-approves, Reviewer rejects again — now over
    # max_revisions=1, so it blocks for a human exactly like the Tester<->Developer
    # loop does.
    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_developer_visit(db_session, card, visit, summary="fixed")
    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_tester_visit_approved(db_session, card, visit, summary="tests still pass")
    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_reviewer_visit_changes_requested(
        db_session, card, visit, feedback="still not fixed", summary="still a security issue"
    )
    assert card.lifecycle_state == LifecycleState.BLOCKED
    assert card.revision_count == 2


async def test_deployer_conflict_bounces_to_developer_and_shares_revision_cap(db_session):
    """run_deploy hitting a merge conflict is not a deploy failure — the merge/push
    never ran, so it doesn't count against max_deploy_attempts — it's 'another round
    of Developer work needed', sharing the same revision_count budget as Reviewer/
    Tester's request_changes."""
    project = await _make_project(db_session, name="deployer-conflict-loop", max_revisions=1)
    card = await card_service.create_card(db_session, project.id, title="Racy card", raw_request="do it")
    card.column = Column.DEPLOYER
    await db_session.flush()

    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_deployer_visit_conflict(db_session, card, visit, conflicted_paths=["app.py"])

    assert card.column == Column.DEVELOPER
    assert card.lifecycle_state == LifecycleState.ACTIVE
    assert card.revision_count == 1
    assert card.deploy_attempt_count == 0
    assert "app.py" in card.latest_feedback
    assert visit.outcome == VisitOutcome.DEPLOY_CONFLICT

    # A second conflict in a row goes over max_revisions=1, same safety valve as
    # the Reviewer/Tester loop.
    card.column = Column.DEPLOYER
    await db_session.flush()
    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_deployer_visit_conflict(db_session, card, visit, conflicted_paths=["app.py"])
    assert card.lifecycle_state == LifecycleState.BLOCKED
    assert card.revision_count == 2


async def test_retry_with_a_note_stores_it_on_the_card(db_session):
    project = await _make_project(db_session, name="retry-note")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    visit = await transitions.start_visit(db_session, card)
    await transitions.fail_visit_with_error(db_session, card, visit, message="boom")
    assert card.lifecycle_state == LifecycleState.BLOCKED

    await card_service.retry_card(db_session, card.id, note="rebase onto main first")

    assert card.lifecycle_state == LifecycleState.ACTIVE
    assert card.retry_note == "rebase onto main first"


async def test_retry_without_a_note_clears_any_stale_one(db_session):
    project = await _make_project(db_session, name="retry-note-clear")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    card.retry_note = "leftover from a previous retry"
    visit = await transitions.start_visit(db_session, card)
    await transitions.fail_visit_with_error(db_session, card, visit, message="boom")

    await card_service.retry_card(db_session, card.id)

    assert card.retry_note is None


async def test_deploy_cap_exhausted_marks_card_failed_with_no_further_action(db_session):
    project = await _make_project(db_session, name="doomed-deploy", max_deploy_attempts=1)
    card = await card_service.create_card(db_session, project.id, title="Doomed deploy", raw_request="do it")
    card.column = Column.DEPLOYER
    await db_session.flush()

    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_deployer_visit(
        db_session, card, visit, success=False, summary="registry unreachable"
    )

    assert card.lifecycle_state == LifecycleState.FAILED
    assert card.deploy_attempt_count == 1
    visits = await card_service.list_column_visits(db_session, card.id)
    assert visits[-1].outcome == VisitOutcome.FAILED


async def test_run_error_blocks_card_regardless_of_column(db_session):
    project = await _make_project(db_session, name="runaway-loop")
    card = await card_service.create_card(db_session, project.id, title="Runaway loop", raw_request="do it")

    visit = await transitions.start_visit(db_session, card)
    await transitions.fail_visit_with_error(
        db_session, card, visit, message="exceeded max_iterations_per_run"
    )

    assert card.lifecycle_state == LifecycleState.BLOCKED
    visits = await card_service.list_column_visits(db_session, card.id)
    assert visits[-1].outcome == VisitOutcome.ERROR


async def test_cannot_retry_an_active_card_or_cancel_a_done_card(db_session):
    project = await _make_project(db_session, name="guard-rails")
    card = await card_service.create_card(db_session, project.id, title="Guarded", raw_request="do it")

    try:
        await transitions.retry_card(db_session, card)
    except ValueError:
        pass
    else:
        raise AssertionError("retrying an ACTIVE card should have raised")

    card.lifecycle_state = LifecycleState.DONE
    try:
        await transitions.cancel_card(db_session, card)
    except ValueError:
        pass
    else:
        raise AssertionError("cancelling a DONE card should have raised")
