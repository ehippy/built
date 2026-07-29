"""has_passing_run_since_last_change — the rigid test gate Developer's submit_for_test
and Tester's approve both check server-side (see agent/loop.py's
_require_passing_test_run). Exercises the domain function directly against real
CardEvent/RunAttempt rows, without going through the full agent loop."""

from built.domain import run_attempts, transitions
from built.domain.enums import EventType
from built.domain.events import append_event
from built.services import card_service, project_service


async def _make_visit(session, *, test_command="pytest -q"):
    project = await project_service.create_project(
        session,
        name=f"run-attempts-{id(session)}",
        overarching_goal="goal",
        repo_remote_url="https://example.invalid/repo.git",
        test_command=test_command,
    )
    card = await card_service.create_card(session, project.id, title="t", raw_request="r")
    visit = await transitions.start_visit(session, card)
    return card, visit


async def _record_bash_call(session, card, visit, *, command, exit_code):
    """Mirrors what agent/loop.py's run_column_visit does per bash tool call: append a
    TOOL_CALL event, then record a RunAttempt tagged with that event's seq — bash never
    carries a commit_sha unless it happened to change tracked files."""
    event = await append_event(
        session,
        card_id=card.id,
        column_visit_id=visit.id,
        type=EventType.TOOL_CALL,
        payload={"name": "bash", "arguments": {"command": command}, "commit_sha": None},
    )
    await run_attempts.record_run_attempt(
        session,
        card_id=card.id,
        column_visit_id=visit.id,
        command=command,
        exit_code=exit_code,
        stdout="",
        stderr="",
        card_event_seq=event.seq,
    )


async def _record_tool_call(session, card, visit, *, name, commit_sha):
    await append_event(
        session,
        card_id=card.id,
        column_visit_id=visit.id,
        type=EventType.TOOL_CALL,
        payload={"name": name, "arguments": {}, "commit_sha": commit_sha},
    )


async def test_no_run_at_all_fails(db_session):
    card, visit = await _make_visit(db_session)
    assert not await run_attempts.has_passing_run_since_last_change(db_session, visit.id, "pytest -q")


async def test_wrong_command_fails_even_after_a_trivial_command_passes(db_session):
    """Closes the original loophole: the real suite failing, followed by a trivial
    command that happens to exit 0, must not satisfy the gate."""
    card, visit = await _make_visit(db_session)
    await _record_bash_call(db_session, card, visit, command="pytest -q", exit_code=1)
    await _record_bash_call(db_session, card, visit, command="echo ok", exit_code=0)
    assert not await run_attempts.has_passing_run_since_last_change(db_session, visit.id, "pytest -q")


async def test_failed_matching_run_fails(db_session):
    card, visit = await _make_visit(db_session)
    await _record_bash_call(db_session, card, visit, command="pytest -q", exit_code=1)
    assert not await run_attempts.has_passing_run_since_last_change(db_session, visit.id, "pytest -q")


async def test_exact_matching_passing_run_succeeds(db_session):
    card, visit = await _make_visit(db_session)
    await _record_bash_call(db_session, card, visit, command="pytest -q", exit_code=0)
    assert await run_attempts.has_passing_run_since_last_change(db_session, visit.id, "pytest -q")


async def test_trailing_redirection_is_normalized_away(db_session):
    card, visit = await _make_visit(db_session)
    await _record_bash_call(db_session, card, visit, command="pytest -q 2>&1", exit_code=0)
    assert await run_attempts.has_passing_run_since_last_change(db_session, visit.id, "pytest -q")


async def test_leading_cd_workspace_prefix_is_normalized_away(db_session):
    """Regression test for a real stuck-in-a-loop card: bash already runs with
    cwd=/workspace, but an agent habitually prefixing "cd /workspace && " onto an
    otherwise-exact, actually-passing command must not make the gate reject it
    forever."""
    card, visit = await _make_visit(db_session)
    await _record_bash_call(db_session, card, visit, command="cd /workspace && pytest -q", exit_code=0)
    assert await run_attempts.has_passing_run_since_last_change(db_session, visit.id, "pytest -q")


async def test_edit_after_a_passing_run_invalidates_it(db_session):
    """The gap a "most recent RunAttempt" check alone can't catch: RunAttempt rows
    only exist for bash calls, so a write_file/edit_file after a green run doesn't
    create a new one — has_passing_run_since_last_change must check the full event
    stream, not just the run_attempts table, to catch this."""
    card, visit = await _make_visit(db_session)
    await _record_bash_call(db_session, card, visit, command="pytest -q", exit_code=0)
    await _record_tool_call(db_session, card, visit, name="write_file", commit_sha="deadbeef")
    assert not await run_attempts.has_passing_run_since_last_change(db_session, visit.id, "pytest -q")


async def test_rerunning_after_the_edit_passes_again(db_session):
    card, visit = await _make_visit(db_session)
    await _record_bash_call(db_session, card, visit, command="pytest -q", exit_code=0)
    await _record_tool_call(db_session, card, visit, name="edit_file", commit_sha="deadbeef")
    await _record_bash_call(db_session, card, visit, command="pytest -q", exit_code=0)
    assert await run_attempts.has_passing_run_since_last_change(db_session, visit.id, "pytest -q")


async def test_read_only_tool_call_after_a_passing_run_does_not_invalidate_it(db_session):
    """git_status/read_file/etc. never carry a commit_sha, so they shouldn't be
    mistaken for a mutation."""
    card, visit = await _make_visit(db_session)
    await _record_bash_call(db_session, card, visit, command="pytest -q", exit_code=0)
    await _record_tool_call(db_session, card, visit, name="git_status", commit_sha=None)
    assert await run_attempts.has_passing_run_since_last_change(db_session, visit.id, "pytest -q")
