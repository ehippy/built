"""End-to-end test of the Developer agent loop against a real toy git repo (real
`git`, real filesystem, real domain/transitions state machine) with the LLM and the
container executor faked — neither a live LLM endpoint nor a Docker daemon is
available in this dev environment. See tests/unit/fakes.py."""

from built.agent.loop import run_developer_visit
from built.domain import transitions
from built.domain.enums import Column, LifecycleState, VisitOutcome
from built.llm.client import LLMResult, ToolCallRequest
from built.sandbox import worktree
from built.sandbox.container import CommandResult
from built.services import card_service, project_service
from built.tools import git_tools
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher
from tests.unit.fakes import FakeCommandExecutor, RaisingLLMClient, ScriptedLLMClient


async def _make_developer_card(
    db_session, toy_repo_remote, *, max_iterations_per_run: int = 25, test_command: str | None = "pytest -q"
):
    project = await project_service.create_project(
        db_session,
        name=f"dev-loop-{max_iterations_per_run}-{test_command}",
        overarching_goal="Add basic arithmetic helpers to app.py.",
        repo_remote_url=str(toy_repo_remote),
        max_iterations_per_run=max_iterations_per_run,
        test_command=test_command,
    )
    card = await card_service.create_card(
        db_session, project.id, title="Add subtract()", raw_request="Add a subtract(a, b) function to app.py"
    )
    # Stand-in for the PM column, which doesn't exist yet (Phase 4).
    card.spec = "Add a subtract(a, b) function to app.py that returns a - b."
    card.acceptance_criteria = ["app.py defines subtract(a, b)", "subtract(2, 1) == 1"]
    card.column = Column.DEVELOPER
    await db_session.flush()

    wt_path = await worktree.create_card_worktree(project, card)
    card.worktree_path = str(wt_path)
    await db_session.flush()
    return project, card, wt_path


async def test_developer_loop_happy_path_reads_writes_runs_bash_and_submits(db_session, toy_repo_remote):
    project, card, wt_path = await _make_developer_card(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card)

    new_app_py = "def greet():\n    return 'hi'\n\n\ndef subtract(a, b):\n    return a - b\n"
    llm = ScriptedLLMClient(
        [
            # Turn 1: no tool call at all — exercises the "must call a tool" nudge.
            LLMResult(content="Let me look at the file first.", tool_calls=[]),
            # Turn 2: reads the file.
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="read_file", arguments={"path": "app.py"})],
                endpoint_used="fake::model",
            ),
            # Turn 3: writes the implementation.
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_2",
                        name="write_file",
                        arguments={"path": "app.py", "content": new_app_py},
                    )
                ],
                endpoint_used="fake::model",
            ),
            # Turn 4: runs the project's test command — required before submit_for_test
            # will be accepted.
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_3", name="bash", arguments={"command": "pytest -q"})],
                endpoint_used="fake::model",
            ),
            # Turn 5: done.
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_4", name="submit_for_test", arguments={"summary": "Added subtract()"}
                    )
                ],
                endpoint_used="fake::model",
            ),
        ]
    )
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="ok", stderr=""))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)

    result = await run_developer_visit(
        db_session,
        project,
        card,
        visit,
        llm_client=llm,
        dispatcher=dispatcher,
        max_iterations=project.max_iterations_per_run,
    )

    assert result.column == Column.TESTER
    assert result.lifecycle_state == LifecycleState.ACTIVE
    assert visit.outcome == VisitOutcome.SUBMITTED
    assert visit.summary == "Added subtract()"
    assert visit.endpoint_used == "fake::model"

    # The write_file call actually landed on disk and was committed.
    assert (wt_path / "app.py").read_text() == new_app_py
    log = await git_tools.run_git("log", "--oneline", cwd=wt_path)
    assert "write_file: app.py" in log

    # The bash call went through the (fake) executor.
    assert executor.calls == ["pytest -q"]

    events = await card_service.list_events(db_session, card.id)
    tool_call_events = [e for e in events if e.type.value == "tool_call"]
    assert [e.payload["name"] for e in tool_call_events] == ["read_file", "write_file", "bash"]
    assert tool_call_events[1].payload["commit_sha"] is not None  # write_file committed
    assert tool_call_events[2].payload["commit_sha"] is None  # bash changed nothing on disk


async def test_developer_submit_for_test_is_rejected_without_a_passing_run_then_succeeds(
    db_session, toy_repo_remote
):
    """Developer used to have no server-side gate at all — submit_for_test just
    trusted the model's word. This is the regression test for that: claiming done
    with no test run must be rejected, exactly like Tester's approve already was."""
    project, card, wt_path = await _make_developer_card(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card)

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_1", name="submit_for_test", arguments={"summary": "done"})
                ],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_2",
                        name="write_file",
                        arguments={
                            "path": "app.py",
                            "content": (
                                "def greet():\n    return 'hi'\n\n\ndef subtract(a, b):\n    return a - b\n"
                            ),
                        },
                    )
                ],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_3", name="bash", arguments={"command": "pytest -q"})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_4", name="submit_for_test", arguments={"summary": "done"})
                ],
                endpoint_used="fake::model",
            ),
        ]
    )
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="1 passed", stderr=""))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)

    result = await run_developer_visit(
        db_session, project, card, visit, llm_client=llm, dispatcher=dispatcher, max_iterations=10
    )

    assert result.column == Column.TESTER
    assert visit.outcome == VisitOutcome.SUBMITTED
    assert llm.calls[3] is not None  # the loop kept going past the rejected first attempt


async def test_developer_submit_for_test_is_rejected_if_nothing_was_ever_changed(db_session, toy_repo_remote):
    """Confirmed in production on a large migration card: a Developer that reads
    the whole repo, never calls write_file/edit_file, then runs the project's
    already-passing test command (which happened to cover none of the actual task)
    could submit_for_test having implemented nothing at all — has_passing_run_since_
    last_change has no way to tell "verified real work" apart from "verified an
    untouched repo still works". This is the gate that catches the difference."""
    project, card, wt_path = await _make_developer_card(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card)

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="read_file", arguments={"path": "app.py"})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_2", name="bash", arguments={"command": "pytest -q"})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_3", name="submit_for_test", arguments={"summary": "done"})
                ],
                endpoint_used="fake::model",
            ),
        ]
    )
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="1 passed", stderr=""))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)

    result = await run_developer_visit(
        db_session, project, card, visit, llm_client=llm, dispatcher=dispatcher, max_iterations=3
    )

    assert result.column == Column.DEVELOPER  # never advanced
    assert result.lifecycle_state == LifecycleState.BLOCKED


async def test_developer_submit_for_test_is_rejected_when_the_run_doesnt_match_the_configured_command(
    db_session, toy_repo_remote
):
    """A passing run of the WRONG command must not satisfy the gate — otherwise
    "run something that happens to exit 0" is just as gameable as no check at all."""
    project, card, wt_path = await _make_developer_card(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card)

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="bash", arguments={"command": "echo ok"})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_2", name="submit_for_test", arguments={"summary": "done"})
                ],
                endpoint_used="fake::model",
            ),
        ]
    )
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="ok", stderr=""))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)

    result = await run_developer_visit(
        db_session, project, card, visit, llm_client=llm, dispatcher=dispatcher, max_iterations=2
    )

    assert result.column == Column.DEVELOPER  # never advanced
    assert result.lifecycle_state == LifecycleState.BLOCKED


async def test_developer_submit_for_test_is_rejected_after_an_edit_since_the_passing_run(
    db_session, toy_repo_remote
):
    """A green run followed by an unverified edit must not still read as tested —
    RunAttempt rows only exist for bash calls, so this has to be checked against the
    full event stream (see has_passing_run_since_last_change), not just the latest
    RunAttempt row."""
    project, card, wt_path = await _make_developer_card(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card)

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="bash", arguments={"command": "pytest -q"})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_2",
                        name="write_file",
                        arguments={
                            "path": "app.py",
                            "content": (
                                "def greet():\n    return 'hi'\n\n\ndef subtract(a, b):\n    return a - b\n"
                            ),
                        },
                    )
                ],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_3", name="submit_for_test", arguments={"summary": "done"})
                ],
                endpoint_used="fake::model",
            ),
        ]
    )
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="1 passed", stderr=""))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)

    result = await run_developer_visit(
        db_session, project, card, visit, llm_client=llm, dispatcher=dispatcher, max_iterations=3
    )

    assert result.column == Column.DEVELOPER  # never advanced
    assert result.lifecycle_state == LifecycleState.BLOCKED


async def test_developer_loop_exceeds_iteration_cap_blocks_the_card(db_session, toy_repo_remote):
    project, card, wt_path = await _make_developer_card(db_session, toy_repo_remote, max_iterations_per_run=2)
    visit = await transitions.start_visit(db_session, card)

    # Never calls a terminal tool — just keeps reading the same file forever.
    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id=f"call_{i}", name="read_file", arguments={"path": "app.py"})],
                endpoint_used="fake::model",
            )
            for i in range(1, 10)
        ]
    )
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)

    result = await run_developer_visit(
        db_session, project, card, visit, llm_client=llm, dispatcher=dispatcher, max_iterations=2
    )

    assert result.lifecycle_state == LifecycleState.BLOCKED
    assert result.column == Column.DEVELOPER  # never advanced
    assert visit.outcome == VisitOutcome.ERROR
    assert "max_iterations_per_run" in (visit.summary or "")


async def test_developer_loop_second_attempt_gets_a_recap_of_the_first(db_session, toy_repo_remote):
    project, card, wt_path = await _make_developer_card(db_session, toy_repo_remote, max_iterations_per_run=1)

    first_visit = await transitions.start_visit(db_session, card)
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="ok", stderr=""))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)
    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="bash", arguments={"command": "npm install"})],
                endpoint_used="fake::model",
            )
        ]
    )
    await run_developer_visit(
        db_session, project, card, first_visit, llm_client=llm, dispatcher=dispatcher, max_iterations=1
    )
    assert card.lifecycle_state == LifecycleState.BLOCKED  # iteration cap of 1 tripped

    await transitions.retry_card(db_session, card)
    second_visit = await transitions.start_visit(db_session, card)
    recap = await card_service.get_previous_attempt_recap(
        db_session, card.id, card.column, before_attempt=second_visit.attempt_number
    )
    assert recap is not None
    assert "npm install" in recap

    llm2 = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_2", name="submit_for_test", arguments={"summary": "done"})
                ],
                endpoint_used="fake::model",
            )
        ]
    )
    await run_developer_visit(
        db_session,
        project,
        card,
        second_visit,
        llm_client=llm2,
        dispatcher=dispatcher,
        max_iterations=5,
        retry_recap=recap,
    )

    sent_user_message = llm2.calls[0]["messages"][1]["content"]
    assert "Context from your previous attempt at this column" in sent_user_message
    assert "npm install" in sent_user_message


async def test_developer_loop_includes_a_human_retry_note_in_the_prompt(db_session, toy_repo_remote):
    project, card, wt_path = await _make_developer_card(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card)
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)
    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_1", name="submit_for_test", arguments={"summary": "done"})
                ],
                endpoint_used="fake::model",
            )
        ]
    )

    await run_developer_visit(
        db_session,
        project,
        card,
        visit,
        llm_client=llm,
        dispatcher=dispatcher,
        max_iterations=5,
        retry_note="rebase onto main and resolve the conflict in app.py",
    )

    sent_user_message = llm.calls[0]["messages"][1]["content"]
    assert "A human left this instruction for this attempt" in sent_user_message
    assert "rebase onto main and resolve the conflict in app.py" in sent_user_message


async def test_developer_loop_endpoint_chain_exhausted_blocks_the_card(db_session, toy_repo_remote):
    project, card, wt_path = await _make_developer_card(db_session, toy_repo_remote)
    visit = await transitions.start_visit(db_session, card)

    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    dispatcher = ToolDispatcher(ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor)

    result = await run_developer_visit(
        db_session,
        project,
        card,
        visit,
        llm_client=RaisingLLMClient(),
        dispatcher=dispatcher,
        max_iterations=project.max_iterations_per_run,
    )

    assert result.lifecycle_state == LifecycleState.BLOCKED
    assert visit.outcome == VisitOutcome.ERROR
    assert "unhandled error" in (visit.summary or "")
