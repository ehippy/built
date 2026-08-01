"""agent/summarizer.py against a real toy git repo + real sqlite — LLM faked. Covers
write_card_postmortem directly (transcript building, the submit_postmortem call,
best-effort failure handling) and the RETRO curation kind that mines the resulting
CardPostmortem rows."""

from sqlalchemy import select

from built.agent.curation import run_curation_pass
from built.agent.summarizer import write_card_postmortem
from built.db.models import CardPostmortem
from built.domain import transitions
from built.domain.enums import ActivityKind, Column, EventType, LifecycleState
from built.llm.client import LLMResult, ToolCallRequest
from built.orchestrator import curator
from built.sandbox import worktree
from built.sandbox.container import CommandResult
from built.services import card_service, project_service
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher
from tests.unit.fakes import FakeCommandExecutor, RaisingLLMClient, ScriptedLLMClient


async def _make_card(db_session, toy_repo_remote, **overrides):
    project = await project_service.create_project(
        db_session,
        name=overrides.pop("name", "summarizer-x"),
        overarching_goal="goal",
        repo_remote_url=str(toy_repo_remote),
    )
    card = await card_service.create_card(db_session, project.id, title="Add farewell", raw_request="r")
    await db_session.flush()
    return project, card


def _dispatcher(wt_path) -> ToolDispatcher:
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    return ToolDispatcher(ctx=ToolContext(card_id="curator-x", worktree_root=wt_path), executor=executor)


# --- write_card_postmortem ----------------------------------------------------------


async def test_write_card_postmortem_creates_row_and_system_note(db_session, toy_repo_remote):
    project, card = await _make_card(db_session, toy_repo_remote, name="summarizer-1")
    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_pm_visit(
        db_session, card, visit, spec="s", acceptance_criteria=["x"], summary="submitted spec"
    )
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="submit_postmortem",
                        arguments={"went_well": "spec was clear", "struggles": "none"},
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )

    postmortem = await write_card_postmortem(
        db_session, project, card, outcome=LifecycleState.DONE, llm_client=llm
    )
    await db_session.commit()

    assert postmortem is not None
    assert postmortem.went_well == "spec was clear"
    assert postmortem.struggles == "none"
    assert postmortem.outcome == LifecycleState.DONE

    row = await db_session.scalar(select(CardPostmortem).where(CardPostmortem.card_id == card.id))
    assert row is not None and row.id == postmortem.id

    events = await card_service.list_events(db_session, card.id)
    assert any(
        e.type == EventType.SYSTEM_NOTE and e.payload.get("action") == "postmortem_written" for e in events
    )

    # The transcript handed to the model is the visit map, not a raw event replay —
    # leans on domain/transitions.py's own summary/outcome, not a reconstruction.
    sent_user_message = llm.calls[0]["messages"][1]["content"]
    assert "[pm attempt 1] submitted: submitted spec" in sent_user_message


async def test_write_card_postmortem_visit_map_shows_the_whole_revision_loop(db_session, toy_repo_remote):
    """A card bounced back to Developer once before Tester finally approved —
    the visit map should show both Developer attempts and the Tester rejection
    in order, with each step's own summary intact, not just the final outcome."""
    project, card = await _make_card(db_session, toy_repo_remote, name="summarizer-6")

    pm_visit = await transitions.start_visit(db_session, card)
    await transitions.complete_pm_visit(
        db_session, card, pm_visit, spec="s", acceptance_criteria=["x"], summary="scoped the change"
    )

    dev_visit_1 = await transitions.start_visit(db_session, card)
    await transitions.complete_developer_visit(db_session, card, dev_visit_1, summary="implemented v1")

    tester_visit_1 = await transitions.start_visit(db_session, card)
    await transitions.complete_tester_visit_changes_requested(
        db_session,
        card,
        tester_visit_1,
        feedback="the new test fails intermittently",
        summary="rejected: flaky test",
    )

    dev_visit_2 = await transitions.start_visit(db_session, card)
    await transitions.complete_developer_visit(
        db_session, card, dev_visit_2, summary="fixed the race in the test setup"
    )
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="submit_postmortem",
                        arguments={
                            "went_well": "eventually landed cleanly",
                            "struggles": "one round of test flakiness",
                        },
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )

    await write_card_postmortem(db_session, project, card, outcome=LifecycleState.DONE, llm_client=llm)

    sent_user_message = llm.calls[0]["messages"][1]["content"]
    assert "[developer attempt 1] submitted: implemented v1" in sent_user_message
    assert "[tester attempt 1] changes_requested: rejected: flaky test" in sent_user_message
    assert "[developer attempt 2] submitted: fixed the race in the test setup" in sent_user_message
    # In visit order, not just present — proves the map is the journey, not a bag of lines.
    assert sent_user_message.index("developer attempt 1") < sent_user_message.index("tester attempt 1")
    assert sent_user_message.index("tester attempt 1") < sent_user_message.index("developer attempt 2")


async def test_write_card_postmortem_returns_none_on_llm_failure(db_session, toy_repo_remote):
    """Best-effort, mirroring run_curation_pass: a broken endpoint loses the
    postmortem but must never raise into the caller — the card's own closure
    can't be allowed to fail because of this."""
    project, card = await _make_card(db_session, toy_repo_remote, name="summarizer-2")

    postmortem = await write_card_postmortem(
        db_session, project, card, outcome=LifecycleState.FAILED, llm_client=RaisingLLMClient()
    )

    assert postmortem is None
    assert await db_session.scalar(select(CardPostmortem).where(CardPostmortem.card_id == card.id)) is None


async def test_write_card_postmortem_returns_none_when_no_submit_call(db_session, toy_repo_remote):
    project, card = await _make_card(db_session, toy_repo_remote, name="summarizer-3")
    llm = ScriptedLLMClient(
        [LLMResult(content="I have nothing to say.", tool_calls=[], endpoint_used="fake::model")]
    )

    postmortem = await write_card_postmortem(
        db_session, project, card, outcome=LifecycleState.DONE, llm_client=llm
    )

    assert postmortem is None


# --- get_visit_detail: digging into a visit whose summary hints at trouble --------


async def test_get_visit_detail_returns_full_feedback_and_tool_calls(db_session, toy_repo_remote):
    project, card = await _make_card(db_session, toy_repo_remote, name="summarizer-7")
    card.column = Column.TESTER
    await db_session.flush()
    tester_visit = await transitions.start_visit(db_session, card)
    await transitions.complete_tester_visit_changes_requested(
        db_session,
        card,
        tester_visit,
        feedback="test_checkout.py fails: expected 200, got 500 — see traceback in the run log",
        summary="rejected: checkout test fails",
    )
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="get_visit_detail",
                        arguments={"column": "tester", "attempt_number": 1},
                    )
                ],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_2",
                        name="submit_postmortem",
                        arguments={
                            "went_well": "n/a",
                            "struggles": "checkout test kept returning 500",
                        },
                    )
                ],
                endpoint_used="fake::model",
            ),
        ]
    )

    postmortem = await write_card_postmortem(
        db_session, project, card, outcome=LifecycleState.FAILED, llm_client=llm
    )

    assert postmortem is not None
    # Second call's messages include the tool result from the first — the detail
    # actually reached the model before it wrote the postmortem. Found by
    # tool_call_id, not position: `messages` is the same list object mutated
    # across iterations, so a later iteration's appends land in this same
    # recorded reference too — position isn't stable, content is.
    tool_result = next(m for m in llm.calls[1]["messages"] if m.get("tool_call_id") == "call_1")
    assert tool_result["role"] == "tool"
    assert "expected 200, got 500" in tool_result["content"]


async def test_get_visit_detail_unknown_visit_returns_a_hint_not_an_error(db_session, toy_repo_remote):
    """A hallucinated (column, attempt_number) pair shouldn't blow up the whole
    postmortem — it should read like a normal tool result the model can recover
    from, matching how every other tool in this codebase reports a bad call."""
    project, card = await _make_card(db_session, toy_repo_remote, name="summarizer-8")

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="get_visit_detail",
                        arguments={"column": "deployer", "attempt_number": 9},
                    )
                ],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_2",
                        name="submit_postmortem",
                        arguments={"went_well": "n/a", "struggles": "n/a"},
                    )
                ],
                endpoint_used="fake::model",
            ),
        ]
    )

    postmortem = await write_card_postmortem(
        db_session, project, card, outcome=LifecycleState.FAILED, llm_client=llm
    )

    assert postmortem is not None
    tool_result = next(m for m in llm.calls[1]["messages"] if m.get("tool_call_id") == "call_1")
    assert "No such visit" in tool_result["content"]


async def test_write_card_postmortem_gives_up_after_max_iterations_of_lookups(db_session, toy_repo_remote):
    """A model that keeps calling get_visit_detail and never submits must not hang
    or loop forever — it's a best-effort background write, not something worth
    burning unbounded calls on."""
    project, card = await _make_card(db_session, toy_repo_remote, name="summarizer-9")

    def _detail_call(call_id: str) -> LLMResult:
        return LLMResult(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id=call_id, name="get_visit_detail", arguments={"column": "pm", "attempt_number": 1}
                )
            ],
            endpoint_used="fake::model",
        )

    llm = ScriptedLLMClient([_detail_call(f"call_{i}") for i in range(10)])

    postmortem = await write_card_postmortem(
        db_session, project, card, outcome=LifecycleState.FAILED, llm_client=llm
    )

    assert postmortem is None
    # Bounded, not exhausted: ScriptedLLMClient had 10 scripted responses but the
    # loop must have stopped well short of consuming all of them.
    assert len(llm.calls) < 10


# --- ActivityKind.RETRO: mines postmortems for a recurring pattern -----------------


async def test_curation_retro_proposes_a_card_from_recent_postmortems(db_session, toy_repo_remote):
    project, _card = await _make_card(db_session, toy_repo_remote, name="summarizer-4")
    dispatcher_wt = await worktree.ensure_tool_worktree(project, tool="curator")

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="propose_tasks",
                        arguments={
                            "tasks": [
                                {
                                    "title": "Fix flaky integration test blocking Tester",
                                    "raw_request": "The same integration test keeps failing intermittently.",
                                }
                            ]
                        },
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )

    created = await run_curation_pass(
        db_session,
        project,
        ActivityKind.RETRO,
        llm_client=llm,
        dispatcher=_dispatcher(dispatcher_wt),
        max_iterations=10,
        run_id="test-run",
        extra_context="- [failed, 2 revision(s)] went well: (nothing notable) | struggles: Tester kept "
        "rejecting on the same flaky integration test",
    )

    assert [c.title for c in created] == ["Fix flaky integration test blocking Tester"]
    events = await card_service.list_events(db_session, created[0].id)
    assert events[0].payload["source"] == "curation:retro"


async def test_needs_run_retro_gated_by_new_postmortems(db_session, toy_repo_remote):
    project, card = await _make_card(db_session, toy_repo_remote, name="summarizer-5")

    should_run, extra_context = await curator._needs_run(db_session, project, ActivityKind.RETRO)
    assert should_run is False
    assert extra_context is None

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="submit_postmortem",
                        arguments={"went_well": "n/a", "struggles": "flaky test again"},
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )
    await write_card_postmortem(db_session, project, card, outcome=LifecycleState.FAILED, llm_client=llm)
    await db_session.commit()

    should_run, extra_context = await curator._needs_run(db_session, project, ActivityKind.RETRO)
    assert should_run is True
    assert extra_context is not None and "flaky test again" in extra_context
