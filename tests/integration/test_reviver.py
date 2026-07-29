"""The Reviver's multi-action pass — LLM faked, everything else real: DB-level tool
handlers (list/read/revive/leave), the auto-revive cap, and that a human-initiated
retry doesn't touch the same budget."""

from built.agent.reviver import run_reviver_pass
from built.domain import transitions
from built.domain.enums import LifecycleState
from built.llm.client import LLMResult, ToolCallRequest
from built.logging_config import get_logs
from built.services import card_service, project_service
from tests.unit.fakes import ScriptedLLMClient


async def _make_project(session, **overrides):
    defaults = {
        "name": f"reviver-{overrides.pop('_n', 'x')}",
        "overarching_goal": "goal",
        "repo_remote_url": "https://example.invalid/repo.git",
    }
    defaults.update(overrides)
    return await project_service.create_project(session, **defaults)


async def _make_blocked_card(session, project, title="stuck card", *, message="boom"):
    card = await card_service.create_card(session, project.id, title=title, raw_request="r")
    visit = await transitions.start_visit(session, card)
    await transitions.fail_visit_with_error(session, card, visit, message=message)
    await session.commit()
    return card


async def test_reviver_ignores_stuck_cards_in_a_paused_project(db_session):
    project = await _make_project(db_session, _n="paused")
    await _make_blocked_card(db_session, project, title="stuck but paused")
    await project_service.pause_project(db_session, project.id)

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_1", name="list_stuck_cards", arguments={}),
                ],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_2", name="done_for_now", arguments={"summary": "nothing to do"})
                ],
                endpoint_used="fake::model",
            ),
        ]
    )

    counts = await run_reviver_pass(db_session, llm_client=llm, max_iterations=10)

    assert counts == {"revived": 0, "left_blocked": 0, "errors": 0}
    assert "No stuck cards" in llm.calls[1]["messages"][3]["content"]


async def test_reviver_does_nothing_when_no_cards_are_stuck(db_session):
    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_1", name="list_stuck_cards", arguments={}),
                ],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_2", name="done_for_now", arguments={"summary": "nothing to do"})
                ],
                endpoint_used="fake::model",
            ),
        ]
    )

    counts = await run_reviver_pass(db_session, llm_client=llm, max_iterations=10)

    assert counts == {"revived": 0, "left_blocked": 0, "errors": 0}
    # index 3: system, user, assistant(tool_calls), tool(list_stuck_cards result) — a
    # fixed index, not [-1]: ScriptedLLMClient stores a reference to the same mutable
    # messages list, so [-1] would pick up messages appended by later calls too.
    assert "No stuck cards" in llm.calls[1]["messages"][3]["content"]


async def test_reviver_revives_a_card_with_a_note(db_session):
    project = await _make_project(db_session, _n="1")
    card = await _make_blocked_card(db_session, project, message="timeout talking to the LLM endpoint")

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="list_stuck_cards", arguments={})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_2",
                        name="revive_card",
                        arguments={"card_id": card.id, "note": "transient timeout, just retry"},
                    )
                ],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_3", name="done_for_now", arguments={"summary": "done"})],
                endpoint_used="fake::model",
            ),
        ]
    )

    prior_logs = get_logs()
    cutoff = prior_logs[-1].seq if prior_logs else 0
    counts = await run_reviver_pass(db_session, llm_client=llm, max_iterations=10)

    assert counts == {"revived": 1, "left_blocked": 0, "errors": 0}
    assert card.lifecycle_state == LifecycleState.ACTIVE
    assert card.retry_note == "transient timeout, just retry"
    assert card.auto_revive_count == 1

    new_logs = get_logs(since_seq=cutoff)
    assert any(card.id in e.message and "transient timeout, just retry" in e.message for e in new_logs)


async def test_reviver_leaves_a_card_blocked_with_a_reason(db_session):
    project = await _make_project(db_session, _n="2")
    card = await _make_blocked_card(
        db_session, project, message="no deploy config — configure one in Project Settings"
    )

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="list_stuck_cards", arguments={})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_2",
                        name="leave_blocked",
                        arguments={"card_id": card.id, "reason": "needs a human to configure deploy"},
                    )
                ],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_3", name="done_for_now", arguments={"summary": "done"})],
                endpoint_used="fake::model",
            ),
        ]
    )

    prior_logs = get_logs()
    cutoff = prior_logs[-1].seq if prior_logs else 0
    counts = await run_reviver_pass(db_session, llm_client=llm, max_iterations=10)

    assert counts == {"revived": 0, "left_blocked": 1, "errors": 0}
    assert card.lifecycle_state == LifecycleState.BLOCKED
    assert card.auto_revive_count == 0

    new_logs = get_logs(since_seq=cutoff)
    assert any(card.id in e.message and "needs a human to configure deploy" in e.message for e in new_logs)

    events = await card_service.list_events(db_session, card.id)
    assert any(e.payload.get("action") == "reviver_left_blocked" for e in events)


async def test_reviver_refuses_to_revive_past_the_cap(db_session):
    project = await _make_project(db_session, _n="3")
    card = await _make_blocked_card(db_session, project)
    card.auto_revive_count = 3  # settings.reviver_max_auto_revives default
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="revive_card", arguments={"card_id": card.id})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_2", name="done_for_now", arguments={"summary": "done"})],
                endpoint_used="fake::model",
            ),
        ]
    )

    counts = await run_reviver_pass(db_session, llm_client=llm, max_iterations=10)

    assert counts == {"revived": 0, "left_blocked": 0, "errors": 1}
    assert card.lifecycle_state == LifecycleState.BLOCKED
    assert card.auto_revive_count == 3


async def test_human_initiated_retry_does_not_touch_auto_revive_count(db_session):
    project = await _make_project(db_session, _n="4")
    card = await _make_blocked_card(db_session, project)

    await card_service.retry_card(db_session, card.id, note="a human's note")

    assert card.lifecycle_state == LifecycleState.ACTIVE
    assert card.auto_revive_count == 0
