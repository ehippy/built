"""ActivityKind.PM_TRIAGE (agent/pm_triage.py) — the one background pass with
authority to act on cards already sitting in the PM column. It browses the repo
read-only via the same explore tools/dispatcher/worktree as every other curation
kind (see test_curator.py), just ending in groom_backlog instead of propose_tasks.
LLM faked, everything else real, same house style as test_curator.py."""

from sqlalchemy import select

from built.agent.pm_triage import run_pm_triage_pass
from built.db.models import CardEvent, CurationEvent
from built.domain import transitions
from built.domain.enums import ActivityKind, Column, EventType, LifecycleState, Priority
from built.llm.client import LLMResult, ToolCallRequest
from built.logging_config import get_logs
from built.orchestrator import curator
from built.sandbox import worktree
from built.sandbox.container import CommandResult
from built.services import card_service, endpoint_service, project_service
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher
from tests.unit.fakes import FakeCommandExecutor, ScriptedLLMClient


async def _make_project(db_session, toy_repo_remote, **overrides):
    defaults = {
        "name": f"pm-triage-{overrides.pop('_n', 'x')}",
        "overarching_goal": "goal",
        "repo_remote_url": str(toy_repo_remote),
    }
    defaults.update(overrides)
    project = await project_service.create_project(db_session, **defaults)
    wt_path = await worktree.ensure_tool_worktree(project, tool="curator")
    return project, wt_path


def _dispatcher(wt_path) -> ToolDispatcher:
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    return ToolDispatcher(ctx=ToolContext(card_id="curator-x", worktree_root=wt_path), executor=executor)


def _groom_call(call_id: str = "call_1", *, reprioritizations=None, duplicate_groups=None) -> ToolCallRequest:
    return ToolCallRequest(
        id=call_id,
        name="groom_backlog",
        arguments={
            "reprioritizations": reprioritizations or [],
            "duplicate_groups": duplicate_groups or [],
        },
    )


async def _events_for(db_session, card_id: str) -> list[CardEvent]:
    return await card_service.list_events(db_session, card_id)


# --- Happy path --------------------------------------------------------------


async def test_groom_backlog_reprioritizes_and_merges_duplicates(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="1")
    stale = await card_service.create_card(
        db_session, project.id, title="Important stale card", raw_request="r"
    )
    keep = await card_service.create_card(db_session, project.id, title="Fix the thing", raw_request="r")
    dup = await card_service.create_card(db_session, project.id, title="Also fix the thing", raw_request="r")
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    _groom_call(
                        reprioritizations=[
                            {"card_id": stale.id, "priority": "high", "reason": "sat untouched too long"}
                        ],
                        duplicate_groups=[
                            {
                                "keep_card_id": keep.id,
                                "duplicate_card_ids": [dup.id],
                                "reason": "same underlying request",
                            }
                        ],
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )

    summary = await run_pm_triage_pass(
        db_session,
        project,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=5,
        run_id="test-run",
    )

    assert "reprioritized 1" in summary
    assert "archived 1" in summary

    await db_session.refresh(stale)
    assert stale.priority == Priority.HIGH
    stale_events = await _events_for(db_session, stale.id)
    assert any(e.payload.get("action") == "pm_triage_reprioritized" for e in stale_events)

    await db_session.refresh(dup)
    assert dup.archived_at is not None
    dup_events = await _events_for(db_session, dup.id)
    assert any(
        e.payload.get("action") == "pm_triage_archived_duplicate" and e.payload.get("duplicate_of") == keep.id
        for e in dup_events
    )

    keep_events = await _events_for(db_session, keep.id)
    assert any(e.payload.get("action") == "pm_triage_merge_kept" for e in keep_events)
    await db_session.refresh(keep)
    assert keep.archived_at is None


async def test_empty_groom_backlog_is_a_legitimate_no_op(db_session, toy_repo_remote):
    """Unlike propose_tasks, groom_backlog isn't forced to always do something —
    "the backlog is fine" must be a valid, non-retried outcome."""
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="2")
    await card_service.create_card(db_session, project.id, title="c1", raw_request="r")
    await card_service.create_card(db_session, project.id, title="c2", raw_request="r")
    await db_session.commit()

    llm = ScriptedLLMClient(
        [LLMResult(content=None, tool_calls=[_groom_call()], endpoint_used="fake::model")]
    )

    summary = await run_pm_triage_pass(
        db_session,
        project,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=5,
        run_id="test-run",
    )

    assert summary == "no changes needed"


async def test_can_browse_the_repo_before_deciding(db_session, toy_repo_remote):
    """Unlike agents_md/retro, PM_TRIAGE gets the same read-only explore tools as
    every other curation kind — it can look at the actual code, not just titles,
    before deciding whether a card is still worth doing."""
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="1b")
    await card_service.create_card(db_session, project.id, title="c1", raw_request="r")
    await card_service.create_card(db_session, project.id, title="c2", raw_request="r")
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="read_file", arguments={"path": "app.py"})],
                endpoint_used="fake::model",
            ),
            LLMResult(content=None, tool_calls=[_groom_call()], endpoint_used="fake::model"),
        ]
    )

    summary = await run_pm_triage_pass(
        db_session,
        project,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=5,
        run_id="test-run",
    )

    assert summary == "no changes needed"
    events = (
        await db_session.scalars(
            select(CurationEvent).where(
                CurationEvent.project_id == project.id, CurationEvent.kind == ActivityKind.PM_TRIAGE
            )
        )
    ).all()
    read_events = [e for e in events if e.payload.get("name") == "read_file"]
    assert read_events and not read_events[0].payload["is_error"]


# --- Server-side validation: never trusts the model's own say-so -------------


async def test_a_card_outside_the_pm_column_is_never_touched(db_session, toy_repo_remote):
    """Even if the model somehow references a card that's moved past PM, it must
    be silently ignored — PM_TRIAGE's authority is scoped to Column.PM only."""
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="3")
    in_dev = await card_service.create_card(db_session, project.id, title="in dev", raw_request="r")
    in_dev.column = Column.DEVELOPER
    await db_session.commit()
    # A second PM card so the backlog looks realistic — irrelevant to this
    # assertion since we call run_pm_triage_pass directly, bypassing _needs_run.
    await card_service.create_card(db_session, project.id, title="pm card", raw_request="r")
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    _groom_call(
                        reprioritizations=[
                            {"card_id": in_dev.id, "priority": "high", "reason": "trying to sneak in"}
                        ]
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )

    summary = await run_pm_triage_pass(
        db_session,
        project,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=5,
        run_id="test-run",
    )

    assert summary == "no changes needed"
    await db_session.refresh(in_dev)
    assert in_dev.priority == Priority.NORMAL
    # The only event is the ordinary "created" note every card gets; no
    # pm_triage_reprioritized note was ever added.
    actions = [e.payload.get("action") for e in await _events_for(db_session, in_dev.id)]
    assert "pm_triage_reprioritized" not in actions


async def test_keep_card_cannot_be_its_own_duplicate(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="4")
    keep = await card_service.create_card(db_session, project.id, title="keep", raw_request="r")
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    _groom_call(
                        duplicate_groups=[
                            {"keep_card_id": keep.id, "duplicate_card_ids": [keep.id], "reason": "oops"}
                        ]
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )

    summary = await run_pm_triage_pass(
        db_session,
        project,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=5,
        run_id="test-run",
    )

    assert summary == "no changes needed"
    await db_session.refresh(keep)
    assert keep.archived_at is None


async def test_a_card_claimed_by_two_groups_is_only_archived_once(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="5")
    keep_a = await card_service.create_card(db_session, project.id, title="keep a", raw_request="r")
    keep_b = await card_service.create_card(db_session, project.id, title="keep b", raw_request="r")
    contested = await card_service.create_card(db_session, project.id, title="contested", raw_request="r")
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    _groom_call(
                        duplicate_groups=[
                            {
                                "keep_card_id": keep_a.id,
                                "duplicate_card_ids": [contested.id],
                                "reason": "dup of a",
                            },
                            {
                                "keep_card_id": keep_b.id,
                                "duplicate_card_ids": [contested.id],
                                "reason": "dup of b",
                            },
                        ]
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )

    summary = await run_pm_triage_pass(
        db_session,
        project,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=5,
        run_id="test-run",
    )

    assert "archived 1" in summary
    await db_session.refresh(contested)
    assert contested.archived_at is not None
    # Only the first group's event landed — the second group's claim on the same
    # card was dropped, not silently double-processed.
    events = await _events_for(db_session, contested.id)
    duplicate_of = [
        e.payload["duplicate_of"] for e in events if e.payload.get("action") == "pm_triage_archived_duplicate"
    ]
    assert duplicate_of == [keep_a.id]


async def test_malformed_entries_are_skipped_not_applied(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="6")
    card = await card_service.create_card(db_session, project.id, title="c", raw_request="r")
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    _groom_call(
                        reprioritizations=[
                            {"card_id": card.id, "priority": "urgent!!", "reason": "bad enum value"},
                            {"card_id": card.id, "priority": "high", "reason": ""},  # empty reason
                            {"card_id": "not-a-real-card", "priority": "high", "reason": "fake id"},
                        ]
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )

    summary = await run_pm_triage_pass(
        db_session,
        project,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=5,
        run_id="test-run",
    )

    assert summary == "no changes needed"
    await db_session.refresh(card)
    assert card.priority == Priority.NORMAL


async def test_no_op_reprioritization_is_not_applied_or_logged(db_session, toy_repo_remote):
    """Setting a card to the priority it already has shouldn't churn an event."""
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="7")
    card = await card_service.create_card(db_session, project.id, title="c", raw_request="r")
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    _groom_call(
                        reprioritizations=[
                            {"card_id": card.id, "priority": "normal", "reason": "already normal"}
                        ]
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )

    summary = await run_pm_triage_pass(
        db_session,
        project,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=5,
        run_id="test-run",
    )

    assert summary == "no changes needed"
    actions = [e.payload.get("action") for e in await _events_for(db_session, card.id)]
    assert "pm_triage_reprioritized" not in actions


# --- Pass mechanics: nudges, event logging, error handling --------------------


async def test_nudges_when_no_tool_call_then_recovers(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="8")
    await card_service.create_card(db_session, project.id, title="c", raw_request="r")
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(content="thinking...", tool_calls=[], endpoint_used="fake::model"),
            LLMResult(content=None, tool_calls=[_groom_call()], endpoint_used="fake::model"),
        ]
    )

    summary = await run_pm_triage_pass(
        db_session,
        project,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=5,
        run_id="test-run",
    )

    assert summary == "no changes needed"


async def test_gives_up_gracefully_when_iterations_exhausted(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="9")
    llm = ScriptedLLMClient(
        [LLMResult(content="still thinking", tool_calls=[], endpoint_used="fake::model") for _ in range(3)]
    )

    summary = await run_pm_triage_pass(
        db_session,
        project,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=3,
        run_id="test-run",
    )

    assert "gave up" in summary


async def test_logs_an_error_event_on_unhandled_failure(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="10")

    class _BoomLLM:
        async def complete(self, *, messages, tools):
            raise RuntimeError("endpoint unreachable")

    summary = await run_pm_triage_pass(
        db_session,
        project,
        llm_client=_BoomLLM(),
        dispatcher=_dispatcher(wt_path),
        max_iterations=5,
        run_id="test-run",
    )

    assert "errored" in summary


async def test_pass_logs_curation_events_for_the_board_panel(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="11")
    await card_service.create_card(db_session, project.id, title="c", raw_request="r")
    await db_session.commit()

    llm = ScriptedLLMClient(
        [LLMResult(content=None, tool_calls=[_groom_call()], endpoint_used="fake::model")]
    )

    await run_pm_triage_pass(
        db_session,
        project,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=5,
        run_id="test-run",
    )

    events = (
        await db_session.scalars(
            select(CurationEvent)
            .where(CurationEvent.project_id == project.id, CurationEvent.kind == ActivityKind.PM_TRIAGE)
            .order_by(CurationEvent.seq)
        )
    ).all()
    assert EventType.LLM_RESPONSE in [e.type for e in events]
    terminal = [e for e in events if e.payload.get("name") == "groom_backlog"]
    assert terminal and terminal[0].payload["result"] == "no changes needed"


# --- Scheduling: orchestrator/curator.py's _needs_run gating ------------------


async def test_needs_run_skips_pm_triage_below_two_cards(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="12")
    should_run, extra_context = await curator._needs_run(db_session, project, ActivityKind.PM_TRIAGE)
    assert should_run is False
    assert extra_context is None

    await card_service.create_card(db_session, project.id, title="only one", raw_request="r")
    await db_session.commit()
    should_run, _ = await curator._needs_run(db_session, project, ActivityKind.PM_TRIAGE)
    assert should_run is False


async def test_needs_run_pm_triage_ignores_the_pm_backlog_cap(db_session, toy_repo_remote, monkeypatch):
    """The WIP cap exists to stop *proposing* more PM work — the opposite of what
    PM_TRIAGE does, so it must run even when the backlog is well past the cap."""
    monkeypatch.setattr(curator.settings, "curator_max_pm_backlog", 2)
    project, _ = await _make_project(
        db_session, toy_repo_remote, _n="13", overseer_prompt="Investigate anything you find worth filing."
    )
    for i in range(5):
        await card_service.create_card(db_session, project.id, title=f"c{i}", raw_request="r")
    await db_session.commit()

    should_run, extra_context = await curator._needs_run(db_session, project, ActivityKind.PM_TRIAGE)

    assert should_run is True
    assert extra_context is not None

    # A propose-more kind is still correctly blocked by the same cap (not by a
    # missing overseer_prompt — this project has one set).
    should_run, _ = await curator._needs_run(db_session, project, ActivityKind.OVERSEER)
    assert should_run is False


async def test_needs_run_pm_triage_extra_context_is_recent_activity_not_the_backlog(
    db_session, toy_repo_remote
):
    """PM_TRIAGE's own pass re-reads the live PM backlog itself (run_pm_triage_pass)
    — _needs_run's extra_context is the *other* signal it needs handed in: recent
    visit outcomes and postmortems, the same material agents_md/retro get."""
    project, _ = await _make_project(db_session, toy_repo_remote, _n="13b")
    for i in range(2):
        await card_service.create_card(db_session, project.id, title=f"c{i}", raw_request="r")
    card = await card_service.create_card(db_session, project.id, title="finished thing", raw_request="r")
    await db_session.commit()
    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_pm_visit(
        db_session, card, visit, spec="s", acceptance_criteria=["x"], summary="wrote the spec"
    )
    await db_session.commit()

    should_run, extra_context = await curator._needs_run(db_session, project, ActivityKind.PM_TRIAGE)

    assert should_run is True
    assert extra_context is not None
    assert "finished thing" in extra_context
    assert "wrote the spec" in extra_context


async def test_needs_run_pm_triage_respects_its_own_interval(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="14")
    for i in range(2):
        await card_service.create_card(db_session, project.id, title=f"c{i}", raw_request="r")
    await db_session.commit()
    await project_service.record_activity_run(db_session, project.id, ActivityKind.PM_TRIAGE)

    should_run, _ = await curator._needs_run(db_session, project, ActivityKind.PM_TRIAGE)
    assert should_run is False

    curator._CURATION_INTERVALS[ActivityKind.PM_TRIAGE] = 0
    try:
        should_run, _ = await curator._needs_run(db_session, project, ActivityKind.PM_TRIAGE)
        assert should_run is True
    finally:
        curator._CURATION_INTERVALS[ActivityKind.PM_TRIAGE] = 900


# --- card_service.list_pm_backlog ---------------------------------------------


async def test_list_pm_backlog_excludes_non_pm_archived_terminal_and_epic_cards(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="15")
    pm_card = await card_service.create_card(db_session, project.id, title="pm", raw_request="r")
    dev_card = await card_service.create_card(db_session, project.id, title="dev", raw_request="r")
    dev_card.column = Column.DEVELOPER
    archived = await card_service.create_card(db_session, project.id, title="archived", raw_request="r")
    await card_service.archive_card(db_session, archived.id)
    done = await card_service.create_card(db_session, project.id, title="done", raw_request="r")
    done.lifecycle_state = LifecycleState.DONE
    epic = await card_service.create_card(db_session, project.id, title="epic", raw_request="r")
    child = await card_service.create_card(db_session, project.id, title="child", raw_request="r")
    await db_session.commit()
    await card_service.link_epic_child(db_session, parent_card_id=epic.id, child_card_id=child.id)
    await db_session.commit()

    backlog = await card_service.list_pm_backlog(db_session, project.id)

    ids = {c.id for c in backlog}
    assert ids == {pm_card.id, child.id}


async def test_list_pm_backlog_orders_oldest_first(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="16")
    first = await card_service.create_card(db_session, project.id, title="first", raw_request="r")
    second = await card_service.create_card(db_session, project.id, title="second", raw_request="r")
    await db_session.commit()

    backlog = await card_service.list_pm_backlog(db_session, project.id)

    assert [c.id for c in backlog] == [first.id, second.id]


# --- orchestrator/curator.py wiring --------------------------------------------


async def test_run_curation_activity_dispatches_to_pm_triage_pass(db_session, toy_repo_remote, monkeypatch):
    """kind == PM_TRIAGE must call run_pm_triage_pass (not run_curation_pass) and
    record its returned summary verbatim, same setup (worktree/dispatcher/
    agents_doc) as every other kind — proven by stubbing the pass function and
    checking record_activity_run picked up its exact summary string."""
    project, _ = await _make_project(db_session, toy_repo_remote, _n="17")
    await endpoint_service.create_endpoint_config(
        db_session,
        base_url="http://127.0.0.1:1",  # never actually called — run_pm_triage_pass is stubbed below
        model="fake-model",
        project_id=project.id,
        role=Column.PM,
    )
    await card_service.create_card(db_session, project.id, title="c1", raw_request="r")
    await card_service.create_card(db_session, project.id, title="c2", raw_request="r")
    await db_session.commit()

    async def _fake_pass(session, project, **kwargs):
        return "reprioritized 3 card(s)"

    monkeypatch.setattr(curator, "run_pm_triage_pass", _fake_pass)

    prior_logs = get_logs()
    cutoff = prior_logs[-1].seq if prior_logs else 0
    await curator.run_curation_activity(project.id, ActivityKind.PM_TRIAGE)
    new_logs = get_logs(since_seq=cutoff)

    assert any(
        "pm_triage" in e.message and "starting" in e.message and project.id in e.message for e in new_logs
    )
    last_run = await project_service.get_activity_last_run(db_session, project.id, ActivityKind.PM_TRIAGE)
    assert last_run is not None
    runs = await project_service.list_activity_runs(db_session, project.id)
    assert runs[ActivityKind.PM_TRIAGE].last_result_summary == "reprioritized 3 card(s)"
