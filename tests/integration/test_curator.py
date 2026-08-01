"""Curation (agent/curation.py + orchestrator/curator.py) against a real toy git
repo — LLM faked, everything else real. Every kind explores read-only and creates
new cards via propose_tasks; none of them ever edit the repo. Covers the shared
mechanics (using bug_sweep as the representative kind), the agents_md kind's
different shape (context from recent visit outcomes, not a repo browse), and the
orchestrator layer: cadence gating, pause-skipping, and the in-progress guard."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from built.agent import curation
from built.agent.curation import run_curation_pass
from built.db.models import CurationEvent
from built.domain import transitions
from built.domain.enums import ActivityKind, Column, EventType, LifecycleState, Severity
from built.domain.events import append_curation_event
from built.llm.client import LLMResult, ToolCallRequest
from built.llm.tool_schemas import MAX_PROPOSED_TASKS
from built.logging_config import get_logs
from built.main import app
from built.orchestrator import curator
from built.sandbox import worktree
from built.sandbox.container import CommandResult
from built.services import card_service, project_service
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher
from tests.unit.fakes import FakeCommandExecutor, ScriptedLLMClient


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _make_project(db_session, toy_repo_remote, **overrides):
    defaults = {
        "name": f"curator-{overrides.pop('_n', 'x')}",
        "overarching_goal": "Add basic arithmetic helpers to app.py.",
        "repo_remote_url": str(toy_repo_remote),
    }
    defaults.update(overrides)
    project = await project_service.create_project(db_session, **defaults)
    wt_path = await worktree.ensure_tool_worktree(project, tool="curator")
    return project, wt_path


def _dispatcher(wt_path) -> ToolDispatcher:
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    return ToolDispatcher(ctx=ToolContext(card_id="curator-x", worktree_root=wt_path), executor=executor)


# --- Shared mechanics (agent/curation.py), exercised via bug_sweep -----------------


async def test_curation_explores_then_proposes_tasks(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="1")

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="list_files", arguments={"path": "."})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_2", name="read_file", arguments={"path": "app.py"})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_3",
                        name="propose_tasks",
                        arguments={
                            "tasks": [
                                {
                                    "title": "Fix divide-by-zero",
                                    "severity": "high",
                                    "raw_request": "greet() crashes on None.",
                                },
                            ]
                        },
                    )
                ],
                endpoint_used="fake::model",
            ),
        ]
    )

    created = await run_curation_pass(
        db_session,
        project,
        ActivityKind.BUG_SWEEP,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=10,
    )

    assert [c.title for c in created] == ["Fix divide-by-zero"]
    assert created[0].column == Column.PM
    assert created[0].lifecycle_state == LifecycleState.ACTIVE
    events = await card_service.list_events(db_session, created[0].id)
    assert events[0].payload["source"] == "curation:bug_sweep"


async def test_curation_nudges_on_empty_tasks_then_recovers(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="2")

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="propose_tasks", arguments={"tasks": []})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_2",
                        name="propose_tasks",
                        arguments={
                            "tasks": [
                                {"title": "Add mod()", "severity": "high", "raw_request": "Add a mod(a, b)."}
                            ]
                        },
                    )
                ],
                endpoint_used="fake::model",
            ),
        ]
    )

    created = await run_curation_pass(
        db_session,
        project,
        ActivityKind.BUG_SWEEP,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=10,
    )

    assert [c.title for c in created] == ["Add mod()"]


async def test_curation_caps_at_max_proposed_tasks(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="3")

    too_many = [
        {"title": f"Task {i}", "severity": "critical", "raw_request": f"Do thing {i}."}
        for i in range(MAX_PROPOSED_TASKS + 5)
    ]
    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_1", name="propose_tasks", arguments={"tasks": too_many})
                ],
                endpoint_used="fake::model",
            )
        ]
    )

    created = await run_curation_pass(
        db_session,
        project,
        ActivityKind.BUG_SWEEP,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=10,
    )

    assert len(created) == MAX_PROPOSED_TASKS


async def test_curation_returns_empty_when_iterations_exhausted(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="4")

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id=f"call_{i}", name="list_files", arguments={"path": "."})],
                endpoint_used="fake::model",
            )
            for i in range(1, 5)
        ]
    )

    created = await run_curation_pass(
        db_session,
        project,
        ActivityKind.BUG_SWEEP,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=3,
    )

    assert created == []
    assert await card_service.list_cards(db_session, project.id) == []


async def test_curation_prompt_lists_existing_card_titles_to_avoid_duplicates(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="5")
    await card_service.create_card(db_session, project.id, title="Add subtract()", raw_request="r")

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="propose_tasks",
                        arguments={
                            "tasks": [{"title": "Add divide()", "severity": "medium", "raw_request": "r"}]
                        },
                    )
                ],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_2", name="report_duplicate_candidates", arguments={"duplicates": []}
                    )
                ],
                endpoint_used="fake::model",
            ),
        ]
    )

    await run_curation_pass(
        db_session,
        project,
        ActivityKind.OPPORTUNITY_BRAINSTORM,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=10,
    )

    sent_user_message = llm.calls[0]["messages"][1]["content"]
    assert "Add subtract()" in sent_user_message


async def test_curation_polish_review_proposes_a_card(db_session, toy_repo_remote):
    """Just proves the fourth read-only kind is wired correctly — full mechanics
    already covered above via bug_sweep."""
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="6")

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="propose_tasks",
                        arguments={"tasks": [{"title": "Consistent button labels", "raw_request": "r"}]},
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )

    created = await run_curation_pass(
        db_session,
        project,
        ActivityKind.POLISH_REVIEW,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=10,
    )

    events = await card_service.list_events(db_session, created[0].id)
    assert events[0].payload["source"] == "curation:polish_review"


@pytest.mark.parametrize(
    ("kind", "title"),
    [
        (ActivityKind.SECURITY_SWEEP, "Sanitize template input in report renderer"),
        (ActivityKind.COVERAGE_SWEEP, "Add test for empty-cart checkout path"),
        (ActivityKind.REFACTOR_SWEEP, "Split oversized handler in routes.py"),
    ],
)
async def test_curation_new_kinds_propose_a_card(db_session, toy_repo_remote, kind, title):
    """Proves the three new kinds (security_sweep, coverage_sweep, refactor_sweep)
    are wired correctly — full mechanics already covered above via bug_sweep."""
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n=f"6-{kind.value}")

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="propose_tasks",
                        arguments={"tasks": [{"title": title, "severity": "high", "raw_request": "r"}]},
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )

    created = await run_curation_pass(
        db_session,
        project,
        kind,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=10,
    )

    assert [c.title for c in created] == [title]
    events = await card_service.list_events(db_session, created[0].id)
    assert events[0].payload["source"] == f"curation:{kind.value}"


# --- agents_md kind: different context, proposes a card instead of editing --------


async def test_curation_agents_md_proposes_a_card_from_recent_outcomes(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="7")

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
                                    "title": "Update AGENTS.md: sandbox needs HOME=/tmp for npm",
                                    "raw_request": "Document the npm/HOME sandbox quirk in AGENTS.md.",
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
        ActivityKind.AGENTS_MD,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=10,
        extra_context="- [developer] Add tetris: sandbox needed HOME=/tmp for npm to work",
    )

    events = await card_service.list_events(db_session, created[0].id)
    assert events[0].payload["source"] == "curation:agents_md"
    assert not (wt_path / "AGENTS.md").exists()  # never edits the repo — only proposes
    sent_user_message = llm.calls[0]["messages"][1]["content"]
    assert "HOME=/tmp" in sent_user_message


# --- Severity-based whittling (agent/curation.py) ----------------------------------


def test_parsed_severity_handles_valid_invalid_and_missing():
    assert curation._parsed_severity({"severity": "critical"}) == Severity.CRITICAL
    assert curation._parsed_severity({"severity": "  HIGH  "}) == Severity.HIGH
    assert curation._parsed_severity({"severity": "urgent"}) is None
    assert curation._parsed_severity({}) is None


async def test_apply_file_policy_all_above_bar_keeps_only_critical_and_high(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="20")
    tasks = [
        {"title": "a", "severity": "critical", "raw_request": "r"},
        {"title": "b", "severity": "high", "raw_request": "r"},
        {"title": "c", "severity": "medium", "raw_request": "r"},
        {"title": "d", "severity": "low", "raw_request": "r"},
        {"title": "e", "raw_request": "r"},  # missing severity — treated as not clearing the bar
    ]
    kept = await curation._apply_file_policy(
        db_session, project, ActivityKind.BUG_SWEEP, tasks, "file_all_above_bar"
    )
    assert [t["title"] for t in kept] == ["a", "b"]


async def test_apply_file_policy_best_only_keeps_the_single_highest_severity(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="21")
    tasks = [
        {"title": "low-one", "severity": "low", "raw_request": "r"},
        {"title": "the-critical-one", "severity": "critical", "raw_request": "r"},
        {"title": "high-one", "severity": "high", "raw_request": "r"},
    ]
    kept = await curation._apply_file_policy(
        db_session, project, ActivityKind.POLISH_REVIEW, tasks, "file_best_only"
    )
    assert [t["title"] for t in kept] == ["the-critical-one"]


async def test_apply_file_policy_best_only_breaks_ties_by_original_order(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="22")
    tasks = [
        {"title": "first-high", "severity": "high", "raw_request": "r"},
        {"title": "second-high", "severity": "high", "raw_request": "r"},
    ]
    kept = await curation._apply_file_policy(
        db_session, project, ActivityKind.POLISH_REVIEW, tasks, "file_best_only"
    )
    assert [t["title"] for t in kept] == ["first-high"]


async def test_bug_sweep_files_every_candidate_above_the_bar(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="23")
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
                                {"title": "critical-bug", "severity": "critical", "raw_request": "r"},
                                {"title": "medium-bug", "severity": "medium", "raw_request": "r"},
                                {"title": "low-bug", "severity": "low", "raw_request": "r"},
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
        ActivityKind.BUG_SWEEP,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=10,
    )

    assert [c.title for c in created] == ["critical-bug"]


async def test_opportunity_brainstorm_files_only_the_highest_severity_candidate(db_session, toy_repo_remote):
    """Candidates are proposed out of severity order (critical isn't first) —
    proves ranking drives selection, not just truncation to the first item."""
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="24")
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
                                {"title": "low-one", "severity": "low", "raw_request": "r"},
                                {"title": "the-critical-one", "severity": "critical", "raw_request": "r"},
                                {"title": "high-one", "severity": "high", "raw_request": "r"},
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
        ActivityKind.OPPORTUNITY_BRAINSTORM,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=10,
    )

    assert [c.title for c in created] == ["the-critical-one"]


# --- Live-DB duplicate detection (agent/curation.py) -------------------------------


async def test_dedup_drops_a_candidate_flagged_as_duplicate(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="25")
    existing = await card_service.create_card(db_session, project.id, title="Existing card", raw_request="r")

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="propose_tasks",
                        arguments={
                            "tasks": [{"title": "Dup of existing", "severity": "high", "raw_request": "r"}]
                        },
                    )
                ],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_2",
                        name="report_duplicate_candidates",
                        arguments={
                            "duplicates": [{"candidate_index": 0, "duplicate_of_card_id": existing.id}]
                        },
                    )
                ],
                endpoint_used="fake::model",
            ),
        ]
    )

    created = await run_curation_pass(
        db_session,
        project,
        ActivityKind.BUG_SWEEP,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=10,
    )

    assert created == []
    events = (
        await db_session.scalars(
            select(CurationEvent)
            .where(CurationEvent.project_id == project.id, CurationEvent.kind == ActivityKind.BUG_SWEEP)
            .order_by(CurationEvent.seq)
        )
    ).all()
    dedup_events = [e for e in events if e.payload.get("stage") == "dedup"]
    assert dedup_events and dedup_events[0].payload["dropped"] == ["Dup of existing"]


async def test_dedup_skips_the_llm_call_when_there_are_no_live_open_cards(db_session, toy_repo_remote):
    """A fresh project has nothing to compare against, so the dedup call must never
    fire — only one response is scripted; if dedup fired anyway, ScriptedLLMClient
    would raise on the unscripted second call."""
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="26")
    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="propose_tasks",
                        arguments={"tasks": [{"title": "New idea", "severity": "high", "raw_request": "r"}]},
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )

    created = await run_curation_pass(
        db_session,
        project,
        ActivityKind.BUG_SWEEP,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=10,
    )

    assert [c.title for c in created] == ["New idea"]


async def test_dedup_fails_open_on_llm_error_so_real_work_isnt_lost(db_session, toy_repo_remote):
    """The dedup call's own failure (here: no second response scripted, so
    ScriptedLLMClient raises) must not lose an otherwise-valid candidate — it's
    logged as an ERROR event and the candidate still gets filed."""
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="27")
    await card_service.create_card(db_session, project.id, title="Some existing card", raw_request="r")

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="propose_tasks",
                        arguments={"tasks": [{"title": "New idea", "severity": "high", "raw_request": "r"}]},
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )

    created = await run_curation_pass(
        db_session,
        project,
        ActivityKind.BUG_SWEEP,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=10,
    )

    assert [c.title for c in created] == ["New idea"]
    events = (
        await db_session.scalars(
            select(CurationEvent)
            .where(CurationEvent.project_id == project.id, CurationEvent.kind == ActivityKind.BUG_SWEEP)
            .order_by(CurationEvent.seq)
        )
    ).all()
    error_events = [e for e in events if e.type == EventType.ERROR and e.payload.get("stage") == "dedup"]
    assert error_events


# --- Orchestrator layer: cadence, pause-skipping, in-progress guard ----------------


async def test_needs_run_agents_md_gated_by_new_visit_outcomes(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="8")

    should_run, extra_context = await curator._needs_run(db_session, project, ActivityKind.AGENTS_MD)
    assert should_run is False
    assert extra_context is None

    card = await card_service.create_card(db_session, project.id, title="c", raw_request="r")
    visit = await transitions.start_visit(db_session, card)
    await transitions.complete_pm_visit(
        db_session, card, visit, spec="s", acceptance_criteria=["x"], summary="s"
    )
    await db_session.commit()

    should_run, extra_context = await curator._needs_run(db_session, project, ActivityKind.AGENTS_MD)
    assert should_run is True
    assert extra_context is not None and "c" in extra_context


async def test_needs_run_bug_sweep_respects_the_global_interval(db_session, toy_repo_remote):
    """bug_sweep has no _CURATION_INTERVALS override (it equals the global default),
    so it's gated by curator_poll_interval_seconds since its last run — not
    immediately due again just because the scheduler woke up."""
    project, _ = await _make_project(db_session, toy_repo_remote, _n="9")

    should_run, _ = await curator._needs_run(db_session, project, ActivityKind.BUG_SWEEP)
    assert should_run is True

    await project_service.record_activity_run(db_session, project.id, ActivityKind.BUG_SWEEP)

    # Not due again immediately — the interval hasn't elapsed.
    should_run, _ = await curator._needs_run(db_session, project, ActivityKind.BUG_SWEEP)
    assert should_run is False

    # Due again once the interval is (effectively) zero.
    curator._CURATION_INTERVALS[ActivityKind.BUG_SWEEP] = 0
    try:
        should_run, _ = await curator._needs_run(db_session, project, ActivityKind.BUG_SWEEP)
        assert should_run is True
    finally:
        curator._CURATION_INTERVALS[ActivityKind.BUG_SWEEP] = 1800


async def test_needs_run_polish_review_uses_its_own_longer_interval(db_session, toy_repo_remote):
    """polish_review (a file_best_only, backlog-grooming kind) is configured with a
    longer interval than the global default — confirms _needs_run actually reads
    the per-kind entry rather than falling back to curator_poll_interval_seconds."""
    project, _ = await _make_project(db_session, toy_repo_remote, _n="9c")
    await project_service.record_activity_run(db_session, project.id, ActivityKind.POLISH_REVIEW)

    should_run, _ = await curator._needs_run(db_session, project, ActivityKind.POLISH_REVIEW)
    assert should_run is False

    curator._CURATION_INTERVALS[ActivityKind.POLISH_REVIEW] = 0
    try:
        should_run, _ = await curator._needs_run(db_session, project, ActivityKind.POLISH_REVIEW)
        assert should_run is True
    finally:
        curator._CURATION_INTERVALS[ActivityKind.POLISH_REVIEW] = 21600


async def test_needs_run_blocked_once_pm_backlog_hits_the_cap(db_session, toy_repo_remote, monkeypatch):
    """The WIP limit: with no per-kind cooldown, curation would otherwise keep
    piling cards into PM faster than a concurrency-capped orchestrator can work
    through them. Applies to every kind, including agents_md."""
    monkeypatch.setattr(curator.settings, "curator_max_pm_backlog", 2)
    project, _ = await _make_project(db_session, toy_repo_remote, _n="9b")
    for i in range(2):
        await card_service.create_card(db_session, project.id, title=f"c{i}", raw_request="r")
    await db_session.commit()

    should_run, _ = await curator._needs_run(db_session, project, ActivityKind.BUG_SWEEP)
    assert should_run is False
    should_run, _ = await curator._needs_run(db_session, project, ActivityKind.AGENTS_MD)
    assert should_run is False


async def test_needs_run_allowed_below_the_pm_backlog_cap(db_session, toy_repo_remote, monkeypatch):
    monkeypatch.setattr(curator.settings, "curator_max_pm_backlog", 2)
    project, _ = await _make_project(db_session, toy_repo_remote, _n="9c")
    await card_service.create_card(db_session, project.id, title="c0", raw_request="r")
    await db_session.commit()

    should_run, _ = await curator._needs_run(db_session, project, ActivityKind.BUG_SWEEP)
    assert should_run is True


async def test_count_column_backlog_ignores_other_columns_and_archived_cards(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="9d")
    pm_card = await card_service.create_card(db_session, project.id, title="pm", raw_request="r")
    archived_pm_card = await card_service.create_card(
        db_session, project.id, title="archived", raw_request="r"
    )
    await card_service.archive_card(db_session, archived_pm_card.id)
    other_column_card = await card_service.create_card(db_session, project.id, title="dev", raw_request="r")
    other_column_card.column = Column.DEVELOPER
    await db_session.commit()

    count = await card_service.count_column_backlog(db_session, project.id, Column.PM)

    assert count == 1
    assert pm_card.column == Column.PM


async def test_count_column_backlog_excludes_epic_parent_cards(db_session, toy_repo_remote):
    """An epic parent sits column=PM forever (services/card_service.py's
    link_epic_child) — without this exclusion it would inflate the WIP gate
    permanently, one unit per surviving epic."""
    project, _ = await _make_project(db_session, toy_repo_remote, _n="9e")
    await card_service.create_card(db_session, project.id, title="pm", raw_request="r")
    epic = await card_service.create_card(db_session, project.id, title="epic", raw_request="r")
    child = await card_service.create_card(db_session, project.id, title="child", raw_request="r")
    other_child = await card_service.create_card(db_session, project.id, title="other", raw_request="r")
    await db_session.commit()
    await card_service.link_epic_child(db_session, parent_card_id=epic.id, child_card_id=child.id)
    await card_service.link_epic_child(db_session, parent_card_id=epic.id, child_card_id=other_child.id)
    await db_session.commit()

    count = await card_service.count_column_backlog(db_session, project.id, Column.PM)

    # pm_card + the two children — epic itself excluded.
    assert count == 3


async def test_run_curator_once_skips_a_paused_project(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="10")
    await project_service.pause_project(db_session, project.id)
    await db_session.commit()

    await curator.run_curator_once()

    for kind in ActivityKind:
        assert await project_service.get_activity_last_run(db_session, project.id, kind) is None


async def test_run_curator_once_skips_curation_when_curation_paused(db_session, toy_repo_remote):
    """pause_curation is a narrower switch than pause_project: it stops only the
    curator's automatic scheduler, not the worker orchestrator or Reviver."""
    project, _ = await _make_project(db_session, toy_repo_remote, _n="10b")
    await project_service.pause_curation(db_session, project.id)
    await db_session.commit()
    assert project.paused_at is None

    await curator.run_curator_once()

    for kind in ActivityKind:
        assert await project_service.get_activity_last_run(db_session, project.id, kind) is None


async def test_run_curator_once_skips_a_disabled_kind_but_attempts_others(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="10c")
    await project_service.set_curation_kind_enabled(
        db_session, project.id, ActivityKind.BUG_SWEEP, enabled=False
    )
    await db_session.commit()

    prior_logs = get_logs()
    cutoff = prior_logs[-1].seq if prior_logs else 0
    await curator.run_curator_once()
    new_logs = get_logs(since_seq=cutoff)

    def _started(kind: ActivityKind) -> bool:
        return any(
            "starting" in e.message and kind.value in e.message and project.id in e.message
            for e in new_logs
        )

    assert _started(ActivityKind.BUG_SWEEP) is False
    assert _started(ActivityKind.OPPORTUNITY_BRAINSTORM) is True


# --- Curation on/off state: project-wide pause + per-kind enable/disable ----------


async def test_curation_state_absent_by_default(db_session, toy_repo_remote):
    """No row == fully active, nothing paused or disabled — see
    ProjectCurationState's docstring."""
    project, _ = await _make_project(db_session, toy_repo_remote, _n="10a")
    assert await project_service.get_curation_state(db_session, project.id) is None


async def test_pause_curation_and_resume_curation_round_trip(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="10d")

    paused = await project_service.pause_curation(db_session, project.id)
    assert paused.paused_at is not None
    state = await project_service.get_curation_state(db_session, project.id)
    assert state is not None and state.paused_at is not None

    resumed = await project_service.resume_curation(db_session, project.id)
    assert resumed.paused_at is None


async def test_set_curation_kind_enabled_toggles_independently(db_session, toy_repo_remote):
    """Disabling one kind never touches another's state, and re-enabling a kind
    removes it from disabled_kinds rather than leaving a stale falsy entry."""
    project, _ = await _make_project(db_session, toy_repo_remote, _n="10e")

    await project_service.set_curation_kind_enabled(
        db_session, project.id, ActivityKind.BUG_SWEEP, enabled=False
    )
    state = await project_service.get_curation_state(db_session, project.id)
    assert state.disabled_kinds == ["bug_sweep"]

    await project_service.set_curation_kind_enabled(
        db_session, project.id, ActivityKind.POLISH_REVIEW, enabled=False
    )
    state = await project_service.get_curation_state(db_session, project.id)
    assert set(state.disabled_kinds) == {"bug_sweep", "polish_review"}

    await project_service.set_curation_kind_enabled(
        db_session, project.id, ActivityKind.BUG_SWEEP, enabled=True
    )
    state = await project_service.get_curation_state(db_session, project.id)
    assert state.disabled_kinds == ["polish_review"]


async def test_in_progress_guard_blocks_same_kind_but_not_different_kind():
    """Unlike the old single-project discovery guard, this one is keyed by
    (project_id, kind) — a bug sweep and a polish review for the same project are
    independent and can run concurrently."""
    project_id = "proj-x"
    curator._curation_in_progress.clear()
    curator._curation_in_progress.add((project_id, ActivityKind.BUG_SWEEP))

    assert curator.is_curation_running(project_id, ActivityKind.BUG_SWEEP) is True
    assert curator.is_curation_running(project_id, ActivityKind.POLISH_REVIEW) is False

    curator._curation_in_progress.clear()


async def test_curation_skips_outright_if_already_marked_in_progress(db_session, toy_repo_remote):
    """Curation doesn't go through claim_next_card, so it isn't covered by
    per-project claim serialization — a separate in-memory guard prevents two runs
    for the same (project, kind) racing each other and proposing near-duplicate
    cards."""
    project, _ = await _make_project(db_session, toy_repo_remote, _n="11")

    curator._curation_in_progress.add((project.id, ActivityKind.BUG_SWEEP))
    try:
        assert curator.is_curation_running(project.id, ActivityKind.BUG_SWEEP) is True
        await curator.run_curation_activity(project.id, ActivityKind.BUG_SWEEP)  # should no-op immediately
    finally:
        curator._curation_in_progress.discard((project.id, ActivityKind.BUG_SWEEP))

    assert await card_service.list_cards(db_session, project.id) == []


async def test_curation_releases_the_guard_even_on_setup_failure(db_session):
    project = await project_service.create_project(
        db_session,
        name="curator-setup-fail",
        overarching_goal="g",
        repo_remote_url="/nonexistent/path/repo.git",
    )
    # run_curation_activity opens its own session (async_session_factory), separate
    # from db_session — without committing, it can't see this project at all and
    # exercises the "missing project" path instead of the setup failure this test
    # is actually meant to cover.
    await db_session.commit()

    prior_logs = get_logs()
    cutoff = prior_logs[-1].seq if prior_logs else 0
    await curator.run_curation_activity(project.id, ActivityKind.BUG_SWEEP)

    assert curator.is_curation_running(project.id, ActivityKind.BUG_SWEEP) is False

    # The "starting" log fires before setup — this is task-lifecycle visibility,
    # not conditional on the pass actually succeeding.
    new_logs = get_logs(since_seq=cutoff)
    assert any(
        "bug_sweep" in e.message and "starting" in e.message and project.id in e.message for e in new_logs
    )


# --- list_activity_runs: what the board page.s curation dropdown reads ------------


async def test_list_activity_runs_keyed_by_kind_missing_kinds_absent(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="12")

    assert await project_service.list_activity_runs(db_session, project.id) == {}

    await project_service.record_activity_run(
        db_session, project.id, ActivityKind.BUG_SWEEP, summary="created 1 card(s)"
    )

    runs = await project_service.list_activity_runs(db_session, project.id)
    assert set(runs) == {ActivityKind.BUG_SWEEP}
    assert runs[ActivityKind.BUG_SWEEP].last_result_summary == "created 1 card(s)"


# --- Curation dropdown: per-kind status readout, folded in alongside the toggles --


async def test_curation_dropdown_shows_per_kind_status(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="13")
    await db_session.commit()

    async with _client() as client:
        before = await client.get(f"/ui/projects/{project.id}/board")
        assert "Bug sweep" in before.text
        assert "never run yet" in before.text

    await project_service.record_activity_run(
        db_session, project.id, ActivityKind.BUG_SWEEP, summary="created 2 card(s)"
    )
    await db_session.commit()
    curator._curation_in_progress.add((project.id, ActivityKind.OPPORTUNITY_BRAINSTORM))
    try:
        async with _client() as client:
            after = await client.get(f"/ui/projects/{project.id}/board")
        assert "created 2 card(s)" in after.text
        assert 'title="Running now"' in after.text
        # No CurationEvent logged yet for the running kind — falls back to "starting…".
        assert "starting…" in after.text
    finally:
        curator._curation_in_progress.discard((project.id, ActivityKind.OPPORTUNITY_BRAINSTORM))


async def test_curation_dropdown_shows_the_latest_event_while_running(db_session, toy_repo_remote):
    """A running kind's readout in the dropdown shows what it's doing right now,
    not just a bare spinner."""
    project, _ = await _make_project(db_session, toy_repo_remote, _n="14")
    await append_curation_event(
        db_session,
        project_id=project.id,
        kind=ActivityKind.BUG_SWEEP,
        type=EventType.TOOL_CALL,
        payload={"name": "read_file", "arguments": {"path": "app.py"}, "result": "...", "is_error": False},
    )
    await db_session.commit()

    curator._curation_in_progress.add((project.id, ActivityKind.BUG_SWEEP))
    try:
        async with _client() as client:
            page = await client.get(f"/ui/projects/{project.id}/board")
        assert "read_file(app.py)" in page.text
    finally:
        curator._curation_in_progress.discard((project.id, ActivityKind.BUG_SWEEP))


async def test_board_curation_pause_resume_and_kind_toggle_round_trip(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="17")
    await db_session.commit()

    async with _client() as client:
        before = await client.get(f"/ui/projects/{project.id}/board")
        assert f'action="/ui/projects/{project.id}/curation/pause"' in before.text
        assert f'action="/ui/projects/{project.id}/curation/kinds/bug_sweep"' in before.text

        pause_resp = await client.post(f"/ui/projects/{project.id}/curation/pause", follow_redirects=False)
        assert pause_resp.status_code == 303

        after_pause = await client.get(f"/ui/projects/{project.id}/board")
        assert "Automatic curation paused" in after_pause.text
        assert f'action="/ui/projects/{project.id}/curation/resume"' in after_pause.text

        resume_resp = await client.post(
            f"/ui/projects/{project.id}/curation/resume", follow_redirects=False
        )
        assert resume_resp.status_code == 303

        # Unchecking a checkbox omits its field entirely — an empty body, not "false".
        uncheck_resp = await client.post(
            f"/ui/projects/{project.id}/curation/kinds/bug_sweep", data={}, follow_redirects=False
        )
        assert uncheck_resp.status_code == 303

        # Checked via the rendered page, not a direct DB read: db_session's identity
        # map would otherwise happily hand back a stale ProjectCurationState object
        # loaded before these client-session commits (see get_curation_state's
        # session.get(), which checks the identity map before querying).
        after_uncheck = await client.get(f"/ui/projects/{project.id}/board")
        assert 'id="curation-enabled-bug_sweep" name="enabled" value="true" checked' not in after_uncheck.text
        assert 'id="curation-enabled-bug_sweep"' in after_uncheck.text
        assert "off — not on the automatic schedule" in after_uncheck.text

        recheck_resp = await client.post(
            f"/ui/projects/{project.id}/curation/kinds/bug_sweep",
            data={"enabled": "true"},
            follow_redirects=False,
        )
        assert recheck_resp.status_code == 303

        after_recheck = await client.get(f"/ui/projects/{project.id}/board")
        assert 'id="curation-enabled-bug_sweep" name="enabled" value="true" checked' in after_recheck.text


async def test_curation_toggle_over_htmx_swaps_the_panel_instead_of_redirecting(
    db_session, toy_repo_remote
):
    """The curation modal's checkbox submits via htmx (see _curation_panel.html.j2's
    onchange="this.form.requestSubmit()") — a 303 redirect would force a full page
    reload and close the Bootstrap modal, since its open/closed state lives only in
    client-side JS/DOM, not anything the server re-renders. An htmx request must get
    the refreshed panel fragment back, not a redirect."""
    project, _ = await _make_project(db_session, toy_repo_remote, _n="18")
    await db_session.commit()

    async with _client() as client:
        resp = await client.post(
            f"/ui/projects/{project.id}/curation/kinds/bug_sweep",
            data={},
            headers={"HX-Request": "true"},
            follow_redirects=False,
        )

    assert resp.status_code == 200
    assert resp.text.strip().startswith('<div id="curation-panel">')
    assert "off — not on the automatic schedule" in resp.text
    # A fragment, not a full page — none of the surrounding board chrome leaked in.
    assert "<nav" not in resp.text


# --- Event logging: what a pass writes for the curation dropdown to read ----------


async def test_curation_pass_logs_events_for_llm_responses_and_tool_calls(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="15")

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="read_file", arguments={"path": "app.py"})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_2",
                        name="propose_tasks",
                        arguments={"tasks": [{"title": "t", "severity": "high", "raw_request": "r"}]},
                    )
                ],
                endpoint_used="fake::model",
            ),
        ]
    )

    await run_curation_pass(
        db_session,
        project,
        ActivityKind.BUG_SWEEP,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=10,
    )

    events = (
        await db_session.scalars(
            select(CurationEvent)
            .where(CurationEvent.project_id == project.id, CurationEvent.kind == ActivityKind.BUG_SWEEP)
            .order_by(CurationEvent.seq)
        )
    ).all()

    assert EventType.LLM_RESPONSE in [e.type for e in events]
    tool_call_events = [e for e in events if e.payload.get("name") == "read_file"]
    assert tool_call_events, events
    terminal_events = [e for e in events if e.payload.get("name") == "propose_tasks"]
    assert terminal_events and terminal_events[0].payload["result"] == "created 1 card(s)"


async def test_curation_pass_logs_an_error_event_on_unhandled_failure(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="16")

    class _BoomLLM:
        async def complete(self, *, messages, tools):
            raise RuntimeError("endpoint unreachable")

    created = await run_curation_pass(
        db_session,
        project,
        ActivityKind.BUG_SWEEP,
        llm_client=_BoomLLM(),
        dispatcher=_dispatcher(wt_path),
        max_iterations=10,
    )

    assert created == []
    latest = await project_service.get_latest_curation_event(db_session, project.id, ActivityKind.BUG_SWEEP)
    assert latest is not None
    assert "endpoint unreachable" in latest.payload["error"]
