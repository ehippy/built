"""Curation (agent/curation.py + orchestrator/curator.py) against a real toy git
repo — LLM faked, everything else real. Every kind explores read-only and creates
new cards via propose_tasks; none of them ever edit the repo. Covers the shared
mechanics (using bug_sweep as the representative kind), the agents_md kind's
different shape (context from recent visit outcomes, not a repo browse), and the
orchestrator layer: cadence gating, pause-skipping, and the in-progress guard."""

from built.agent.curation import run_curation_pass
from built.domain import transitions
from built.domain.enums import ActivityKind, Column, LifecycleState
from built.llm.client import LLMResult, ToolCallRequest
from built.llm.tool_schemas import MAX_PROPOSED_TASKS
from built.orchestrator import curator
from built.sandbox import worktree
from built.sandbox.container import CommandResult
from built.services import card_service, project_service
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher
from tests.unit.fakes import FakeCommandExecutor, ScriptedLLMClient


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
                                {"title": "Fix divide-by-zero", "raw_request": "greet() crashes on None."},
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
                        arguments={"tasks": [{"title": "Add mod()", "raw_request": "Add a mod(a, b)."}]},
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
        {"title": f"Task {i}", "raw_request": f"Do thing {i}."} for i in range(MAX_PROPOSED_TASKS + 5)
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
                        arguments={"tasks": [{"title": "Add divide()", "raw_request": "r"}]},
                    )
                ],
                endpoint_used="fake::model",
            )
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


async def test_needs_run_explore_kinds_gated_by_flat_cadence(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="9")

    # Never run before — due immediately.
    should_run, _ = await curator._needs_run(db_session, project, ActivityKind.BUG_SWEEP)
    assert should_run is True

    await project_service.record_activity_run(db_session, project.id, ActivityKind.BUG_SWEEP)

    # Just ran — not due again yet.
    should_run, _ = await curator._needs_run(db_session, project, ActivityKind.BUG_SWEEP)
    assert should_run is False


async def test_run_curator_once_skips_a_paused_project(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="10")
    await project_service.pause_project(db_session, project.id)
    await db_session.commit()

    await curator.run_curator_once()

    for kind in ActivityKind:
        assert await project_service.get_activity_last_run(db_session, project.id, kind) is None


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

    await curator.run_curation_activity(project.id, ActivityKind.BUG_SWEEP)

    assert curator.is_curation_running(project.id, ActivityKind.BUG_SWEEP) is False
