"""CurationRun (db/models.py) — the per-invocation history record that lets a
past curation/pm_triage pass be inspected after the fact instead of only the
single latest CurationEvent. Covers the service-layer lifecycle
(start/finish/list/get), orchestrator/curator.py's wiring of it into
run_curation_activity (including the setup-failure path), and the two new UI
pages that render it."""

from httpx import ASGITransport, AsyncClient

from built.agent.curation import run_curation_pass
from built.domain.enums import ActivityKind, Column, CurationRunOutcome, EventType
from built.domain.events import append_curation_event
from built.llm.client import LLMResult, ToolCallRequest
from built.main import app
from built.orchestrator import curator
from built.sandbox import worktree
from built.sandbox.container import CommandResult
from built.services import endpoint_service, project_service
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher
from tests.unit.fakes import FakeCommandExecutor, ScriptedLLMClient


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _make_project(db_session, toy_repo_remote, **overrides):
    defaults = {
        "name": f"curation-runs-{overrides.pop('_n', 'x')}",
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


# --- Service layer: start/finish/list/get -------------------------------------


async def test_start_curation_run_opens_with_no_outcome(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="1")

    run = await project_service.start_curation_run(db_session, project.id, ActivityKind.BUG_SWEEP)

    assert run.started_at is not None
    assert run.ended_at is None
    assert run.outcome is None


async def test_finish_curation_run_sums_token_usage_from_its_own_events(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="2")
    run = await project_service.start_curation_run(db_session, project.id, ActivityKind.BUG_SWEEP)
    await append_curation_event(
        db_session,
        project_id=project.id,
        kind=ActivityKind.BUG_SWEEP,
        run_id=run.id,
        type=EventType.LLM_RESPONSE,
        payload={},
        tokens_in=100,
        tokens_out=20,
    )
    await append_curation_event(
        db_session,
        project_id=project.id,
        kind=ActivityKind.BUG_SWEEP,
        run_id=run.id,
        type=EventType.LLM_RESPONSE,
        payload={},
        tokens_in=50,
        tokens_out=10,
    )
    await db_session.commit()

    finished = await project_service.finish_curation_run(db_session, run.id, summary="created 1 card(s)")

    assert finished.ended_at is not None
    assert finished.tokens_in == 150
    assert finished.tokens_out == 30
    assert finished.outcome == CurationRunOutcome.OK
    assert finished.summary == "created 1 card(s)"


async def test_finish_curation_run_detects_error_regardless_of_summary_text(db_session, toy_repo_remote):
    """An ERROR-type event always wins the outcome, even if the summary text
    happens to look success-shaped — the event is ground truth, the summary
    string is not."""
    project, _ = await _make_project(db_session, toy_repo_remote, _n="3")
    run = await project_service.start_curation_run(db_session, project.id, ActivityKind.BUG_SWEEP)
    await append_curation_event(
        db_session,
        project_id=project.id,
        kind=ActivityKind.BUG_SWEEP,
        run_id=run.id,
        type=EventType.ERROR,
        payload={"error": "boom"},
    )
    await db_session.commit()

    finished = await project_service.finish_curation_run(db_session, run.id, summary="created 3 card(s)")

    assert finished.outcome == CurationRunOutcome.ERROR


async def test_finish_curation_run_outcome_from_summary_text(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="4")

    async def _finish(summary: str) -> CurationRunOutcome:
        run = await project_service.start_curation_run(db_session, project.id, ActivityKind.PM_TRIAGE)
        await db_session.commit()
        finished = await project_service.finish_curation_run(db_session, run.id, summary=summary)
        return finished.outcome

    assert await _finish("created 0 card(s)") == CurationRunOutcome.NO_CHANGE
    assert await _finish("no changes needed") == CurationRunOutcome.NO_CHANGE
    assert await _finish("no changes — gave up without calling groom_backlog") == CurationRunOutcome.GAVE_UP
    assert await _finish("reprioritized 2 card(s)") == CurationRunOutcome.OK
    assert await _finish("created 2 card(s)") == CurationRunOutcome.OK


async def test_list_curation_runs_orders_newest_first(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="5")
    first = await project_service.start_curation_run(db_session, project.id, ActivityKind.BUG_SWEEP)
    await db_session.commit()
    second = await project_service.start_curation_run(db_session, project.id, ActivityKind.BUG_SWEEP)
    await db_session.commit()

    runs = await project_service.list_curation_runs(db_session, project.id, ActivityKind.BUG_SWEEP)

    assert [r.id for r in runs] == [second.id, first.id]


async def test_list_curation_runs_scoped_to_project_and_kind(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="6")
    other_project, _ = await _make_project(db_session, toy_repo_remote, _n="6b")
    await project_service.start_curation_run(db_session, project.id, ActivityKind.BUG_SWEEP)
    await project_service.start_curation_run(db_session, project.id, ActivityKind.POLISH_REVIEW)
    await project_service.start_curation_run(db_session, other_project.id, ActivityKind.BUG_SWEEP)
    await db_session.commit()

    runs = await project_service.list_curation_runs(db_session, project.id, ActivityKind.BUG_SWEEP)

    assert len(runs) == 1


async def test_get_curation_run_eager_loads_its_events(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="7")
    run = await project_service.start_curation_run(db_session, project.id, ActivityKind.BUG_SWEEP)
    await append_curation_event(
        db_session,
        project_id=project.id,
        kind=ActivityKind.BUG_SWEEP,
        run_id=run.id,
        type=EventType.LLM_RESPONSE,
        payload={"content": "thinking"},
    )
    await db_session.commit()

    fetched = await project_service.get_curation_run(db_session, run.id)

    assert len(fetched.events) == 1
    assert fetched.events[0].payload["content"] == "thinking"


async def test_get_curation_run_404s_for_missing_run(db_session):
    try:
        await project_service.get_curation_run(db_session, "does-not-exist")
        raise AssertionError("expected NotFoundError")
    except project_service.NotFoundError:
        pass


# --- orchestrator/curator.py wiring -------------------------------------------


async def test_run_curation_activity_creates_a_finished_ok_run(db_session, toy_repo_remote, monkeypatch):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="8")
    await endpoint_service.create_endpoint_config(
        db_session,
        base_url="http://127.0.0.1:1",  # never actually called — run_curation_pass is stubbed below
        model="fake-model",
        project_id=project.id,
        role=Column.PM,
    )
    await db_session.commit()

    async def _fake_run_curation_pass(session, project, kind, *, run_id, **kwargs):
        await append_curation_event(
            session,
            project_id=project.id,
            kind=kind,
            run_id=run_id,
            type=EventType.LLM_RESPONSE,
            payload={},
            tokens_in=42,
            tokens_out=7,
        )
        await session.commit()
        return ["fake-card"]

    monkeypatch.setattr(curator, "run_curation_pass", _fake_run_curation_pass)

    await curator.run_curation_activity(project.id, ActivityKind.BUG_SWEEP)

    runs = await project_service.list_curation_runs(db_session, project.id, ActivityKind.BUG_SWEEP)
    assert len(runs) == 1
    run = runs[0]
    assert run.ended_at is not None
    assert run.outcome == CurationRunOutcome.OK
    assert run.summary == "created 1 card(s)"
    assert run.tokens_in == 42
    assert run.tokens_out == 7


async def test_run_curation_activity_setup_failure_records_an_error_run(db_session):
    """A bad git remote (or any setup failure) must still close out a CurationRun
    as ERROR — not leave the row open forever with no explanation, and not skip
    creating one at all."""
    project = await project_service.create_project(
        db_session,
        name="curation-runs-setup-fail",
        overarching_goal="g",
        repo_remote_url="/nonexistent/path/repo.git",
    )
    await db_session.commit()

    await curator.run_curation_activity(project.id, ActivityKind.BUG_SWEEP)

    runs = await project_service.list_curation_runs(db_session, project.id, ActivityKind.BUG_SWEEP)
    assert len(runs) == 1
    assert runs[0].ended_at is not None
    assert runs[0].outcome == CurationRunOutcome.ERROR


async def test_run_curation_pass_tags_every_event_with_the_run_id(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="9")
    run = await project_service.start_curation_run(db_session, project.id, ActivityKind.BUG_SWEEP)
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="propose_tasks",
                        arguments={"tasks": [{"title": "t", "severity": "high", "raw_request": "r"}]},
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )

    await run_curation_pass(
        db_session,
        project,
        ActivityKind.BUG_SWEEP,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_iterations=5,
        run_id=run.id,
    )

    fetched = await project_service.get_curation_run(db_session, run.id)
    assert len(fetched.events) >= 1
    assert all(e.run_id == run.id for e in fetched.events)


# --- UI: history + detail pages ------------------------------------------------


async def test_curation_kind_history_page_renders_runs(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="10")
    run = await project_service.start_curation_run(db_session, project.id, ActivityKind.BUG_SWEEP)
    await project_service.finish_curation_run(db_session, run.id, summary="created 2 card(s)")
    await db_session.commit()

    async with _client() as client:
        resp = await client.get(f"/ui/projects/{project.id}/curation/bug_sweep")

    assert resp.status_code == 200
    assert "created 2 card(s)" in resp.text
    assert "Bug sweep" in resp.text


async def test_curation_run_detail_page_renders_transcript(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="11")
    run = await project_service.start_curation_run(db_session, project.id, ActivityKind.BUG_SWEEP)
    await append_curation_event(
        db_session,
        project_id=project.id,
        kind=ActivityKind.BUG_SWEEP,
        run_id=run.id,
        type=EventType.LLM_RESPONSE,
        payload={"content": "looked at app.py and found nothing worth filing"},
    )
    await project_service.finish_curation_run(db_session, run.id, summary="created 0 card(s)")
    await db_session.commit()

    async with _client() as client:
        resp = await client.get(f"/ui/projects/{project.id}/curation/bug_sweep/runs/{run.id}")

    assert resp.status_code == 200
    assert "looked at app.py and found nothing worth filing" in resp.text


async def test_orphaned_run_shows_as_interrupted_not_running(db_session, toy_repo_remote):
    """A run whose process crashed mid-pass never gets ended_at/outcome set — with
    is_curation_running now false for it (nothing in the in-memory guard, since
    that's lost on restart too), the UI must call it interrupted, not running
    forever."""
    project, _ = await _make_project(db_session, toy_repo_remote, _n="12")
    await project_service.start_curation_run(db_session, project.id, ActivityKind.BUG_SWEEP)
    await db_session.commit()
    assert curator.is_curation_running(project.id, ActivityKind.BUG_SWEEP) is False

    async with _client() as client:
        resp = await client.get(f"/ui/projects/{project.id}/curation/bug_sweep")

    assert resp.status_code == 200
    assert "interrupted" in resp.text
    assert "running" not in resp.text.lower()


async def test_curation_panel_links_to_history_page(db_session, toy_repo_remote):
    project, _ = await _make_project(db_session, toy_repo_remote, _n="13")
    await db_session.commit()

    async with _client() as client:
        resp = await client.get(f"/ui/projects/{project.id}/board")

    assert f'/ui/projects/{project.id}/curation/bug_sweep"' in resp.text
