import asyncio

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse, Response

from built.api.deps import SessionDep
from built.db.models import CurationEvent
from built.domain.enums import ActivityKind, Column, EventType, Priority
from built.orchestrator.curator import is_curation_running, run_curation_activity
from built.services import card_service, project_service
from built.ui.templates import templates, tool_descriptor

router = APIRouter(prefix="/ui", tags=["ui"])

_CURATION_LABELS: dict[ActivityKind, str] = {
    ActivityKind.BUG_SWEEP: "Bug sweep",
    ActivityKind.OPPORTUNITY_BRAINSTORM: "Opportunities",
    ActivityKind.POLISH_REVIEW: "Polish review",
    ActivityKind.STAY_DRY: "Duplication sweep",
    ActivityKind.SECURITY_SWEEP: "Security sweep",
    ActivityKind.COVERAGE_SWEEP: "Coverage sweep",
    ActivityKind.REFACTOR_SWEEP: "Refactor sweep",
    ActivityKind.AGENTS_MD: "Tend AGENTS.md",
    ActivityKind.RETRO: "Retro",
    ActivityKind.PM_TRIAGE: "PM triage",
}


def _describe_curation_event(event: CurationEvent) -> str:
    """One short line for the status panel's live "doing: ..." display — not the
    full recap format card_service uses for retry context, just enough to show
    something is actually happening."""
    if event.type == EventType.ERROR:
        return f"error: {event.payload.get('error', '?')}"
    name = event.payload.get("name")
    if name:
        if name == "propose_tasks":
            return str(event.payload.get("result") or "propose_tasks")
        descriptor = tool_descriptor(event.payload.get("arguments"))
        return f"{name}({descriptor})"
    tool_calls = event.payload.get("tool_calls") or []
    if tool_calls:
        return f"calling {tool_calls[0]}…"
    return "thinking…"


def _any_curation_running(project_id: str) -> bool:
    """Cheap, in-memory-only check (is_curation_running just tests set membership,
    no DB query) — used to drive the Curation button's spinner on the board's
    existing 3s poll without adding a DB round trip to that cadence."""
    return any(is_curation_running(project_id, kind) for kind in ActivityKind)


async def _curation_statuses(session: SessionDep, project_id: str) -> list[dict]:
    runs = await project_service.list_activity_runs(session, project_id)
    curation_state = await project_service.get_curation_state(session, project_id)
    disabled_kinds = set(curation_state.disabled_kinds) if curation_state else set()
    statuses = []
    for kind in ActivityKind:
        running = is_curation_running(project_id, kind)
        current_activity = None
        if running:
            latest_event = await project_service.get_latest_curation_event(session, project_id, kind)
            if latest_event is not None:
                current_activity = _describe_curation_event(latest_event)
        statuses.append(
            {
                "kind": kind.value,
                "label": _CURATION_LABELS[kind],
                "enabled": kind.value not in disabled_kinds,
                "running": running,
                "current_activity": current_activity,
                "last_run_at": runs[kind].last_run_at if kind in runs else None,
                "summary": runs[kind].last_result_summary if kind in runs else None,
            }
        )
    return statuses


async def _curation_context(session: SessionDep, project_id: str) -> dict:
    """Shared by the full board page and the curation modal's own htmx fragment
    response — both render _curation_panel.html.j2 off the same data."""
    curation_state = await project_service.get_curation_state(session, project_id)
    return {
        "statuses": await _curation_statuses(session, project_id),
        "curation_running": _any_curation_running(project_id),
        "curation_paused_at": curation_state.paused_at if curation_state else None,
    }


async def _curation_action_response(project_id: str, request: Request, session: SessionDep) -> Response:
    """Every curation-panel form (pause/resume/kind-toggle/run-now) submits via
    htmx now — on success, swap the refreshed panel fragment back in place rather
    than redirecting the whole page, which would reload the document and lose the
    Bootstrap modal's open state. Non-htmx submits (JS disabled) still fall back
    to the old redirect-to-board behavior, same pattern as cards.py's
    _card_action_response."""
    if request.headers.get("HX-Request") == "true":
        project = await project_service.get_project(session, project_id)
        return templates.TemplateResponse(
            request,
            "_curation_panel.html.j2",
            {"project": project, **await _curation_context(session, project_id)},
        )
    return RedirectResponse(
        request.headers.get("referer") or f"/ui/projects/{project_id}/board", status_code=303
    )


@router.get("/projects/{project_id}/board")
async def board(project_id: str, request: Request, session: SessionDep, show_archived: bool = False):
    project = await project_service.get_project(session, project_id)
    board = await card_service.get_board(session, project_id, include_archived=show_archived)
    epics = await card_service.list_epics(session, project_id)
    return templates.TemplateResponse(
        request,
        "board.html.j2",
        {
            "project": project,
            "board": board,
            "show_archived": show_archived,
            "epics": epics,
            "current_epic_id": project.current_epic_id,
            **await _curation_context(session, project_id),
        },
    )


@router.get("/projects/{project_id}/board/fragment")
async def board_fragment(project_id: str, request: Request, session: SessionDep, show_archived: bool = False):
    project = await project_service.get_project(session, project_id)
    board = await card_service.get_board(session, project_id, include_archived=show_archived)
    return templates.TemplateResponse(
        request,
        "_board_fragment.html.j2",
        {
            "project": project,
            "board": board,
            "show_archived": show_archived,
            "curation_running": _any_curation_running(project_id),
            "is_polling_fragment": True,
        },
    )


@router.get("/projects/{project_id}/board/epics-fragment")
async def board_epics_fragment(project_id: str, request: Request, session: SessionDep):
    project = await project_service.get_project(session, project_id)
    epics = await card_service.list_epics(session, project_id)
    return templates.TemplateResponse(
        request,
        "_epics_fragment.html.j2",
        {"project": project, "epics": epics, "current_epic_id": project.current_epic_id},
    )


@router.post("/projects/{project_id}/current-epic")
async def set_current_epic(
    project_id: str, session: SessionDep, request: Request, epic_card_id: str = Form("")
) -> RedirectResponse:
    """Board page's current-epic <select>. Human-set-only (see
    services/project_service.py's set_current_epic) — empty string clears it, same
    convention as the checkbox-omission handling on set_curation_kind_enabled."""
    await project_service.set_current_epic(session, project_id, epic_card_id or None)
    return RedirectResponse(
        request.headers.get("referer") or f"/ui/projects/{project_id}/board", status_code=303
    )


@router.post("/projects/{project_id}/cards")
async def create_card(
    project_id: str,
    session: SessionDep,
    title: str = Form(...),
    raw_request: str = Form(...),
    priority: str = Form(Priority.NORMAL.value),
) -> RedirectResponse:
    await card_service.create_card(
        session, project_id, title=title, raw_request=raw_request, priority=Priority(priority)
    )
    return RedirectResponse(f"/ui/projects/{project_id}/board", status_code=303)


@router.post("/projects/{project_id}/board/columns/{column}/archive-all")
async def archive_all_in_column(
    project_id: str, column: Column, session: SessionDep, request: Request
) -> RedirectResponse:
    await card_service.archive_all_in_column(session, project_id, column)
    return RedirectResponse(
        request.headers.get("referer") or f"/ui/projects/{project_id}/board", status_code=303
    )


@router.post("/projects/{project_id}/curation/pause")
async def pause_curation(project_id: str, session: SessionDep, request: Request) -> Response:
    await project_service.pause_curation(session, project_id)
    return await _curation_action_response(project_id, request, session)


@router.post("/projects/{project_id}/curation/resume")
async def resume_curation(project_id: str, session: SessionDep, request: Request) -> Response:
    await project_service.resume_curation(session, project_id)
    return await _curation_action_response(project_id, request, session)


@router.post("/projects/{project_id}/curation/kinds/{kind}")
async def set_curation_kind_enabled(
    project_id: str,
    kind: ActivityKind,
    session: SessionDep,
    request: Request,
    enabled: bool = Form(False),
) -> Response:
    """Backs each kind's checkbox in the board's Curation panel. A checkbox omits
    its form field entirely when unchecked (not "false"), which is exactly what
    Form(False)'s default handles — no explicit off-value needed on the input."""
    await project_service.set_curation_kind_enabled(session, project_id, kind, enabled=enabled)
    return await _curation_action_response(project_id, request, session)


@router.post("/projects/{project_id}/curate/{kind}")
async def curate(project_id: str, kind: ActivityKind, request: Request, session: SessionDep) -> Response:
    """Kicks off one curation pass (agent/curation.py) in the background — a full
    pass is an LLM agentic loop and can take a while, so this responds immediately.
    Any new cards appear on the board as they're created via the existing polling
    fragment."""
    await project_service.get_project(session, project_id)  # 404s early if missing
    if not is_curation_running(project_id, kind):
        asyncio.create_task(run_curation_activity(project_id, kind))
    return await _curation_action_response(project_id, request, session)


def _run_status(run, *, is_running_now: bool) -> str:
    """A run's outcome column value never written for a still-open row (see
    CurationRun's own docstring) — is_curation_running (checked once per row by
    the caller, true only for the single most-recently-started run of this kind,
    if any) is what disambiguates "still going" from "the process died before it
    could close this out"."""
    if run.ended_at is not None:
        return run.outcome.value if run.outcome else "ok"
    return "running" if is_running_now else "interrupted"


@router.get("/projects/{project_id}/curation/{kind}")
async def curation_kind_history(project_id: str, kind: ActivityKind, request: Request, session: SessionDep):
    project = await project_service.get_project(session, project_id)
    runs = await project_service.list_curation_runs(session, project_id, kind)
    running = is_curation_running(project_id, kind)
    rows = [
        {
            "id": run.id,
            "started_at": run.started_at,
            "ended_at": run.ended_at,
            "status": _run_status(run, is_running_now=running and i == 0),
            "summary": run.summary,
            "tokens_in": run.tokens_in,
            "tokens_out": run.tokens_out,
        }
        for i, run in enumerate(runs)
    ]
    return templates.TemplateResponse(
        request,
        "curation_kind_history.html.j2",
        {
            "project": project,
            "kind": kind.value,
            "label": _CURATION_LABELS[kind],
            "running": running,
            "runs": rows,
        },
    )


@router.get("/projects/{project_id}/curation/{kind}/runs/{run_id}")
async def curation_run_detail(
    project_id: str, kind: ActivityKind, run_id: str, request: Request, session: SessionDep
):
    project = await project_service.get_project(session, project_id)
    run = await project_service.get_curation_run(session, run_id)
    # is_curation_running only says "some run of this kind is in flight" — confirm
    # THIS row is that one (not an older orphaned run) before calling it "running"
    # rather than "interrupted".
    most_recent = await project_service.list_curation_runs(session, project_id, kind, limit=1)
    is_the_current_run = bool(most_recent) and most_recent[0].id == run.id
    running = is_curation_running(project_id, kind) and is_the_current_run
    return templates.TemplateResponse(
        request,
        "curation_run_detail.html.j2",
        {
            "project": project,
            "kind": kind.value,
            "label": _CURATION_LABELS[kind],
            "run": run,
            "status": _run_status(run, is_running_now=running),
            "events": run.events,
        },
    )
