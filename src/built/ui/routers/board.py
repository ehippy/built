import asyncio

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from built.api.deps import SessionDep
from built.db.models import CurationEvent
from built.domain.enums import ActivityKind, EventType, Priority
from built.orchestrator.curator import is_curation_running, run_curation_activity
from built.services import card_service, project_service
from built.ui.templates import templates, tool_descriptor

router = APIRouter(prefix="/ui", tags=["ui"])

_CURATION_LABELS: dict[ActivityKind, str] = {
    ActivityKind.BUG_SWEEP: "Bug sweep",
    ActivityKind.OPPORTUNITY_BRAINSTORM: "Opportunities",
    ActivityKind.POLISH_REVIEW: "Polish review",
    ActivityKind.AGENTS_MD: "Tend AGENTS.md",
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


async def _curation_statuses(session: SessionDep, project_id: str) -> list[dict]:
    runs = await project_service.list_activity_runs(session, project_id)
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
                "running": running,
                "current_activity": current_activity,
                "last_run_at": runs[kind].last_run_at if kind in runs else None,
                "summary": runs[kind].last_result_summary if kind in runs else None,
            }
        )
    return statuses


@router.get("/projects/{project_id}/board")
async def board(project_id: str, request: Request, session: SessionDep, show_archived: bool = False):
    project = await project_service.get_project(session, project_id)
    board = await card_service.get_board(session, project_id, include_archived=show_archived)
    statuses = await _curation_statuses(session, project_id)
    return templates.TemplateResponse(
        request,
        "board.html.j2",
        {"project": project, "board": board, "show_archived": show_archived, "statuses": statuses},
    )


@router.get("/projects/{project_id}/board/fragment")
async def board_fragment(project_id: str, request: Request, session: SessionDep, show_archived: bool = False):
    project = await project_service.get_project(session, project_id)
    board = await card_service.get_board(session, project_id, include_archived=show_archived)
    return templates.TemplateResponse(
        request,
        "_board_fragment.html.j2",
        {"project": project, "board": board, "show_archived": show_archived},
    )


@router.get("/projects/{project_id}/curation-status/fragment")
async def curation_status_fragment(project_id: str, request: Request, session: SessionDep):
    project = await project_service.get_project(session, project_id)
    statuses = await _curation_statuses(session, project_id)
    return templates.TemplateResponse(
        request, "_curation_status_fragment.html.j2", {"project": project, "statuses": statuses}
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


@router.post("/projects/{project_id}/curate/{kind}")
async def curate(project_id: str, kind: ActivityKind, session: SessionDep) -> RedirectResponse:
    """Kicks off one curation pass (agent/curation.py) in the background — a full
    pass is an LLM agentic loop and can take a while, so this redirects immediately.
    Any new cards appear on the board as they're created via the existing polling
    fragment."""
    await project_service.get_project(session, project_id)  # 404s early if missing
    if not is_curation_running(project_id, kind):
        asyncio.create_task(run_curation_activity(project_id, kind))
    return RedirectResponse(f"/ui/projects/{project_id}/board", status_code=303)
