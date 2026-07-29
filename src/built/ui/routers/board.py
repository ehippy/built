import asyncio

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from built.api.deps import SessionDep
from built.domain.enums import ActivityKind
from built.orchestrator.curator import is_curation_running, run_curation_activity
from built.services import card_service, project_service
from built.ui.templates import templates

router = APIRouter(prefix="/ui", tags=["ui"])


@router.get("/projects/{project_id}/board")
async def board(project_id: str, request: Request, session: SessionDep, show_archived: bool = False):
    project = await project_service.get_project(session, project_id)
    board = await card_service.get_board(session, project_id, include_archived=show_archived)
    return templates.TemplateResponse(
        request, "board.html.j2", {"project": project, "board": board, "show_archived": show_archived}
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


@router.post("/projects/{project_id}/cards")
async def create_card(
    project_id: str,
    session: SessionDep,
    title: str = Form(...),
    raw_request: str = Form(...),
) -> RedirectResponse:
    await card_service.create_card(session, project_id, title=title, raw_request=raw_request)
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
