import asyncio

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from built.api.deps import SessionDep
from built.orchestrator.worker import is_discovery_running, run_project_discovery
from built.services import card_service, project_service
from built.ui.templates import templates

router = APIRouter(prefix="/ui", tags=["ui"])


@router.get("/projects/{project_id}/board")
async def board(project_id: str, request: Request, session: SessionDep):
    project = await project_service.get_project(session, project_id)
    board = await card_service.get_board(session, project_id)
    return templates.TemplateResponse(request, "board.html.j2", {"project": project, "board": board})


@router.get("/projects/{project_id}/board/fragment")
async def board_fragment(project_id: str, request: Request, session: SessionDep):
    project = await project_service.get_project(session, project_id)
    board = await card_service.get_board(session, project_id)
    return templates.TemplateResponse(
        request, "_board_fragment.html.j2", {"project": project, "board": board}
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


@router.post("/projects/{project_id}/discover-tasks")
async def discover_tasks(project_id: str, session: SessionDep) -> RedirectResponse:
    """Kicks off one autonomous PM discovery pass in the background — a full pass is
    an LLM agentic loop and can take a while, so this redirects immediately. Any new
    cards appear on the board as they're created via the existing polling fragment."""
    await project_service.get_project(session, project_id)  # 404s early if missing
    if not is_discovery_running(project_id):
        asyncio.create_task(run_project_discovery(project_id))
    return RedirectResponse(f"/ui/projects/{project_id}/board", status_code=303)
