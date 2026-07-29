from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from built.api.deps import SessionDep
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
