"""Dashboard view into the unified "built" logger (logging_config.py) — worker
claims, Reviver/Curator/archiver passes, crash recovery, etc. Polled the same way
the per-card transcript is (see ui/routers/cards.py's events fragment): always
re-fetch the recent window and swap it in, no incremental diffing."""

from fastapi import APIRouter, Request

from built.api.deps import SessionDep
from built.logging_config import get_logs
from built.services import project_service
from built.ui.templates import templates

router = APIRouter(prefix="/ui", tags=["ui"])


@router.get("/logs")
async def logs_page(request: Request, session: SessionDep):
    all_projects = await project_service.list_projects(session)
    return templates.TemplateResponse(
        request, "logs.html.j2", {"logs": get_logs(), "all_projects": all_projects}
    )


@router.get("/logs/fragment")
async def logs_fragment(request: Request):
    return templates.TemplateResponse(request, "_logs_fragment.html.j2", {"logs": get_logs()})
