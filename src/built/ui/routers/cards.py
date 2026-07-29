from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from built.api.deps import SessionDep
from built.services import card_service, project_service
from built.ui.templates import templates

router = APIRouter(prefix="/ui", tags=["ui"])


@router.get("/cards/{card_id}")
async def card_detail(card_id: str, request: Request, session: SessionDep):
    card = await card_service.get_card(session, card_id, with_last_activity=True)
    project = await project_service.get_project(session, card.project_id)
    visits = await card_service.list_column_visits(session, card_id)
    events = await card_service.list_events(session, card_id, limit=200)
    return templates.TemplateResponse(
        request,
        "card_detail.html.j2",
        {"card": card, "project": project, "visits": visits, "events": events},
    )


@router.get("/cards/{card_id}/events/fragment")
async def card_events_fragment(card_id: str, request: Request, session: SessionDep):
    card = await card_service.get_card(session, card_id)
    events = await card_service.list_events(session, card_id, limit=200)
    return templates.TemplateResponse(request, "_events_fragment.html.j2", {"card": card, "events": events})


@router.post("/cards/{card_id}/retry")
async def retry_card(card_id: str, session: SessionDep, note: str = Form("")) -> RedirectResponse:
    await card_service.retry_card(session, card_id, note=note or None)
    return RedirectResponse(f"/ui/cards/{card_id}", status_code=303)


@router.post("/cards/{card_id}/cancel")
async def cancel_card(card_id: str, session: SessionDep) -> RedirectResponse:
    await card_service.cancel_card(session, card_id)
    return RedirectResponse(f"/ui/cards/{card_id}", status_code=303)


@router.post("/cards/{card_id}/edit")
async def edit_card(
    card_id: str, session: SessionDep, title: str = Form(...), raw_request: str = Form(...)
) -> RedirectResponse:
    await card_service.update_card(session, card_id, title=title, raw_request=raw_request)
    return RedirectResponse(f"/ui/cards/{card_id}", status_code=303)


@router.post("/cards/{card_id}/archive")
async def archive_card(card_id: str, session: SessionDep) -> RedirectResponse:
    await card_service.archive_card(session, card_id)
    return RedirectResponse(f"/ui/cards/{card_id}", status_code=303)


@router.post("/cards/{card_id}/unarchive")
async def unarchive_card(card_id: str, session: SessionDep) -> RedirectResponse:
    await card_service.unarchive_card(session, card_id)
    return RedirectResponse(f"/ui/cards/{card_id}", status_code=303)
