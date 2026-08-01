from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse, Response

from built.api.deps import SessionDep
from built.domain.enums import Priority
from built.services import card_service, project_service
from built.ui.templates import templates

router = APIRouter(prefix="/ui", tags=["ui"])


async def _card_detail_context(session: SessionDep, card_id: str) -> dict:
    """Shared by the full-page card_detail view and the board's slide-over panel —
    both render the same _card_detail_body.html.j2 partial off the same data."""
    card = await card_service.get_card(session, card_id, with_last_activity=True)
    visits = await card_service.list_column_visits(session, card_id)
    events = await card_service.list_recent_events(session, card_id, limit=200)
    is_epic = await card_service.is_epic(session, card_id)
    epic_parent = await card_service.get_epic_parent(session, card_id)
    epic_children = (
        await card_service.list_epic_children(session, card_id, include_archived=True) if is_epic else []
    )
    dependencies = await card_service.list_dependencies(session, card_id)
    dependents = await card_service.list_dependents(session, card_id)
    return {
        "card": card,
        "visits": visits,
        "events": events,
        "is_epic": is_epic,
        "epic_parent": epic_parent,
        "epic_children": epic_children,
        "dependencies": dependencies,
        "dependents": dependents,
    }


@router.get("/cards/{card_id}")
async def card_detail(card_id: str, request: Request, session: SessionDep):
    context = await _card_detail_context(session, card_id)
    project = await project_service.get_project(session, context["card"].project_id)
    return templates.TemplateResponse(request, "card_detail.html.j2", {**context, "project": project})


@router.get("/cards/{card_id}/panel")
async def card_detail_panel(card_id: str, request: Request, session: SessionDep):
    context = await _card_detail_context(session, card_id)
    return templates.TemplateResponse(request, "_card_detail_panel.html.j2", context)


async def _card_action_response(card_id: str, request: Request, session: SessionDep) -> Response:
    """Both the standalone card page and the board's slide-over panel submit these action
    forms via htmx now, so on success we swap the refreshed detail fragment back in place
    rather than redirecting — a redirect would force a full top-level navigation, kicking the
    user out of the board's popover onto the standalone /ui/cards/{id} page. Non-htmx submits
    (JS disabled) still fall back to the old redirect-to-self behavior."""
    if request.headers.get("HX-Request") == "true":
        context = await _card_detail_context(session, card_id)
        return templates.TemplateResponse(request, "_card_detail_action_response.html.j2", context)
    return RedirectResponse(f"/ui/cards/{card_id}", status_code=303)


@router.get("/cards/{card_id}/events/fragment")
async def card_events_fragment(card_id: str, request: Request, session: SessionDep):
    card = await card_service.get_card(session, card_id)
    events = await card_service.list_recent_events(session, card_id, limit=200)
    return templates.TemplateResponse(
        request, "_events_fragment.html.j2", {"card": card, "events": events, "is_polling_fragment": True}
    )


@router.post("/cards/{card_id}/retry")
async def retry_card(card_id: str, request: Request, session: SessionDep, note: str = Form("")) -> Response:
    await card_service.retry_card(session, card_id, note=note or None)
    return await _card_action_response(card_id, request, session)


@router.post("/cards/{card_id}/nudge")
async def nudge_card(card_id: str, request: Request, session: SessionDep, note: str = Form(...)) -> Response:
    await card_service.add_nudge(session, card_id, note=note)
    return await _card_action_response(card_id, request, session)


@router.post("/cards/{card_id}/cancel")
async def cancel_card(card_id: str, request: Request, session: SessionDep) -> Response:
    await card_service.cancel_card(session, card_id)
    return await _card_action_response(card_id, request, session)


@router.post("/cards/{card_id}/edit")
async def edit_card(
    card_id: str, request: Request, session: SessionDep, title: str = Form(...), raw_request: str = Form(...)
) -> Response:
    await card_service.update_card(session, card_id, title=title, raw_request=raw_request)
    return await _card_action_response(card_id, request, session)


@router.post("/cards/{card_id}/priority")
async def set_card_priority(
    card_id: str, request: Request, session: SessionDep, priority: str = Form(...)
) -> Response:
    await card_service.set_priority(session, card_id, Priority(priority))
    return await _card_action_response(card_id, request, session)


@router.post("/cards/{card_id}/archive")
async def archive_card(card_id: str, request: Request, session: SessionDep) -> Response:
    await card_service.archive_card(session, card_id)
    return await _card_action_response(card_id, request, session)


@router.post("/cards/{card_id}/unarchive")
async def unarchive_card(card_id: str, request: Request, session: SessionDep) -> Response:
    await card_service.unarchive_card(session, card_id)
    return await _card_action_response(card_id, request, session)
