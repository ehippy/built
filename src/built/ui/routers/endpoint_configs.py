from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from built.api.deps import SessionDep
from built.config import settings
from built.domain.enums import Column
from built.llm.client import check_endpoint_health
from built.services import endpoint_service
from built.ui.templates import templates

router = APIRouter(prefix="/ui", tags=["ui"])


@router.get("/endpoint-configs")
async def list_global_endpoint_configs(request: Request, session: SessionDep):
    endpoint_configs = await endpoint_service.list_endpoint_configs(session, only_global=True)
    return templates.TemplateResponse(
        request,
        "endpoint_configs.html.j2",
        {"endpoint_configs": endpoint_configs, "default_max_tokens": settings.default_max_tokens},
    )


@router.post("/endpoint-configs")
async def create_global_endpoint_config(
    session: SessionDep,
    base_url: str = Form(...),
    model: str = Form(...),
    role: str = Form(""),
    priority: int = Form(0),
    api_key_ref: str = Form(""),
    max_concurrency: int = Form(1),
    context_window: str = Form(""),
    supports_tool_calling: bool = Form(True),
) -> RedirectResponse:
    await endpoint_service.create_endpoint_config(
        session,
        base_url=base_url,
        model=model,
        project_id=None,
        role=Column(role) if role else None,
        priority=priority,
        api_key_ref=api_key_ref or None,
        max_concurrency=max_concurrency,
        context_window=int(context_window) if context_window.strip() else None,
        supports_tool_calling=supports_tool_calling,
    )
    return RedirectResponse("/ui/endpoint-configs", status_code=303)


@router.post("/endpoint-configs/{endpoint_id}/edit")
async def edit_endpoint_config(
    endpoint_id: str,
    session: SessionDep,
    base_url: str = Form(...),
    model: str = Form(...),
    role: str = Form(""),
    priority: int = Form(0),
    api_key_ref: str = Form(""),
    max_concurrency: int = Form(1),
    context_window: str = Form(""),
    supports_tool_calling: bool = Form(False),
    next: str = "/ui/endpoint-configs",
) -> RedirectResponse:
    await endpoint_service.edit_endpoint_config(
        session,
        endpoint_id,
        base_url=base_url,
        model=model,
        role=Column(role) if role else None,
        priority=priority,
        api_key_ref=api_key_ref or None,
        max_concurrency=max_concurrency,
        context_window=int(context_window) if context_window.strip() else None,
        supports_tool_calling=supports_tool_calling,
    )
    return RedirectResponse(next, status_code=303)


@router.get("/endpoint-configs/{endpoint_id}/health")
async def endpoint_config_health(endpoint_id: str, request: Request, session: SessionDep):
    """Backs the health dot's htmx fragment. Fired on row load and on manual
    re-check click only — deliberately no auto-poll interval, since a real
    completion call competes for the same scarce max_concurrency slot as actual
    card work; see check_endpoint_health's own "busy" short-circuit for the other
    half of that guard."""
    endpoint = await endpoint_service.get_endpoint_config(session, endpoint_id)
    if not endpoint.enabled:
        health = {"state": "disabled", "latency_ms": None, "error": None}
    else:
        health = await check_endpoint_health(endpoint)
    return templates.TemplateResponse(
        request, "_endpoint_health_fragment.html.j2", {"endpoint": endpoint, "health": health}
    )


@router.post("/endpoint-configs/{endpoint_id}/context-window")
async def set_endpoint_context_window(
    endpoint_id: str,
    session: SessionDep,
    context_window: int = Form(...),
    next: str = "/ui/endpoint-configs",
) -> RedirectResponse:
    """The model's real context window isn't discoverable from here — an operator has
    to tell the app what it is, or every project resolving to this endpoint silently
    caps at settings.default_max_tokens (128k) regardless of what the model actually
    supports."""
    await endpoint_service.update_endpoint_config(session, endpoint_id, context_window=context_window)
    return RedirectResponse(next, status_code=303)


@router.post("/endpoint-configs/{endpoint_id}/toggle")
async def toggle_endpoint_config(
    endpoint_id: str, session: SessionDep, next: str = "/ui/endpoint-configs"
) -> RedirectResponse:
    endpoint = await endpoint_service.get_endpoint_config(session, endpoint_id)
    await endpoint_service.update_endpoint_config(session, endpoint_id, enabled=not endpoint.enabled)
    return RedirectResponse(next, status_code=303)


@router.post("/endpoint-configs/{endpoint_id}/delete")
async def delete_endpoint_config(
    endpoint_id: str, session: SessionDep, next: str = "/ui/endpoint-configs"
) -> RedirectResponse:
    await endpoint_service.delete_endpoint_config(session, endpoint_id)
    return RedirectResponse(next, status_code=303)
