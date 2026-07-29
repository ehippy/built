from fastapi import APIRouter, HTTPException, status

from built.api.deps import RequireApiKey, SessionDep
from built.api.schemas import EndpointConfigCreate, EndpointConfigOut, EndpointConfigUpdate
from built.domain.enums import Column
from built.services import endpoint_service
from built.services.project_service import NotFoundError

router = APIRouter(prefix="/api/v1", tags=["endpoint-configs"])


@router.post("/endpoint-configs", response_model=EndpointConfigOut, status_code=status.HTTP_201_CREATED)
async def create_endpoint_config(
    body: EndpointConfigCreate, session: SessionDep, _: RequireApiKey
) -> EndpointConfigOut:
    endpoint = await endpoint_service.create_endpoint_config(session, **body.model_dump())
    return EndpointConfigOut.model_validate(endpoint)


@router.get("/endpoint-configs", response_model=list[EndpointConfigOut])
async def list_endpoint_configs(
    session: SessionDep, scope: str = "all", project_id: str | None = None
) -> list[EndpointConfigOut]:
    endpoints = await endpoint_service.list_endpoint_configs(
        session, project_id=project_id, only_global=(scope == "global")
    )
    return [EndpointConfigOut.model_validate(e) for e in endpoints]


@router.patch("/endpoint-configs/{endpoint_id}", response_model=EndpointConfigOut)
async def update_endpoint_config(
    endpoint_id: str, body: EndpointConfigUpdate, session: SessionDep, _: RequireApiKey
) -> EndpointConfigOut:
    try:
        endpoint = await endpoint_service.update_endpoint_config(session, endpoint_id, **body.model_dump())
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return EndpointConfigOut.model_validate(endpoint)


@router.delete("/endpoint-configs/{endpoint_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_endpoint_config(endpoint_id: str, session: SessionDep, _: RequireApiKey) -> None:
    try:
        await endpoint_service.delete_endpoint_config(session, endpoint_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.get("/projects/{project_id}/endpoint-chain/{role}", response_model=list[EndpointConfigOut])
async def resolved_endpoint_chain(
    project_id: str, role: Column, session: SessionDep
) -> list[EndpointConfigOut]:
    """Debug endpoint: shows the effective fallback chain for a (project, role) pair
    after tier resolution — useful for confirming a project's endpoint config actually
    resolves the way you expect before an agent run depends on it."""
    chain = await endpoint_service.get_resolved_chain(session, project_id=project_id, role=role)
    return [EndpointConfigOut.model_validate(e) for e in chain]
