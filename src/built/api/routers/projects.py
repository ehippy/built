from fastapi import APIRouter, HTTPException, status

from built.api.deps import RequireApiKey, SessionDep
from built.api.schemas import (
    DeployConfigIn,
    DeployConfigOut,
    ProjectCreate,
    ProjectOut,
    ProjectUpdate,
)
from built.services import project_service
from built.services.project_service import NotFoundError

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])


@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
async def create_project(body: ProjectCreate, session: SessionDep, _: RequireApiKey) -> ProjectOut:
    project = await project_service.create_project(session, **body.model_dump())
    return ProjectOut.model_validate(project)


@router.get("", response_model=list[ProjectOut])
async def list_projects(session: SessionDep, include_archived: bool = False) -> list[ProjectOut]:
    projects = await project_service.list_projects(session, include_archived=include_archived)
    return [ProjectOut.model_validate(p) for p in projects]


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(project_id: str, session: SessionDep) -> ProjectOut:
    try:
        project = await project_service.get_project(session, project_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return ProjectOut.model_validate(project)


@router.patch("/{project_id}", response_model=ProjectOut)
async def update_project(
    project_id: str, body: ProjectUpdate, session: SessionDep, _: RequireApiKey
) -> ProjectOut:
    try:
        project = await project_service.update_project(session, project_id, **body.model_dump())
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return ProjectOut.model_validate(project)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def archive_project(project_id: str, session: SessionDep, _: RequireApiKey) -> None:
    try:
        await project_service.archive_project(session, project_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.put("/{project_id}/deploy-config", response_model=DeployConfigOut)
async def set_deploy_config(
    project_id: str, body: DeployConfigIn, session: SessionDep, _: RequireApiKey
) -> DeployConfigOut:
    try:
        config = await project_service.set_deploy_config(session, project_id, **body.model_dump())
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return DeployConfigOut.model_validate(config)
