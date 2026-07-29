import asyncio

from fastapi import APIRouter, HTTPException, status

from built.api.deps import RequireApiKey, SessionDep
from built.api.schemas import (
    DeployConfigIn,
    DeployConfigOut,
    ProjectCreate,
    ProjectOut,
    ProjectUpdate,
)
from built.orchestrator.worker import is_discovery_running, run_project_discovery
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


@router.post("/{project_id}/pause", response_model=ProjectOut)
async def pause_project(project_id: str, session: SessionDep, _: RequireApiKey) -> ProjectOut:
    try:
        project = await project_service.pause_project(session, project_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return ProjectOut.model_validate(project)


@router.post("/{project_id}/resume", response_model=ProjectOut)
async def resume_project(project_id: str, session: SessionDep, _: RequireApiKey) -> ProjectOut:
    try:
        project = await project_service.resume_project(session, project_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return ProjectOut.model_validate(project)


@router.post("/{project_id}/discover-tasks", status_code=status.HTTP_202_ACCEPTED)
async def discover_tasks(project_id: str, session: SessionDep, _: RequireApiKey) -> dict:
    """Kicks off one autonomous PM discovery pass in the background and returns
    immediately — a full pass is an LLM agentic loop and can take a while. New cards
    (if any) appear on the board as they're created; poll GET /projects/{id}/board."""
    try:
        await project_service.get_project(session, project_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    if is_discovery_running(project_id):
        raise HTTPException(status.HTTP_409_CONFLICT, "discovery is already running for this project")
    asyncio.create_task(run_project_discovery(project_id))
    return {"status": "started"}


@router.put("/{project_id}/deploy-config", response_model=DeployConfigOut)
async def set_deploy_config(
    project_id: str, body: DeployConfigIn, session: SessionDep, _: RequireApiKey
) -> DeployConfigOut:
    try:
        config = await project_service.set_deploy_config(session, project_id, **body.model_dump())
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return DeployConfigOut.model_validate(config)
