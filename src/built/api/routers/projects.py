import asyncio

from fastapi import APIRouter, HTTPException, status

from built.agent.curation import assess_overseer_prompt
from built.api.deps import RequireApiKey, SessionDep
from built.api.schemas import (
    DeployConfigIn,
    DeployConfigOut,
    OverseerPromptIn,
    ProjectCreate,
    ProjectOut,
    ProjectUpdate,
)
from built.domain.enums import ActivityKind, Column
from built.llm.client import FallbackLLMClient
from built.orchestrator.curator import is_curation_running, run_curation_activity
from built.services import endpoint_service, project_service
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


@router.put("/{project_id}/overseer-prompt", response_model=ProjectOut)
async def set_overseer_prompt(
    project_id: str, body: OverseerPromptIn, session: SessionDep, _: RequireApiKey
) -> ProjectOut:
    """Deliberately not folded into the generic PATCH /{project_id} — without this
    dedicated endpoint, the comprehensiveness gate (agent/curation.py's
    assess_overseer_prompt) would be UI-only theater that any direct API client
    bypasses. Mirrors ui/routers/projects.py's update_overseer_prompt exactly: a
    blank prompt always saves unconditionally; a non-blank prompt without
    force=True is judged first, and both "not comprehensive" and "judge call
    failed" block with the same 422 shape, distinguished by issues vs. error — the
    client fixes the prompt or resubmits with force=True, same escape hatch as the
    UI's "save anyway" button."""
    try:
        project = await project_service.get_project(session, project_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    prompt = (body.prompt or "").strip() or None
    if prompt is not None and not body.force:
        try:
            chain = await endpoint_service.get_resolved_chain(session, project_id=project_id, role=Column.PM)
            assessment = await assess_overseer_prompt(prompt, project, llm_client=FallbackLLMClient(chain))
        except Exception as exc:  # noqa: BLE001 — same soft-block contract as the UI route
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                {"comprehensive": None, "issues": [], "error": str(exc)},
            ) from exc
        if not assessment.comprehensive:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                {"comprehensive": False, "issues": assessment.issues, "error": None},
            )
    project = await project_service.set_overseer_prompt(session, project_id, prompt)
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


@router.post("/{project_id}/curate/{kind}", status_code=status.HTTP_202_ACCEPTED)
async def curate(project_id: str, kind: ActivityKind, session: SessionDep, _: RequireApiKey) -> dict:
    """Kicks off one curation pass (agent/curation.py) in the background and returns
    immediately — a full pass is an LLM agentic loop and can take a while. New cards
    (if any) appear on the board as they're created; poll GET /projects/{id}/board."""
    try:
        await project_service.get_project(session, project_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    if is_curation_running(project_id, kind):
        raise HTTPException(status.HTTP_409_CONFLICT, f"{kind.value} is already running for this project")
    asyncio.create_task(run_curation_activity(project_id, kind))
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
