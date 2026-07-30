"""Dashboard routes for projects: list, create, and per-project settings (endpoint
fallback chains, deploy config, archive). Calls services/ directly, like the REST
API — never HTTP-calls the REST API itself.

No API-key check here (unlike the REST API's mutating endpoints): the dashboard is
meant for interactive local use by whoever is running the service, and plain HTML
forms can't easily send a custom header. Don't expose this service to an untrusted
network without adding real session-based auth in front of it."""

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from built.api.deps import SessionDep
from built.config import settings
from built.domain.enums import Column, DeployKind, DeployMode
from built.services import card_service, endpoint_service, project_service
from built.ui.templates import templates

router = APIRouter(prefix="/ui", tags=["ui"])


async def _projects_with_activity(session: AsyncSession) -> list[dict]:
    projects = await project_service.list_projects(session)
    return [
        {"project": project, "activity": await card_service.get_project_activity_summary(session, project.id)}
        for project in projects
    ]


@router.get("/")
async def index() -> RedirectResponse:
    return RedirectResponse("/ui/projects", status_code=303)


@router.get("/projects")
async def list_projects(request: Request, session: SessionDep):
    rows = await _projects_with_activity(session)
    return templates.TemplateResponse(request, "projects_list.html.j2", {"rows": rows})


@router.get("/projects/fragment")
async def projects_list_fragment(request: Request, session: SessionDep):
    rows = await _projects_with_activity(session)
    return templates.TemplateResponse(request, "_projects_list_fragment.html.j2", {"rows": rows})


@router.post("/projects")
async def create_project(
    session: SessionDep,
    name: str = Form(...),
    overarching_goal: str = Form(...),
    repo_remote_url: str = Form(...),
    default_branch: str = Form("main"),
    sandbox_image: str = Form(""),
    test_command: str = Form(""),
    max_revisions: int | None = Form(None),
    max_deploy_attempts: int | None = Form(None),
    max_iterations_per_run: int | None = Form(None),
) -> RedirectResponse:
    project = await project_service.create_project(
        session,
        name=name,
        overarching_goal=overarching_goal,
        repo_remote_url=repo_remote_url,
        default_branch=default_branch or "main",
        sandbox_image=sandbox_image or None,
        test_command=test_command or None,
        max_revisions=max_revisions,
        max_deploy_attempts=max_deploy_attempts,
        max_iterations_per_run=max_iterations_per_run,
    )
    return RedirectResponse(f"/ui/projects/{project.id}/board", status_code=303)


@router.get("/projects/{project_id}/settings")
async def project_settings(project_id: str, request: Request, session: SessionDep):
    project = await project_service.get_project(session, project_id)
    endpoint_configs = await endpoint_service.list_endpoint_configs(session, project_id=project_id)
    return templates.TemplateResponse(
        request,
        "project_settings.html.j2",
        {
            "project": project,
            "endpoint_configs": endpoint_configs,
            "deploy_config": project.deploy_config,
            "default_max_tokens": settings.default_max_tokens,
        },
    )


@router.post("/projects/{project_id}/settings")
async def update_project_settings(
    project_id: str,
    session: SessionDep,
    name: str = Form(...),
    overarching_goal: str = Form(...),
    default_branch: str = Form("main"),
    sandbox_image: str = Form(""),
    test_command: str = Form(""),
    max_revisions: int = Form(...),
    max_deploy_attempts: int = Form(...),
    max_iterations_per_run: int = Form(...),
) -> RedirectResponse:
    await project_service.update_project(
        session,
        project_id,
        name=name,
        overarching_goal=overarching_goal,
        default_branch=default_branch,
        sandbox_image=sandbox_image or None,
        test_command=test_command or None,
        max_revisions=max_revisions,
        max_deploy_attempts=max_deploy_attempts,
        max_iterations_per_run=max_iterations_per_run,
    )
    return RedirectResponse(f"/ui/projects/{project_id}/settings", status_code=303)


@router.post("/projects/{project_id}/archive")
async def archive_project(project_id: str, session: SessionDep) -> RedirectResponse:
    await project_service.archive_project(session, project_id)
    return RedirectResponse("/ui/projects", status_code=303)


@router.post("/projects/{project_id}/pause")
async def pause_project(project_id: str, session: SessionDep, request: Request) -> RedirectResponse:
    await project_service.pause_project(session, project_id)
    return RedirectResponse(request.headers.get("referer") or "/ui/projects", status_code=303)


@router.post("/projects/{project_id}/resume")
async def resume_project(project_id: str, session: SessionDep, request: Request) -> RedirectResponse:
    await project_service.resume_project(session, project_id)
    return RedirectResponse(request.headers.get("referer") or "/ui/projects", status_code=303)


@router.post("/projects/{project_id}/endpoint-configs")
async def add_project_endpoint_config(
    project_id: str,
    session: SessionDep,
    base_url: str = Form(...),
    model: str = Form(...),
    role: str = Form(""),
    priority: int = Form(0),
    api_key_ref: str = Form(""),
    max_concurrency: int = Form(1),
    context_window: str = Form(""),
) -> RedirectResponse:
    await endpoint_service.create_endpoint_config(
        session,
        base_url=base_url,
        model=model,
        project_id=project_id,
        role=Column(role) if role else None,
        priority=priority,
        api_key_ref=api_key_ref or None,
        max_concurrency=max_concurrency,
        context_window=int(context_window) if context_window.strip() else None,
    )
    return RedirectResponse(f"/ui/projects/{project_id}/settings", status_code=303)


@router.post("/projects/{project_id}/deploy-config")
async def set_project_deploy_config(
    project_id: str,
    session: SessionDep,
    kind: str = Form(...),
    mode: str = Form("pr_to_operator"),
    command: str = Form(""),
    script_path: str = Form(""),
    webhook_url: str = Form(""),
    timeout_seconds: int = Form(600),
    github_token_ref: str = Form(""),
) -> RedirectResponse:
    await project_service.set_deploy_config(
        session,
        project_id,
        kind=DeployKind(kind),
        mode=DeployMode(mode),
        command=command or None,
        script_path=script_path or None,
        webhook_url=webhook_url or None,
        timeout_seconds=timeout_seconds,
        github_token_ref=github_token_ref or None,
    )
    return RedirectResponse(f"/ui/projects/{project_id}/settings", status_code=303)
