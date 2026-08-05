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

from built.agent.curation import assess_overseer_prompt
from built.api.deps import SessionDep
from built.config import settings
from built.domain.enums import Column, DeployKind, DeployMode
from built.llm.client import FallbackLLMClient
from built.sandbox import deploy_runner
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


@router.get("/projects/nav-fragment")
async def projects_nav_fragment(request: Request, session: SessionDep):
    """The nav bar's Projects dropdown fetches this itself (see base.html.j2) —
    it's the same list on every page regardless of which route rendered it, so
    it has no business being plumbed through every UI route's context by hand."""
    all_projects = await project_service.list_projects(session)
    return templates.TemplateResponse(
        request, "_projects_nav_fragment.html.j2", {"all_projects": all_projects}
    )


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


@router.get("/projects/{project_id}/ci-status")
async def project_ci_status(project_id: str, request: Request, session: SessionDep):
    """The nav bar's GitHub-icon status dot — polled by htmx, not computed on the
    initial page render, so an external GitHub round-trip never blocks a normal
    page load. Checked against the tip of default_branch (GitHub's check-runs
    endpoint accepts a branch name as well as a sha), not any particular card's
    deploying_commit_sha — this is "is the repo green right now", independent of
    whether `built` itself is mid-deploy on anything."""
    project = await project_service.get_project(session, project_id)
    try:
        check_runs = await deploy_runner.fetch_check_runs(project, project.default_branch)
    except deploy_runner.CIStatusUnavailableError:
        state = "unknown"
    else:
        if not check_runs:
            state = "none"
        elif any(run.status != "completed" for run in check_runs):
            state = "pending"
        elif any(run.conclusion in deploy_runner.FAILING_CONCLUSIONS for run in check_runs):
            state = "failure"
        else:
            state = "success"
    return templates.TemplateResponse(
        request, "_ci_status_fragment.html.j2", {"project": project, "ci_state": state}
    )


async def _project_settings_context(session: SessionDep, project_id: str) -> dict:
    """Shared by the plain GET render and update_overseer_prompt's failed-validation
    re-render below — the latter can't redirect (there's no flash/session mechanism
    to carry the judge's feedback + the operator's draft text across one), so it
    renders this same template directly instead and needs the same context."""
    project = await project_service.get_project(session, project_id)
    endpoint_configs = await endpoint_service.list_endpoint_configs(session, project_id=project_id)
    return {
        "project": project,
        "endpoint_configs": endpoint_configs,
        "deploy_config": project.deploy_config,
        "default_max_tokens": settings.default_max_tokens,
    }


@router.get("/projects/{project_id}/settings")
async def project_settings(project_id: str, request: Request, session: SessionDep):
    return templates.TemplateResponse(
        request, "project_settings.html.j2", await _project_settings_context(session, project_id)
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
    return RedirectResponse(f"/ui/projects/{project_id}/settings#details-pane", status_code=303)


@router.post("/projects/{project_id}/prompts")
async def update_project_prompts(
    project_id: str,
    session: SessionDep,
    pm_guidance: str = Form(""),
    developer_guidance: str = Form(""),
    tester_guidance: str = Form(""),
    reviewer_guidance: str = Form(""),
    deployer_guidance: str = Form(""),
) -> RedirectResponse:
    await project_service.update_project(
        session,
        project_id,
        pm_guidance=pm_guidance or None,
        developer_guidance=developer_guidance or None,
        tester_guidance=tester_guidance or None,
        reviewer_guidance=reviewer_guidance or None,
        deployer_guidance=deployer_guidance or None,
    )
    return RedirectResponse(f"/ui/projects/{project_id}/settings#prompts-pane", status_code=303)


@router.post("/projects/{project_id}/overseer-prompt")
async def update_overseer_prompt(
    project_id: str,
    request: Request,
    session: SessionDep,
    overseer_prompt: str = Form(""),
    force: bool = Form(False),
):
    """A blank submission always saves unconditionally (no judge call — an empty
    mandate can't fail "comprehensive enough", it's just off). A non-blank
    submission without force=true is judged first (agent/curation.py's
    assess_overseer_prompt); on a non-comprehensive verdict, or if the judge call
    itself fails, this re-renders the settings page directly (not a redirect —
    see _project_settings_context's docstring) with the feedback and the draft
    text preserved, plus a second submit button that resubmits with force=true.
    force is a one-shot escape hatch for that submission only, not a persisted
    approval — editing and resubmitting later gets freshly judged."""
    prompt = overseer_prompt.strip() or None
    if prompt is not None and not force:
        project = await project_service.get_project(session, project_id)
        try:
            chain = await endpoint_service.get_resolved_chain(session, project_id=project_id, role=Column.PM)
            assessment = await assess_overseer_prompt(prompt, project, llm_client=FallbackLLMClient(chain))
        except Exception as exc:  # noqa: BLE001 — soft-blocked, not swallowed; see assess_overseer_prompt
            context = await _project_settings_context(session, project_id)
            context.update(
                overseer_validation={"error": str(exc), "issues": []},
                overseer_prompt_draft=prompt,
                active_pane="overseer-pane",
            )
            return templates.TemplateResponse(request, "project_settings.html.j2", context)
        if not assessment.comprehensive:
            context = await _project_settings_context(session, project_id)
            context.update(
                overseer_validation={"error": None, "issues": assessment.issues},
                overseer_prompt_draft=prompt,
                active_pane="overseer-pane",
            )
            return templates.TemplateResponse(request, "project_settings.html.j2", context)
    await project_service.set_overseer_prompt(session, project_id, prompt)
    return RedirectResponse(f"/ui/projects/{project_id}/settings#overseer-pane", status_code=303)


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
    return RedirectResponse(f"/ui/projects/{project_id}/settings#endpoints-pane", status_code=303)


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
    return RedirectResponse(f"/ui/projects/{project_id}/settings#deploy-pane", status_code=303)
