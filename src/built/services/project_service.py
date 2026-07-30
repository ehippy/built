import re
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from built.config import settings
from built.db.models import (
    CurationEvent,
    DeployConfig,
    EpicLink,
    Project,
    ProjectActivityRun,
    ProjectCurationState,
)
from built.domain.enums import ActivityKind, DeployKind, DeployMode


class NotFoundError(Exception):
    pass


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "project"


async def _unique_slug(session: AsyncSession, name: str) -> str:
    base = _slugify(name)
    slug = base
    suffix = 2
    while await session.scalar(select(Project.id).where(Project.slug == slug)):
        slug = f"{base}-{suffix}"
        suffix += 1
    return slug


async def create_project(
    session: AsyncSession,
    *,
    name: str,
    overarching_goal: str,
    repo_remote_url: str,
    default_branch: str = "main",
    sandbox_image: str | None = None,
    test_command: str | None = None,
    max_revisions: int | None = None,
    max_deploy_attempts: int | None = None,
    max_iterations_per_run: int | None = None,
    max_tokens: int | None = None,
) -> Project:
    project = Project(
        name=name,
        slug=await _unique_slug(session, name),
        overarching_goal=overarching_goal,
        repo_remote_url=repo_remote_url,
        default_branch=default_branch,
        sandbox_image=sandbox_image,
        test_command=test_command,
        max_revisions=max_revisions if max_revisions is not None else settings.default_max_revisions,
        max_deploy_attempts=(
            max_deploy_attempts if max_deploy_attempts is not None else settings.default_max_deploy_attempts
        ),
        max_iterations_per_run=(
            max_iterations_per_run
            if max_iterations_per_run is not None
            else settings.default_max_iterations_per_run
        ),
        max_tokens=max_tokens if max_tokens is not None else None,
    )
    # A brand-new object was never loaded via a query, so the selectin strategy on
    # Project.deploy_config never fires for it — set it directly (we know it's None,
    # there's no way to create one in the same call) rather than let ProjectOut
    # serialization try to lazy-load it later and hit MissingGreenlet.
    project.deploy_config = None
    session.add(project)
    await session.flush()
    return project


async def get_project(session: AsyncSession, project_id: str) -> Project:
    # Explicit selectinload rather than relying on session.get()'s handling of the
    # mapper-level lazy="selectin" default — belt and suspenders around a relationship
    # that ProjectOut always serializes.
    project = await session.get(Project, project_id, options=[selectinload(Project.deploy_config)])
    if project is None:
        raise NotFoundError(f"no project {project_id!r}")
    return project


async def list_projects(session: AsyncSession, *, include_archived: bool = False) -> list[Project]:
    stmt = select(Project).options(selectinload(Project.deploy_config)).order_by(Project.created_at)
    if not include_archived:
        stmt = stmt.where(Project.archived_at.is_(None))
    return list((await session.scalars(stmt)).all())


async def update_project(session: AsyncSession, project_id: str, **fields) -> Project:
    project = await get_project(session, project_id)
    for key, value in fields.items():
        if value is not None:
            setattr(project, key, value)
    await session.flush()
    return project


async def archive_project(session: AsyncSession, project_id: str) -> Project:
    project = await get_project(session, project_id)
    project.archived_at = datetime.now(UTC)
    await session.flush()
    return project


async def pause_project(session: AsyncSession, project_id: str) -> Project:
    project = await get_project(session, project_id)
    project.paused_at = datetime.now(UTC)
    await session.flush()
    return project


async def resume_project(session: AsyncSession, project_id: str) -> Project:
    project = await get_project(session, project_id)
    project.paused_at = None
    await session.flush()
    return project


async def set_current_epic(session: AsyncSession, project_id: str, epic_card_id: str | None) -> Project:
    """The board page's "current epic" control — human-set-only, never touched by
    define_epic itself (agent/loop.py), so a human's chosen focus is never silently
    swapped out from under them. Nudges PM/curation prompts (agent/context.py)
    alongside overarching_goal, never gates which cards the orchestrator claims.
    epic_card_id=None clears it."""
    project = await get_project(session, project_id)
    if epic_card_id is not None:
        # Raw query rather than services.card_service.is_epic — card_service already
        # imports this module (get_project/NotFoundError), so importing back would
        # be circular.
        is_epic = await session.scalar(
            select(EpicLink.card_id).where(EpicLink.parent_card_id == epic_card_id).limit(1)
        )
        if is_epic is None:
            raise ValueError(f"card {epic_card_id!r} is not an epic")
    project.current_epic_id = epic_card_id
    await session.flush()
    return project


async def get_curation_state(session: AsyncSession, project_id: str) -> ProjectCurationState | None:
    """None means fully active — nothing paused, no kind disabled (see
    ProjectCurationState's docstring). Callers that need concrete paused_at/
    disabled_kinds values should treat a None row the same as one with both unset."""
    return await session.get(ProjectCurationState, project_id)


async def _get_or_create_curation_state(session: AsyncSession, project_id: str) -> ProjectCurationState:
    state = await session.get(ProjectCurationState, project_id)
    if state is None:
        state = ProjectCurationState(project_id=project_id)
        session.add(state)
        await session.flush()
    return state


async def pause_curation(session: AsyncSession, project_id: str) -> ProjectCurationState:
    """Pauses only the curator's automatic scheduler for this project — the worker
    orchestrator and Reviver keep running, unlike pause_project. A human's manual
    "run now" trigger still always fires (see ProjectCurationState's docstring)."""
    state = await _get_or_create_curation_state(session, project_id)
    state.paused_at = datetime.now(UTC)
    await session.flush()
    return state


async def resume_curation(session: AsyncSession, project_id: str) -> ProjectCurationState:
    state = await _get_or_create_curation_state(session, project_id)
    state.paused_at = None
    await session.flush()
    return state


async def set_curation_kind_enabled(
    session: AsyncSession, project_id: str, kind: ActivityKind, *, enabled: bool
) -> ProjectCurationState:
    """Turns one ActivityKind on/off in the automatic scheduler for this project,
    independent of the others and of pause_curation's project-wide switch. Always
    reassigns a new list (rather than mutating disabled_kinds in place) — JSON
    columns aren't change-tracked on in-place mutation, only on reassignment."""
    state = await _get_or_create_curation_state(session, project_id)
    disabled = set(state.disabled_kinds)
    if enabled:
        disabled.discard(kind.value)
    else:
        disabled.add(kind.value)
    state.disabled_kinds = sorted(disabled)
    await session.flush()
    return state


async def get_activity_last_run(
    session: AsyncSession, project_id: str, kind: ActivityKind
) -> datetime | None:
    """When a curation activity (orchestrator/curator.py) last ran for this project
    and kind — None if it's never run. Replaces the old single-purpose
    Project.agents_doc_tended_at, generalized to every ActivityKind."""
    stmt = select(ProjectActivityRun.last_run_at).where(
        ProjectActivityRun.project_id == project_id, ProjectActivityRun.kind == kind
    )
    return await session.scalar(stmt)


async def list_activity_runs(
    session: AsyncSession, project_id: str
) -> dict[ActivityKind, ProjectActivityRun]:
    """Every curation kind's last-run row for this project, keyed by kind — what the
    board page's curation status panel renders. A kind with no row yet just isn't a
    key in the returned dict."""
    stmt = select(ProjectActivityRun).where(ProjectActivityRun.project_id == project_id)
    rows = (await session.scalars(stmt)).all()
    return {row.kind: row for row in rows}


async def get_latest_curation_event(
    session: AsyncSession, project_id: str, kind: ActivityKind
) -> CurationEvent | None:
    """The single most recent transcript entry for one curation kind — what the
    status panel shows as "what it's doing right now" while a pass is running (and
    what it did last, once it's finished)."""
    stmt = (
        select(CurationEvent)
        .where(CurationEvent.project_id == project_id, CurationEvent.kind == kind)
        .order_by(CurationEvent.seq.desc())
        .limit(1)
    )
    return await session.scalar(stmt)


async def record_activity_run(
    session: AsyncSession, project_id: str, kind: ActivityKind, *, summary: str | None = None
) -> ProjectActivityRun:
    stmt = select(ProjectActivityRun).where(
        ProjectActivityRun.project_id == project_id, ProjectActivityRun.kind == kind
    )
    run = await session.scalar(stmt)
    if run is None:
        run = ProjectActivityRun(project_id=project_id, kind=kind, last_run_at=datetime.now(UTC))
        session.add(run)
    else:
        run.last_run_at = datetime.now(UTC)
    run.last_result_summary = summary
    await session.flush()
    return run


async def set_deploy_config(
    session: AsyncSession,
    project_id: str,
    *,
    kind: DeployKind,
    mode: DeployMode = DeployMode.PR_TO_OPERATOR,
    command: str | None = None,
    script_path: str | None = None,
    webhook_url: str | None = None,
    env_var_refs: list[str] | None = None,
    timeout_seconds: int = 600,
    github_token_ref: str | None = None,
) -> DeployConfig:
    project = await get_project(session, project_id)
    if project.deploy_config is None:
        project.deploy_config = DeployConfig(project_id=project.id, kind=kind)
    config = project.deploy_config
    config.kind = kind
    config.mode = mode
    config.command = command
    config.script_path = script_path
    config.webhook_url = webhook_url
    config.env_var_refs = env_var_refs or []
    config.timeout_seconds = timeout_seconds
    config.github_token_ref = github_token_ref
    await session.flush()
    return config
