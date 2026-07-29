import re
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from built.config import settings
from built.db.models import DeployConfig, Project
from built.domain.enums import DeployKind, DeployMode


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
    max_revisions: int | None = None,
    max_deploy_attempts: int | None = None,
    max_iterations_per_run: int | None = None,
) -> Project:
    project = Project(
        name=name,
        slug=await _unique_slug(session, name),
        overarching_goal=overarching_goal,
        repo_remote_url=repo_remote_url,
        default_branch=default_branch,
        sandbox_image=sandbox_image,
        max_revisions=max_revisions if max_revisions is not None else settings.default_max_revisions,
        max_deploy_attempts=(
            max_deploy_attempts if max_deploy_attempts is not None else settings.default_max_deploy_attempts
        ),
        max_iterations_per_run=(
            max_iterations_per_run
            if max_iterations_per_run is not None
            else settings.default_max_iterations_per_run
        ),
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
