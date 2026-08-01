from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from built.db.models import EndpointConfig
from built.domain.endpoint_resolution import resolve_endpoint_chain
from built.domain.enums import Column
from built.services.project_service import NotFoundError


async def create_endpoint_config(
    session: AsyncSession,
    *,
    base_url: str,
    model: str,
    project_id: str | None = None,
    role: Column | None = None,
    priority: int = 0,
    api_key_ref: str | None = None,
    supports_tool_calling: bool = True,
    extra_params: dict | None = None,
    enabled: bool = True,
    max_concurrency: int = 1,
    context_window: int | None = None,
) -> EndpointConfig:
    endpoint = EndpointConfig(
        project_id=project_id,
        role=role,
        priority=priority,
        base_url=base_url,
        model=model,
        api_key_ref=api_key_ref,
        supports_tool_calling=supports_tool_calling,
        extra_params=extra_params or {},
        enabled=enabled,
        max_concurrency=max_concurrency,
        context_window=context_window,
    )
    session.add(endpoint)
    await session.flush()
    return endpoint


async def get_endpoint_config(session: AsyncSession, endpoint_id: str) -> EndpointConfig:
    endpoint = await session.get(EndpointConfig, endpoint_id)
    if endpoint is None:
        raise NotFoundError(f"no endpoint config {endpoint_id!r}")
    return endpoint


async def list_endpoint_configs(
    session: AsyncSession, *, project_id: str | None = None, only_global: bool = False
) -> list[EndpointConfig]:
    stmt = select(EndpointConfig).order_by(EndpointConfig.priority)
    if only_global:
        stmt = stmt.where(EndpointConfig.project_id.is_(None))
    elif project_id is not None:
        stmt = stmt.where(EndpointConfig.project_id == project_id)
    return list((await session.scalars(stmt)).all())


async def update_endpoint_config(session: AsyncSession, endpoint_id: str, **fields) -> EndpointConfig:
    endpoint = await get_endpoint_config(session, endpoint_id)
    for key, value in fields.items():
        if value is not None:
            setattr(endpoint, key, value)
    await session.flush()
    return endpoint


async def edit_endpoint_config(
    session: AsyncSession,
    endpoint_id: str,
    *,
    base_url: str,
    model: str,
    role: Column | None,
    priority: int,
    api_key_ref: str | None,
    max_concurrency: int,
    context_window: int | None,
    supports_tool_calling: bool,
) -> EndpointConfig:
    """Full-row edit for the UI's edit form. Unlike `update_endpoint_config` — whose
    "skip None" semantics exist so the JSON PATCH API can treat an omitted field as
    "leave unchanged" — every field here is always overwritten, so clearing
    api_key_ref/context_window back to unset, or role back to "all roles", actually
    takes effect instead of silently no-oping."""
    endpoint = await get_endpoint_config(session, endpoint_id)
    endpoint.base_url = base_url
    endpoint.model = model
    endpoint.role = role
    endpoint.priority = priority
    endpoint.api_key_ref = api_key_ref
    endpoint.max_concurrency = max_concurrency
    endpoint.context_window = context_window
    endpoint.supports_tool_calling = supports_tool_calling
    await session.flush()
    return endpoint


async def delete_endpoint_config(session: AsyncSession, endpoint_id: str) -> None:
    endpoint = await get_endpoint_config(session, endpoint_id)
    await session.delete(endpoint)
    await session.flush()


async def get_resolved_chain(
    session: AsyncSession, *, project_id: str | None, role: Column | None
) -> list[EndpointConfig]:
    return await resolve_endpoint_chain(session, project_id=project_id, role=role)
