from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from built.db.models import EndpointConfig
from built.domain.enums import Column


async def resolve_endpoint_chain(
    session: AsyncSession, *, project_id: str | None, role: Column
) -> list[EndpointConfig]:
    """Resolve the fallback chain for (project, role): the first non-empty tier wins —
    tiers are never merged. Order: project+role rows -> project rows with role IS NULL
    -> global rows (project_id IS NULL, any role)."""

    async def _tier(*filters) -> list[EndpointConfig]:
        stmt = (
            select(EndpointConfig)
            .where(EndpointConfig.enabled.is_(True), *filters)
            .order_by(EndpointConfig.priority)
        )
        return list((await session.scalars(stmt)).all())

    if project_id is not None:
        tier = await _tier(EndpointConfig.project_id == project_id, EndpointConfig.role == role)
        if tier:
            return tier

        tier = await _tier(EndpointConfig.project_id == project_id, EndpointConfig.role.is_(None))
        if tier:
            return tier

    return await _tier(EndpointConfig.project_id.is_(None))
