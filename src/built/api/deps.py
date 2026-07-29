from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from built.config import settings
from built.db.base import async_session_factory


async def get_session() -> AsyncIterator[AsyncSession]:
    async with async_session_factory() as session:
        async with session.begin():
            yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """v1 security floor for mutating endpoints. There's no human-approval gate
    anywhere in the pipeline and Deployer can trigger a real deployment, so this API
    must not be open by default even for a local/dev deployment — if no key is
    configured, mutating requests are refused rather than silently allowed."""
    if not settings.api_key:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "BUILT_API_KEY is not configured on the server — mutating endpoints are disabled until it is.",
        )
    if x_api_key != settings.api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key header.")


RequireApiKey = Annotated[None, Depends(require_api_key)]
