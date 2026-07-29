import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

# Imported for its side effect: registers all ORM models on Base.metadata before
# create_all() runs. Nothing in this module references the import directly.
import built.db.models  # noqa: F401
from built.api.routers import board, cards, endpoint_configs, health, projects
from built.config import settings
from built.db.base import create_all
from built.orchestrator.worker import run_worker_pool

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await create_all()

    worker_task: asyncio.Task | None = None
    stop_event = asyncio.Event()
    if settings.orchestrator_enabled:
        worker_task = asyncio.create_task(
            run_worker_pool(
                concurrency=settings.orchestrator_concurrency,
                poll_interval=settings.orchestrator_poll_interval_seconds,
                stop_event=stop_event,
            )
        )

    try:
        yield
    finally:
        if worker_task is not None:
            stop_event.set()
            await worker_task


def create_app() -> FastAPI:
    app = FastAPI(title="built", description="Agentic software factory", lifespan=lifespan)

    app.include_router(health.router)
    app.include_router(projects.router)
    app.include_router(endpoint_configs.router)
    app.include_router(cards.router)
    app.include_router(board.router)

    return app


app = create_app()
