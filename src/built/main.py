import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Imported for its side effect: registers all ORM models on Base.metadata before
# create_all() runs. Nothing in this module references the import directly.
import built.db.models  # noqa: F401
from built.api.routers import board as api_board
from built.api.routers import cards as api_cards
from built.api.routers import endpoint_configs as api_endpoint_configs
from built.api.routers import health
from built.api.routers import projects as api_projects
from built.config import settings
from built.db.base import create_all
from built.orchestrator.reviver import run_reviver_loop
from built.orchestrator.worker import run_worker_pool
from built.ui.routers import board as ui_board
from built.ui.routers import cards as ui_cards
from built.ui.routers import endpoint_configs as ui_endpoint_configs
from built.ui.routers import projects as ui_projects

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "ui" / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await create_all()

    worker_task: asyncio.Task | None = None
    reviver_task: asyncio.Task | None = None
    stop_event = asyncio.Event()
    if settings.orchestrator_enabled:
        worker_task = asyncio.create_task(
            run_worker_pool(
                concurrency=settings.orchestrator_concurrency,
                poll_interval=settings.orchestrator_poll_interval_seconds,
                stop_event=stop_event,
            )
        )
    if settings.reviver_enabled:
        reviver_task = asyncio.create_task(
            run_reviver_loop(
                stop_event=stop_event,
                poll_interval=settings.reviver_poll_interval_seconds,
            )
        )

    try:
        yield
    finally:
        if worker_task is not None or reviver_task is not None:
            stop_event.set()
        if worker_task is not None:
            await worker_task
        if reviver_task is not None:
            await reviver_task


def create_app() -> FastAPI:
    app = FastAPI(title="built", description="Agentic software factory", lifespan=lifespan)

    app.mount("/ui/static", StaticFiles(directory=str(STATIC_DIR)), name="ui-static")

    app.include_router(health.router)
    app.include_router(api_projects.router)
    app.include_router(api_endpoint_configs.router)
    app.include_router(api_cards.router)
    app.include_router(api_board.router)

    app.include_router(ui_projects.router)
    app.include_router(ui_board.router)
    app.include_router(ui_cards.router)
    app.include_router(ui_endpoint_configs.router)

    return app


app = create_app()
