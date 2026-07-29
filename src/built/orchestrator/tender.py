"""Runs the Tender (agent/tender.py) on a timer — wakes on its own, no manual
trigger. One project at a time: for each project with visits closed since its last
pass, review and maybe edit AGENTS.md, then push if anything changed."""

import asyncio
import logging
from datetime import UTC, datetime

from built.agent.tender import run_tender_pass
from built.config import settings
from built.db.base import async_session_factory
from built.db.models import Project
from built.domain.enums import Column
from built.llm.client import FallbackLLMClient
from built.sandbox.container import DockerCommandExecutor
from built.sandbox.worktree import ensure_tool_worktree
from built.services import card_service, endpoint_service, project_service
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher
from built.tools.git_tools import GitCommandError, run_git

logger = logging.getLogger(__name__)


async def _needs_tending(session, project: Project) -> bool:
    outcomes = await card_service.list_recent_visit_outcomes(
        session, project.id, since=project.agents_doc_tended_at, limit=1
    )
    return bool(outcomes)


async def _tend_one_project(session, project: Project) -> None:
    try:
        chain = await endpoint_service.get_resolved_chain(session, project_id=project.id, role=Column.PM)
        llm_client = FallbackLLMClient(chain)
        wt_path = await ensure_tool_worktree(project, tool="tender")
    except Exception:
        logger.exception("tender: setup failed for project %s", project.id)
        return

    executor_kwargs = {"image": project.sandbox_image} if project.sandbox_image else {}
    dispatcher = ToolDispatcher(
        ctx=ToolContext(card_id=f"tender-{project.id}", worktree_root=wt_path),
        executor=DockerCommandExecutor(**executor_kwargs),
    )
    result = await run_tender_pass(
        session,
        project,
        llm_client=llm_client,
        dispatcher=dispatcher,
        max_iterations=settings.tender_max_iterations,
    )

    if result["edited"]:
        try:
            await run_git("push", "origin", f"HEAD:{project.default_branch}", cwd=wt_path)
        except GitCommandError:
            logger.exception("tender: push failed for project %s — edit stays local for now", project.id)

    project.agents_doc_tended_at = datetime.now(UTC)
    await session.commit()
    logger.info(
        "tender pass for project %s: edited=%s summary=%r", project.id, result["edited"], result["summary"]
    )


async def run_tender_once() -> None:
    """One pass over every project that has something new to review, in its own
    session — meant to be called by the timer loop below."""
    async with async_session_factory() as session:
        projects = await project_service.list_projects(session)
        for project in projects:
            if await _needs_tending(session, project):
                await _tend_one_project(session, project)


async def run_tender_loop(*, stop_event: asyncio.Event, poll_interval: float) -> None:
    while not stop_event.is_set():
        try:
            await run_tender_once()
        except Exception:
            # Crash-isolation: one bad cycle shouldn't kill the Tender forever — log
            # and try again at the next scheduled wake.
            logger.exception("tender: unhandled error during cycle")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
        except TimeoutError:
            pass
