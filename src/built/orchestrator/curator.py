"""Curation (agent/curation.py) on a schedule: wakes on its own, and for every
non-paused project checks which of the ActivityKinds are due. Also exposes
run_curation_activity for on-demand manual triggers (the board page's per-kind
"run now" buttons) — same underlying pass, just invoked once outside the timer.

Curation never edits the repo — every kind ends with propose_tasks, so unlike the
old Tender this replaces, its worktree needs no git identity, no commit, no push."""

import asyncio
import logging

from built.agent.context_window import ContextWindowConfig
from built.agent.curation import run_curation_pass
from built.config import settings
from built.db.base import async_session_factory
from built.db.models import Project
from built.domain.enums import ActivityKind, Column
from built.llm.client import FallbackLLMClient
from built.sandbox.container import DockerCommandExecutor
from built.sandbox.worktree import ensure_tool_worktree, read_default_branch_file
from built.services import card_service, endpoint_service, project_service
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)

# (project_id, kind) pairs with a curation pass currently in flight. In-memory only
# — a single-process app doesn't need this to survive a restart (see
# orchestrator/worker.py's claim_next_card docstring for the same reasoning).
# Keyed by kind too, not just project_id like the old discovery guard it replaces —
# a bug sweep and a polish review for the same project are independent and can run
# concurrently; only two triggers of the *same* kind for the *same* project collide.
_curation_in_progress: set[tuple[str, ActivityKind]] = set()


def is_curation_running(project_id: str, kind: ActivityKind) -> bool:
    return (project_id, kind) in _curation_in_progress


def _format_recent_outcomes(outcomes: list[dict]) -> str:
    lines = [f"- [{o['column']}] {o['card_title']}: {o['summary']}" for o in outcomes if o.get("summary")]
    return "\n".join(lines) or "(nothing new)"


async def _needs_run(session, project: Project, kind: ActivityKind) -> tuple[bool, str | None]:
    """Returns (should_run, extra_context). Every kind is WIP-limited by the PM
    column's backlog first — with no per-kind cooldown, curation could otherwise
    pile up cards far faster than a concurrency-capped orchestrator can ever work
    through. This only gates the *automatic* scheduler loop below; a human
    explicitly clicking "run now" (run_curation_activity called directly from the
    UI) still always fires.

    Past that: agents_md is gated by "has anything closed since last run" (its
    extra_context is that closed work, formatted) — the natural throttle for a
    sparse-signal activity. The other three have no cooldown at all: every
    scheduler wake is another chance to find new work, so
    curator_poll_interval_seconds is the only pacing — keep proposing as long as
    there's room in the backlog."""
    pm_backlog = await card_service.count_column_backlog(session, project.id, Column.PM)
    if pm_backlog >= settings.curator_max_pm_backlog:
        return False, None

    if kind != ActivityKind.AGENTS_MD:
        return True, None
    last_run = await project_service.get_activity_last_run(session, project.id, kind)
    outcomes = await card_service.list_recent_visit_outcomes(session, project.id, since=last_run)
    if not outcomes:
        return False, None
    return True, _format_recent_outcomes(outcomes)


async def run_curation_activity(
    project_id: str, kind: ActivityKind, *, extra_context: str | None = None
) -> None:
    """Runs one curation pass for one project/kind and returns — meant to be fired
    as a detached background task (manual trigger) or awaited inline from the
    scheduler loop below. Never raises — run_curation_pass already swallows its own
    failures."""
    if is_curation_running(project_id, kind):
        logger.info("curation %s already running for project %s — skipping", kind.value, project_id)
        return
    _curation_in_progress.add((project_id, kind))
    try:
        async with async_session_factory() as session:
            project = await session.get(Project, project_id)
            if project is None:
                logger.warning("curation requested for missing project %s", project_id)
                return
            logger.info("curation %s starting for project %s (%r)", kind.value, project_id, project.name)
            try:
                chain = await endpoint_service.get_resolved_chain(
                    session, project_id=project.id, role=Column.PM
                )
                llm_client = FallbackLLMClient(chain)
                # Own dedicated worktree + branch, not Deployer's/Discovery's — git
                # refuses to check out the same branch in two worktrees at once.
                wt_path = await ensure_tool_worktree(project, tool="curator")
            except Exception:
                logger.exception("curation setup failed for project %s", project_id)
                return

            executor_kwargs = {"image": project.sandbox_image} if project.sandbox_image else {}
            dispatcher = ToolDispatcher(
                ctx=ToolContext(card_id=f"curator-{kind.value}-{project.id}", worktree_root=wt_path),
                executor=DockerCommandExecutor(**executor_kwargs),
            )

            if project.max_tokens:
                max_tokens = project.max_tokens
            elif any(e.context_window for e in chain):
                max_tokens = max(e.context_window for e in chain if e.context_window)
            else:
                max_tokens = settings.default_max_tokens

            agents_doc = await read_default_branch_file(project, "AGENTS.md")

            created = await run_curation_pass(
                session,
                project,
                kind,
                llm_client=llm_client,
                dispatcher=dispatcher,
                max_iterations=settings.curator_max_iterations,
                agents_doc=agents_doc,
                extra_context=extra_context,
                context_window_config=ContextWindowConfig(
                    max_tokens=max_tokens,
                    keep_messages=settings.default_keep_messages,
                ),
            )
            await project_service.record_activity_run(
                session, project.id, kind, summary=f"created {len(created)} card(s)"
            )
            await session.commit()
            logger.info("curation %s for project %s created %d card(s)", kind.value, project_id, len(created))
    finally:
        _curation_in_progress.discard((project_id, kind))


async def run_curator_once() -> None:
    """One scheduler wake: for every non-paused project, run whichever kinds are
    due. Meant to be called by the timer loop below.

    Two independent, project-scoped pause switches gate this, neither of which
    ever blocks a manual "run now" trigger (run_curation_activity): Project.paused_at
    (also stops the worker orchestrator and Reviver) skips the project outright;
    ProjectCurationState.paused_at skips just the curator for this project, and its
    disabled_kinds skips individual ActivityKinds while the rest stay on schedule."""
    async with async_session_factory() as session:
        projects = await project_service.list_projects(session)
        due: list[tuple[str, ActivityKind, str | None]] = []
        for project in projects:
            if project.paused_at is not None:
                continue
            curation_state = await project_service.get_curation_state(session, project.id)
            if curation_state is not None and curation_state.paused_at is not None:
                continue
            disabled_kinds = set(curation_state.disabled_kinds) if curation_state else set()
            for kind in ActivityKind:
                if kind.value in disabled_kinds:
                    continue
                should_run, extra_context = await _needs_run(session, project, kind)
                if should_run:
                    due.append((project.id, kind, extra_context))
    for project_id, kind, extra_context in due:
        await run_curation_activity(project_id, kind, extra_context=extra_context)


async def run_curator_loop(*, stop_event: asyncio.Event, poll_interval: float) -> None:
    while not stop_event.is_set():
        try:
            await run_curator_once()
        except Exception:
            # Crash-isolation: one bad cycle shouldn't kill the curator forever —
            # log and try again at the next scheduled wake.
            logger.exception("curator: unhandled error during cycle")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
        except TimeoutError:
            pass
