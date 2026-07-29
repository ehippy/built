"""RunAttempt bookkeeping — the source of truth server-verified terminal transitions
check against (Tester's `approve`, and Deployer's `run_deploy` in Phase 5), so the
model can't just claim success in prose."""

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from built.db.models import RunAttempt
from built.domain.enums import RunAttemptStatus


async def record_run_attempt(
    session: AsyncSession,
    *,
    card_id: str,
    column_visit_id: str,
    command: str,
    exit_code: int,
    stdout: str,
    stderr: str,
) -> RunAttempt:
    prior = await session.scalar(
        select(func.count()).select_from(RunAttempt).where(RunAttempt.column_visit_id == column_visit_id)
    )
    attempt = RunAttempt(
        card_id=card_id,
        column_visit_id=column_visit_id,
        attempt_number=(prior or 0) + 1,
        status=RunAttemptStatus.SUCCEEDED if exit_code == 0 else RunAttemptStatus.FAILED,
        command_executed=command,
        exit_code=exit_code,
        stdout_ref=stdout[:20_000],
        stderr_ref=stderr[:20_000],
        ended_at=datetime.now(UTC),
    )
    session.add(attempt)
    await session.flush()
    return attempt


async def latest_run_attempt(session: AsyncSession, column_visit_id: str) -> RunAttempt | None:
    stmt = (
        select(RunAttempt)
        .where(RunAttempt.column_visit_id == column_visit_id)
        .order_by(RunAttempt.attempt_number.desc())
        .limit(1)
    )
    return await session.scalar(stmt)


async def latest_run_attempt_succeeded(session: AsyncSession, column_visit_id: str) -> bool:
    attempt = await latest_run_attempt(session, column_visit_id)
    return attempt is not None and attempt.status == RunAttemptStatus.SUCCEEDED
