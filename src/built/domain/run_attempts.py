"""RunAttempt bookkeeping — the source of truth server-verified terminal transitions
check against (Developer's `submit_for_test`, Tester's `approve`, and Deployer's
`run_deploy` in Phase 5), so the model can't just claim success in prose."""

import re
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from built.db.models import CardEvent, RunAttempt
from built.domain.enums import EventType, RunAttemptStatus

# Strips a trailing stderr/output redirection an agent commonly tacks onto an
# otherwise-exact command (e.g. "pytest -q 2>&1") — doesn't change what ran or its
# exit code, so it shouldn't cause an exact-match comparison to fail.
_TRAILING_REDIRECT_RE = re.compile(r"\s*(?:2>&1|2>/dev/null|>\s*/dev/null(?:\s+2>&1)?)\s*$")


def _normalize_command(command: str) -> str:
    return _TRAILING_REDIRECT_RE.sub("", command.strip()).strip()


async def record_run_attempt(
    session: AsyncSession,
    *,
    card_id: str,
    column_visit_id: str,
    command: str,
    exit_code: int,
    stdout: str,
    stderr: str,
    card_event_seq: int | None = None,
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
        card_event_seq=card_event_seq,
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


async def has_passing_run_since_last_change(
    session: AsyncSession, column_visit_id: str, test_command: str
) -> bool:
    """The rigid test gate Developer's `submit_for_test` and Tester's `approve` both
    check server-side. True only if ALL of:
    - the most recent bash call in this visit is, mod trailing redirection, exactly
      the project's configured test_command (not just "some command exited 0" —
      that was the old, gameable check: run the suite, see red, then run `true`);
    - it exited 0;
    - nothing has mutated the repo (write_file/edit_file, or a bash call that
      changed tracked files) since — otherwise a green run followed by an
      unverified edit would still read as "tested", which is exactly the failure
      mode the Tester's new job of actively strengthening tests runs into."""
    attempt = await latest_run_attempt(session, column_visit_id)
    if attempt is None or attempt.status != RunAttemptStatus.SUCCEEDED:
        return False
    if _normalize_command(attempt.command_executed or "") != _normalize_command(test_command):
        return False
    if attempt.card_event_seq is None:
        return False
    later_payloads = await session.scalars(
        select(CardEvent.payload).where(
            CardEvent.column_visit_id == column_visit_id,
            CardEvent.type == EventType.TOOL_CALL,
            CardEvent.seq > attempt.card_event_seq,
        )
    )
    return not any(payload.get("commit_sha") for payload in later_payloads)
