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
# Strips a leading "cd <path> && "/"cd <path>; " an agent commonly tacks on out of
# habit (confirmed live: a Developer visit stuck retrying forever because it kept
# prefixing "cd /workspace && " onto an otherwise-exact, actually-passing command).
# bash already runs with cwd=/workspace (see sandbox/container.py), so this is
# always redundant — and if it ever weren't (a cd to some other directory), the
# command it wraps would just fail on its own relative paths and surface as a real
# nonzero exit code anyway, not something normalization needs to guard against.
_LEADING_CD_RE = re.compile(r"^\s*cd\s+\S+\s*(?:&&|;)\s*")


def _normalize_command(command: str) -> str:
    stripped = _LEADING_CD_RE.sub("", command.strip())
    return _TRAILING_REDIRECT_RE.sub("", stripped).strip()


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


async def has_made_any_change_this_visit(session: AsyncSession, column_visit_id: str) -> bool:
    """A second, narrower gate on top of has_passing_run_since_last_change: a
    Developer visit that never calls write_file/edit_file (or a bash command that
    changes tracked files) can still trivially satisfy that check by running an
    already-passing test command against an untouched repo — confirmed in
    production on a large migration card where four consecutive Developer visits
    each read the whole repo repeatedly and then submitted having written nothing
    at all, because the project's test command only covered pre-existing game
    logic the task never touched. Same failure shape as the gameable check this
    module already closed once (claim success without real verification) — this
    closes the sibling loophole (verify something true but irrelevant, then claim
    success) rather than assuming any passing run implies real work happened."""
    result = await session.scalars(
        select(CardEvent.payload).where(
            CardEvent.column_visit_id == column_visit_id,
            CardEvent.type == EventType.TOOL_CALL,
        )
    )
    return any(payload.get("commit_sha") for payload in result)


async def has_completed_plan_this_visit(session: AsyncSession, column_visit_id: str) -> bool:
    """Developer's submit_for_test gate on planning (see llm.tool_schemas.UPDATE_PLAN):
    true only if the most recent update_plan call in this visit has at least one
    step and every step marked done. Only the latest call counts — an earlier
    all-done snapshot doesn't survive a later update_plan call that adds a new
    pending step, exactly like has_passing_run_since_last_change only trusts the
    latest bash run rather than any run that ever passed."""
    result = await session.scalars(
        select(CardEvent.payload)
        .where(
            CardEvent.column_visit_id == column_visit_id,
            CardEvent.type == EventType.TOOL_CALL,
        )
        .order_by(CardEvent.seq)
    )
    latest_steps: list | None = None
    for payload in result:
        if payload.get("name") == "update_plan":
            latest_steps = payload.get("arguments", {}).get("steps") or []
    if not latest_steps:
        return False
    return all(step.get("status") == "done" for step in latest_steps)
