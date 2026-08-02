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


def _top_level_operator_spans(command: str) -> list[tuple[int, int, str]] | None:
    """Scans `command` for top-level (outside quotes) `&&`, bare `;`, and bare `|`
    occurrences — the only shell structure this module reasons about — and returns
    each as (start, end, operator) in order of appearance. Returns None if the
    command contains unbalanced quotes, or any construct outside that short list
    (`||`, backgrounding `&`, subshells, command substitution): callers then fall
    back to requiring a whole-string exact match, which is always safe, just
    occasionally stricter than a human would insist on. Deliberately not a full
    shell parser — wide enough to recognize the shapes agents actually produce,
    narrow enough that "can't confidently parse it" is the default, not the
    exception."""
    spans: list[tuple[int, int, str]] = []
    quote: str | None = None
    i, n = 0, len(command)
    while i < n:
        ch = command[i]
        if quote:
            if quote == '"' and ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch == "\\":
            i += 2
            continue
        if ch in "()`":
            return None
        if ch == "$" and command[i : i + 2] == "$(":
            return None
        if command[i : i + 2] == "&&":
            spans.append((i, i + 2, "&&"))
            i += 2
            continue
        if ch == "&":
            return None
        if command[i : i + 2] == "||":
            return None
        if ch == "|":
            spans.append((i, i + 1, "|"))
            i += 1
            continue
        if ch == ";":
            spans.append((i, i + 1, ";"))
            i += 1
            continue
        i += 1
    if quote is not None:
        return None
    return spans


def _match_test_command_shape(command: str, test_command: str) -> str | None:
    """Recognizes the shapes of `command` this module can safely verify against
    `test_command`, beyond a plain whole-string match:
    - "chained": test_command is the final `&&`/`;`-separated clause, with no pipe
      in it — safe with no extra bookkeeping, since bash's own exit code after a
      plain `&&`/`;` chain already reflects only the last command run, whether or
      not something else ran first in the same call.
    - "piped": test_command is the first stage of a pipe that is (or is the tail
      of a `&&`/`;` chain and is) the last thing the call runs — the pipeline's
      *overall* exit code reflects whatever ran last in the pipe, not
      test_command's own, so this shape is only trustworthy when paired with a
      captured $PIPESTATUS (see RunAttempt.pipestatus / sandbox/container.py's
      _wrap_for_pipestatus) — callers must check pipestatus[0], not exit_code.
    Returns None if `command` doesn't match test_command in any shape this module
    understands — including whenever _top_level_operator_spans itself can't
    confidently parse it, which fails closed to "not recognized" rather than
    guessing."""
    expected = _normalize_command(test_command)
    if _normalize_command(command) == expected:
        return "whole"
    spans = _top_level_operator_spans(command)
    if not spans:
        return None
    chain_spans = [s for s in spans if s[2] in ("&&", ";")]
    clause_start = chain_spans[-1][1] if chain_spans else 0
    pipe_spans = [s for s in spans if s[2] == "|" and s[0] >= clause_start]
    if pipe_spans:
        first_stage = command[clause_start : pipe_spans[0][0]]
        return "piped" if _normalize_command(first_stage) == expected else None
    last_clause = command[clause_start:]
    if chain_spans and _normalize_command(last_clause) == expected:
        return "chained"
    return None


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
    pipestatus: list[int] | None = None,
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
        pipestatus=pipestatus,
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


async def diagnose_missing_passing_run(
    session: AsyncSession, column_visit_id: str, test_command: str
) -> str | None:
    """The rigid test gate Developer's `submit_for_test` and Tester's `approve` both
    check server-side, as a specific human-readable reason (None if the gate is
    satisfied) rather than a bare bool — so a rejected model is told what it
    actually did wrong instead of having to guess. Confirmed in production: a
    Developer visit ground through 500+ iterations misreading a generic rejection
    as "my edit is a no-op", because nothing told it its bash call was piped
    (`npm test | grep ...`) rather than the bare command the gate requires.

    Satisfied only if ALL of:
    - the most recent bash call in this visit runs test_command, in one of the
      shapes _match_test_command_shape recognizes: verbatim; as the final
      `&&`/`;`-separated clause; or as the first stage of a pipe that is (or ends)
      the call — not just "some command exited 0", which was the old, gameable
      check (run the suite, see red, then run `true`);
    - the test command's own exit code — pipestatus[0] for the piped shape,
      exit_code otherwise, since a pipeline's overall exit_code reflects whatever
      ran last in it, not test_command's — was 0;
    - nothing has mutated the repo (write_file/edit_file, or a bash call that
      changed tracked files) since — otherwise a green run followed by an
      unverified edit would still read as "tested", which is exactly the failure
      mode the Tester's job of actively strengthening tests runs into."""
    attempt = await latest_run_attempt(session, column_visit_id)
    if attempt is None:
        return f"You haven't run the test command (`{test_command}`) via bash yet this visit."
    command = attempt.command_executed or ""
    shape = _match_test_command_shape(command, test_command)
    if shape is None:
        culprit = ""
        if _top_level_operator_spans(command) is None:
            culprit = (
                " — it uses shell features (background `&`, `||`, subshells, or command "
                "substitution) this project can't safely verify piece-by-piece"
            )
        elif "|" in command:
            culprit = (
                " — it's piped into something else, but the part before the first pipe isn't "
                f"exactly `{test_command}`. The test command must be the very first stage of the "
                "pipe for its own exit code to count, e.g. "
                f"`{test_command} | grep ...` is fine, `... | {test_command}` is not"
            )
        elif "&&" in command or ";" in command:
            culprit = (
                " — it's chained with another command, but the final clause isn't exactly "
                f"`{test_command}`. The test command must be the last thing the call runs, e.g. "
                f"`edit-command && {test_command}` is fine, `{test_command} && other-command` is not"
            )
        return (
            f"Your most recent bash call was `{command}`, not the exact command `{test_command}`"
            f"{culprit}."
        )
    if shape == "piped":
        stage_rc = attempt.pipestatus[0] if attempt.pipestatus else None
        if stage_rc is None:
            return (
                f"Your last bash call piped `{test_command}` into something else, but its own exit "
                "code wasn't captured — rerun it."
            )
        if stage_rc != 0:
            return (
                f"Your last bash call piped `{test_command}` into something else, and the test "
                f"command's own exit code was {stage_rc}, not 0 — the pipeline's overall exit code "
                "doesn't count here, only the test command's own. Fix the failure and rerun."
            )
    elif attempt.status != RunAttemptStatus.SUCCEEDED:
        return (
            f"Your most recent bash run (`{command}`) exited "
            f"{attempt.exit_code}, not 0 — fix the failure and rerun `{test_command}` before submitting."
        )
    if attempt.card_event_seq is None:
        return f"Your most recent `{test_command}` run wasn't recorded with an event — rerun it."
    later_payloads = await session.scalars(
        select(CardEvent.payload).where(
            CardEvent.column_visit_id == column_visit_id,
            CardEvent.type == EventType.TOOL_CALL,
            CardEvent.seq > attempt.card_event_seq,
        )
    )
    if any(payload.get("commit_sha") for payload in later_payloads):
        return (
            f"You've changed a file since your last passing `{test_command}` run — rerun it "
            "after your final edit, with nothing else in between, before submitting."
        )
    return None


async def has_passing_run_since_last_change(
    session: AsyncSession, column_visit_id: str, test_command: str
) -> bool:
    """Bool wrapper around diagnose_missing_passing_run — see that docstring for
    exactly what this checks."""
    return await diagnose_missing_passing_run(session, column_visit_id, test_command) is None


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
