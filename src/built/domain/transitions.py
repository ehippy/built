"""The card state machine. This is the one place revision-cap, deploy-cap, and
run-error handling live — the safety valves in an otherwise fully-autonomous pipeline.
Every function here is pure business logic over ORM objects; it doesn't care whether
it's called by an integration test (Phase 2) or the agent loop after a tool call
(Phase 3+)."""

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from built.db.models import Card, CardColumnVisit, EpicLink, Project
from built.domain.enums import Column, EventType, LifecycleState, VisitOutcome
from built.domain.events import append_event


async def start_visit(session: AsyncSession, card: Card) -> CardColumnVisit:
    """Open a new CardColumnVisit for the card's current column, numbering re-entries
    (e.g. a Developer visit after a revision bounce) with an incrementing attempt.
    Queries CardColumnVisit directly rather than touching `card.column_visits` — the
    caller may have fetched `card` without eager-loading that relationship, and async
    SQLAlchemy has no implicit lazy-load to fall back on."""
    prior_attempts = await session.scalar(
        select(func.count())
        .select_from(CardColumnVisit)
        .where(CardColumnVisit.card_id == card.id, CardColumnVisit.column == card.column)
    )
    visit = CardColumnVisit(card_id=card.id, column=card.column, attempt_number=(prior_attempts or 0) + 1)
    session.add(visit)
    await session.flush()
    return visit


async def _close_visit(
    session: AsyncSession,
    visit: CardColumnVisit,
    *,
    outcome: VisitOutcome,
    summary: str,
    endpoint_used: str | None = None,
    feedback: str | None = None,
) -> None:
    visit.ended_at = datetime.now(UTC)
    visit.outcome = outcome
    visit.summary = summary
    visit.endpoint_used = endpoint_used
    payload = {"column": visit.column.value, "outcome": outcome.value, "summary": summary}
    # Only request_changes callers pass this — the transcript previously showed just
    # the one-line `summary` for a changes_requested transition, never the fuller
    # `feedback` that actually became card.latest_feedback for the Developer's next
    # attempt, so anyone reading the transcript saw far less detail than the
    # Developer itself got.
    if feedback is not None:
        payload["feedback"] = feedback
    await append_event(
        session,
        card_id=visit.card_id,
        column_visit_id=visit.id,
        type=EventType.TRANSITION,
        payload=payload,
    )


async def complete_pm_visit(
    session: AsyncSession,
    card: Card,
    visit: CardColumnVisit,
    *,
    spec: str,
    acceptance_criteria: list[str],
    summary: str,
    endpoint_used: str | None = None,
) -> Card:
    """PM's submit_spec: record the spec + acceptance criteria, advance to Developer."""
    card.spec = spec
    card.acceptance_criteria = acceptance_criteria
    card.column = Column.DEVELOPER
    await _close_visit(
        session, visit, outcome=VisitOutcome.SUBMITTED, summary=summary, endpoint_used=endpoint_used
    )
    return card


async def split_pm_visit(
    session: AsyncSession,
    card: Card,
    visit: CardColumnVisit,
    *,
    summary: str,
    endpoint_used: str | None = None,
) -> Card:
    """PM's split_into_subtasks: this card was too broad for one Developer visit to
    implement and one Tester visit to verify coherently. The replacement cards
    already exist on the backlog by the time this is called (agent/loop.py's handler
    creates them first) — archive this one rather than advancing it, the same way a
    human archiving a card works: history/events/visits stay intact and it's still
    reachable at its own URL, it just drops off the board and out of claiming."""
    card.archived_at = datetime.now(UTC)
    await _close_visit(
        session, visit, outcome=VisitOutcome.SPLIT, summary=summary, endpoint_used=endpoint_used
    )
    return card


async def define_epic_visit(
    session: AsyncSession,
    card: Card,
    visit: CardColumnVisit,
    *,
    spec: str,
    summary: str,
    endpoint_used: str | None = None,
) -> Card:
    """PM's define_epic: this card becomes a live epic tracker for the child cards
    agent/loop.py's handler already created and linked (via services/card_service.py's
    link_epic_child) before calling this. Unlike split_pm_visit, NOT archived — it
    stays on the board, and `column` stays PM forever (it has no Developer/Tester/
    Reviewer/Deployer work of its own): get_board and count_column_backlog exclude
    it from the ordinary PM swimlane/backlog via EpicLink, and
    orchestrator/worker.py's claim_next_card excludes it from ever being claimed.
    It reaches lifecycle_state=DONE only via _maybe_complete_epic below, once every
    non-archived child does — never by an agent calling a terminal tool directly."""
    card.spec = spec
    await _close_visit(
        session, visit, outcome=VisitOutcome.EPIC_DEFINED, summary=summary, endpoint_used=endpoint_used
    )
    return card


async def _maybe_complete_epic(session: AsyncSession, card: Card) -> None:
    """Call right after a card's lifecycle_state is set to DONE (see the three call
    sites below). If `card` is an epic's child and every non-archived sibling has
    also reached DONE, the parent completes too — its own job (tracking the
    initiative) is now done. Zero non-archived siblings left doesn't count as "all
    done" (nothing to judge by — shouldn't happen in practice since define_epic
    requires at least 2 children, but guards against every child ending up
    archived). A child that ends FAILED or stays BLOCKED simply means the epic
    never auto-completes — conservative default, surfaced on the board's epic
    panel so a human notices why.

    Deliberately self-contained (raw select() over EpicLink/Card, not a
    services.card_service call) — services/card_service.py already imports this
    module, so importing back would be circular."""
    parent_id = await session.scalar(select(EpicLink.parent_card_id).where(EpicLink.card_id == card.id))
    if parent_id is None:
        return
    parent = await session.get(Card, parent_id)
    if parent is None or parent.lifecycle_state != LifecycleState.ACTIVE:
        return
    sibling_states = (
        await session.execute(
            select(Card.lifecycle_state)
            .join(EpicLink, EpicLink.card_id == Card.id)
            .where(EpicLink.parent_card_id == parent_id, Card.archived_at.is_(None))
        )
    ).scalars().all()
    if sibling_states and all(s == LifecycleState.DONE for s in sibling_states):
        parent.lifecycle_state = LifecycleState.DONE
        await append_event(
            session,
            card_id=parent.id,
            type=EventType.SYSTEM_NOTE,
            payload={"action": "epic_auto_completed", "completed_via_child": card.id},
        )


async def complete_developer_visit(
    session: AsyncSession,
    card: Card,
    visit: CardColumnVisit,
    *,
    summary: str,
    endpoint_used: str | None = None,
) -> Card:
    """Developer's submit_for_test: advance to Tester."""
    card.column = Column.TESTER
    await _close_visit(
        session, visit, outcome=VisitOutcome.SUBMITTED, summary=summary, endpoint_used=endpoint_used
    )
    return card


async def complete_tester_visit_approved(
    session: AsyncSession,
    card: Card,
    visit: CardColumnVisit,
    *,
    summary: str,
    endpoint_used: str | None = None,
) -> Card:
    """Tester's approve: advance to Reviewer. The caller (Phase 3+ tool dispatcher) is
    responsible for rejecting `approve` server-side unless the latest RunAttempt for
    this visit actually succeeded — this function trusts that check already happened.
    Passing tests only establishes the implementation behaves as specified; Reviewer
    is the separate gate on whether the diff itself (design, security, maintainability)
    is something worth shipping."""
    card.column = Column.REVIEWER
    await _close_visit(
        session, visit, outcome=VisitOutcome.APPROVED, summary=summary, endpoint_used=endpoint_used
    )
    return card


async def complete_reviewer_visit_approved(
    session: AsyncSession,
    card: Card,
    visit: CardColumnVisit,
    *,
    summary: str,
    endpoint_used: str | None = None,
) -> Card:
    """Reviewer's approve: advance to Deployer. Reviewer has no file-write or bash
    tools (see llm/tool_schemas.REVIEWER_TOOLS) — it can only read the diff and
    either approve or bounce it back, so this is a real second opinion on the
    implementation, not the same test-passing check Tester already did."""
    card.column = Column.DEPLOYER
    await _close_visit(
        session, visit, outcome=VisitOutcome.APPROVED, summary=summary, endpoint_used=endpoint_used
    )
    return card


async def complete_reviewer_visit_changes_requested(
    session: AsyncSession,
    card: Card,
    visit: CardColumnVisit,
    *,
    feedback: str,
    summary: str,
    endpoint_used: str | None = None,
) -> Card:
    """Reviewer's request_changes: bounce back to Developer, sharing the same
    revision_count budget as Tester's request_changes — both represent "another
    round of Developer work needed" against the same safety valve."""
    card.revision_count += 1
    card.latest_feedback = feedback
    card.column = Column.DEVELOPER
    project = await session.get(Project, card.project_id)
    if card.revision_count > project.max_revisions:
        card.lifecycle_state = LifecycleState.BLOCKED
    await _close_visit(
        session,
        visit,
        outcome=VisitOutcome.CHANGES_REQUESTED,
        summary=summary,
        endpoint_used=endpoint_used,
        feedback=feedback,
    )
    return card


async def complete_tester_visit_changes_requested(
    session: AsyncSession,
    card: Card,
    visit: CardColumnVisit,
    *,
    feedback: str,
    summary: str,
    endpoint_used: str | None = None,
) -> Card:
    """Tester's request_changes: bounce back to Developer and bump the revision
    counter. Exceeding the project's max_revisions blocks the card for a human —
    the safety valve in the Tester<->Developer loop, since nothing else bounds it."""
    card.revision_count += 1
    card.latest_feedback = feedback
    card.column = Column.DEVELOPER
    # session.get() hits the identity map, not a query, since the project is already
    # loaded in this session — async SQLAlchemy has no implicit lazy-load, so
    # `card.project.max_revisions` would raise MissingGreenlet instead.
    project = await session.get(Project, card.project_id)
    if card.revision_count > project.max_revisions:
        card.lifecycle_state = LifecycleState.BLOCKED
    await _close_visit(
        session,
        visit,
        outcome=VisitOutcome.CHANGES_REQUESTED,
        summary=summary,
        endpoint_used=endpoint_used,
        feedback=feedback,
    )
    return card


async def complete_deployer_visit(
    session: AsyncSession,
    card: Card,
    visit: CardColumnVisit,
    *,
    success: bool,
    summary: str,
    deploy_url: str | None = None,
    pending_ci_commit_sha: str | None = None,
    pending_pr_number: int | None = None,
    endpoint_used: str | None = None,
) -> Card:
    """Deployer's run_deploy (auto_main) or open_pull_request (pr_to_operator).

    Failure retries up to the project's max_deploy_attempts, then is terminal
    (failed) with no further action — there is no human-approval gate to fall back
    to in this pipeline.

    Success is terminal (done) UNLESS pending_ci_commit_sha (auto_main) or
    pending_pr_number (pr_to_operator) is given. In both those cases the Deployer's
    own job — the git-level push / the PR opened — is genuinely finished, but the
    card's overall completion isn't: it stays ACTIVE with the external handoff
    tracked for orchestrator/ci_watcher.py (CI) or orchestrator/pr_watcher.py (PR
    review) to poll. The CI side either confirms it DONE (confirm_ci_passed),
    blocks it for a human if CI never resolves (mark_ci_wait_timed_out), or reopens
    it back to Developer if CI comes back red (reopen_after_ci_failure below); the
    PR side confirms it DONE (confirm_pr_merged), bounces it back to Developer on
    review feedback (request_pr_changes), or blocks it for a human
    (mark_pr_wait_timed_out and friends)."""
    if success:
        if deploy_url is not None:
            card.deploy_url = deploy_url
        if pending_ci_commit_sha is not None:
            card.deploying_commit_sha = pending_ci_commit_sha
            card.deploying_since = datetime.now(UTC)
            await _close_visit(
                session,
                visit,
                outcome=VisitOutcome.DEPLOYED_PENDING_CI,
                summary=summary,
                endpoint_used=endpoint_used,
            )
            return card
        if pending_pr_number is not None:
            card.pr_number = pending_pr_number
            card.pr_waiting_since = datetime.now(UTC)
            await _close_visit(
                session,
                visit,
                outcome=VisitOutcome.DEPLOYED_PENDING_PR,
                summary=summary,
                endpoint_used=endpoint_used,
            )
            return card
        card.lifecycle_state = LifecycleState.DONE
        await _close_visit(
            session, visit, outcome=VisitOutcome.DONE, summary=summary, endpoint_used=endpoint_used
        )
        await _maybe_complete_epic(session, card)
        return card

    card.deploy_attempt_count += 1
    project = await session.get(Project, card.project_id)
    if card.deploy_attempt_count >= project.max_deploy_attempts:
        card.lifecycle_state = LifecycleState.FAILED
    await _close_visit(
        session, visit, outcome=VisitOutcome.FAILED, summary=summary, endpoint_used=endpoint_used
    )
    return card


async def complete_deployer_visit_conflict(
    session: AsyncSession,
    card: Card,
    visit: CardColumnVisit,
    *,
    conflicted_paths: list[str],
    endpoint_used: str | None = None,
) -> Card:
    """Deployer's run_deploy hit a merge conflict against default_branch. Rather
    than have the Deployer agent guess at a resolution with no bash/test tools and
    no further Tester/Reviewer pass (see sandbox/deploy_runner.py's
    DeployRunResult.conflict), bounce the card back to Developer — sharing the same
    revision_count budget as Reviewer/Tester's request_changes, since it's still
    'another round of Developer work needed' against the same safety valve. This
    does NOT count against max_deploy_attempts: the merge/push/deploy itself never
    ran, so it isn't a deploy failure.

    The next Developer visit's own branch-sync step (orchestrator/worker.py +
    sandbox/worktree.sync_card_branch_with_default) reproduces this same conflict
    directly in the card's own worktree, where Developer has the full read/write/
    bash toolset — and its test gate — to actually fix and re-verify it. Because it
    flows through submit_for_test again, Tester and Reviewer both see the fix too,
    unlike the old in-place Deployer resolution this replaces."""
    project = await session.get(Project, card.project_id)
    feedback = (
        f"An automatic deploy attempt failed: merging this card's branch into "
        f"{project.default_branch} produced a conflict in: {', '.join(conflicted_paths)}. Someone "
        f"else's change landed on {project.default_branch} since this branch was last synced with "
        f"it. Resolve the conflict against the current {project.default_branch}, re-verify the "
        "tests still pass, and resubmit."
    )
    summary = f"deploy blocked on a merge conflict in {', '.join(conflicted_paths)} — sent back to Developer"
    card.revision_count += 1
    card.latest_feedback = feedback
    card.column = Column.DEVELOPER
    if card.revision_count > project.max_revisions:
        card.lifecycle_state = LifecycleState.BLOCKED
    await _close_visit(
        session,
        visit,
        outcome=VisitOutcome.DEPLOY_CONFLICT,
        summary=summary,
        endpoint_used=endpoint_used,
        feedback=feedback,
    )
    return card


async def confirm_ci_passed(session: AsyncSession, card: Card, *, note: str) -> Card:
    """orchestrator/ci_watcher.py: CI came back green, or the repo turned out to
    have no CI at all for this commit — either way, the deploy this card produced
    is now genuinely confirmed, so this is where DONE actually happens for an
    auto_main card that had CI to wait on."""
    card.lifecycle_state = LifecycleState.DONE
    card.deploying_commit_sha = None
    card.deploying_since = None
    await append_event(
        session, card_id=card.id, type=EventType.SYSTEM_NOTE, payload={"action": "ci_confirmed", "note": note}
    )
    await _maybe_complete_epic(session, card)
    return card


async def reopen_after_ci_failure(
    session: AsyncSession, card: Card, project: Project, *, feedback: str
) -> Card:
    """orchestrator/ci_watcher.py: CI came back red on the commit this card's own
    auto_main deploy produced. The push itself genuinely succeeded, but the result
    it shipped is broken — and unlike a bug a stranger might file, there's no
    mystery about which card's work caused it or why: ci_watcher.py already fetched
    the failing job's actual error output. So reopen THIS card and bounce it back
    to Developer with that diagnosis as feedback, instead of leaving it DONE and
    filing a new one to reinvestigate from zero evidence. This is deliberately NOT
    a revert of default_branch (an automated revert of shared history that may
    already have other work built on top of it is a bigger, riskier action than
    this pipeline should take unsupervised) — it fixes forward, on the same branch,
    the same way a Reviewer/Tester rejection or a Deployer merge-conflict bounce
    does, and shares that revision_count budget: a card whose fix keeps re-breaking
    CI eventually BLOCKs for a human instead of silently spawning an unbounded
    chain of look-alike cards. Does NOT touch deploy_attempt_count — the deploy
    mechanics themselves didn't fail, the code they shipped did."""
    card.revision_count += 1
    card.latest_feedback = feedback
    card.column = Column.DEVELOPER
    card.lifecycle_state = LifecycleState.ACTIVE
    card.deploying_commit_sha = None
    card.deploying_since = None
    if card.revision_count > project.max_revisions:
        card.lifecycle_state = LifecycleState.BLOCKED
    await append_event(
        session,
        card_id=card.id,
        type=EventType.SYSTEM_NOTE,
        payload={"action": "ci_failed_reopened", "note": feedback},
    )
    return card


async def mark_ci_wait_timed_out(session: AsyncSession, card: Card, *, note: str) -> Card:
    """orchestrator/ci_watcher.py: CI never resolved within the configured window —
    stuck runner, misconfigured workflow, whatever. Blocks for a human rather than
    polling forever; the Reviver can also pick this up like any other blocked card."""
    card.lifecycle_state = LifecycleState.BLOCKED
    card.deploying_commit_sha = None
    card.deploying_since = None
    await append_event(
        session,
        card_id=card.id,
        type=EventType.SYSTEM_NOTE,
        payload={"action": "ci_wait_timed_out", "note": note},
    )
    return card


async def confirm_pr_merged(session: AsyncSession, card: Card, *, note: str) -> Card:
    """orchestrator/pr_watcher.py: the PR this card opened has been merged — either
    by the watcher after an approving review, or by a human. The deploy the card
    was holding the pipeline open for is now genuinely done, so this is where DONE
    actually happens for a pr_to_operator card."""
    card.lifecycle_state = LifecycleState.DONE
    card.pr_number = None
    card.pr_waiting_since = None
    await append_event(
        session, card_id=card.id, type=EventType.SYSTEM_NOTE, payload={"action": "pr_merged", "note": note}
    )
    await _maybe_complete_epic(session, card)
    return card


async def request_pr_changes(session: AsyncSession, card: Card, *, feedback: str, note: str) -> Card:
    """orchestrator/pr_watcher.py: a reviewer requested changes on the PR this card
    opened. Bounce back to Developer with the review feedback, sharing the same
    revision_count safety valve as Tester/Reviewer's request_changes (exceeding
    max_revisions blocks the card for a human). pr_number is cleared so the card is
    claimable again; when it flows back through to Deployer, open_pull_request
    finds the still-open PR (same branch) and updates it rather than opening a
    duplicate."""
    card.revision_count += 1
    card.latest_feedback = feedback
    card.column = Column.DEVELOPER
    card.pr_number = None
    card.pr_waiting_since = None
    project = await session.get(Project, card.project_id)
    if card.revision_count > project.max_revisions:
        card.lifecycle_state = LifecycleState.BLOCKED
    await append_event(
        session,
        card_id=card.id,
        type=EventType.SYSTEM_NOTE,
        payload={"action": "pr_changes_requested", "feedback": feedback, "note": note},
    )
    return card


async def mark_pr_wait_timed_out(session: AsyncSession, card: Card, *, note: str) -> Card:
    """orchestrator/pr_watcher.py: the PR never got reviewed and merged within the
    configured window. Blocks for a human rather than polling forever; the Reviver
    can pick this up like any other blocked card."""
    card.lifecycle_state = LifecycleState.BLOCKED
    card.pr_number = None
    card.pr_waiting_since = None
    await append_event(
        session,
        card_id=card.id,
        type=EventType.SYSTEM_NOTE,
        payload={"action": "pr_wait_timed_out", "note": note},
    )
    return card


async def mark_pr_closed_unmerged(session: AsyncSession, card: Card, *, note: str) -> Card:
    """orchestrator/pr_watcher.py: a human closed the PR without merging. The card's
    work is neither shipped nor back in the pipeline — block for a human to decide
    whether it should be reopened, retried, or cancelled."""
    card.lifecycle_state = LifecycleState.BLOCKED
    card.pr_number = None
    card.pr_waiting_since = None
    await append_event(
        session,
        card_id=card.id,
        type=EventType.SYSTEM_NOTE,
        payload={"action": "pr_closed_unmerged", "note": note},
    )
    return card


async def mark_pr_merge_conflicted(session: AsyncSession, card: Card, *, note: str) -> Card:
    """orchestrator/pr_watcher.py: the PR was approved but default_branch has
    advanced into a real conflict with the card's branch, so the merge API refused.
    An autonomous rebase/merge of shared history is a bigger, riskier action than
    this pipeline should take unsupervised (the same call ci_watcher makes about
    reverting a bad commit) — block for a human to resolve or merge by hand."""
    card.lifecycle_state = LifecycleState.BLOCKED
    card.pr_number = None
    card.pr_waiting_since = None
    await append_event(
        session,
        card_id=card.id,
        type=EventType.SYSTEM_NOTE,
        payload={"action": "pr_merge_conflicted", "note": note},
    )
    return card


async def fail_visit_with_error(
    session: AsyncSession, card: Card, visit: CardColumnVisit, *, message: str
) -> Card:
    """Iteration cap exceeded, endpoint fallback chain exhausted, or an unhandled
    exception mid-run — always blocks the card for a human, regardless of column."""
    card.lifecycle_state = LifecycleState.BLOCKED
    await _close_visit(session, visit, outcome=VisitOutcome.ERROR, summary=message)
    return card


async def mark_visit_interrupted(session: AsyncSession, card: Card, visit: CardColumnVisit) -> Card:
    """The worker process died mid-visit. The card stays ACTIVE — on restart the
    orchestrator starts a fresh attempt of the same column rather than trying to
    resume mid-tool-call (accepted v1 simplification)."""
    await _close_visit(
        session, visit, outcome=VisitOutcome.INTERRUPTED, summary="Worker process interrupted."
    )
    return card


async def abandon_visit_for_lifecycle_change(
    session: AsyncSession, card: Card, visit: CardColumnVisit
) -> Card:
    """agent/loop.py: the card's lifecycle_state no longer matches what it was when
    this visit started (currently only ever a human cancelling it mid-run —
    cancel_card sets FAILED without touching the running visit or its claim/lease at
    all, since nothing else previously noticed). Closes out the visit without
    touching lifecycle_state or column — whatever set it that way already did the
    right thing, this is just making sure the agent loop stops burning iterations
    (and, with orchestrator_concurrency's default of 1, blocking every other card)
    on work nobody wants anymore."""
    await _close_visit(
        session,
        visit,
        outcome=VisitOutcome.CANCELLED,
        summary=f"Card lifecycle_state changed to {card.lifecycle_state.value!r} mid-visit — stopped.",
    )
    return card


async def retry_card(session: AsyncSession, card: Card, *, note: str | None = None) -> Card:
    """The one human touchpoint that exists *outside* the autonomous pipeline: un-stick
    a blocked or failed card with a clean safety-valve budget. An optional note is
    surfaced to whichever column runs next (see agent/context.py's retry_note
    handling) and cleared after that one visit — see orchestrator/worker.py."""
    if card.lifecycle_state not in (LifecycleState.BLOCKED, LifecycleState.FAILED):
        raise ValueError(f"cannot retry a card in state {card.lifecycle_state.value!r}")
    card.lifecycle_state = LifecycleState.ACTIVE
    card.revision_count = 0
    card.deploy_attempt_count = 0
    card.latest_feedback = None
    card.retry_note = note
    await append_event(
        session, card_id=card.id, type=EventType.SYSTEM_NOTE, payload={"action": "retry", "note": note}
    )
    return card


async def cancel_card(session: AsyncSession, card: Card) -> Card:
    """Also a human-only action. There's no separate CANCELLED state — a cancelled
    card is recorded as FAILED with a system note explaining why."""
    if card.lifecycle_state in (LifecycleState.DONE, LifecycleState.FAILED):
        raise ValueError(f"cannot cancel a card in state {card.lifecycle_state.value!r}")
    card.lifecycle_state = LifecycleState.FAILED
    await append_event(session, card_id=card.id, type=EventType.SYSTEM_NOTE, payload={"action": "cancel"})
    return card
