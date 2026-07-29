from enum import StrEnum


class Column(StrEnum):
    """The four fixed kanban columns a card moves through, in order."""

    PM = "pm"
    DEVELOPER = "developer"
    TESTER = "tester"
    DEPLOYER = "deployer"


COLUMN_ORDER: list[Column] = [Column.PM, Column.DEVELOPER, Column.TESTER, Column.DEPLOYER]


class LifecycleState(StrEnum):
    """Whether a card is currently claimable by the orchestrator."""

    ACTIVE = "active"  # claimable
    BLOCKED = "blocked"  # a safety valve tripped (revision/deploy cap, run error) — needs a human
    DONE = "done"  # deployed successfully — terminal
    FAILED = "failed"  # exhausted deploy retries, or cancelled — terminal


class VisitOutcome(StrEnum):
    """How a single CardColumnVisit ended. Meaning is contextual to visit.column."""

    SUBMITTED = "submitted"  # PM's submit_spec, or Developer's submit_for_test
    APPROVED = "approved"  # Tester's approve
    CHANGES_REQUESTED = "changes_requested"  # Tester's request_changes
    DONE = "done"  # Deployer's run_deploy succeeded (pr_to_operator, or auto_main with no CI to watch)
    DEPLOYED_PENDING_CI = "deployed_pending_ci"  # auto_main run_deploy succeeded; CI still has to confirm
    FAILED = "failed"  # Deployer's run_deploy failed (this attempt, or terminally over cap)
    ERROR = "error"  # iteration cap / endpoint chain exhausted / unhandled exception
    INTERRUPTED = "interrupted"  # process crashed mid-visit; orchestrator restarts fresh


class EventType(StrEnum):
    """CardEvent.type — the transcript entry kinds the dashboard renders."""

    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TRANSITION = "transition"
    SYSTEM_NOTE = "system_note"
    ERROR = "error"


class RunAttemptStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class DeployKind(StrEnum):
    NONE = "none"  # merge + push is the whole job — no separate deploy step
    SCRIPT = "script"
    COMMAND = "command"
    WEBHOOK = "webhook"


class DeployMode(StrEnum):
    """How Deployer ships an approved card. auto_main merges to default_branch,
    pushes, and runs the project's configured deploy command — zero human gate.
    pr_to_operator pushes the card's branch as-is and opens a GitHub PR for a human to
    review and merge; no merge and no deploy command run automatically."""

    AUTO_MAIN = "auto_main"
    PR_TO_OPERATOR = "pr_to_operator"


class ActivityKind(StrEnum):
    """A periodic curation pass over a project. Every kind does the same mechanical
    thing — explore read-only, decide, call propose_tasks — differing only in system
    prompt and (for AGENTS_MD) what context it's fed. None of them ever edit the
    repo directly; the only thing any kind can do is create new cards, exactly like
    a human PM filing a ticket. See agent/curation.py and orchestrator/curator.py."""

    BUG_SWEEP = "bug_sweep"
    OPPORTUNITY_BRAINSTORM = "opportunity_brainstorm"
    POLISH_REVIEW = "polish_review"
    AGENTS_MD = "agents_md"  # proposes an AGENTS.md-update card; replaces the old Tender
