import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from built.db.base import Base
from built.domain.enums import (
    ActivityKind,
    ChatRole,
    CurationRunOutcome,
    DeployKind,
    DeployMode,
    EventType,
    LifecycleState,
    Priority,
    RunAttemptStatus,
    VisitOutcome,
)
from built.domain.enums import (
    Column as ColumnEnum,
)


def _new_id() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(primary_key=True, default=_new_id)
    name: Mapped[str]
    slug: Mapped[str] = mapped_column(unique=True, index=True)
    overarching_goal: Mapped[str] = mapped_column(Text)
    repo_remote_url: Mapped[str]
    managed_repo_path: Mapped[str | None] = mapped_column(default=None)
    default_branch: Mapped[str] = mapped_column(default="main")
    # Docker image the bash tool runs in for this project's Developer/Tester columns —
    # e.g. "node:22-slim" for a JS/TS repo. NULL = sandbox.container.DEFAULT_IMAGE
    # (python:3.12-slim), which has no Node/Go/Rust/etc. toolchain.
    sandbox_image: Mapped[str | None] = mapped_column(default=None)
    # The exact command that runs this project's whole test suite, e.g. "pytest -q" or
    # "npm test". NULL blocks Developer's submit_for_test and Tester's approve outright
    # (see domain/run_attempts.py) rather than falling back to a weaker check — without
    # this, "some bash command exited 0" was the only gate, and it didn't verify the
    # command run was actually the test suite.
    test_command: Mapped[str | None] = mapped_column(Text, default=None)
    # The operator's entire investigation mandate for ActivityKind.OVERSEER — unlike
    # every other curation kind, built supplies no built-in definition of what this
    # role looks for (see agent/context.py's build_curation_prompt). NULL/blank blocks
    # OVERSEER from running at all (orchestrator/curator.py's _needs_run and
    # run_curation_activity both gate on this) — same nullable-blocks-a-capability
    # shape as test_command above, no silent built-authored fallback. Changing this via
    # project settings fires a single-shot LLM comprehensiveness check before it's
    # allowed to save (see agent/curation.py's assess_overseer_prompt) — set through
    # services/project_service.py's set_overseer_prompt, not update_project's generic
    # **fields passthrough, because that passthrough silently skips None and could
    # never actually clear this back to blank.
    overseer_prompt: Mapped[str | None] = mapped_column(Text, default=None)

    # Safety-valve caps — copied from service-level defaults at creation, overridable per project.
    max_revisions: Mapped[int] = mapped_column(Integer, default=3)
    max_deploy_attempts: Mapped[int] = mapped_column(Integer, default=2)
    max_iterations_per_run: Mapped[int] = mapped_column(Integer, default=25)
    # Context window size in tokens. Falls back to settings.default_max_tokens (128k)
    # if NULL — useful for smaller models that need a tighter budget.
    max_tokens: Mapped[int | None] = mapped_column(Integer, default=None)

    # Python-side defaults (not server_default=func.now()): a server-computed default
    # is only known to SQLAlchemy after a post-flush refresh, which is itself a lazy
    # DB round-trip and hits the exact same async MissingGreenlet problem as an
    # unloaded relationship. A client-side default is known immediately, no query needed.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    # Distinct from archived_at: a paused project stays visible everywhere (list, board,
    # settings) but the orchestrator/reviver/tender all skip it — a human wants the repo
    # left alone for a while without hiding or losing its cards. An in-flight visit still
    # finishes; pause only blocks new claims.
    paused_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    # The board's "current epic" selector (see EpicLink) — a hint fed into PM/curation
    # prompts (agent/context.py's _with_current_epic) alongside overarching_goal, never
    # a replacement for it, and never consulted by claim_next_card. Human-set-only:
    # define_epic never sets this itself, so a human's chosen focus is never silently
    # swapped out from under them. Added via db/base.py's ADDITIVE_COLUMNS, not a new
    # table — a single scalar FK, always read where `project` is already loaded, with
    # no list/JSON-reassign concern that would justify a side table.
    # use_alter=True: this FK closes a genuine cycle between projects and cards
    # (cards.project_id already points the other way) — without it, SQLAlchemy's
    # metadata-level table sorter (used by tests/conftest.py's per-test table wipe,
    # via Base.metadata.sorted_tables) can't find a safe dependency order and warns
    # every test run. Doesn't change actual runtime behavior on SQLite (this app
    # never turns on PRAGMA foreign_keys, so nothing enforces the constraint anyway)
    # — purely resolves the bookkeeping cycle.
    current_epic_id: Mapped[str | None] = mapped_column(
        ForeignKey("cards.id", use_alter=True, name="fk_projects_current_epic_id"), default=None
    )
    # A "clear chat" cutoff, not a delete: ChatMessage rows with seq <= this value are
    # excluded from both the rendered transcript and what agent/chat.py replays into the
    # model, but stay in the table — same archive-not-delete posture as Card.archived_at.
    # Added via ADDITIVE_COLUMNS, same precedent as current_epic_id above.
    chat_cleared_before_seq: Mapped[int | None] = mapped_column(Integer, default=None)

    # Per-role custom instructions, set via project settings (services/project_service.py)
    # and appended to that role's system prompt by agent/context.py's
    # _with_role_guidance. Distinct from AGENTS.md (_with_agents_doc): AGENTS.md is a
    # convention the repo itself documents and every role reads identically, kept
    # current by curation; these are a human's direct, role-targeted instructions,
    # entered here rather than committed to the repo. pm_guidance also applies to
    # curation passes and project chat (agent/curation.py, agent/chat.py) since both
    # are PM-role activities, not just the PM column visit on a card.
    pm_guidance: Mapped[str | None] = mapped_column(Text, default=None)
    developer_guidance: Mapped[str | None] = mapped_column(Text, default=None)
    tester_guidance: Mapped[str | None] = mapped_column(Text, default=None)
    reviewer_guidance: Mapped[str | None] = mapped_column(Text, default=None)
    deployer_guidance: Mapped[str | None] = mapped_column(Text, default=None)

    # selectin: async SQLAlchemy has no implicit lazy-load, and this relationship is
    # read unconditionally by ProjectOut serialization — selectin issues its own
    # awaited follow-up query at load time instead of an attribute-access-time one.
    # (create_project still sets it directly — see the comment there.)
    deploy_config: Mapped["DeployConfig | None"] = relationship(
        back_populates="project", uselist=False, cascade="all, delete-orphan", lazy="selectin"
    )
    # raise: never accessed via this relationship — services query EndpointConfig/Card
    # directly. lazy="raise" turns an accidental future lazy-access into a clear error
    # at the access site instead of an async MissingGreenlet crash somewhere unrelated.
    endpoint_configs: Mapped[list["EndpointConfig"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", lazy="raise"
    )
    cards: Mapped[list["Card"]] = relationship(
        back_populates="project",
        # Project.current_epic_id is a second FK between projects and cards (the
        # reverse direction) — without this, SQLAlchemy can't infer which FK this
        # relationship should join on and raises AmbiguousForeignKeysError.
        foreign_keys="Card.project_id",
        cascade="all, delete-orphan",
        lazy="raise",
    )


class DeployConfig(Base):
    """1:1 with Project. Deployer's `run_deploy` tool executes exactly this, with zero
    LLM-supplied arguments — the model never controls what actually runs."""

    __tablename__ = "deploy_configs"

    id: Mapped[str] = mapped_column(primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), unique=True)
    # auto_main: kind/command/script_path/webhook_url/env_var_refs/timeout_seconds
    # below apply. pr_to_operator: only github_token_ref applies — the rest are unused.
    mode: Mapped[DeployMode] = mapped_column(default=DeployMode.PR_TO_OPERATOR)
    kind: Mapped[DeployKind]
    command: Mapped[str | None] = mapped_column(Text, default=None)
    script_path: Mapped[str | None] = mapped_column(default=None)
    webhook_url: Mapped[str | None] = mapped_column(default=None)
    # Names of env vars the deploy execution path should inject — never raw secret values.
    env_var_refs: Mapped[list[str]] = mapped_column(JSON, default=list)
    timeout_seconds: Mapped[int] = mapped_column(Integer, default=600)
    # Env var name holding a GitHub PAT (repo-scoped, PR-write) — never the raw token.
    github_token_ref: Mapped[str | None] = mapped_column(default=None)

    project: Mapped["Project"] = relationship(back_populates="deploy_config", lazy="raise")


class EndpointConfig(Base):
    """One entry in a fallback chain. `project_id IS NULL` = global default;
    `role IS NULL` = applies to every column role for that project (or globally)."""

    __tablename__ = "endpoint_configs"

    id: Mapped[str] = mapped_column(primary_key=True, default=_new_id)
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), nullable=True, default=None)
    role: Mapped[ColumnEnum | None] = mapped_column(nullable=True, default=None)
    priority: Mapped[int] = mapped_column(Integer, default=0)

    base_url: Mapped[str]
    model: Mapped[str]
    api_key_ref: Mapped[str | None] = mapped_column(default=None)
    supports_tool_calling: Mapped[bool] = mapped_column(default=True)
    extra_params: Mapped[dict] = mapped_column(JSON, default=dict)
    enabled: Mapped[bool] = mapped_column(default=True)
    # Max simultaneous in-flight requests this physical (base_url, model) backend
    # should ever receive from this app — see llm/client.py's per-endpoint semaphore.
    # Default of 1 matches a single local model instance with no continuous batching;
    # a hosted/scaled endpoint can be configured higher.
    max_concurrency: Mapped[int] = mapped_column(Integer, default=1)
    # Context window size in tokens for this specific model. Falls back to
    # Project.max_tokens, then to settings.default_max_tokens if NULL. Helps the
    # agent loop know when to compact before hitting a context-error from the API.
    context_window: Mapped[int | None] = mapped_column(Integer, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    project: Mapped["Project | None"] = relationship(back_populates="endpoint_configs", lazy="raise")


class ProjectActivityRun(Base):
    """One row per (project, curation activity kind) — when it last ran, so
    orchestrator/curator.py can ask "is overseer due for project X" uniformly
    across every kind (agent/curation.py, orchestrator/curator.py). Replaces the old
    single-purpose Project.agents_doc_tended_at, which only ever tracked one kind."""

    __tablename__ = "project_activity_runs"
    __table_args__ = (UniqueConstraint("project_id", "kind"),)

    id: Mapped[str] = mapped_column(primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    kind: Mapped[ActivityKind]
    last_run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_result_summary: Mapped[str | None] = mapped_column(Text, default=None)


class ProjectCurationState(Base):
    """Curator on/off state for one project — a separate concern from
    Project.paused_at, which also stops the worker orchestrator and Reviver.
    paused_at here pauses only the curator's automatic scheduler (orchestrator/
    curator.py's run_curator_once) for the whole project; disabled_kinds turns off
    individual ActivityKinds while leaving the rest on the schedule. Neither ever
    blocks a human's manual "run now" trigger (run_curation_activity) — only the
    automatic scheduler loop reads this row. A project with no row here is fully
    active, nothing paused or disabled — see project_service.get_curation_state.

    A new table rather than new columns on Project deliberately: this app has no
    migration framework (db/base.py's create_all() only creates missing tables), so
    a new table needs no manual ALTER TABLE against an already-deployed database."""

    __tablename__ = "project_curation_states"

    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), primary_key=True)
    paused_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    disabled_kinds: Mapped[list[str]] = mapped_column(JSON, default=list)


class CurationRun(Base):
    """One row per curation pass invocation (orchestrator/curator.py's
    run_curation_activity) — CurationEvent rows link into it via run_id, the same
    relationship CardColumnVisit has to CardEvent. Gives the board a real
    per-invocation history (started_at/ended_at/outcome/summary/token usage)
    instead of only the single latest CurationEvent and one overwritten
    ProjectActivityRun row. outcome/ended_at stay NULL while the run is in
    progress; a row that's still NULL long after started_at, with
    orchestrator.curator.is_curation_running now false for it, is a crash, not a
    slow pass — see ui/routers/board.py's history view for that check."""

    __tablename__ = "curation_runs"

    id: Mapped[str] = mapped_column(primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    kind: Mapped[ActivityKind]
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    outcome: Mapped[CurationRunOutcome | None] = mapped_column(default=None)
    summary: Mapped[str | None] = mapped_column(Text, default=None)
    tokens_in: Mapped[int | None] = mapped_column(Integer, default=None)
    tokens_out: Mapped[int | None] = mapped_column(Integer, default=None)
    # Never populated yet — same as CardEvent.cost_estimate below; no per-model
    # pricing table exists anywhere in this codebase. Kept for schema parity so
    # whenever that's added, both event streams already have somewhere to put it.
    cost_estimate: Mapped[float | None] = mapped_column(default=None)

    events: Mapped[list["CurationEvent"]] = relationship(
        back_populates="run", order_by="CurationEvent.seq", lazy="raise"
    )


class CurationEvent(Base):
    """Append-only transcript for curation passes (agent/curation.py) — mirrors
    CardEvent's shape, but scoped to (project, kind) rather than card_id: a
    curation pass isn't tied to any card until (and unless) propose_tasks actually
    creates one. seq is monotonic per (project_id, kind), across every pass, same as
    CardEvent's per-card seq — what lets the board page's status panel show what a
    kind is doing right now, and what its last pass actually did. run_id groups
    events by invocation (CurationRun) — nullable since it was added after this
    table already had rows; historical events predating CurationRun just have
    none."""

    __tablename__ = "curation_events"
    __table_args__ = (
        UniqueConstraint("project_id", "kind", "seq", name="uq_curation_events_project_kind_seq"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    kind: Mapped[ActivityKind]
    run_id: Mapped[str | None] = mapped_column(ForeignKey("curation_runs.id"), index=True, default=None)
    seq: Mapped[int] = mapped_column(Integer)
    type: Mapped[EventType]
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    tokens_in: Mapped[int | None] = mapped_column(Integer, default=None)
    tokens_out: Mapped[int | None] = mapped_column(Integer, default=None)
    latency_ms: Mapped[int | None] = mapped_column(Integer, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    run: Mapped["CurationRun | None"] = relationship(back_populates="events", lazy="raise")


class CardPostmortem(Base):
    """Written once by agent/summarizer.py, right as a card reaches a terminal
    lifecycle_state (DONE or FAILED) — see the terminal handlers in agent/loop.py
    and the CI-confirmation paths in orchestrator/ci_watcher.py, both of which call
    write_card_postmortem before the transition that closes the card actually
    commits. No unique constraint on card_id: a card that fails, gets retried, and
    later succeeds accumulates one row per closure, which is exactly the history
    ActivityKind.RETRO wants to mine, not something to collapse to a single row.

    Deliberately not folded into CardEvent: CardEvent is an append-only transcript
    scoped to one card for the dashboard to tail, while this is small, structured,
    cross-card data a periodic pass reads in bulk (services/card_service.py's
    list_recent_postmortems) — a shape closer to CurationEvent's role than
    CardEvent's."""

    __tablename__ = "card_postmortems"

    id: Mapped[str] = mapped_column(primary_key=True, default=_new_id)
    card_id: Mapped[str] = mapped_column(ForeignKey("cards.id"), index=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    outcome: Mapped[LifecycleState]
    went_well: Mapped[str] = mapped_column(Text)
    struggles: Mapped[str] = mapped_column(Text)
    # Denormalized off Card at write time so a later mining pass can reason about
    # "how much friction did this card cost" without joining back to cards.
    revision_count: Mapped[int] = mapped_column(Integer, default=0)
    deploy_attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ChatMessage(Base):
    """One row per turn in a project's single, ongoing brainstorming chat (agent/
    chat.py) — unlike CardEvent/CurationEvent, this must be exactly replayable back
    into llm/client.py's complete(messages=...), not just good for rendering, so it
    stores OpenAI message shape directly (role/content/tool_calls/tool_call_id)
    instead of a freeform payload dict. seq is monotonic per project_id — there is
    deliberately no session-grouping table; scope is one chat per project, so
    project_id alone is the key. See Project.chat_cleared_before_seq for how "clear
    chat" works without deleting these rows."""

    __tablename__ = "chat_messages"
    __table_args__ = (UniqueConstraint("project_id", "seq", name="uq_chat_messages_project_seq"),)

    id: Mapped[str] = mapped_column(primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    role: Mapped[ChatRole]
    content: Mapped[str | None] = mapped_column(Text, default=None)
    # Assistant rows only: [{"id": tool_call_id, "name": str, "arguments": dict}, ...].
    # Kept as a plain dict here, not the OpenAI wire shape (which double-JSON-encodes
    # `arguments` as a string) — services/chat_service.py's to_openai_messages does
    # that encoding only transiently, at the point of calling complete().
    tool_calls: Mapped[list | None] = mapped_column(JSON, default=None)
    tool_call_id: Mapped[str | None] = mapped_column(default=None)  # role == TOOL only
    tool_name: Mapped[str | None] = mapped_column(default=None)  # role == TOOL only
    is_error: Mapped[bool] = mapped_column(default=False)
    # Set only on a create_ticket/update_ticket tool-result row — lets the chat
    # fragment link straight to the card it just filed/edited.
    card_id: Mapped[str | None] = mapped_column(ForeignKey("cards.id"), default=None)
    tokens_in: Mapped[int | None] = mapped_column(Integer, default=None)
    tokens_out: Mapped[int | None] = mapped_column(Integer, default=None)
    latency_ms: Mapped[int | None] = mapped_column(Integer, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Card(Base):
    __tablename__ = "cards"

    id: Mapped[str] = mapped_column(primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    title: Mapped[str]
    raw_request: Mapped[str] = mapped_column(Text)

    column: Mapped[ColumnEnum] = mapped_column(default=ColumnEnum.PM)
    lifecycle_state: Mapped[LifecycleState] = mapped_column(default=LifecycleState.ACTIVE)
    # A human's manual bless/deprioritize signal — see orchestrator/worker.py's
    # _CLAIM_PRIORITY_ORDER, which sorts on this before column-depth or recency.
    priority: Mapped[Priority] = mapped_column(default=Priority.NORMAL)

    branch_name: Mapped[str | None] = mapped_column(default=None)
    worktree_path: Mapped[str | None] = mapped_column(default=None)

    spec: Mapped[str | None] = mapped_column(Text, default=None)
    acceptance_criteria: Mapped[list[str]] = mapped_column(JSON, default=list)

    revision_count: Mapped[int] = mapped_column(Integer, default=0)
    latest_feedback: Mapped[str | None] = mapped_column(Text, default=None)
    deploy_attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    # A human instruction attached to a retry (the one touchpoint outside the
    # autonomous pipeline) — surfaced to whichever column runs next, then cleared
    # after that one visit so it doesn't linger across later, unrelated retries.
    retry_note: Mapped[str | None] = mapped_column(Text, default=None)
    # A human note dropped in while the card is active, regardless of whether a
    # visit is running right now — unlike retry_note it needs no blocked/failed
    # state and no retry. run_column_visit checks this every iteration (it already
    # refreshes the card every iteration for the cancellation check) and, once
    # seen, injects it as a message and clears it — surfaced within one iteration
    # of a running visit, or at the start of the next one if the card is idle.
    pending_nudge: Mapped[str | None] = mapped_column(Text, default=None)
    # How many times the autonomous Reviver (agent/reviver.py) has retried this card,
    # capped at settings.reviver_max_auto_revives — once reached, the Reviver leaves
    # it alone permanently and it waits for a human. Not touched by a human-initiated
    # retry, which has its own unlimited touchpoint outside the pipeline.
    auto_revive_count: Mapped[int] = mapped_column(Integer, default=0)
    # PR URL (pr_to_operator mode) so the dashboard can link straight to it. Unused
    # (stays None) in auto_main mode.
    deploy_url: Mapped[str | None] = mapped_column(default=None)

    # Set the moment an auto_main deploy pushes successfully; cleared once CI
    # resolves (or there turns out to be none) — see orchestrator/ci_watcher.py.
    # While set, the card stays ACTIVE but isn't reclaimed by the worker pool (the
    # Deployer's own job — the git-level push — is genuinely done; the card's
    # overall completion isn't, until CI actually confirms it). NULL for
    # pr_to_operator deploys, which track a pending PR via pr_number instead.
    deploying_commit_sha: Mapped[str | None] = mapped_column(default=None)
    deploying_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    # Set the moment a pr_to_operator deploy opens (or re-uses) a GitHub PR;
    # cleared once that PR merges, or the card bounces back to Developer on a
    # "changes requested" review, or the wait times out — see
    # orchestrator/pr_watcher.py. Mirrors deploying_commit_sha's
    # wait-for-external-confirmation shape: while set, the card stays ACTIVE in
    # Deployer but isn't reclaimed by the worker pool. NULL for auto_main deploys.
    pr_number: Mapped[int | None] = mapped_column(default=None)
    pr_waiting_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    # Orchestrator claim/lease — see orchestrator/worker.py (Phase 4).
    claimed_by_worker_id: Mapped[str | None] = mapped_column(default=None)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    # Soft-hide, not delete: an archived card drops off the board and out of claiming
    # but its history/events/visits stay intact and it's still reachable at its direct
    # URL — the one place its own Archive/Unarchive control lives.
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    # raise on both: transitions.py fetches the project explicitly via session.get()
    # instead, and callers that need visits ask for them with an explicit
    # selectinload(Card.column_visits) query option (see card_service.get_card).
    project: Mapped["Project"] = relationship(
        back_populates="cards", foreign_keys=[project_id], lazy="raise"
    )
    column_visits: Mapped[list["CardColumnVisit"]] = relationship(
        back_populates="card",
        cascade="all, delete-orphan",
        order_by="CardColumnVisit.started_at",
        lazy="raise",
    )

    @property
    def is_being_worked(self) -> bool:
        """True while an orchestrator worker actually holds this card's claim/lease —
        distinct from lifecycle_state == ACTIVE, which also covers a card sitting idle
        in the queue waiting for a free worker. Drives the "still working" spinner in
        the dashboard."""
        if self.claimed_by_worker_id is None or self.lease_expires_at is None:
            return False
        lease_expires_at = self.lease_expires_at
        if lease_expires_at.tzinfo is None:
            # SQLite doesn't reliably round-trip tzinfo (see _timeago) — a Card just
            # read back from the DB can have a naive lease_expires_at even though it
            # was written as UTC.
            lease_expires_at = lease_expires_at.replace(tzinfo=UTC)
        return lease_expires_at > _utcnow()


class CardDependency(Base):
    """One row means: card_id cannot be claimed until depends_on_card_id reaches
    DONE — see orchestrator/worker.py's claim_next_card. A real join table, not a
    JSON list on Card, because claim_next_card needs to test this as a SQL subquery
    against every candidate card on every poll tick, the same way it already
    excludes busy/paused projects — a JSON array column can't be joined/indexed the
    same way. Authored today by the PM's define_epic (agent/loop.py); see
    services/card_service.py's add_dependency for the general-purpose entry point
    (cycle/cross-project/self-dependency guards live there, not here — this table
    is deliberately dumb storage)."""

    __tablename__ = "card_dependencies"
    __table_args__ = (UniqueConstraint("card_id", "depends_on_card_id", name="uq_card_dependencies_pair"),)

    id: Mapped[str] = mapped_column(primary_key=True, default=_new_id)
    card_id: Mapped[str] = mapped_column(ForeignKey("cards.id"), index=True)
    # Indexed separately from the (card_id, depends_on_card_id) unique constraint's
    # composite index — claim-gating's exclusion subquery joins on this column, not
    # card_id, and a composite index isn't useful for a lookup that leads with its
    # second column.
    depends_on_card_id: Mapped[str] = mapped_column(ForeignKey("cards.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class EpicLink(Base):
    """One row per epic child card. card_id (the child) is the primary key — a card
    can belong to at most one epic, structurally, not just by convention. "Is this
    card an epic" is EXISTS(select 1 from epic_links where parent_card_id =
    card.id), not a stored flag on Card — see services/card_service.py's is_epic.
    Created by the PM's define_epic (agent/loop.py) alongside each child card; the
    parent itself is never archived and stays column=PM forever (it has no
    Developer/Tester/Reviewer/Deployer work of its own) — get_board and
    count_column_backlog exclude it from the ordinary PM swimlane/backlog via this
    table, and claim_next_card excludes it from ever being claimed. It reaches
    lifecycle_state=DONE only via domain/transitions.py's _maybe_complete_epic, once
    every non-archived child does."""

    __tablename__ = "epic_links"

    card_id: Mapped[str] = mapped_column(ForeignKey("cards.id"), primary_key=True)
    parent_card_id: Mapped[str] = mapped_column(ForeignKey("cards.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class CardColumnVisit(Base):
    """One row per time a card enters a column — a revision loop creates multiple
    Developer visits for the same card, each with an incrementing attempt_number."""

    __tablename__ = "card_column_visits"

    id: Mapped[str] = mapped_column(primary_key=True, default=_new_id)
    card_id: Mapped[str] = mapped_column(ForeignKey("cards.id"), index=True)
    column: Mapped[ColumnEnum]
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    outcome: Mapped[VisitOutcome | None] = mapped_column(default=None)
    endpoint_used: Mapped[str | None] = mapped_column(default=None)
    summary: Mapped[str | None] = mapped_column(Text, default=None)

    card: Mapped["Card"] = relationship(back_populates="column_visits", lazy="raise")
    events: Mapped[list["CardEvent"]] = relationship(
        back_populates="column_visit", cascade="all, delete-orphan", order_by="CardEvent.seq", lazy="raise"
    )
    run_attempts: Mapped[list["RunAttempt"]] = relationship(
        back_populates="column_visit", cascade="all, delete-orphan", lazy="raise"
    )


class CardEvent(Base):
    """Append-only transcript the dashboard tails. `seq` is monotonic per card (not
    global) so `GET /cards/{id}/events?since_seq=` is a simple, stable cursor."""

    __tablename__ = "card_events"
    __table_args__ = (UniqueConstraint("card_id", "seq", name="uq_card_events_card_seq"),)

    id: Mapped[str] = mapped_column(primary_key=True, default=_new_id)
    card_id: Mapped[str] = mapped_column(ForeignKey("cards.id"), index=True)
    # Nullable: pipeline-level events (card created, retried/cancelled before any agent
    # picked it up) aren't tied to a specific column visit.
    column_visit_id: Mapped[str | None] = mapped_column(
        ForeignKey("card_column_visits.id"), index=True, default=None
    )
    seq: Mapped[int] = mapped_column(Integer)
    type: Mapped[EventType]
    # Size-capped and secret-scrubbed by the caller before persisting — see services/card_service.py.
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    tokens_in: Mapped[int | None] = mapped_column(Integer, default=None)
    tokens_out: Mapped[int | None] = mapped_column(Integer, default=None)
    latency_ms: Mapped[int | None] = mapped_column(Integer, default=None)
    cost_estimate: Mapped[float | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    column_visit: Mapped["CardColumnVisit"] = relationship(back_populates="events", lazy="raise")


class RunAttempt(Base):
    """A bounded external command execution — Tester's test-suite runs and Deployer's
    deploy runs both record here. Terminal transitions are verified against this, not
    against what the model claims in prose."""

    __tablename__ = "run_attempts"

    id: Mapped[str] = mapped_column(primary_key=True, default=_new_id)
    card_id: Mapped[str] = mapped_column(ForeignKey("cards.id"), index=True)
    column_visit_id: Mapped[str] = mapped_column(ForeignKey("card_column_visits.id"), index=True)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[RunAttemptStatus] = mapped_column(default=RunAttemptStatus.RUNNING)
    command_executed: Mapped[str | None] = mapped_column(Text, default=None)
    exit_code: Mapped[int | None] = mapped_column(Integer, default=None)
    stdout_ref: Mapped[str | None] = mapped_column(Text, default=None)
    stderr_ref: Mapped[str | None] = mapped_column(Text, default=None)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    # The CardEvent.seq of this bash call's own tool_call event — lets
    # has_passing_run_since_last_change() tell whether a file was touched (a later
    # tool_call event with a commit_sha) after this run, without a timestamp join.
    card_event_seq: Mapped[int | None] = mapped_column(Integer, default=None)
    # bash's own $PIPESTATUS after this command ran, one entry per top-level pipeline
    # stage (e.g. [1, 0] for `npm test | grep ...` where the suite actually failed but
    # grep still matched something and exited 0). Only ever populated for a command
    # domain/run_attempts.py recognizes as "the configured test_command piped into
    # something else" — lets that gate check the test command's own exit code
    # (pipestatus[0]) instead of the pipeline's overall exit_code, which reflects
    # whatever ran last and is not trustworthy for this. None for every other command.
    pipestatus: Mapped[list[int] | None] = mapped_column(JSON, default=None)

    column_visit: Mapped["CardColumnVisit"] = relationship(back_populates="run_attempts", lazy="raise")
