import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from built.db.base import Base
from built.domain.enums import (
    ActivityKind,
    DeployKind,
    DeployMode,
    EventType,
    LifecycleState,
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
        back_populates="project", cascade="all, delete-orphan", lazy="raise"
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
    orchestrator/curator.py can ask "is bug_sweep due for project X" uniformly
    across every kind (agent/curation.py, orchestrator/curator.py). Replaces the old
    single-purpose Project.agents_doc_tended_at, which only ever tracked one kind."""

    __tablename__ = "project_activity_runs"
    __table_args__ = (UniqueConstraint("project_id", "kind"),)

    id: Mapped[str] = mapped_column(primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    kind: Mapped[ActivityKind]
    last_run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_result_summary: Mapped[str | None] = mapped_column(Text, default=None)


class CurationEvent(Base):
    """Append-only transcript for curation passes (agent/curation.py) — mirrors
    CardEvent's shape, but scoped to (project, kind) rather than card_id: a
    curation pass isn't tied to any card until (and unless) propose_tasks actually
    creates one. seq is monotonic per (project_id, kind), across every pass, same as
    CardEvent's per-card seq — what lets the board page's status panel show what a
    kind is doing right now, and what its last pass actually did."""

    __tablename__ = "curation_events"
    __table_args__ = (
        UniqueConstraint("project_id", "kind", "seq", name="uq_curation_events_project_kind_seq"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    kind: Mapped[ActivityKind]
    seq: Mapped[int] = mapped_column(Integer)
    type: Mapped[EventType]
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Card(Base):
    __tablename__ = "cards"

    id: Mapped[str] = mapped_column(primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    title: Mapped[str]
    raw_request: Mapped[str] = mapped_column(Text)

    column: Mapped[ColumnEnum] = mapped_column(default=ColumnEnum.PM)
    lifecycle_state: Mapped[LifecycleState] = mapped_column(default=LifecycleState.ACTIVE)

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
    # How many times the autonomous Reviver (agent/reviver.py) has retried this card,
    # capped at settings.reviver_max_auto_revives — once reached, the Reviver leaves
    # it alone permanently and it waits for a human. Not touched by a human-initiated
    # retry, which has its own unlimited touchpoint outside the pipeline.
    auto_revive_count: Mapped[int] = mapped_column(Integer, default=0)
    # PR URL (pr_to_operator mode) so the dashboard can link straight to it. Unused
    # (stays None) in auto_main mode.
    deploy_url: Mapped[str | None] = mapped_column(default=None)

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
    project: Mapped["Project"] = relationship(back_populates="cards", lazy="raise")
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

    column_visit: Mapped["CardColumnVisit"] = relationship(back_populates="run_attempts", lazy="raise")
