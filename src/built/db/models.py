import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from built.db.base import Base
from built.domain.enums import (
    Column as ColumnEnum,
)
from built.domain.enums import (
    DeployKind,
    EventType,
    LifecycleState,
    RunAttemptStatus,
    VisitOutcome,
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

    # Safety-valve caps — copied from service-level defaults at creation, overridable per project.
    max_revisions: Mapped[int] = mapped_column(Integer, default=3)
    max_deploy_attempts: Mapped[int] = mapped_column(Integer, default=2)
    max_iterations_per_run: Mapped[int] = mapped_column(Integer, default=25)

    # Python-side defaults (not server_default=func.now()): a server-computed default
    # is only known to SQLAlchemy after a post-flush refresh, which is itself a lazy
    # DB round-trip and hits the exact same async MissingGreenlet problem as an
    # unloaded relationship. A client-side default is known immediately, no query needed.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

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
    kind: Mapped[DeployKind]
    command: Mapped[str | None] = mapped_column(Text, default=None)
    script_path: Mapped[str | None] = mapped_column(default=None)
    webhook_url: Mapped[str | None] = mapped_column(default=None)
    # Names of env vars the deploy execution path should inject — never raw secret values.
    env_var_refs: Mapped[list[str]] = mapped_column(JSON, default=list)
    timeout_seconds: Mapped[int] = mapped_column(Integer, default=600)

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

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    project: Mapped["Project | None"] = relationship(back_populates="endpoint_configs", lazy="raise")


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

    # Orchestrator claim/lease — see orchestrator/worker.py (Phase 4).
    claimed_by_worker_id: Mapped[str | None] = mapped_column(default=None)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

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

    column_visit: Mapped["CardColumnVisit"] = relationship(back_populates="run_attempts", lazy="raise")
