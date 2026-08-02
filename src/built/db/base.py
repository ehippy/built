from collections.abc import AsyncIterator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from built.config import settings


class Base(DeclarativeBase):
    """Shared declarative base — every ORM model in built.db.models inherits from this."""


engine = create_async_engine(settings.database_url, echo=False)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


@event.listens_for(engine.sync_engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record) -> None:
    """Several code paths (orchestrator/curator/deployer loops, and detached
    asyncio.create_task background jobs like the manual curate trigger) hold a DB
    session open concurrently with API request handling. SQLite's default rollback
    journal serializes a writer's COMMIT behind any other connection's open read
    transaction, which without a busy_timeout raises 'database is locked'
    immediately instead of waiting. WAL lets readers and the one writer proceed
    without blocking each other in the first place; busy_timeout is the fallback
    for the writer-vs-writer case WAL alone doesn't solve."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


ADDITIVE_COLUMNS = [
    ("projects", "test_command", "TEXT"),
    ("run_attempts", "card_event_seq", "INTEGER"),
    ("cards", "deploying_commit_sha", "TEXT"),
    ("cards", "deploying_since", "DATETIME"),
    # Nullable, no DEFAULT needed — only pr_to_operator cards waiting on a PR
    # review ever set these (see db/models.py's Card.pr_number).
    ("cards", "pr_number", "INTEGER"),
    ("cards", "pr_waiting_since", "DATETIME"),
    # DEFAULT applies to existing rows too (unlike the ORM-side mapped_column
    # default, which only fires on new inserts) — without it, every pre-existing
    # card would read back as priority=NULL instead of Priority.NORMAL. Must be
    # the enum MEMBER NAME ('NORMAL'), not its value ('normal') — SQLAlchemy's
    # Enum type stores and looks up by .name by default (confirmed against every
    # other enum column in this app: lifecycle_state holds 'ACTIVE'/'DONE'/etc,
    # not 'active'/'done'), so a lowercase default here reads back as a
    # LookupError instead of Priority.NORMAL.
    ("cards", "priority", "TEXT DEFAULT 'NORMAL'"),
    # Nullable, no DEFAULT needed — every existing project simply has no current
    # epic until a human sets one (see db/models.py's Project.current_epic_id).
    ("projects", "current_epic_id", "TEXT"),
    # Nullable, no DEFAULT needed — every existing project has never had its chat
    # cleared (see db/models.py's Project.chat_cleared_before_seq).
    ("projects", "chat_cleared_before_seq", "INTEGER"),
    # Nullable, no DEFAULT needed — every existing card simply has no pending nudge
    # until a human leaves one (see db/models.py's Card.pending_nudge).
    ("cards", "pending_nudge", "TEXT"),
    # Nullable, no DEFAULT needed — every existing project simply has no per-role
    # guidance until a human sets it (see db/models.py's Project.pm_guidance etc).
    ("projects", "pm_guidance", "TEXT"),
    ("projects", "developer_guidance", "TEXT"),
    ("projects", "tester_guidance", "TEXT"),
    ("projects", "reviewer_guidance", "TEXT"),
    ("projects", "deployer_guidance", "TEXT"),
    # Nullable, no DEFAULT needed — every curation_events row written before
    # CurationRun existed simply has no run to group into (see
    # db/models.py's CurationEvent.run_id/tokens_in/tokens_out/latency_ms).
    ("curation_events", "run_id", "TEXT"),
    ("curation_events", "tokens_in", "INTEGER"),
    ("curation_events", "tokens_out", "INTEGER"),
    ("curation_events", "latency_ms", "INTEGER"),
    # Nullable, no DEFAULT needed — every run_attempts row recorded before pipestatus
    # capture existed simply has none (see db/models.py's RunAttempt.pipestatus).
    ("run_attempts", "pipestatus", "TEXT"),
]


def _add_missing_columns(sync_conn) -> None:
    """create_all() only creates whole tables that don't exist yet — it never adds a
    column to a table that's already there. Nullable columns added to a model after
    its table already exists on disk need this nudge, or the app 500s the first time
    it touches the new column against a pre-existing SQLite file."""
    for table, column, sql_type in ADDITIVE_COLUMNS:
        existing = {row[1] for row in sync_conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            sync_conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")


async def create_all() -> None:
    """v1 has no Alembic — schema is (re)created from current model metadata at startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with async_session_factory() as session:
        yield session
