from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from built.config import settings


class Base(DeclarativeBase):
    """Shared declarative base — every ORM model in built.db.models inherits from this."""


engine = create_async_engine(settings.database_url, echo=False)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


ADDITIVE_COLUMNS = [
    ("projects", "test_command", "TEXT"),
    ("run_attempts", "card_event_seq", "INTEGER"),
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
