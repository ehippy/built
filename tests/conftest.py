import os
import subprocess
import tempfile
from pathlib import Path

import pytest

# Point the app at an isolated temp SQLite file before `built.config` (and its
# module-level `settings` singleton) is imported by any test module.
_tmp_dir = Path(tempfile.mkdtemp(prefix="built-test-"))
os.environ.setdefault("BUILT_DATA_DIR", str(_tmp_dir))
os.environ.setdefault("BUILT_DATABASE_URL", f"sqlite+aiosqlite:///{_tmp_dir / 'test.db'}")
os.environ.setdefault("BUILT_API_KEY", "test-api-key")

import pytest_asyncio  # noqa: E402

# Imported for its side effect: registers all ORM models on Base.metadata before
# create_all() runs below (mirrors built.main's own comment on this same import).
# Previously this only happened incidentally, as a side effect of some other test
# module importing something under built.db.models first during collection — a
# test file that (like test_logging_config.py) never touches the DB at all could
# run in isolation with an empty Base.metadata and fail create_all() outright.
import built.db.models  # noqa: E402, F401
from built.db.base import Base, async_session_factory, create_all  # noqa: E402
from built.logging_config import configure_logging  # noqa: E402

# Otherwise this only happens as a side effect of importing built.main, which not
# every test module does — whether a given test's logger.info() calls actually
# reach the ring buffer would silently depend on test collection/import order.
configure_logging()


@pytest_asyncio.fixture(autouse=True)
async def _clean_database():
    """Runs before every test. Several code paths commit mid-operation now (see
    agent/loop.py — needed so a card's progress is visible to anything polling
    CardEvent mid-run, e.g. the dashboard), which means the old rollback-only
    teardown no longer actually isolates tests from each other within the shared
    test-session SQLite file: a card left ACTIVE by one test would otherwise be
    picked up by an unrelated orchestrator claim test. Clear every table up front
    instead, so each test starts from a genuinely empty database."""
    await create_all()
    async with async_session_factory() as session:
        async with session.begin():
            for table in reversed(Base.metadata.sorted_tables):
                await session.execute(table.delete())


@pytest_asyncio.fixture
async def db_session(_clean_database):
    """A DB session for tests that drive built.domain functions directly (not through
    the HTTP API). Depends on _clean_database explicitly (even though it's autouse)
    to guarantee ordering: the table wipe must happen before this session opens."""
    async with async_session_factory() as session:
        yield session


@pytest.fixture
def toy_repo_remote(tmp_path: Path) -> Path:
    """A local git repo usable as a `repo_remote_url` — a plain filesystem path is a
    valid git clone source, so worktree/clone tests need no network access."""
    repo_dir = tmp_path / "toy-repo"
    repo_dir.mkdir()

    def run(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True)

    run("init", "-q", "-b", "main")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "Test")
    (repo_dir / "README.md").write_text("# Toy repo\n")
    (repo_dir / "app.py").write_text("def greet():\n    return 'hi'\n")
    run("add", "-A")
    run("commit", "-q", "-m", "init")
    return repo_dir
