"""Manual, ad hoc smoke test that drives PM -> Developer -> Tester against a real
local OpenAI-compatible endpoint and a real toy git repo. NOT part of the automated
test suite (pytest never imports this) and NOT run in CI — it needs network access to
a live LLM endpoint, takes real wall-clock time, and its exact tool-call sequence is
whatever the model decides, so it isn't suitable as a deterministic regression test.
Run it by hand with `python scripts/live_smoke_test.py` to sanity-check the whole
pipeline against a real model when you have one available.

Uses a local, unsandboxed CommandExecutor for the `bash` tool instead of
DockerCommandExecutor — Docker isn't installed in every dev environment (including
the one this was written in). DO NOT reuse LocalCommandExecutor outside this script;
running arbitrary LLM-generated shell commands directly on the host, with no
container, no resource limits, and no capability drops, is exactly the risk
sandbox/container.py exists to contain. It's acceptable here only because the task is
fixed by us, the repo is a disposable toy fixture, and nobody but us runs this script.
"""

import asyncio
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("live_smoke_test")

ENDPOINT_BASE_URL = "http://neuralforge:13305/v1"
ENDPOINT_MODEL = "Qwen3.6-35B-A3B-GGUF"


class LocalCommandExecutor:
    """Runs bash commands directly on the host. See the module docstring — dev-only,
    never wire this into orchestrator/worker.py."""

    async def run(self, *, worktree: Path, command: str, timeout_seconds: int):
        from built.sandbox.container import CommandResult

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(worktree),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
            return CommandResult(
                exit_code=proc.returncode or 0,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
            )
        except TimeoutError:
            return CommandResult(exit_code=-1, stdout="", stderr="timed out", timed_out=True)


def make_toy_repo(root: Path) -> Path:
    repo_dir = root / "toy-repo"
    repo_dir.mkdir()

    def run(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True)

    run("init", "-q", "-b", "main")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "Test")
    (repo_dir / "README.md").write_text("# Toy calculator\n\nA tiny arithmetic library.\n")
    (repo_dir / "app.py").write_text("def add(a, b):\n    return a + b\n")
    (repo_dir / "test_app.py").write_text(
        "from app import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    run("add", "-A")
    run("commit", "-q", "-m", "init")
    return repo_dir


async def main() -> None:
    tmp_root = Path(tempfile.mkdtemp(prefix="built-smoke-"))
    import os

    os.environ["BUILT_DATA_DIR"] = str(tmp_root / "data")
    os.environ["BUILT_DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp_root / 'data' / 'smoke.db'}"
    os.environ["BUILT_API_KEY"] = "smoke-test"
    os.environ["BUILT_ORCHESTRATOR_ENABLED"] = "false"  # we're driving visits by hand

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    import built.db.models  # noqa: F401
    from built.agent.loop import run_developer_visit, run_pm_visit, run_tester_visit
    from built.db.base import async_session_factory, create_all
    from built.db.models import EndpointConfig
    from built.domain import transitions
    from built.domain.enums import Column
    from built.llm.client import FallbackLLMClient
    from built.sandbox import worktree
    from built.services import card_service, project_service
    from built.tools.base import ToolContext
    from built.tools.dispatcher import ToolDispatcher

    await create_all()
    repo_dir = make_toy_repo(tmp_root)
    executor = LocalCommandExecutor()

    async with async_session_factory() as session:
        project = await project_service.create_project(
            session,
            name="Smoke Test Calculator",
            overarching_goal="A small, well-tested arithmetic library.",
            repo_remote_url=str(repo_dir),
            max_iterations_per_run=15,
        )
        card = await card_service.create_card(
            session,
            project.id,
            title="Add subtract()",
            raw_request="Add a subtract(a, b) function to app.py that returns a - b, with a test.",
        )
        await session.commit()

        endpoint = EndpointConfig(
            base_url=ENDPOINT_BASE_URL, model=ENDPOINT_MODEL, supports_tool_calling=True
        )
        llm_client = FallbackLLMClient([endpoint])

        wt_path = await worktree.create_card_worktree(project, card)
        card.worktree_path = str(wt_path)
        await session.commit()
        dispatcher = ToolDispatcher(
            ctx=ToolContext(card_id=card.id, worktree_root=wt_path), executor=executor
        )

        logger.info("=== PM ===")
        visit = await transitions.start_visit(session, card)
        await session.commit()
        card = await run_pm_visit(
            session, project, card, visit, llm_client=llm_client, dispatcher=dispatcher, max_iterations=15
        )
        logger.info("column=%s lifecycle=%s", card.column, card.lifecycle_state)
        logger.info("spec: %s", card.spec)
        logger.info("acceptance_criteria: %s", card.acceptance_criteria)

        if card.column == Column.DEVELOPER and card.lifecycle_state.value == "active":
            logger.info("=== Developer ===")
            visit = await transitions.start_visit(session, card)
            await session.commit()
            card = await run_developer_visit(
                session, project, card, visit, llm_client=llm_client, dispatcher=dispatcher, max_iterations=15
            )
            logger.info("column=%s lifecycle=%s", card.column, card.lifecycle_state)
            log = subprocess.run(
                ["git", "log", "--oneline"], cwd=wt_path, capture_output=True, text=True
            ).stdout
            logger.info("git log:\n%s", log)
            logger.info("app.py:\n%s", (wt_path / "app.py").read_text())

        if card.column == Column.TESTER and card.lifecycle_state.value == "active":
            logger.info("=== Tester ===")
            developer_summary = await card_service.get_latest_visit_summary(
                session, card.id, Column.DEVELOPER
            )
            visit = await transitions.start_visit(session, card)
            await session.commit()
            card = await run_tester_visit(
                session,
                project,
                card,
                visit,
                llm_client=llm_client,
                dispatcher=dispatcher,
                max_iterations=15,
                developer_summary=developer_summary,
            )
            logger.info("column=%s lifecycle=%s", card.column, card.lifecycle_state)

        events = await card_service.list_events(session, card.id, limit=1000)
        logger.info("=== Transcript (%d events) ===", len(events))
        for event in events:
            logger.info("[%s] %s: %s", event.type.value, event.seq, str(event.payload)[:300])

        logger.info("=== FINAL: column=%s lifecycle=%s ===", card.column, card.lifecycle_state)

    shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
