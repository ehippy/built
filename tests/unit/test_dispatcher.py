from built.sandbox.container import CommandResult
from built.tools import dispatcher as dispatcher_module
from built.tools.base import ToolContext, ToolResult
from built.tools.dispatcher import ToolDispatcher
from tests.unit.fakes import FakeCommandExecutor


async def test_dispatch_fetch_docs_forwards_url_and_does_not_auto_commit(tmp_path, monkeypatch):
    calls = []

    async def fake_fetch_docs(url: str) -> ToolResult:
        calls.append(url)
        return ToolResult.ok("Fetched https://docs.example.com/api:\n\nsome docs")

    monkeypatch.setattr(dispatcher_module.web_tools, "fetch_docs", fake_fetch_docs)

    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    dispatcher = ToolDispatcher(
        ctx=ToolContext(card_id="card1", worktree_root=tmp_path), executor=executor
    )

    outcome = await dispatcher.dispatch("fetch_docs", {"url": "https://docs.example.com/api"})

    assert calls == ["https://docs.example.com/api"]
    assert not outcome.result.is_error
    assert "some docs" in outcome.result.output
    assert outcome.commit_sha is None
    assert executor.calls == []
