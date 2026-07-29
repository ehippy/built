"""Fakes for the two pieces this dev environment can't exercise live: a real LLM
endpoint and a Docker daemon. Both LLMClient and CommandExecutor are Protocols
specifically so the agent loop and dispatcher can be fully tested without either."""

from built.llm.client import LLMResult
from built.sandbox.container import CommandResult


class ScriptedLLMClient:
    """Returns pre-scripted LLMResults in order, one per `.complete()` call."""

    def __init__(self, responses: list[LLMResult]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, *, messages: list[dict], tools: list[dict]) -> LLMResult:
        self.calls.append({"messages": messages, "tools": tools})
        if not self._responses:
            raise AssertionError("ScriptedLLMClient ran out of scripted responses")
        return self._responses.pop(0)


class RaisingLLMClient:
    """Simulates every endpoint in the fallback chain failing."""

    async def complete(self, *, messages: list[dict], tools: list[dict]) -> LLMResult:
        raise RuntimeError("simulated: all endpoints in the fallback chain failed")


class FakeCommandExecutor:
    """Returns a fixed CommandResult without touching Docker or the filesystem."""

    def __init__(self, result: CommandResult):
        self._result = result
        self.calls: list[str] = []

    async def run(self, *, worktree, command: str, timeout_seconds: int) -> CommandResult:
        self.calls.append(command)
        return self._result
