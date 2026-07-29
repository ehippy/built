"""Formats a CommandResult into a ToolResult for the model. Execution itself happens
in the dispatcher (which calls the CommandExecutor directly) — kept separate here so
callers that need the raw CommandResult back (Tester's RunAttempt bookkeeping, which
gates the server-verified `approve` transition) don't have to re-parse formatted text
to recover the exit code."""

from built.sandbox.container import CommandResult
from built.tools.base import ToolResult

MAX_OUTPUT_CHARS = 20_000
DEFAULT_TIMEOUT_SECONDS = 120


def format_bash_result(result: CommandResult) -> ToolResult:
    output = (
        f"exit code: {result.exit_code}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(output)} chars total]"
    if result.timed_out:
        return ToolResult.error(output)
    return ToolResult.ok(output) if result.exit_code == 0 else ToolResult.error(output)
