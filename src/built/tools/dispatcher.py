"""Executes a tool call by name against a column's non-terminal toolset. Terminal
tools (submit_spec, submit_for_test, approve, request_changes, run_deploy) are NOT
handled here — the agent loop intercepts those directly, since calling one ends the
run and triggers a domain state transition rather than producing a ToolResult fed
back to the model."""

from dataclasses import dataclass

from built.sandbox.container import CommandExecutor, CommandResult
from built.tools import git_tools, read_tools, web_tools, write_tools
from built.tools.base import ToolContext, ToolResult
from built.tools.bash_tool import DEFAULT_TIMEOUT_SECONDS, format_bash_result

MUTATING_TOOLS = frozenset({"write_file", "edit_file", "bash"})


@dataclass
class DispatchOutcome:
    result: ToolResult
    commit_sha: str | None = None
    # Only set when name == "bash" — the raw exit code/output, for callers that need
    # more than the formatted ToolResult text (e.g. Tester's RunAttempt bookkeeping).
    command_result: CommandResult | None = None


@dataclass
class ToolDispatcher:
    ctx: ToolContext
    executor: CommandExecutor

    async def dispatch(self, name: str, arguments: dict) -> DispatchOutcome:
        try:
            result, command_result = await self._call(name, arguments)
        except KeyError as exc:
            return DispatchOutcome(
                result=ToolResult.error(f"missing required argument {exc} for tool {name!r}")
            )

        commit_sha = None
        if name in MUTATING_TOOLS and self.ctx.auto_commit and not result.is_error:
            commit_sha = await git_tools.commit_all(
                self.ctx.worktree_root, message=_commit_message(name, arguments)
            )
        return DispatchOutcome(result=result, commit_sha=commit_sha, command_result=command_result)

    async def _call(self, name: str, arguments: dict) -> tuple[ToolResult, CommandResult | None]:
        if name == "read_file":
            kwargs = {}
            if "offset" in arguments:
                kwargs["offset"] = arguments["offset"]
            if "limit" in arguments:
                kwargs["limit"] = arguments["limit"]
            return read_tools.read_file(self.ctx, arguments["path"], **kwargs), None
        if name == "list_files":
            return read_tools.list_files(self.ctx, arguments.get("path", ".")), None
        if name == "glob_files":
            return read_tools.glob_files(self.ctx, arguments["pattern"]), None
        if name == "grep_files":
            return read_tools.grep_files(self.ctx, arguments["pattern"], arguments.get("path", ".")), None
        if name == "write_file":
            return write_tools.write_file(self.ctx, arguments["path"], arguments["content"]), None
        if name == "edit_file":
            return (
                write_tools.edit_file(
                    self.ctx,
                    arguments["path"],
                    arguments["old_str"],
                    arguments["new_str"],
                    replace_all=arguments.get("replace_all", False),
                ),
                None,
            )
        if name == "bash":
            command_result = await self.executor.run(
                worktree=self.ctx.worktree_root,
                command=arguments["command"],
                timeout_seconds=arguments.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
            )
            return format_bash_result(command_result), command_result
        if name == "git_status":
            return ToolResult.ok(await git_tools.status(self.ctx.worktree_root)), None
        if name == "git_diff":
            return ToolResult.ok(await git_tools.diff(self.ctx.worktree_root)), None
        if name == "review_diff":
            assert self.ctx.base_ref is not None, "review_diff requires ToolContext.base_ref"
            diff = await git_tools.diff_against_ref(self.ctx.worktree_root, self.ctx.base_ref)
            return ToolResult.ok(diff), None
        if name == "fetch_docs":
            return await web_tools.fetch_docs(arguments["url"]), None
        if name == "update_plan":
            # Pure bookkeeping — no worktree/git access, nothing to auto-commit. The
            # steps themselves are recorded as this TOOL_CALL event's `arguments` by
            # run_column_visit like any other tool call; run_attempts.
            # has_completed_plan_this_visit reads them back from there.
            steps = arguments["steps"]
            done = sum(1 for step in steps if step.get("status") == "done")
            return ToolResult.ok(f"Plan recorded: {len(steps)} step(s), {done} marked done."), None
        return ToolResult.error(f"unknown tool: {name!r}"), None


def _commit_message(name: str, arguments: dict) -> str:
    if name in ("write_file", "edit_file"):
        return f"{name}: {arguments.get('path', '?')}"
    if name == "bash":
        command = str(arguments.get("command", ""))
        return f"bash: {command[:72]}"
    return name
