"""Read-only tools: safe to run in-process (no container) since they can't mutate
anything — path confinement via ToolContext.resolve() is protection enough."""

import re
from pathlib import Path

from built.tools.base import PathEscapesWorktreeError, ToolContext, ToolResult

MAX_READ_BYTES = 200_000
MAX_MATCHES = 500


def read_file(ctx: ToolContext, path: str) -> ToolResult:
    try:
        resolved = ctx.resolve(path)
    except PathEscapesWorktreeError as exc:
        return ToolResult.error(str(exc))
    if not resolved.is_file():
        return ToolResult.error(f"no such file: {path!r}")
    data = resolved.read_bytes()
    if len(data) > MAX_READ_BYTES:
        return ToolResult.error(f"{path!r} is too large to read ({len(data)} bytes)")
    try:
        return ToolResult.ok(data.decode("utf-8"))
    except UnicodeDecodeError:
        return ToolResult.error(f"{path!r} is not valid UTF-8 text")


def list_files(ctx: ToolContext, path: str = ".") -> ToolResult:
    try:
        resolved = ctx.resolve(path)
    except PathEscapesWorktreeError as exc:
        return ToolResult.error(str(exc))
    if not resolved.is_dir():
        return ToolResult.error(f"no such directory: {path!r}")
    entries = sorted(
        p.relative_to(ctx.root).as_posix() + ("/" if p.is_dir() else "")
        for p in resolved.iterdir()
        if p.name != ".git"
    )
    return ToolResult.ok("\n".join(entries) or "(empty directory)")


def glob_files(ctx: ToolContext, pattern: str) -> ToolResult:
    if Path(pattern).is_absolute():
        return ToolResult.error(f"pattern {pattern!r} must be relative to the worktree")
    matches = sorted(
        p.relative_to(ctx.root).as_posix() for p in ctx.root.glob(pattern) if ".git" not in p.parts
    )
    return ToolResult.ok("\n".join(matches[:MAX_MATCHES]) or "(no matches)")


def grep_files(ctx: ToolContext, pattern: str, path: str = ".") -> ToolResult:
    try:
        resolved = ctx.resolve(path)
        regex = re.compile(pattern)
    except PathEscapesWorktreeError as exc:
        return ToolResult.error(str(exc))
    except re.error as exc:
        return ToolResult.error(f"invalid regex: {exc}")

    files = (
        [resolved]
        if resolved.is_file()
        else [p for p in sorted(resolved.rglob("*")) if p.is_file() and ".git" not in p.parts]
    )
    hits: list[str] = []
    for file in files:
        try:
            text = file.read_text("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                hits.append(f"{file.relative_to(ctx.root).as_posix()}:{lineno}:{line.strip()}")
                if len(hits) >= MAX_MATCHES:
                    break
        if len(hits) >= MAX_MATCHES:
            break
    return ToolResult.ok("\n".join(hits) or "(no matches)")
