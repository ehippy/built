"""Read-only tools: safe to run in-process (no container) since they can't mutate
anything — path confinement via ToolContext.resolve() is protection enough."""

import re
from pathlib import Path

from built.tools.base import PathEscapesWorktreeError, ToolContext, ToolResult

# A hard ceiling on what we'll even open — catches a binary/generated artifact
# rather than dumping megabytes of tokens into the conversation. Distinct from
# DEFAULT_READ_LINES: a file under this ceiling but longer than that default is
# still fully readable, just paged via offset/limit instead of all at once.
MAX_FILE_BYTES = 5_000_000
DEFAULT_READ_LINES = 2000
MAX_MATCHES = 500


def read_file(ctx: ToolContext, path: str, offset: int = 1, limit: int | None = None) -> ToolResult:
    try:
        resolved = ctx.resolve(path)
    except PathEscapesWorktreeError as exc:
        return ToolResult.error(str(exc))
    if not resolved.is_file():
        return ToolResult.error(f"no such file: {path!r}")
    data = resolved.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        return ToolResult.error(
            f"{path!r} is {len(data)} bytes — too large to open at all. If it's a text file, use "
            "grep_files to find the lines you actually need, or shell out to `sed`/`head` via bash."
        )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return ToolResult.error(f"{path!r} is not valid UTF-8 text")

    lines = text.splitlines()
    total = len(lines)
    start = max(offset, 1) - 1
    if total and start >= total:
        return ToolResult.error(f"{path!r} has only {total} lines — offset {offset} is past the end")
    end = min(start + (limit if limit else DEFAULT_READ_LINES), total)

    numbered = "\n".join(f"{i:>6}\t{lines[i - 1]}" for i in range(start + 1, end + 1))
    if end < total:
        numbered += f"\n\n[{total - end} more line(s) below — pass offset={end + 1} to continue reading]"
    return ToolResult.ok(numbered)


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
