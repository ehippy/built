"""Path-confined file mutation tools. Not containerized in this build — see
sandbox/container.py for the bash executor, which is where containerization matters
most (arbitrary shell trivially escapes path confinement; a fixed write/edit schema
cannot, since every path argument is checked by ToolContext.resolve())."""

from built.tools.base import PathEscapesWorktreeError, ToolContext, ToolResult


def write_file(ctx: ToolContext, path: str, content: str) -> ToolResult:
    try:
        resolved = ctx.resolve(path)
    except PathEscapesWorktreeError as exc:
        return ToolResult.error(str(exc))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return ToolResult.ok(f"wrote {len(content)} chars to {path}")


def edit_file(ctx: ToolContext, path: str, old_str: str, new_str: str) -> ToolResult:
    try:
        resolved = ctx.resolve(path)
    except PathEscapesWorktreeError as exc:
        return ToolResult.error(str(exc))
    if not resolved.is_file():
        return ToolResult.error(f"no such file: {path!r}")
    text = resolved.read_text(encoding="utf-8")
    count = text.count(old_str)
    if count == 0:
        return ToolResult.error(f"old_str not found in {path!r}")
    if count > 1:
        return ToolResult.error(
            f"old_str is not unique in {path!r} ({count} occurrences) — include more context"
        )
    resolved.write_text(text.replace(old_str, new_str, 1), encoding="utf-8")
    return ToolResult.ok(f"edited {path}")
