"""Path-confined file mutation tools. Not containerized in this build — see
sandbox/container.py for the bash executor, which is where containerization matters
most (arbitrary shell trivially escapes path confinement; a fixed write/edit schema
cannot, since every path argument is checked by ToolContext.resolve())."""

import re

from built.tools.base import PathEscapesWorktreeError, ToolContext, ToolResult, format_numbered_lines

# Lines of unchanged context to show above/below an edit in the post-edit snippet —
# enough to confirm the change landed in the right place without a follow-up
# read_file, not so much that a routine edit balloons the tool result.
SNIPPET_CONTEXT_LINES = 4
# write_file's confirmation preview is deliberately short: the model just supplied
# `content` itself in this very tool call, so echoing much of it back is pure waste —
# this is a sanity check that it landed on disk intact, not a re-read.
WRITE_PREVIEW_LINES = 10


def write_file(ctx: ToolContext, path: str, content: str) -> ToolResult:
    try:
        resolved = ctx.resolve(path)
    except PathEscapesWorktreeError as exc:
        return ToolResult.error(str(exc))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")

    lines = content.splitlines()
    preview = format_numbered_lines(lines[:WRITE_PREVIEW_LINES], 1)
    summary = f"wrote {len(content)} chars ({len(lines)} lines) to {path}"
    if len(lines) > WRITE_PREVIEW_LINES:
        preview += f"\n\n[{len(lines) - WRITE_PREVIEW_LINES} more line(s) — read_file to review the rest]"
    return ToolResult.ok(f"{summary}\n{preview}" if lines else summary)


def edit_file(
    ctx: ToolContext, path: str, old_str: str, new_str: str, *, replace_all: bool = False
) -> ToolResult:
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
    if count > 1 and not replace_all:
        escaped = re.escape(old_str)
        hit_lines = sorted({text.count("\n", 0, m.start()) + 1 for m in re.finditer(escaped, text)})
        return ToolResult.error(
            f"old_str is not unique in {path!r} — matches at line(s) {', '.join(map(str, hit_lines))}. "
            "Either include more surrounding context to target a single occurrence, or pass "
            "replace_all=true if you actually mean to change all of them."
        )

    if replace_all:
        resolved.write_text(text.replace(old_str, new_str), encoding="utf-8")
        return ToolResult.ok(f"replaced {count} occurrence(s) of old_str in {path!r}")

    start_index = text.index(old_str)
    start_line = text.count("\n", 0, start_index) + 1
    end_line = start_line + new_str.count("\n")
    new_text = text.replace(old_str, new_str, 1)
    resolved.write_text(new_text, encoding="utf-8")

    new_lines = new_text.splitlines()
    snippet_start = max(start_line - SNIPPET_CONTEXT_LINES, 1)
    snippet_end = min(end_line + SNIPPET_CONTEXT_LINES, len(new_lines))
    snippet = format_numbered_lines(new_lines[snippet_start - 1 : snippet_end], snippet_start)
    return ToolResult.ok(f"edited {path} — resulting lines {snippet_start}-{snippet_end}:\n{snippet}")
