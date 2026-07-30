"""Every tool function is scoped to exactly one card's worktree via a ToolContext.
Path arguments are resolved and confined here — the primary safety mechanism standing
between an autonomous, tool-calling LLM and the rest of the filesystem, and the only
protection for tools that run in-process rather than in a container."""

from dataclasses import dataclass
from pathlib import Path


class PathEscapesWorktreeError(Exception):
    pass


@dataclass(frozen=True)
class ToolContext:
    card_id: str
    worktree_root: Path
    # Off only for the Deployer's merge-conflict-resolution worktree: a partial fix
    # (one of several conflicted files) must never auto-commit and silently complete
    # a merge with leftover conflict markers still baked into an untouched file.
    # deploy_runner completes that commit itself once every conflicted path is clear.
    auto_commit: bool = True
    # Reviewer only: the branch to diff HEAD against (project.default_branch). Every
    # other role's worktree either has no committed history worth comparing (PM) or
    # auto-commits after each tool call, so an ordinary uncommitted `git diff` is
    # meaningless for them — Reviewer needs the full since-branched diff instead.
    base_ref: str | None = None

    @property
    def root(self) -> Path:
        """The canonicalized worktree root. Use this — not `worktree_root` directly —
        anywhere a path gets compared against or made relative to the root. On macOS,
        `$TMPDIR` (and `tempfile.mkdtemp()`) resolve through a /var -> /private/var
        symlink; `worktree_root` may be the unresolved form while paths discovered via
        `resolve()`/`iterdir()`/`rglob()` are already canonical, and mixing the two
        makes `Path.relative_to()` raise even though both sides name the same file."""
        return self.worktree_root.resolve()

    def resolve(self, relative_path: str) -> Path:
        """Resolve a model-supplied path against the worktree root, rejecting any
        attempt to escape it: absolute paths, `..` traversal, or a symlink that
        resolves outside the root. `Path(root) / "/abs/path"` silently discards the
        root in plain pathlib — the absolute-path check below exists specifically to
        catch that before it ever reaches the join."""
        if Path(relative_path).is_absolute():
            raise PathEscapesWorktreeError(f"path {relative_path!r} must be relative to the worktree")
        candidate = (self.root / relative_path).resolve()
        root = self.root
        if candidate != root and root not in candidate.parents:
            raise PathEscapesWorktreeError(f"path {relative_path!r} escapes the worktree")
        return candidate


@dataclass
class ToolResult:
    output: str
    is_error: bool = False

    @classmethod
    def ok(cls, output: str) -> "ToolResult":
        return cls(output=output, is_error=False)

    @classmethod
    def error(cls, message: str) -> "ToolResult":
        return cls(output=message, is_error=True)


def format_numbered_lines(lines: list[str], start_line: int) -> str:
    """Render already-sliced `lines` with a 'right-aligned line number, tab, text'
    prefix — the one format every tool that echoes file content back (read_file,
    edit_file's post-edit snippet) uses, so output looks and aligns identically no
    matter which tool produced it. `start_line` is the 1-indexed line number of
    lines[0]. This prefix is never part of the file itself — callers that build
    old_str for edit_file from this output must strip it first."""
    return "\n".join(f"{i:>6}\t{line}" for i, line in enumerate(lines, start=start_line))
