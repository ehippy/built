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
