import pytest

from built.tools.base import PathEscapesWorktreeError, ToolContext


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "file.txt").write_text("hi")
    return ToolContext(card_id="card1", worktree_root=tmp_path)


def test_resolve_allows_paths_inside_the_worktree(ctx):
    assert ctx.resolve("sub/file.txt") == ctx.worktree_root / "sub" / "file.txt"
    assert ctx.resolve(".") == ctx.worktree_root.resolve()


def test_resolve_rejects_parent_traversal(ctx):
    with pytest.raises(PathEscapesWorktreeError):
        ctx.resolve("../../etc/passwd")


def test_resolve_rejects_absolute_paths(ctx):
    with pytest.raises(PathEscapesWorktreeError):
        ctx.resolve("/etc/passwd")


def test_resolve_rejects_symlink_escape(ctx, tmp_path):
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("secret")
    (ctx.worktree_root / "escape").symlink_to(outside)
    with pytest.raises(PathEscapesWorktreeError):
        ctx.resolve("escape")
