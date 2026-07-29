import pytest

from built.tools import read_tools, write_tools
from built.tools.base import ToolContext


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def greet():\n    return 'hi'\n")
    (tmp_path / "README.md").write_text("# Project\n")
    return ToolContext(card_id="card1", worktree_root=tmp_path)


def test_read_file_ok_and_missing(ctx):
    result = read_tools.read_file(ctx, "src/app.py")
    assert not result.is_error
    assert "def greet" in result.output

    missing = read_tools.read_file(ctx, "src/nope.py")
    assert missing.is_error


def test_read_file_rejects_escape(ctx):
    result = read_tools.read_file(ctx, "../../etc/passwd")
    assert result.is_error
    assert "escapes the worktree" in result.output


def test_list_files(ctx):
    result = read_tools.list_files(ctx, ".")
    assert not result.is_error
    assert "README.md" in result.output
    assert "src/" in result.output


def test_glob_files(ctx):
    result = read_tools.glob_files(ctx, "**/*.py")
    assert result.output == "src/app.py"


def test_grep_files(ctx):
    result = read_tools.grep_files(ctx, "greet", ".")
    assert not result.is_error
    assert "src/app.py:1:def greet():" in result.output

    no_match = read_tools.grep_files(ctx, "nonexistent_symbol_xyz", ".")
    assert no_match.output == "(no matches)"


def test_write_file_creates_and_overwrites(ctx):
    result = write_tools.write_file(ctx, "src/new.py", "x = 1\n")
    assert not result.is_error
    assert (ctx.worktree_root / "src" / "new.py").read_text() == "x = 1\n"

    write_tools.write_file(ctx, "src/new.py", "x = 2\n")
    assert (ctx.worktree_root / "src" / "new.py").read_text() == "x = 2\n"


def test_write_file_rejects_escape(ctx):
    result = write_tools.write_file(ctx, "../outside.txt", "pwned")
    assert result.is_error
    assert not (ctx.worktree_root.parent / "outside.txt").exists()


def test_edit_file_replaces_unique_match(ctx):
    result = write_tools.edit_file(ctx, "src/app.py", "return 'hi'", "return 'hello'")
    assert not result.is_error
    assert "return 'hello'" in (ctx.worktree_root / "src" / "app.py").read_text()


def test_edit_file_rejects_ambiguous_match(ctx):
    (ctx.worktree_root / "dup.py").write_text("x = 1\nx = 1\n")
    result = write_tools.edit_file(ctx, "dup.py", "x = 1", "x = 2")
    assert result.is_error
    assert "not unique" in result.output


def test_edit_file_rejects_no_match(ctx):
    result = write_tools.edit_file(ctx, "src/app.py", "not in the file", "replacement")
    assert result.is_error
    assert "not found" in result.output
