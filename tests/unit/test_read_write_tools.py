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


def test_read_file_prefixes_line_numbers(ctx):
    result = read_tools.read_file(ctx, "src/app.py")
    lines = result.output.splitlines()
    assert lines[0] == "     1\tdef greet():"
    assert lines[1] == "     2\t    return 'hi'"


def test_read_file_offset_and_limit_page_through_a_file(ctx, tmp_path):
    (tmp_path / "big.txt").write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n")

    first_page = read_tools.read_file(ctx, "big.txt", limit=3)
    assert not first_page.is_error
    assert [line.split("\t")[1] for line in first_page.output.splitlines()[:3]] == [
        "line1",
        "line2",
        "line3",
    ]
    assert "7 more line(s) below" in first_page.output
    assert "offset=4" in first_page.output

    second_page = read_tools.read_file(ctx, "big.txt", offset=4, limit=3)
    assert not second_page.is_error
    body, _, _trailer = second_page.output.partition("\n\n")
    assert [line.split("\t")[1] for line in body.splitlines()] == ["line4", "line5", "line6"]

    past_the_end = read_tools.read_file(ctx, "big.txt", offset=100)
    assert past_the_end.is_error
    assert "only 10 lines" in past_the_end.output


def test_read_file_default_limit_truncates_long_files(ctx, tmp_path):
    (tmp_path / "long.txt").write_text("\n".join(f"line{i}" for i in range(1, 2100)) + "\n")

    result = read_tools.read_file(ctx, "long.txt")
    assert not result.is_error
    body, _, trailer = result.output.partition("\n\n")
    body_lines = body.splitlines()
    assert len(body_lines) == read_tools.DEFAULT_READ_LINES
    assert body_lines[-1].split("\t")[1] == "line2000"
    assert "99 more line(s) below" in trailer
    assert "offset=2001" in trailer


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


def test_write_file_confirmation_includes_stats_and_preview(ctx):
    result = write_tools.write_file(ctx, "src/new.py", "a = 1\nb = 2\n")
    assert not result.is_error
    assert "wrote 12 chars (2 lines) to src/new.py" in result.output
    assert "     1\ta = 1" in result.output
    assert "     2\tb = 2" in result.output


def test_write_file_preview_truncates_long_content(ctx):
    content = "\n".join(f"x{i} = {i}" for i in range(1, 30)) + "\n"
    result = write_tools.write_file(ctx, "src/new.py", content)
    assert not result.is_error
    assert len(result.output.splitlines()) < 30
    assert "more line(s)" in result.output


def test_edit_file_replaces_unique_match(ctx):
    result = write_tools.edit_file(ctx, "src/app.py", "return 'hi'", "return 'hello'")
    assert not result.is_error
    assert "return 'hello'" in (ctx.worktree_root / "src" / "app.py").read_text()


def test_edit_file_shows_post_edit_snippet(ctx):
    result = write_tools.edit_file(ctx, "src/app.py", "return 'hi'", "return 'hello'")
    assert not result.is_error
    assert "edited src/app.py" in result.output
    # The snippet shows the resulting file content, line-numbered, not just a bare confirmation.
    assert "     2\t    return 'hello'" in result.output


def test_edit_file_rejects_ambiguous_match(ctx):
    (ctx.worktree_root / "dup.py").write_text("x = 1\nx = 1\n")
    result = write_tools.edit_file(ctx, "dup.py", "x = 1", "x = 2")
    assert result.is_error
    assert "not unique" in result.output
    # Line numbers of every match, so the model can disambiguate without a grep round trip.
    assert "line(s) 1, 2" in result.output


def test_edit_file_replace_all(ctx):
    (ctx.worktree_root / "dup.py").write_text("x = 1\ny = 1\nz = 1\n")
    result = write_tools.edit_file(ctx, "dup.py", "= 1", "= 2", replace_all=True)
    assert not result.is_error
    assert "replaced 3 occurrence(s)" in result.output
    assert (ctx.worktree_root / "dup.py").read_text() == "x = 2\ny = 2\nz = 2\n"


def test_edit_file_rejects_no_match(ctx):
    result = write_tools.edit_file(ctx, "src/app.py", "not in the file", "replacement")
    assert result.is_error
    assert "not found" in result.output
