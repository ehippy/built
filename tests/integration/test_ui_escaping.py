"""Regression test for a real stored-XSS gap: every dashboard template is named
*.html.j2, which Starlette's default `select_autoescape()` does not recognize (it only
matches names ending in .html/.htm/.xml), so autoescaping was silently off for the
whole app. Event payloads routinely contain arbitrary content an agent read or
produced — source files, bash output, LLM text — and the transcript view rendered it
as live HTML/JS. templates.py now forces autoescape on explicitly."""

from httpx import ASGITransport, AsyncClient

from built.domain.enums import EventType
from built.domain.events import append_event
from built.main import app
from built.services import card_service, project_service


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_card_transcript_escapes_html_in_event_payloads(db_session):
    project = await project_service.create_project(
        db_session,
        name="XSS check",
        overarching_goal="goal",
        repo_remote_url="https://example.invalid/repo.git",
    )
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await db_session.commit()

    # Simulate a tool_call event carrying content an agent read straight off disk —
    # e.g. a source file containing a <script> tag, which is completely ordinary.
    await append_event(
        db_session,
        card_id=card.id,
        type=EventType.TOOL_CALL,
        payload={
            "name": "read_file",
            "arguments": {"path": "evil.js"},
            "result": "<script>window.__pwned = true;</script>",
            "is_error": False,
        },
    )
    await db_session.commit()

    async with _client() as client:
        page = await client.get(f"/ui/cards/{card.id}")

    assert page.status_code == 200
    assert "<script>window.__pwned" not in page.text
    assert "&lt;script&gt;window.__pwned" in page.text


async def test_card_spec_renders_as_markdown_but_escapes_raw_html(db_session):
    project = await project_service.create_project(
        db_session,
        name="Markdown check",
        overarching_goal="goal",
        repo_remote_url="https://example.invalid/repo.git",
    )
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    card.spec = (
        "# Heading\n\nDo the **thing**, then:\n\n- step one\n- step two\n\n"
        "<script>window.__pwned = true;</script>"
    )
    await db_session.commit()

    async with _client() as client:
        page = await client.get(f"/ui/cards/{card.id}")

    assert page.status_code == 200
    # Markdown syntax is converted into real HTML elements, not shown as a raw blob.
    # Headings render as a bold block rather than an actual <h1>-<h6> — a card's
    # spec is user content inside this page, not a document with its own outline,
    # see templates._CardMarkdownRenderer.
    assert "Heading</strong>" in page.text
    assert "<strong>thing</strong>" in page.text
    assert "<li>step one</li>" in page.text
    # Raw HTML embedded in the spec source is still escaped, not executed.
    assert "<script>window.__pwned" not in page.text
    assert "&lt;script&gt;window.__pwned" in page.text
