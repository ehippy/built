"""GET /ui/logs and /ui/logs/fragment — the dashboard view into the unified
"built" logger."""

import logging

from httpx import ASGITransport, AsyncClient

from built.main import app


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_logs_page_renders_a_captured_log_line():
    logging.getLogger("built.orchestrator.reviver").info("reviver pass: revived=2 left_blocked=1 errors=0")

    async with _client() as client:
        page = await client.get("/ui/logs")

    assert page.status_code == 200
    assert "reviver pass: revived=2 left_blocked=1 errors=0" in page.text
    assert "built.orchestrator.reviver" in page.text


async def test_logs_fragment_polls_independently_of_the_full_page():
    logging.getLogger("built.orchestrator.curator").warning("curator: something odd")

    async with _client() as client:
        fragment = await client.get("/ui/logs/fragment")

    assert fragment.status_code == 200
    assert "curator: something odd" in fragment.text
    assert 'id="logs"' in fragment.text
    assert 'hx-get="/ui/logs/fragment"' in fragment.text


async def test_logs_page_escapes_html_in_a_log_message():
    logging.getLogger("built.x").error("<script>window.__pwned = true;</script>")

    async with _client() as client:
        page = await client.get("/ui/logs")

    assert page.status_code == 200
    assert "<script>window.__pwned" not in page.text
    assert "&lt;script&gt;window.__pwned" in page.text
