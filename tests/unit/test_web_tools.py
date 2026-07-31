import httpx
import pytest

from built.tools import web_tools


def _resolve_to(*ips):
    def fake(hostname: str) -> list[str]:
        return list(ips)

    return fake


async def test_fetch_docs_rejects_non_http_scheme(monkeypatch):
    result = await web_tools.fetch_docs("ftp://docs.example.com/file")
    assert result.is_error
    assert "scheme" in result.output


async def test_fetch_docs_rejects_private_address(monkeypatch):
    monkeypatch.setattr(web_tools, "_resolve_host", _resolve_to("127.0.0.1"))
    result = await web_tools.fetch_docs("https://internal.example/")
    assert result.is_error
    assert "non-public address" in result.output


async def test_fetch_docs_rejects_cloud_metadata_address(monkeypatch):
    monkeypatch.setattr(web_tools, "_resolve_host", _resolve_to("169.254.169.254"))
    result = await web_tools.fetch_docs("http://metadata.internal/latest/meta-data/")
    assert result.is_error
    assert "non-public address" in result.output


async def test_fetch_docs_rejects_unresolvable_host(monkeypatch):
    import socket

    def fake(hostname: str) -> list[str]:
        raise socket.gaierror("name or service not known")

    monkeypatch.setattr(web_tools, "_resolve_host", fake)
    result = await web_tools.fetch_docs("https://nowhere.invalid/")
    assert result.is_error
    assert "could not resolve" in result.output


async def test_fetch_docs_strips_html_and_returns_text(monkeypatch):
    monkeypatch.setattr(web_tools, "_resolve_host", _resolve_to("93.184.216.34"))
    html = (
        "<html><head><style>body{color:red}</style></head>"
        "<body><script>evil()</script><h1>Widget API</h1><p>Call widget.create()</p></body></html>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, text=html)

    result = await web_tools.fetch_docs(
        "https://docs.example.com/widgets", transport=httpx.MockTransport(handler)
    )
    assert not result.is_error
    assert "Widget API" in result.output
    assert "widget.create()" in result.output
    assert "evil()" not in result.output
    assert "color:red" not in result.output


async def test_fetch_docs_follows_redirect_to_allowed_host(monkeypatch):
    hosts = {"old.example.com": "93.184.216.1", "new.example.com": "93.184.216.2"}
    monkeypatch.setattr(web_tools, "_resolve_host", lambda h: [hosts[h]])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "old.example.com":
            return httpx.Response(302, headers={"location": "https://new.example.com/docs"})
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="moved docs body")

    result = await web_tools.fetch_docs(
        "https://old.example.com/docs", transport=httpx.MockTransport(handler)
    )
    assert not result.is_error
    assert "moved docs body" in result.output
    assert "new.example.com" in result.output


async def test_fetch_docs_blocks_redirect_to_private_host(monkeypatch):
    hosts = {"old.example.com": "93.184.216.1", "internal.example.com": "10.0.0.5"}
    monkeypatch.setattr(web_tools, "_resolve_host", lambda h: [hosts[h]])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "old.example.com":
            return httpx.Response(302, headers={"location": "https://internal.example.com/secrets"})
        raise AssertionError("should never reach the redirect target")

    result = await web_tools.fetch_docs(
        "https://old.example.com/docs", transport=httpx.MockTransport(handler)
    )
    assert result.is_error
    assert "non-public address" in result.output


async def test_fetch_docs_too_many_redirects(monkeypatch):
    monkeypatch.setattr(web_tools, "_resolve_host", _resolve_to("93.184.216.34"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": str(request.url)})

    result = await web_tools.fetch_docs(
        "https://docs.example.com/loop", transport=httpx.MockTransport(handler)
    )
    assert result.is_error
    assert "too many redirects" in result.output


async def test_fetch_docs_reports_http_error_status(monkeypatch):
    monkeypatch.setattr(web_tools, "_resolve_host", _resolve_to("93.184.216.34"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    result = await web_tools.fetch_docs(
        "https://docs.example.com/missing", transport=httpx.MockTransport(handler)
    )
    assert result.is_error
    assert "404" in result.output


async def test_fetch_docs_truncates_long_content(monkeypatch):
    monkeypatch.setattr(web_tools, "_resolve_host", _resolve_to("93.184.216.34"))
    monkeypatch.setattr(web_tools, "MAX_DOC_CHARS", 50)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="x" * 500)

    result = await web_tools.fetch_docs(
        "https://docs.example.com/huge", transport=httpx.MockTransport(handler)
    )
    assert not result.is_error
    assert "truncated" in result.output
    assert "500 chars total" in result.output


def test_html_to_text_collapses_whitespace_and_drops_scripts():
    html = "<div>\n  <p>Hello   world</p>\n  <script>bad()</script>\n</div>"
    text = web_tools._html_to_text(html)
    assert "Hello   world" in text
    assert "bad()" not in text


def test_truncate_leaves_short_text_untouched():
    assert web_tools._truncate("short") == "short"


@pytest.mark.parametrize("ip", ["8.8.8.8", "1.1.1.1"])
async def test_rejection_reason_allows_public_ip_literal(ip):
    assert await web_tools._rejection_reason(f"https://{ip}/") is None


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.1.2.3", "169.254.169.254", "192.168.1.1"])
async def test_rejection_reason_blocks_non_public_ip_literal(ip):
    reason = await web_tools._rejection_reason(f"https://{ip}/")
    assert reason is not None
    assert "non-public address" in reason
