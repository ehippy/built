"""fetch_docs: the one tool that reaches outside the worktree and outside the sandbox
container entirely, straight from the orchestrator process — safe to run in-process
(like read_tools) because URL/IP validation before every request is the confinement
mechanism, the same role ToolContext.resolve() plays for paths. Every hop (including
redirects) is re-validated: scheme must be http/https and the resolved IP must be
globally routable, which rejects loopback/private/link-local addresses — including
169.254.169.254, the cloud metadata address sandbox/container.py's DockerCommandExecutor
docstring flags as a known, unclosed gap for the `bash` tool. This tool closes that gap
for itself rather than inheriting it.
"""

import asyncio
import ipaddress
import socket
from html.parser import HTMLParser

import httpx

from built.tools.base import ToolResult

MAX_REDIRECTS = 5
FETCH_TIMEOUT_SECONDS = 20.0
MAX_RESPONSE_BYTES = 3_000_000
MAX_DOC_CHARS = 20_000
USER_AGENT = "built-fetch-docs/1.0 (+autonomous software factory doc lookup)"


def _resolve_host(hostname: str) -> list[str]:
    """Thin wrapper around socket.getaddrinfo so tests can fake DNS resolution without
    touching the network or relying on IP-literal URLs."""
    return [info[4][0] for info in socket.getaddrinfo(hostname, None)]


async def _rejection_reason(url: str) -> str | None:
    """None if url is safe to fetch, else the reason it's rejected."""
    parsed = httpx.URL(url)
    if parsed.scheme not in ("http", "https"):
        return f"unsupported URL scheme {parsed.scheme!r} in {url!r} — only http/https are allowed"
    hostname = parsed.host
    if not hostname:
        return f"{url!r} has no hostname"
    try:
        addresses = await asyncio.to_thread(_resolve_host, hostname)
    except socket.gaierror as exc:
        return f"could not resolve host {hostname!r}: {exc}"
    for raw_ip in addresses:
        if not ipaddress.ip_address(raw_ip).is_global:
            return f"{url!r} resolves to a non-public address ({raw_ip}) and is blocked"
    return None


class _TextExtractor(HTMLParser):
    """Strips tags and drops <script>/<style>/<noscript> contents, keeping the rest as
    plain text — enough to make a docs page readable without a new HTML/markdown
    dependency for what's fundamentally just tag-stripping."""

    _SKIPPED_TAGS = frozenset({"script", "style", "noscript"})

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIPPED_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIPPED_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._chunks.append(data)

    def text(self) -> str:
        lines = (line.strip() for line in "".join(self._chunks).splitlines())
        return "\n".join(line for line in lines if line)


def _html_to_text(html: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(html)
    return extractor.text()


def _truncate(text: str) -> str:
    if len(text) <= MAX_DOC_CHARS:
        return text
    return text[:MAX_DOC_CHARS] + f"\n... [truncated, {len(text)} chars total]"


async def fetch_docs(url: str, *, transport: httpx.AsyncBaseTransport | None = None) -> ToolResult:
    current_url = url
    async with httpx.AsyncClient(
        transport=transport, timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=False
    ) as client:
        for _ in range(MAX_REDIRECTS + 1):
            reason = await _rejection_reason(current_url)
            if reason:
                return ToolResult.error(reason)
            try:
                async with client.stream(
                    "GET", current_url, headers={"User-Agent": USER_AGENT}
                ) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            return ToolResult.error(f"{current_url!r} redirected with no Location header")
                        current_url = str(response.url.join(location))
                        continue
                    if response.status_code >= 400:
                        return ToolResult.error(f"{current_url} returned HTTP {response.status_code}")
                    content_type = response.headers.get("content-type", "")
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_RESPONSE_BYTES:
                            break
                    final_url = str(response.url)
                    encoding = response.encoding or "utf-8"
            except httpx.HTTPError as exc:
                return ToolResult.error(f"failed to fetch {current_url!r}: {exc}")
            break
        else:
            return ToolResult.error(f"too many redirects fetching {url!r} (>{MAX_REDIRECTS})")

    text = bytes(body).decode(encoding, errors="replace")
    if "html" in content_type:
        text = _html_to_text(text)
    return ToolResult.ok(f"Fetched {final_url}:\n\n{_truncate(text)}")
