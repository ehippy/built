"""Jinja2 templates for the dashboard. UI routers call the same services/ layer as
the REST API directly — no self-HTTP-calling."""

import re
from pathlib import Path

import mistune
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Same shapes sandbox/deploy_runner.py's parse_github_owner_repo matches, kept as
# its own copy so the UI layer doesn't reach into the deploy trust boundary just
# for a plain string transform.
_GITHUB_URL_PATTERNS = [
    re.compile(r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(\.git)?/?$"),
    re.compile(r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(\.git)?$"),
]


class _CardMarkdownRenderer(mistune.HTMLRenderer):
    """A card's spec is user content embedded inside this app's own page — not a
    document with its own outline — so a stray '# Heading' in it shouldn't render
    as an actual <h1>, competing with (or outsizing) the page's real heading
    hierarchy. Every markdown heading level renders identically: bold and
    modestly sized, distinguishable from body text but never louder than it."""

    def heading(self, text: str, level: int, **attrs: object) -> str:
        return f'<strong class="d-block mt-3 mb-1">{text}</strong>\n'


# escape=True (the default) escapes any raw HTML embedded in the markdown source
# rather than passing it through, so agent-generated spec text can't inject markup.
_markdown = mistune.create_markdown(renderer=_CardMarkdownRenderer())

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
# Starlette's default autoescape only fires for names ending in .html/.htm/.xml —
# every template here is named *.html.j2, so it silently falls through to "off"
# unless forced on explicitly. With it off, arbitrary content an agent reads or
# produces (source files, bash output, LLM text) renders as live HTML/JS in the
# transcript view.
templates.env.autoescape = True


def _timeago(dt) -> str:
    from datetime import UTC, datetime

    if dt is None:
        return "—"
    now = datetime.now(UTC)
    delta = now - (dt if dt.tzinfo else dt.replace(tzinfo=UTC))
    seconds = delta.total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def _render_markdown(text: str | None) -> Markup:
    if not text:
        return Markup("")
    return Markup(_markdown(text))


def _clocktime(dt) -> str:
    """Local wall-clock HH:MM:SS — unlike timeago, distinguishable at
    sub-minute granularity, which a live-tailing log view needs (timeago would
    show almost every line as "just now")."""
    if dt is None:
        return "—"
    return dt.astimezone().strftime("%H:%M:%S")


def tool_descriptor(arguments: dict | None) -> str:
    """The single most useful argument to show next to a tool name in a compact
    label, e.g. "read_file(app.py)" or "grep_files(TODO)" — not used by the bash
    tool, which shows its full command instead of a one-word descriptor. Shared
    between the card event transcript and the board's curation status panel
    (built.ui.routers.board._describe_curation_event) so a tool call reads
    identically wherever it's shown."""
    args = arguments or {}
    return args.get("path") or args.get("pattern") or args.get("command") or ""


def _github_url(remote_url: str | None) -> str | None:
    """A project's repo_remote_url as a browsable https://github.com/owner/repo
    link, or None if it isn't a GitHub remote (a local path, GitLab, ...)."""
    if not remote_url:
        return None
    for pattern in _GITHUB_URL_PATTERNS:
        match = pattern.match(remote_url.strip())
        if match:
            return f"https://github.com/{match.group('owner')}/{match.group('repo')}"
    return None


templates.env.filters["timeago"] = _timeago
templates.env.filters["markdown"] = _render_markdown
templates.env.filters["clocktime"] = _clocktime
templates.env.filters["tool_descriptor"] = tool_descriptor
templates.env.filters["github_url"] = _github_url
