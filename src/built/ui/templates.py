"""Jinja2 templates for the dashboard. UI routers call the same services/ layer as
the REST API directly — no self-HTTP-calling."""

from pathlib import Path

import mistune
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

TEMPLATES_DIR = Path(__file__).parent / "templates"

# escape=True (the default) escapes any raw HTML embedded in the markdown source
# rather than passing it through, so agent-generated spec text can't inject markup.
_markdown = mistune.create_markdown()

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


templates.env.filters["timeago"] = _timeago
templates.env.filters["markdown"] = _render_markdown
