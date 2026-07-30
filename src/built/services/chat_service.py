import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from built.db.models import ChatMessage, Project
from built.domain.enums import ChatRole
from built.domain.events import append_chat_message
from built.services.project_service import get_project


async def append_user_message(session: AsyncSession, project_id: str, *, content: str) -> ChatMessage:
    return await append_chat_message(session, project_id=project_id, role=ChatRole.USER, content=content)


async def append_assistant_message(
    session: AsyncSession,
    project_id: str,
    *,
    content: str | None,
    tool_calls: list[dict] | None = None,
    is_error: bool = False,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    latency_ms: int | None = None,
) -> ChatMessage:
    return await append_chat_message(
        session,
        project_id=project_id,
        role=ChatRole.ASSISTANT,
        content=content,
        tool_calls=tool_calls,
        is_error=is_error,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
    )


async def append_tool_message(
    session: AsyncSession,
    project_id: str,
    *,
    tool_call_id: str,
    tool_name: str,
    content: str,
    is_error: bool = False,
    card_id: str | None = None,
) -> ChatMessage:
    return await append_chat_message(
        session,
        project_id=project_id,
        role=ChatRole.TOOL,
        content=content,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        is_error=is_error,
        card_id=card_id,
    )


async def list_recent_messages(
    session: AsyncSession, project_id: str, *, cleared_before_seq: int = 0, limit: int = 200
) -> list[ChatMessage]:
    """The most recent `limit` messages since the last "clear chat", oldest first —
    a chat reads top-to-bottom like a messaging app, unlike _events_fragment.html.j2's
    newest-on-top transcript convention, so callers should NOT `|reverse` this again."""
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.project_id == project_id, ChatMessage.seq > cleared_before_seq)
        .order_by(ChatMessage.seq.desc())
        .limit(limit)
    )
    messages = list((await session.scalars(stmt)).all())
    messages.reverse()
    return messages


def to_openai_messages(rows: list[ChatMessage]) -> list[dict]:
    """Reconstructs persisted rows into the exact role/content/tool_calls/tool_call_id
    shape llm/client.py's complete() expects — the entire reason ChatMessage has
    structured columns instead of a freeform payload dict like CardEvent/CurationEvent."""
    messages: list[dict] = []
    for row in rows:
        if row.role == ChatRole.TOOL:
            messages.append({"role": "tool", "tool_call_id": row.tool_call_id, "content": row.content or ""})
            continue
        message: dict = {"role": row.role.value, "content": row.content}
        if row.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
                }
                for call in row.tool_calls
            ]
        messages.append(message)
    return messages


async def clear_chat(session: AsyncSession, project_id: str) -> Project:
    """Marks every message up to the current tail as cleared — a fresh conversation
    starts from here, but nothing is deleted (consistent with this app's
    archive-not-delete convention, e.g. card_service.archive_card)."""
    project = await get_project(session, project_id)
    current_max = await session.scalar(
        select(ChatMessage.seq)
        .where(ChatMessage.project_id == project_id)
        .order_by(ChatMessage.seq.desc())
        .limit(1)
    )
    project.chat_cleared_before_seq = current_max or 0
    await session.flush()
    return project
