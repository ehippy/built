import asyncio

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from built.agent.chat import is_chat_turn_running, lock_for, run_chat_turn_activity
from built.api.deps import SessionDep
from built.db.base import async_session_factory
from built.services import chat_service, project_service
from built.ui.templates import templates

router = APIRouter(prefix="/ui", tags=["ui"])


@router.get("/projects/{project_id}/chat")
async def chat_page(project_id: str, request: Request, session: SessionDep):
    project = await project_service.get_project(session, project_id)
    messages = await chat_service.list_recent_messages(
        session, project_id, cleared_before_seq=project.chat_cleared_before_seq or 0
    )
    return templates.TemplateResponse(
        request,
        "chat.html.j2",
        {"project": project, "messages": messages, "is_turn_running": is_chat_turn_running(project_id)},
    )


@router.get("/projects/{project_id}/chat/fragment")
async def chat_fragment(project_id: str, request: Request, session: SessionDep):
    project = await project_service.get_project(session, project_id)
    messages = await chat_service.list_recent_messages(
        session, project_id, cleared_before_seq=project.chat_cleared_before_seq or 0
    )
    return templates.TemplateResponse(
        request,
        "_chat_fragment.html.j2",
        {"project": project, "messages": messages, "is_turn_running": is_chat_turn_running(project_id)},
    )


@router.post("/projects/{project_id}/chat/messages")
async def post_chat_message(project_id: str, content: str = Form(...)) -> RedirectResponse:
    """Appends the human's message and returns immediately (303) — the reply is an
    LLM agentic turn and can take a while, mirroring board.py's curate() handler.
    Opens its own session rather than using SessionDep: SessionDep's session only
    commits after this handler returns (as part of its dependency teardown), which
    would race the asyncio.create_task below reading the message back with a fresh
    session of its own before it's actually durable. Held under lock_for so two
    posts landing at the same instant can't compute the same next `seq`."""
    text = content.strip()
    if text:
        async with lock_for(project_id):
            async with async_session_factory() as session:
                await project_service.get_project(session, project_id)  # 404s early if missing
                await chat_service.append_user_message(session, project_id, content=text)
                await session.commit()
        asyncio.create_task(run_chat_turn_activity(project_id))
    return RedirectResponse(f"/ui/projects/{project_id}/chat", status_code=303)


@router.post("/projects/{project_id}/chat/clear")
async def clear_chat(project_id: str) -> RedirectResponse:
    async with lock_for(project_id):  # don't race an in-flight turn still appending rows
        async with async_session_factory() as session:
            await chat_service.clear_chat(session, project_id)
            await session.commit()
    return RedirectResponse(f"/ui/projects/{project_id}/chat", status_code=303)
