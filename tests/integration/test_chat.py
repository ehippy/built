"""Project chat (agent/chat.py) against a real toy git repo — LLM faked, everything
else real. A chat turn explores read-only and can create_ticket/update_ticket, but
unlike Curator's run-to-completion pass, a turn ends as soon as the model replies
with no further tool calls (there's no terminal tool) and is meant to be called
again for the next human message, not looped internally."""

from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from built.agent.chat import run_chat_turn
from built.db.models import ChatMessage
from built.domain.enums import ChatRole
from built.llm.client import LLMResult, ToolCallRequest
from built.llm.tool_schemas import MAX_TICKETS_PER_CHAT_TURN
from built.main import app
from built.sandbox import worktree
from built.sandbox.container import CommandResult
from built.services import card_service, chat_service, project_service
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher
from tests.unit.fakes import FakeCommandExecutor, ScriptedLLMClient


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _make_project(db_session, toy_repo_remote, **overrides):
    defaults = {
        "name": f"chat-{overrides.pop('_n', 'x')}",
        "overarching_goal": "Add basic arithmetic helpers to app.py.",
        "repo_remote_url": str(toy_repo_remote),
    }
    defaults.update(overrides)
    project = await project_service.create_project(db_session, **defaults)
    wt_path = await worktree.ensure_tool_worktree(project, tool="chat")
    return project, wt_path


def _dispatcher(wt_path) -> ToolDispatcher:
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    return ToolDispatcher(ctx=ToolContext(card_id="chat-x", worktree_root=wt_path), executor=executor)


async def test_chat_explores_then_creates_ticket_then_replies(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="1")
    await chat_service.append_user_message(db_session, project.id, content="what's missing from app.py?")
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="read_file", arguments={"path": "app.py"})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_2",
                        name="create_ticket",
                        arguments={"title": "Add subtract()", "raw_request": "app.py has no subtract()."},
                    )
                ],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content="Filed it — want me to look for anything else?",
                tool_calls=[],
                endpoint_used="fake::model",
            ),
        ]
    )

    appended = await run_chat_turn(
        db_session, project, llm_client=llm, dispatcher=_dispatcher(wt_path), max_tool_iterations=10
    )

    cards = await card_service.list_cards(db_session, project.id)
    assert len(cards) == 1
    assert cards[0].title == "Add subtract()"
    assert cards[0].column == "pm"

    roles = [m.role for m in appended]
    assert roles == [ChatRole.ASSISTANT, ChatRole.TOOL, ChatRole.ASSISTANT, ChatRole.TOOL, ChatRole.ASSISTANT]
    tool_result = appended[3]
    assert tool_result.card_id == cards[0].id
    assert not tool_result.is_error
    assert appended[-1].content == "Filed it — want me to look for anything else?"


async def test_chat_update_ticket_edits_existing_card(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="2")
    card = await card_service.create_card(
        db_session, project.id, title="Add subtract()", raw_request="original"
    )
    await chat_service.append_user_message(db_session, project.id, content="actually call it minus()")
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="update_ticket",
                        arguments={"card_id": card.id, "title": "Add minus()", "raw_request": "renamed"},
                    )
                ],
                endpoint_used="fake::model",
            ),
            LLMResult(content="Updated.", tool_calls=[], endpoint_used="fake::model"),
        ]
    )

    await run_chat_turn(
        db_session, project, llm_client=llm, dispatcher=_dispatcher(wt_path), max_tool_iterations=10
    )

    updated = await card_service.get_card(db_session, card.id)
    assert updated.title == "Add minus()"
    assert updated.raw_request == "renamed"


async def test_chat_update_ticket_missing_card_yields_error_tool_result_not_exception(
    db_session, toy_repo_remote
):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="3")
    await chat_service.append_user_message(db_session, project.id, content="rename the nonexistent card")
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="update_ticket",
                        arguments={"card_id": "no-such-card", "title": "x", "raw_request": "y"},
                    )
                ],
                endpoint_used="fake::model",
            ),
            LLMResult(content="Couldn't find that card.", tool_calls=[], endpoint_used="fake::model"),
        ]
    )

    appended = await run_chat_turn(
        db_session, project, llm_client=llm, dispatcher=_dispatcher(wt_path), max_tool_iterations=10
    )

    tool_result = next(m for m in appended if m.role == ChatRole.TOOL)
    assert tool_result.is_error
    assert "no-such-card" in tool_result.content


async def test_chat_search_tickets_finds_matching_card_by_keyword(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="3a")
    match = await card_service.create_card(
        db_session, project.id, title="Fix login bug", raw_request="Users can't log in on mobile."
    )
    await card_service.create_card(db_session, project.id, title="Add dark mode", raw_request="unrelated")
    await chat_service.append_user_message(db_session, project.id, content="what's up with that login bug?")
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_1", name="search_tickets", arguments={"query": "login"})
                ],
                endpoint_used="fake::model",
            ),
            LLMResult(content="Found it — still open.", tool_calls=[], endpoint_used="fake::model"),
        ]
    )

    appended = await run_chat_turn(
        db_session, project, llm_client=llm, dispatcher=_dispatcher(wt_path), max_tool_iterations=10
    )

    tool_result = next(m for m in appended if m.role == ChatRole.TOOL)
    assert not tool_result.is_error
    assert match.id in tool_result.content
    assert "Add dark mode" not in tool_result.content


async def test_chat_search_tickets_lists_recent_when_query_empty(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="3b")
    card = await card_service.create_card(db_session, project.id, title="Add dark mode", raw_request="r")
    await chat_service.append_user_message(db_session, project.id, content="what's already on the board?")
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="search_tickets", arguments={})],
                endpoint_used="fake::model",
            ),
            LLMResult(content="Just one card so far.", tool_calls=[], endpoint_used="fake::model"),
        ]
    )

    appended = await run_chat_turn(
        db_session, project, llm_client=llm, dispatcher=_dispatcher(wt_path), max_tool_iterations=10
    )

    tool_result = next(m for m in appended if m.role == ChatRole.TOOL)
    assert not tool_result.is_error
    assert card.id in tool_result.content


async def test_chat_get_ticket_returns_full_detail(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="3c")
    card = await card_service.create_card(
        db_session, project.id, title="Fix login bug", raw_request="Users can't log in on mobile."
    )
    await chat_service.append_user_message(db_session, project.id, content="what does that card say exactly?")
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="get_ticket", arguments={"card_id": card.id})],
                endpoint_used="fake::model",
            ),
            LLMResult(content="Here's what it says.", tool_calls=[], endpoint_used="fake::model"),
        ]
    )

    appended = await run_chat_turn(
        db_session, project, llm_client=llm, dispatcher=_dispatcher(wt_path), max_tool_iterations=10
    )

    tool_result = next(m for m in appended if m.role == ChatRole.TOOL)
    assert not tool_result.is_error
    assert "Users can't log in on mobile." in tool_result.content
    assert "pm" in tool_result.content


async def test_chat_get_ticket_missing_card_yields_error_tool_result(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="3d")
    await chat_service.append_user_message(db_session, project.id, content="show me that other card")
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_1", name="get_ticket", arguments={"card_id": "no-such-card"})
                ],
                endpoint_used="fake::model",
            ),
            LLMResult(content="Couldn't find that one.", tool_calls=[], endpoint_used="fake::model"),
        ]
    )

    appended = await run_chat_turn(
        db_session, project, llm_client=llm, dispatcher=_dispatcher(wt_path), max_tool_iterations=10
    )

    tool_result = next(m for m in appended if m.role == ChatRole.TOOL)
    assert tool_result.is_error
    assert "no-such-card" in tool_result.content


async def test_chat_no_op_guard_when_history_already_ends_on_assistant(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="4")
    await chat_service.append_user_message(db_session, project.id, content="hello")
    await chat_service.append_assistant_message(db_session, project.id, content="hi there")
    await db_session.commit()

    llm = ScriptedLLMClient([])  # would raise if asked for a completion

    appended = await run_chat_turn(db_session, project, llm_client=llm, dispatcher=_dispatcher(wt_path))

    assert appended == []


async def test_chat_max_tool_iterations_cap(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="5")
    await chat_service.append_user_message(db_session, project.id, content="keep exploring forever")
    await db_session.commit()

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id=f"call_{i}", name="list_files", arguments={"path": "."})],
                endpoint_used="fake::model",
            )
            for i in range(1, 4)
        ]
    )

    appended = await run_chat_turn(
        db_session, project, llm_client=llm, dispatcher=_dispatcher(wt_path), max_tool_iterations=3
    )

    assert "limit" in appended[-1].content


async def test_chat_caps_tickets_created_per_turn(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="6")
    await chat_service.append_user_message(db_session, project.id, content="file a bunch of tickets")
    await db_session.commit()

    create_calls = [
        LLMResult(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id=f"call_{i}",
                    name="create_ticket",
                    arguments={"title": f"Task {i}", "raw_request": f"Do thing {i}."},
                )
            ],
            endpoint_used="fake::model",
        )
        for i in range(MAX_TICKETS_PER_CHAT_TURN + 2)
    ]
    done_reply = LLMResult(content="done", tool_calls=[], endpoint_used="fake::model")
    llm = ScriptedLLMClient([*create_calls, done_reply])

    await run_chat_turn(
        db_session,
        project,
        llm_client=llm,
        dispatcher=_dispatcher(wt_path),
        max_tool_iterations=MAX_TICKETS_PER_CHAT_TURN + 5,
    )

    cards = await card_service.list_cards(db_session, project.id)
    assert len(cards) == MAX_TICKETS_PER_CHAT_TURN


async def test_clear_chat_hides_old_messages_but_keeps_the_rows(db_session, toy_repo_remote):
    project, wt_path = await _make_project(db_session, toy_repo_remote, _n="7")
    await chat_service.append_user_message(db_session, project.id, content="first message")
    await chat_service.append_assistant_message(db_session, project.id, content="first reply")
    await db_session.commit()

    project = await chat_service.clear_chat(db_session, project.id)
    await db_session.commit()

    visible = await chat_service.list_recent_messages(
        db_session, project.id, cleared_before_seq=project.chat_cleared_before_seq or 0
    )
    assert visible == []

    stmt = select(ChatMessage).where(ChatMessage.project_id == project.id)
    all_rows = (await db_session.scalars(stmt)).all()
    assert len(all_rows) == 2  # nothing deleted

    await chat_service.append_user_message(db_session, project.id, content="fresh start")
    await db_session.commit()
    fresh_reply = LLMResult(content="Hi, starting fresh.", tool_calls=[], endpoint_used="fake::model")
    llm = ScriptedLLMClient([fresh_reply])

    appended = await run_chat_turn(db_session, project, llm_client=llm, dispatcher=_dispatcher(wt_path))

    assert len(appended) == 1
    # The model only ever saw the post-clear history, not the earlier turn.
    assert "first message" not in llm.calls[0]["messages"][-1]["content"]


async def test_post_chat_message_persists_immediately_and_redirects(db_session):
    # An invalid repo URL, not toy_repo_remote: the human's message must be visible
    # via the fragment endpoint regardless of what the background LLM turn does, and
    # this keeps that background task's own worktree setup (and thus its DB writes)
    # failing fast rather than racing the next test's table wipe — same pattern as
    # test_api.py's test_curate_endpoint_requires_auth_and_starts_in_background.
    project = await project_service.create_project(
        db_session,
        name="chat-ui",
        overarching_goal="goal",
        repo_remote_url="https://example.invalid/chat-project.git",
    )
    await db_session.commit()

    async with _client() as client:
        response = await client.post(
            f"/ui/projects/{project.id}/chat/messages",
            data={"content": "hello there"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == f"/ui/projects/{project.id}/chat"

        fragment = await client.get(f"/ui/projects/{project.id}/chat/fragment")
        assert "hello there" in fragment.text
