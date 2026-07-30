"""chat_service.to_openai_messages: reconstructing persisted ChatMessage rows back
into the role/content/tool_calls/tool_call_id shape llm/client.py's complete()
expects. Pure logic, no DB — rows are constructed directly, same technique
tests/unit/test_agent_context.py uses for Project/Card."""

import json

from built.db.models import ChatMessage
from built.domain.enums import ChatRole
from built.services.chat_service import to_openai_messages


def _row(**overrides) -> ChatMessage:
    defaults = {"project_id": "p", "seq": 1, "role": ChatRole.USER, "content": "hi"}
    defaults.update(overrides)
    return ChatMessage(**defaults)


def test_user_row_maps_to_plain_role_and_content():
    [message] = to_openai_messages([_row(role=ChatRole.USER, content="what should we build next?")])
    assert message == {"role": "user", "content": "what should we build next?"}


def test_assistant_row_with_no_tool_calls_has_no_tool_calls_key():
    [message] = to_openai_messages([_row(role=ChatRole.ASSISTANT, content="Sure, tell me more.")])
    assert message == {"role": "assistant", "content": "Sure, tell me more."}


def test_assistant_row_with_tool_calls_json_encodes_arguments():
    call_args = {"title": "t", "raw_request": "r"}
    row = _row(
        role=ChatRole.ASSISTANT,
        content=None,
        tool_calls=[{"id": "call_1", "name": "create_ticket", "arguments": call_args}],
    )
    [message] = to_openai_messages([row])
    assert message["role"] == "assistant"
    assert message["content"] is None
    assert message["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "create_ticket", "arguments": json.dumps(call_args)},
        }
    ]


def test_tool_row_maps_to_tool_role_with_call_id():
    row = _row(
        role=ChatRole.TOOL, content="Created card abc123.", tool_call_id="call_1", tool_name="create_ticket"
    )
    [message] = to_openai_messages([row])
    assert message == {"role": "tool", "tool_call_id": "call_1", "content": "Created card abc123."}


def test_tool_row_with_no_content_becomes_empty_string():
    row = _row(role=ChatRole.TOOL, content=None, tool_call_id="call_1", tool_name="create_ticket")
    [message] = to_openai_messages([row])
    assert message["content"] == ""


def test_preserves_row_order():
    rows = [
        _row(seq=1, role=ChatRole.USER, content="first"),
        _row(seq=2, role=ChatRole.ASSISTANT, content="second"),
    ]
    messages = to_openai_messages(rows)
    assert [m["content"] for m in messages] == ["first", "second"]
