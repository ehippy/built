"""PM's autonomous discovery mode against a real toy git repo — LLM faked, everything
else real. Not tied to any card: run_pm_discovery explores the repo on its own
initiative and creates new cards via propose_tasks."""

from built.agent.discovery import run_pm_discovery
from built.domain.enums import Column, LifecycleState
from built.llm.client import LLMResult, ToolCallRequest
from built.llm.tool_schemas import MAX_PROPOSED_TASKS
from built.sandbox import worktree
from built.sandbox.container import CommandResult
from built.services import card_service, project_service
from built.tools.base import ToolContext
from built.tools.dispatcher import ToolDispatcher
from tests.unit.fakes import FakeCommandExecutor, ScriptedLLMClient


async def _make_discovery_project(db_session, toy_repo_remote):
    project = await project_service.create_project(
        db_session,
        name="discovery",
        overarching_goal="Add basic arithmetic helpers to app.py.",
        repo_remote_url=str(toy_repo_remote),
    )
    wt_path = await worktree.ensure_default_branch_worktree(project)
    return project, wt_path


def _dispatcher(wt_path) -> ToolDispatcher:
    executor = FakeCommandExecutor(CommandResult(exit_code=0, stdout="", stderr=""))
    return ToolDispatcher(ctx=ToolContext(card_id="discovery-x", worktree_root=wt_path), executor=executor)


async def test_discovery_explores_then_proposes_tasks(db_session, toy_repo_remote):
    project, wt_path = await _make_discovery_project(db_session, toy_repo_remote)

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="list_files", arguments={"path": "."})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_2", name="read_file", arguments={"path": "app.py"})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_3",
                        name="propose_tasks",
                        arguments={
                            "tasks": [
                                {"title": "Add multiply()", "raw_request": "Add a multiply(a, b) helper."},
                                {"title": "Add divide()", "raw_request": "Add a divide(a, b) helper."},
                            ]
                        },
                    )
                ],
                endpoint_used="fake::model",
            ),
        ]
    )

    created = await run_pm_discovery(
        db_session, project, llm_client=llm, dispatcher=_dispatcher(wt_path), max_iterations=10
    )

    assert [c.title for c in created] == ["Add multiply()", "Add divide()"]
    for card in created:
        assert card.column == Column.PM
        assert card.lifecycle_state == LifecycleState.ACTIVE

    all_cards = await card_service.list_cards(db_session, project.id)
    assert len(all_cards) == 2


async def test_discovery_nudges_on_empty_tasks_then_recovers(db_session, toy_repo_remote):
    project, wt_path = await _make_discovery_project(db_session, toy_repo_remote)

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="propose_tasks", arguments={"tasks": []})],
                endpoint_used="fake::model",
            ),
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_2",
                        name="propose_tasks",
                        arguments={
                            "tasks": [{"title": "Add mod()", "raw_request": "Add a mod(a, b) helper."}]
                        },
                    )
                ],
                endpoint_used="fake::model",
            ),
        ]
    )

    created = await run_pm_discovery(
        db_session, project, llm_client=llm, dispatcher=_dispatcher(wt_path), max_iterations=10
    )

    assert [c.title for c in created] == ["Add mod()"]


async def test_discovery_caps_at_max_proposed_tasks(db_session, toy_repo_remote):
    project, wt_path = await _make_discovery_project(db_session, toy_repo_remote)

    too_many = [
        {"title": f"Task {i}", "raw_request": f"Do thing {i}."} for i in range(MAX_PROPOSED_TASKS + 5)
    ]
    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_1", name="propose_tasks", arguments={"tasks": too_many})
                ],
                endpoint_used="fake::model",
            )
        ]
    )

    created = await run_pm_discovery(
        db_session, project, llm_client=llm, dispatcher=_dispatcher(wt_path), max_iterations=10
    )

    assert len(created) == MAX_PROPOSED_TASKS


async def test_discovery_returns_empty_when_iterations_exhausted(db_session, toy_repo_remote):
    project, wt_path = await _make_discovery_project(db_session, toy_repo_remote)

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[ToolCallRequest(id=f"call_{i}", name="list_files", arguments={"path": "."})],
                endpoint_used="fake::model",
            )
            for i in range(1, 5)
        ]
    )

    created = await run_pm_discovery(
        db_session, project, llm_client=llm, dispatcher=_dispatcher(wt_path), max_iterations=3
    )

    assert created == []
    all_cards = await card_service.list_cards(db_session, project.id)
    assert all_cards == []


async def test_discovery_prompt_lists_existing_card_titles_to_avoid_duplicates(db_session, toy_repo_remote):
    project, wt_path = await _make_discovery_project(db_session, toy_repo_remote)
    await card_service.create_card(db_session, project.id, title="Add subtract()", raw_request="r")

    llm = ScriptedLLMClient(
        [
            LLMResult(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="propose_tasks",
                        arguments={"tasks": [{"title": "Add divide()", "raw_request": "r"}]},
                    )
                ],
                endpoint_used="fake::model",
            )
        ]
    )

    await run_pm_discovery(
        db_session, project, llm_client=llm, dispatcher=_dispatcher(wt_path), max_iterations=10
    )

    sent_user_message = llm.calls[0]["messages"][1]["content"]
    assert "Add subtract()" in sent_user_message
