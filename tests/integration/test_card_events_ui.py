"""Confirmed in production: a card's transcript looked frozen ~18 minutes stale no
matter how many more events happened, on every 2-second poll of
/ui/cards/{id}/events/fragment. Root cause: list_events(since_seq=0, limit=200)
always returns the *oldest* 200 events for a card, not the newest — fine for an API
consumer paging forward with an advancing cursor, wrong for a live dashboard that
always wants "what's happening now". list_recent_events (and the UI routes that now
call it instead) is the fix; this file covers both the service function directly and
the actual UI routes end to end."""

from httpx import ASGITransport, AsyncClient

from built.domain.enums import EventType
from built.domain.events import append_event
from built.main import app
from built.services import card_service, project_service


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _make_project(session, **overrides):
    defaults = {
        "name": f"events-ui-{overrides.pop('_n', 'x')}",
        "overarching_goal": "goal",
        "repo_remote_url": "https://example.invalid/repo.git",
    }
    defaults.update(overrides)
    return await project_service.create_project(session, **defaults)


async def _add_events(session, card_id, count, *, prefix):
    for i in range(count):
        await append_event(
            session, card_id=card_id, type=EventType.SYSTEM_NOTE, payload={"note": f"{prefix}-{i}"}
        )


async def test_list_recent_events_returns_the_newest_not_the_oldest(db_session):
    project = await _make_project(db_session, _n="1")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    # create_card itself appends one "created" event — pad well past a 10-event limit.
    await _add_events(db_session, card.id, 25, prefix="e")

    recent = await card_service.list_recent_events(db_session, card.id, limit=10)

    assert len(recent) == 10
    # Chronological order within the window, and the window is the tail end, not the head.
    assert [e.payload["note"] for e in recent] == [f"e-{i}" for i in range(15, 25)]
    assert [e.seq for e in recent] == sorted(e.seq for e in recent)


async def test_list_recent_events_returns_everything_when_under_the_limit(db_session):
    project = await _make_project(db_session, _n="2")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await _add_events(db_session, card.id, 3, prefix="e")

    recent = await card_service.list_recent_events(db_session, card.id, limit=200)

    # 1 "created" event (from create_card) + 3 added.
    assert len(recent) == 4
    assert recent[-1].payload["note"] == "e-2"


async def test_card_detail_page_shows_the_latest_event_not_a_stale_window(db_session):
    project = await _make_project(db_session, _n="3")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await _add_events(db_session, card.id, 250, prefix="marker")
    await db_session.commit()

    async with _client() as client:
        detail = await client.get(f"/ui/cards/{card.id}")
        fragment = await client.get(f"/ui/cards/{card.id}/events/fragment")

    # The most recent event must be visible...
    assert "marker-249" in detail.text
    assert "marker-249" in fragment.text
    # ...and the stale first-200 window (list_events' old behavior) must not be
    # what's shown — marker-0 sat behind the 200-event cap and should be gone.
    assert "marker-0" not in detail.text
    assert "marker-0" not in fragment.text


async def test_update_plan_renders_as_a_checklist_not_a_generic_tool_result(db_session):
    """The Developer's update_plan tool (see llm.tool_schemas.UPDATE_PLAN) is only
    useful as live context for a human watching a card if its actual steps show up
    in the transcript — the generic tool_call rendering only shows the terse
    confirmation string ("Plan recorded: N step(s)..."), not the plan itself."""
    project = await _make_project(db_session, _n="4")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await append_event(
        db_session,
        card_id=card.id,
        type=EventType.TOOL_CALL,
        payload={
            "name": "update_plan",
            "arguments": {
                "steps": [
                    {"step": "Add _config.yml", "status": "done"},
                    {
                        "step": "Convert games/index.html to use the games-index layout",
                        "status": "in_progress",
                    },
                    {"step": "Update CI to build via Jekyll", "status": "pending"},
                ]
            },
            "result": "Plan recorded: 3 step(s), 1 marked done.",
            "is_error": False,
            "commit_sha": None,
        },
    )
    await db_session.commit()

    async with _client() as client:
        fragment = await client.get(f"/ui/cards/{card.id}/events/fragment")

    assert "Add _config.yml" in fragment.text
    assert "Convert games/index.html to use the games-index layout" in fragment.text
    assert "Update CI to build via Jekyll" in fragment.text
    assert 'class="plan-step plan-status-done"' in fragment.text
    assert 'class="plan-step plan-status-in_progress"' in fragment.text
    assert 'class="plan-step plan-status-pending"' in fragment.text


async def test_bash_tool_call_shows_the_command_alongside_its_output(db_session):
    """Only the output was ever rendered for a bash tool call — the command that
    produced it (the actual interesting part of "what is it running") was never
    shown anywhere in the transcript."""
    project = await _make_project(db_session, _n="5")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await append_event(
        db_session,
        card_id=card.id,
        type=EventType.TOOL_CALL,
        payload={
            "name": "bash",
            "arguments": {"command": "bundle exec jekyll build --destination _site"},
            "result": "exit code: 0\nConfiguration file: /workspace/_config.yml",
            "is_error": False,
            "commit_sha": None,
        },
    )
    await db_session.commit()

    async with _client() as client:
        fragment = await client.get(f"/ui/cards/{card.id}/events/fragment")

    assert "bundle exec jekyll build --destination _site" in fragment.text
    assert 'class="bash-command"' in fragment.text
    # The output is still there too, just no longer the only thing shown.
    assert "Configuration file: /workspace/_config.yml" in fragment.text


async def test_read_file_tool_call_shows_which_file_it_read(db_session):
    """The generic tool_call rendering used to show only the bare tool name — no
    way to tell which file a read_file (or grep_files/glob_files/write_file/
    edit_file) call actually touched without expanding the raw Output details.
    The board's curation status panel already solved this (board.py's
    _describe_curation_event); the card transcript never got the same fix."""
    project = await _make_project(db_session, _n="9")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await append_event(
        db_session,
        card_id=card.id,
        type=EventType.TOOL_CALL,
        payload={
            "name": "read_file",
            "arguments": {"path": "games/snake.html"},
            "result": "     1\t---\n     2\tlayout: game\n",
            "is_error": False,
            "commit_sha": None,
        },
    )
    await db_session.commit()

    async with _client() as client:
        fragment = await client.get(f"/ui/cards/{card.id}/events/fragment")

    assert "read_file(games/snake.html)" in fragment.text


async def test_changes_requested_transition_shows_the_full_feedback_not_just_summary(db_session):
    """A changes_requested transition's payload previously carried only the
    one-line `summary` — the fuller `feedback` (what actually becomes
    card.latest_feedback for the Developer's next attempt) never showed up in the
    transcript itself."""
    project = await _make_project(db_session, _n="6")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await append_event(
        db_session,
        card_id=card.id,
        type=EventType.TRANSITION,
        payload={
            "column": "tester",
            "outcome": "changes_requested",
            "summary": "4 test failures, missing README",
            "feedback": (
                "1. tests/test_whats_new.js fails: expected 5 entries in _data/whats-new.yml, "
                "found 4 — the Simon Says entry from the spec is missing.\n"
                "2. README.md does not exist at the repo root; the spec requires one documenting "
                "the Jekyll build steps."
            ),
        },
    )
    await db_session.commit()

    async with _client() as client:
        fragment = await client.get(f"/ui/cards/{card.id}/events/fragment")

    assert "4 test failures, missing README" in fragment.text
    assert "expected 5 entries in _data/whats-new.yml, found 4" in fragment.text
    assert "README.md does not exist at the repo root" in fragment.text


async def test_compaction_event_shows_summary_as_an_expandable_entry(db_session):
    """Compaction used to be completely invisible — no CardEvent at all. Once
    logged, it needs to actually be legible: before/after counts up front, and
    the summary of what got dropped available to expand, not just a raw dict dump."""
    project = await _make_project(db_session, _n="7")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await append_event(
        db_session,
        card_id=card.id,
        type=EventType.COMPACTION,
        payload={
            "messages_before": 42,
            "messages_after": 7,
            "tokens_before": 9000,
            "tokens_after": 1500,
            "summary": "The agent explored games/ and found the Jekyll layouts already in place.",
        },
    )
    await db_session.commit()

    async with _client() as client:
        fragment = await client.get(f"/ui/cards/{card.id}/events/fragment")

    assert "Context compacted" in fragment.text
    assert "42" in fragment.text
    assert "7" in fragment.text
    assert "9000" in fragment.text
    assert "1500" in fragment.text
    assert "The agent explored games/ and found the Jekyll layouts already in place." in fragment.text


async def test_compaction_event_without_a_summary_says_so(db_session):
    """A fallback path (summarizer failed, or too little to summarize) still
    drops messages but has no summary text — must not look identical to a
    successful compaction with an empty expandable section."""
    project = await _make_project(db_session, _n="8")
    card = await card_service.create_card(db_session, project.id, title="t", raw_request="r")
    await append_event(
        db_session,
        card_id=card.id,
        type=EventType.COMPACTION,
        payload={
            "messages_before": 20,
            "messages_after": 6,
            "tokens_before": 5000,
            "tokens_after": 800,
            "summary": None,
        },
    )
    await db_session.commit()

    async with _client() as client:
        fragment = await client.get(f"/ui/cards/{card.id}/events/fragment")

    assert "Context compacted" in fragment.text
    assert "dropped without a summary" in fragment.text
