"""retry_note handling in the prompt builders: a human instruction attached to a
retry should reach whichever column's prompt is built next, and stay out of the
prompt entirely when there isn't one."""

from built.agent.context import (
    build_chat_prompt,
    build_deployer_prompt,
    build_developer_prompt,
    build_pm_prompt,
    build_tester_prompt,
)
from built.db.models import Card, Project
from built.domain.enums import DeployMode


def _project() -> Project:
    return Project(
        name="p", slug="p", overarching_goal="goal", repo_remote_url="https://example.invalid/r.git"
    )


def _card(**overrides) -> Card:
    defaults = {"project_id": "p", "title": "t", "raw_request": "r", "acceptance_criteria": []}
    defaults.update(overrides)
    return Card(**defaults)


def test_pm_prompt_includes_retry_note_when_present():
    _, user = build_pm_prompt(_project(), _card(), retry_note="focus on the auth flow only")
    assert "focus on the auth flow only" in user


def test_pm_prompt_omits_retry_note_section_when_absent():
    _, user = build_pm_prompt(_project(), _card())
    assert "A human left this instruction" not in user


def test_developer_prompt_includes_retry_note_when_present():
    _, user = build_developer_prompt(_project(), _card(), retry_note="rebase onto main first")
    assert "rebase onto main first" in user


def test_developer_prompt_leads_with_rejection_feedback_not_appends_it():
    """card.latest_feedback used to be appended at the very end of the user
    message, after Card/Request/Spec/Acceptance criteria — same tier as
    retry_recap/retry_note. It needs to lead instead: the most specific,
    up-to-date signal of what's actually wrong shouldn't be the thing most
    likely to get deprioritized against everything else in the prompt."""
    card = _card(latest_feedback="tests/test_whats_new.js fails: expected 5 entries, found 4")
    _, user = build_developer_prompt(_project(), card)

    assert "THIS IS A REVISION" in user
    feedback_pos = user.index("expected 5 entries, found 4")
    card_pos = user.index("Card: t")
    assert feedback_pos < card_pos


def test_developer_prompt_omits_revision_framing_when_no_feedback():
    _, user = build_developer_prompt(_project(), _card())
    assert "THIS IS A REVISION" not in user


def test_developer_prompt_tells_the_plan_to_cover_every_feedback_item():
    system, _ = build_developer_prompt(_project(), _card(latest_feedback="x"))
    assert "step for every item in that feedback" in system


def test_tester_prompt_includes_retry_note_when_present():
    _, user = build_tester_prompt(
        _project(), _card(), developer_summary="did stuff", retry_note="re-run flaky test"
    )
    assert "re-run flaky test" in user


def test_deployer_prompt_includes_retry_note_when_present():
    _, user = build_deployer_prompt(
        _project(),
        _card(branch_name="card/x"),
        mode=DeployMode.AUTO_MAIN,
        retry_note="rebase onto main and resolve the conflict in index.html",
    )
    assert "rebase onto main and resolve the conflict in index.html" in user


def test_chat_prompt_includes_project_goal():
    system = build_chat_prompt(_project(), [])
    assert "goal" in system


def test_chat_prompt_lists_existing_titles_to_avoid_duplicates():
    system = build_chat_prompt(_project(), ["Add a health check endpoint"])
    assert "Add a health check endpoint" in system


def test_chat_prompt_says_none_yet_when_no_existing_titles():
    system = build_chat_prompt(_project(), [])
    assert "(none yet)" in system


def test_chat_prompt_includes_agents_doc_when_present():
    system = build_chat_prompt(_project(), [], agents_doc="Always run the linter before committing.")
    assert "Always run the linter before committing." in system


def test_chat_prompt_includes_current_epic_when_set():
    epic = _card(title="Ship v2 onboarding", spec="Redesign the onboarding flow.")
    system = build_chat_prompt(_project(), [], current_epic=epic)
    assert "Ship v2 onboarding" in system
