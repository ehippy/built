"""retry_note handling in the prompt builders: a human instruction attached to a
retry should reach whichever column's prompt is built next, and stay out of the
prompt entirely when there isn't one."""

from built.agent.context import (
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
