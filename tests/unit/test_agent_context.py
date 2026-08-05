"""retry_note handling in the prompt builders: a human instruction attached to a
retry should reach whichever column's prompt is built next, and stay out of the
prompt entirely when there isn't one."""

import pytest

from built.agent.context import (
    build_chat_prompt,
    build_curation_prompt,
    build_deployer_prompt,
    build_developer_prompt,
    build_pm_prompt,
    build_reviewer_prompt,
    build_tester_prompt,
)
from built.db.models import Card, Project
from built.domain.enums import ActivityKind, DeployMode


def _project(**overrides) -> Project:
    defaults = {
        "name": "p", "slug": "p", "overarching_goal": "goal", "repo_remote_url": "https://example.invalid/r.git"
    }
    defaults.update(overrides)
    return Project(**defaults)


def _card(**overrides) -> Card:
    defaults = {"project_id": "p", "title": "t", "raw_request": "r", "acceptance_criteria": []}
    defaults.update(overrides)
    return Card(**defaults)


def test_developer_prompt_tells_agent_how_to_bootstrap_a_missing_sandbox():
    """Nothing writes Dockerfile.built-sandbox into a project by default (see
    sandbox/container.py's _ensure_worktree_sandbox — it no-ops if the file isn't
    there) — the system prompt is the only thing that can reliably point the
    Developer at build_sandbox instead of it being a one-line tool description
    the model may or may not connect to a missing-package failure on its own."""
    system, _ = build_developer_prompt(_project(), _card())
    assert "Dockerfile.built-sandbox" in system
    assert "build_sandbox" in system


def test_tester_prompt_tells_agent_how_to_bootstrap_a_missing_sandbox():
    system, _ = build_tester_prompt(_project(), _card(), developer_summary="did stuff")
    assert "Dockerfile.built-sandbox" in system
    assert "build_sandbox" in system


def test_pm_prompt_includes_retry_note_when_present():
    _, user = build_pm_prompt(_project(), _card(), retry_note="focus on the auth flow only")
    assert "focus on the auth flow only" in user


def test_pm_prompt_omits_retry_note_section_when_absent():
    _, user = build_pm_prompt(_project(), _card())
    assert "A human left this instruction" not in user


def test_developer_prompt_includes_retry_note_when_present():
    _, user = build_developer_prompt(_project(), _card(), retry_note="rebase onto main first")
    assert "rebase onto main first" in user


def test_curation_prompt_retro_uses_postmortem_digest_not_repo_browse():
    """retro is shaped like agents_md: fed extra_context (here, a postmortem
    digest) instead of the generic 'explore the repo' instructions, and caps
    itself at one proposed card."""
    _, user = build_curation_prompt(
        _project(),
        ActivityKind.RETRO,
        existing_titles=[],
        extra_context="- [failed, 3 revision(s)] went well: (nothing notable) | struggles: Tester kept "
        "rejecting on the same flaky integration test",
    )
    assert "Tester kept rejecting on the same flaky integration test" in user


def test_curation_prompt_retro_reports_no_new_postmortems_when_extra_context_absent():
    _, user = build_curation_prompt(_project(), ActivityKind.RETRO, existing_titles=[])
    assert "(nothing new)" in user


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


def test_developer_prompt_leads_with_sync_conflict_before_feedback():
    """A merge conflict from the pre-visit branch sync (orchestrator/worker.py +
    sandbox/worktree.sync_card_branch_with_default) is a blocking prerequisite —
    it must come before even the revision feedback block, since nothing else this
    visit does will actually get committed until it's resolved."""
    card = _card(latest_feedback="tests/test_whats_new.js fails: expected 5 entries, found 4")
    _, user = build_developer_prompt(_project(), card, sync_conflict_paths=["app.py", "gameStore.js"])

    assert "app.py" in user
    assert "gameStore.js" in user
    conflict_pos = user.index("BEFORE ANYTHING ELSE")
    revision_pos = user.index("THIS IS A REVISION")
    assert conflict_pos < revision_pos


def test_developer_prompt_omits_sync_conflict_note_when_none():
    _, user = build_developer_prompt(_project(), _card())
    assert "BEFORE ANYTHING ELSE" not in user


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


# --- Per-role guidance (Project.pm_guidance/developer_guidance/etc, set via project
# settings — services/project_service.py) — distinct from agents_doc: a human
# instruction targeted at one specific role, not a repo-documented convention. ---


def test_pm_prompt_includes_pm_guidance_when_set():
    system, _ = build_pm_prompt(_project(pm_guidance="Always prefer feature flags over branches."), _card())
    assert "Always prefer feature flags over branches." in system


def test_pm_prompt_omits_guidance_section_when_unset():
    system, _ = build_pm_prompt(_project(), _card())
    assert "Additional project-specific instructions for this role" not in system


def test_developer_prompt_includes_developer_guidance_when_set():
    system, _ = build_developer_prompt(
        _project(developer_guidance="Never touch the legacy billing module."), _card()
    )
    assert "Never touch the legacy billing module." in system


def test_developer_prompt_ignores_other_roles_guidance():
    project = _project(pm_guidance="PM-only note", tester_guidance="tester-only")
    system, _ = build_developer_prompt(project, _card())
    assert "PM-only note" not in system
    assert "tester-only" not in system


def test_tester_prompt_includes_tester_guidance_when_set():
    system, _ = build_tester_prompt(
        _project(tester_guidance="Always test against Safari, not just Chrome."),
        _card(),
        developer_summary=None,
    )
    assert "Always test against Safari, not just Chrome." in system


def test_reviewer_prompt_includes_reviewer_guidance_when_set():
    system, _ = build_reviewer_prompt(
        _project(reviewer_guidance="Reject anything that adds a new npm dependency."),
        _card(),
        tester_summary=None,
    )
    assert "Reject anything that adds a new npm dependency." in system


def test_deployer_prompt_includes_deployer_guidance_when_set():
    system, _ = build_deployer_prompt(
        _project(deployer_guidance="Deploys should never run on Fridays."),
        _card(branch_name="card/x"),
        mode=DeployMode.AUTO_MAIN,
    )
    assert "Deploys should never run on Fridays." in system


def test_chat_prompt_includes_pm_guidance_when_set():
    system = build_chat_prompt(_project(pm_guidance="Keep tickets under 500 words."), [])
    assert "Keep tickets under 500 words." in system


def test_curation_prompt_includes_pm_guidance_for_overseer_kind():
    system, _ = build_curation_prompt(
        _project(
            pm_guidance="Skip anything touching the payments module.",
            overseer_prompt="Investigate app.py for arithmetic bugs.",
        ),
        ActivityKind.OVERSEER,
        [],
    )
    assert "Skip anything touching the payments module." in system


def test_curation_prompt_overseer_uses_the_operators_exact_mandate_text():
    """Unlike every other curation kind, built supplies no built-in definition of
    what OVERSEER looks for — the operator's own words must appear verbatim, not
    paraphrased or summarized."""
    system, _ = build_curation_prompt(
        _project(overseer_prompt="Audit the payment webhook handler for idempotency bugs."),
        ActivityKind.OVERSEER,
        [],
    )
    assert "Audit the payment webhook handler for idempotency bugs." in system


def test_curation_prompt_overseer_asserts_when_prompt_is_blank():
    """Callers (orchestrator/curator.py) must never reach this branch with a blank
    overseer_prompt — this assert is the belt-and-suspenders guard against that
    invariant drifting."""
    with pytest.raises(AssertionError, match="overseer_prompt"):
        build_curation_prompt(_project(overseer_prompt=None), ActivityKind.OVERSEER, [])


def test_curation_prompt_includes_pm_guidance_for_agents_md_kind():
    system, _ = build_curation_prompt(
        _project(pm_guidance="Skip anything touching the payments module."),
        ActivityKind.AGENTS_MD,
        [],
    )
    assert "Skip anything touching the payments module." in system
