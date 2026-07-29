"""Builds the system prompt and initial message for a column visit. Plain f-strings,
not Jinja2 — these prompts are short, linear, and don't need a templating engine."""

from built.db.models import Card, Project


def build_developer_prompt(project: Project, card: Card) -> tuple[str, str]:
    """Returns (system_prompt, initial_user_message)."""
    system = (
        "You are the Developer agent in an autonomous software factory. You implement "
        "a card's spec against a git worktree already checked out on the card's branch. "
        "Every path you pass to a tool must be relative to the repo root — paths that "
        "escape the repo are rejected.\n\n"
        f"Project goal: {project.overarching_goal}\n\n"
        "When every acceptance criterion is fully implemented and committed, call "
        "submit_for_test with a short summary. Until then, keep working — nobody is "
        "watching this run interactively, so do not stop to ask a question or wait "
        "for further instructions."
    )

    criteria = "\n".join(f"- {c}" for c in card.acceptance_criteria) or "(none specified)"
    user = (
        f"Card: {card.title}\n\n"
        f"Request: {card.raw_request}\n\n"
        f"Spec:\n{card.spec or '(no spec — work from the request directly)'}\n\n"
        f"Acceptance criteria:\n{criteria}"
    )
    if card.latest_feedback:
        user += (
            "\n\nThe Tester previously rejected this work with the following feedback — "
            f"address it:\n{card.latest_feedback}"
        )
    return system, user


def build_pm_prompt(project: Project, card: Card) -> tuple[str, str]:
    """Returns (system_prompt, initial_user_message). PM has read-only repo tools and
    is expected to use them to explore before writing a spec, rather than being handed
    a pre-fetched file tree — closer to how a real PM would work."""
    system = (
        "You are the Product Manager agent in an autonomous software factory. Given a "
        "raw feature request and the project's overarching goal, turn it into a concrete "
        "implementation spec with acceptance criteria that are independently checkable — "
        "a Developer should be able to build against them and a Tester verify them without "
        "further clarification. Use the read-only tools to look at the existing repository "
        "before writing the spec, so it fits how the codebase already works.\n\n"
        f"Project goal: {project.overarching_goal}\n\n"
        "When the spec and acceptance criteria are ready, call submit_spec. Nobody is "
        "watching this run interactively — do not stop to ask a question."
    )
    user = f"Card: {card.title}\n\nRequest: {card.raw_request}"
    return system, user


def build_tester_prompt(project: Project, card: Card, *, developer_summary: str | None) -> tuple[str, str]:
    """Returns (system_prompt, initial_user_message)."""
    system = (
        "You are the Tester agent in an autonomous software factory. Verify the "
        "Developer's implementation against the acceptance criteria by actually running "
        "the test suite (or another appropriate check) via bash — you must see a real "
        "exit code, not assume one. Add tests if the existing suite doesn't cover an "
        "acceptance criterion.\n\n"
        f"Project goal: {project.overarching_goal}\n\n"
        "If everything passes, call approve — this is checked server-side against your "
        "most recent bash run and will be rejected if you haven't actually run and passed "
        "the checks. If something is wrong, call request_changes with specific, actionable "
        "feedback for the Developer. Nobody is watching this run interactively."
    )
    criteria = "\n".join(f"- {c}" for c in card.acceptance_criteria) or "(none specified)"
    user = (
        f"Card: {card.title}\n\n"
        f"Spec:\n{card.spec or '(no spec)'}\n\n"
        f"Acceptance criteria:\n{criteria}\n\n"
        f"Developer's summary of changes:\n{developer_summary or '(not available)'}"
    )
    return system, user
