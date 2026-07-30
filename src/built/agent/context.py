"""Builds the system prompt and initial message for a column visit. Plain f-strings,
not Jinja2 — these prompts are short, linear, and don't need a templating engine."""

from built.db.models import Card, Project
from built.domain.enums import ActivityKind, DeployMode
from built.llm.tool_schemas import MAX_PROPOSED_TASKS


def _with_agents_doc(system: str, agents_doc: str | None) -> str:
    """Appends the project's AGENTS.md (kept current by the agents_md curation kind
    — agent/curation.py) to a system prompt, if one exists — project-specific
    practices and conventions every column should know about before doing its own
    work."""
    if not agents_doc:
        return system
    return f"{system}\n\nProject practices (from this repo's AGENTS.md):\n{agents_doc}"


_CURATION_FOCUS: dict[ActivityKind, str] = {
    ActivityKind.BUG_SWEEP: (
        "working in bug-sweep mode: look specifically for defects — broken behavior, unhandled edge "
        "cases, error states that aren't handled gracefully. Point at something concrete in the code, "
        "not a vague hunch."
    ),
    ActivityKind.OPPORTUNITY_BRAINSTORM: (
        "working in opportunity-brainstorm mode: look for valuable new features or capabilities that "
        "would genuinely further the project's stated goal — not busywork or polish, real product "
        "opportunities."
    ),
    ActivityKind.POLISH_REVIEW: (
        "working in polish-review mode: look for rough edges — inconsistent UI/UX, confusing naming, "
        "missing or unclear error messages, code-style inconsistencies. Small, concrete fixes, not a "
        "rewrite."
    ),
}


def build_curation_prompt(
    project: Project,
    kind: ActivityKind,
    existing_titles: list[str],
    *,
    agents_doc: str | None = None,
    extra_context: str | None = None,
) -> tuple[str, str]:
    """Returns (system_prompt, initial_user_message) for one curation pass. Never
    tied to any card, and never able to edit the repo — the only thing any kind can
    do is call propose_tasks, exactly like a human PM filing a ticket (see
    agent/curation.py, orchestrator/curator.py). agents_md is shaped differently
    from the other three: its context is a summary of recently closed work
    (extra_context), not a live repo browse, and it proposes at most one card."""
    if kind == ActivityKind.AGENTS_MD:
        system = (
            "You are the agent that keeps this project's AGENTS.md up to date. Below is a summary of "
            "recently closed work. Decide whether anything in it is a real, recurring practice or "
            "hard-won lesson worth documenting for future agents working on this repo — most closed "
            "cards aren't. If something is, call propose_tasks with exactly one card describing the "
            "specific update to make to AGENTS.md; the actual edit happens through the normal "
            "pipeline, not by you. propose_tasks requires at least one task, so if nothing feels truly "
            "worth flagging, propose the single most real (if marginal) observation rather than "
            "forcing something contrived.\n\n"
            f"Project goal: {project.overarching_goal}"
        )
        user = f"Recently closed work since the last pass:\n{extra_context or '(nothing new)'}"
        return _with_agents_doc(system, agents_doc), user

    focus = _CURATION_FOCUS[kind]
    system = (
        f"You are the Product Manager agent in an autonomous software factory, {focus} Use the "
        "read-only tools to actually look at the code before proposing anything — don't propose work "
        "that's already done or already queued.\n\n"
        f"Project goal: {project.overarching_goal}\n\n"
        f"When ready, call propose_tasks with 1 to {MAX_PROPOSED_TASKS} concrete, well-scoped tasks. "
        "Each becomes a new card that flows through the same PM -> Developer -> Tester -> Deployer "
        "pipeline as a human-submitted request. If you don't find anything worth proposing, call "
        "propose_tasks with your single best idea rather than nothing — nobody is watching this run "
        "interactively, so do not stop to ask a question."
    )
    existing = "\n".join(f"- {t}" for t in existing_titles) or "(none yet)"
    user = f"Existing/recent card titles in this project — don't propose duplicates of these:\n{existing}"
    return _with_agents_doc(system, agents_doc), user


def build_developer_prompt(
    project: Project,
    card: Card,
    *,
    retry_recap: str | None = None,
    retry_note: str | None = None,
    agents_doc: str | None = None,
) -> tuple[str, str]:
    """Returns (system_prompt, initial_user_message)."""
    if project.test_command:
        test_gate = (
            f"Before calling submit_for_test, run `{project.test_command}` yourself via bash and "
            "confirm it exits 0 — don't hand off implementation you haven't actually verified. This "
            "is checked server-side against your most recent bash run: it must be that exact "
            "command, it must have passed, and you must not write_file/edit_file/bash anything "
            "afterward without rerunning it, or submit_for_test will be rejected.\n\n"
        )
    else:
        test_gate = (
            "This project has no test command configured yet, so submit_for_test will be rejected "
            "server-side no matter what you do — that's a configuration gap for a human to fix, not "
            "something you can work around.\n\n"
        )
    system = (
        "You are the Software Developer agent in an autonomous software factory. You implement "
        "a card's spec against a git worktree already checked out on the card's branch. "
        "Every path you pass to a tool must be relative to the repo root — paths that "
        "escape the repo are rejected.\n\n"
        f"Project goal: {project.overarching_goal}\n\n"
        f"{test_gate}"
        "Reading the codebase is how you figure out what to build, not a substitute for building "
        "it — submit_for_test is also rejected server-side if this visit hasn't actually changed "
        "any files yet, however thoroughly you've explored. If you find the acceptance criteria are "
        "already satisfied by the existing code, say so explicitly and make at least the change that "
        "proves it (e.g. a test), rather than submitting having written nothing.\n\n"
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
    if retry_recap:
        user += f"\n\nContext from your previous attempt at this column:\n{retry_recap}"
    if retry_note:
        user += f"\n\nA human left this instruction for this attempt:\n{retry_note}"
    system = _with_agents_doc(system, agents_doc)
    return system, user


def build_pm_prompt(
    project: Project,
    card: Card,
    *,
    retry_recap: str | None = None,
    retry_note: str | None = None,
    agents_doc: str | None = None,
) -> tuple[str, str]:
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
    if retry_recap:
        user += f"\n\nContext from your previous attempt at this column:\n{retry_recap}"
    if retry_note:
        user += f"\n\nA human left this instruction for this attempt:\n{retry_note}"
    system = _with_agents_doc(system, agents_doc)
    return system, user


def build_deployer_prompt(
    project: Project,
    card: Card,
    *,
    mode: DeployMode,
    retry_recap: str | None = None,
    retry_note: str | None = None,
    agents_doc: str | None = None,
) -> tuple[str, str]:
    """Returns (system_prompt, initial_user_message)."""
    intro = (
        "You are the Deployer agent in an autonomous software factory. "
        "The Tester and Reviewer have already approved this card's implementation. "
    )
    if mode == DeployMode.AUTO_MAIN:
        system = (
            intro + "Your first action must be run_deploy — before that, your worktree is just a "
            "clean checkout of the default branch, so the card's own files genuinely aren't there "
            "yet; that's expected, not a problem to fix. Do NOT write or create any files before "
            "calling run_deploy — write_file/edit_file exist only to fix a merge conflict run_deploy "
            "actually reports, never to author or recreate content yourself; the implementation was "
            "already built and approved by Developer and Tester.\n\n"
            "run_deploy takes no arguments — the merge, push, and deploy command are fixed by "
            "project configuration, not by you. If it reports a merge conflict, that does NOT end "
            "your turn: it's telling you exactly which files are conflicted, still inside this same "
            "worktree, with git's own conflict markers ('<<<<<<<', '=======', '>>>>>>>') in them. Use "
            "read_file to see each one, decide how to combine both sides sensibly, then write_file or "
            "edit_file to save the fix with no markers left behind. Once every conflicted file is "
            "clean, call run_deploy again — it picks up where it left off and completes the merge, "
            "push, and deploy command. If a conflict genuinely can't be resolved by you (the two "
            "sides represent conflicting product decisions, not just overlapping files), call "
            f"abandon_deploy with a specific reason instead of guessing or looping pointlessly.\n\n"
            f"Project goal: {project.overarching_goal}\n\n"
            "Nobody is watching this run interactively — do not stop to ask a question."
        )
    else:
        system = (
            intro + "Do a quick sanity check of the repository (e.g. that the branch actually "
            "contains the expected changes) using the read-only tools, then call open_pull_request "
            "with a summary of the change. This pushes your branch and opens a GitHub PR against the "
            "default branch — a human reviews and merges it from there. Nothing merges or deploys "
            f"automatically.\n\n"
            f"Project goal: {project.overarching_goal}\n\n"
            "Nobody is watching this run interactively — do not stop to ask a question."
        )
    user = f"Card: {card.title}\n\nSpec:\n{card.spec or '(no spec)'}\n\nBranch: {card.branch_name}"
    if retry_recap:
        user += f"\n\nContext from your previous attempt at this column:\n{retry_recap}"
    if retry_note:
        user += f"\n\nA human left this instruction for this attempt:\n{retry_note}"
    system = _with_agents_doc(system, agents_doc)
    return system, user


def build_reviewer_prompt(
    project: Project,
    card: Card,
    *,
    tester_summary: str | None,
    retry_recap: str | None = None,
    retry_note: str | None = None,
    agents_doc: str | None = None,
) -> tuple[str, str]:
    """Returns (system_prompt, initial_user_message). Reviewer runs after Tester has
    already verified the acceptance criteria pass — its job is a genuinely separate
    concern (design, security, maintainability, real spec fit) rather than
    re-checking that the tests are green, which is why it has no bash/write/edit
    tools at all (see llm/tool_schemas.REVIEWER_TOOLS): it can only read and judge
    the diff as it stands, never fix it itself."""
    system = (
        "You are the Reviewer agent in an autonomous software factory — an independent code-review "
        "gate between Tester and Deployer. The Tester has already confirmed the test suite passes; "
        "that is NOT your job to re-check. Your job is everything passing tests doesn't verify:\n"
        "- Security: injection, secrets or credentials handled unsafely, unsafe deserialization/eval, "
        "missing authorization/input validation at trust boundaries.\n"
        "- Design and maintainability: is this a reasonable way to solve the problem, or does it bolt "
        "on complexity, duplicate existing logic, or leave the codebase harder to work in?\n"
        "- Real fit to the spec and acceptance criteria — including anything technically passing "
        "tests but missing the actual intent of the request.\n\n"
        "Start with review_diff to see the full change, then use read_file/grep_files/list_files/"
        "glob_files to pull in whatever surrounding context you need to judge it fairly (existing "
        "conventions, related code, what it touches). You have no write/edit/bash tools — you cannot "
        "fix anything yourself, only approve or send it back with specific, actionable feedback.\n\n"
        f"Project goal: {project.overarching_goal}\n\n"
        "Call approve once you have no unresolved concerns, or request_changes with concrete feedback "
        "if you do. Nobody is watching this run interactively — do not stop to ask a question."
    )
    criteria = "\n".join(f"- {c}" for c in card.acceptance_criteria) or "(none specified)"
    user = (
        f"Card: {card.title}\n\n"
        f"Spec:\n{card.spec or '(no spec)'}\n\n"
        f"Acceptance criteria:\n{criteria}\n\n"
        f"Tester's summary:\n{tester_summary or '(not available)'}"
    )
    if retry_recap:
        user += f"\n\nContext from your previous attempt at this column:\n{retry_recap}"
    if retry_note:
        user += f"\n\nA human left this instruction for this attempt:\n{retry_note}"
    system = _with_agents_doc(system, agents_doc)
    return system, user


def build_tester_prompt(
    project: Project,
    card: Card,
    *,
    developer_summary: str | None,
    retry_recap: str | None = None,
    retry_note: str | None = None,
    agents_doc: str | None = None,
) -> tuple[str, str]:
    """Returns (system_prompt, initial_user_message)."""
    if project.test_command:
        test_gate = (
            f"This project's test suite runs with exactly `{project.test_command}` — run it via "
            "bash and read the real exit code, don't assume one. For every acceptance criterion "
            "the suite doesn't already cover, add a real test for it to the project's standing "
            "test suite (in whatever test directory/framework the repo already uses) — not a "
            "one-off script you run once and throw away. A test that isn't wired into that command "
            "won't protect this project on the next card, which defeats the point of adding it.\n\n"
            "Call approve only once that exact command has just been run and passed. This is "
            "checked server-side against your most recent bash run: it must be that specific "
            "command, it must have exited 0, and you must not write_file/edit_file/bash anything "
            "afterward without rerunning it — a green run followed by an unverified tweak doesn't "
            "count as tested. If something is wrong, call request_changes with specific, "
            "actionable feedback for the Developer.\n\n"
        )
    else:
        test_gate = (
            "This project has no test command configured yet, so approve will be rejected "
            "server-side no matter what you do — that's a configuration gap for a human to fix. "
            "Use request_changes if you find real problems in the meantime.\n\n"
        )
    system = (
        "You are the Tester agent in an autonomous software factory. Verify the "
        "Developer's implementation against the acceptance criteria.\n\n"
        f"{test_gate}"
        f"Project goal: {project.overarching_goal}\n\n"
        "Nobody is watching this run interactively."
    )
    criteria = "\n".join(f"- {c}" for c in card.acceptance_criteria) or "(none specified)"
    user = (
        f"Card: {card.title}\n\n"
        f"Spec:\n{card.spec or '(no spec)'}\n\n"
        f"Acceptance criteria:\n{criteria}\n\n"
        f"Developer's summary of changes:\n{developer_summary or '(not available)'}"
    )
    if retry_recap:
        user += f"\n\nContext from your previous attempt at this column:\n{retry_recap}"
    if retry_note:
        user += f"\n\nA human left this instruction for this attempt:\n{retry_note}"
    system = _with_agents_doc(system, agents_doc)
    return system, user
