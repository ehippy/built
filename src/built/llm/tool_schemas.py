"""OpenAI-compatible function-calling tool schemas, one list per column role."""

from built.domain.enums import DeployMode

READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a text file from the repository. Paths are relative to the repo root.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file, relative to the repo root."}
            },
            "required": ["path"],
        },
    },
}

LIST_FILES = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List the files and subdirectories directly inside a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list, relative to the repo root. Defaults to the repo root.",
                }
            },
        },
    },
}

GLOB_FILES = {
    "type": "function",
    "function": {
        "name": "glob_files",
        "description": "Find files matching a glob pattern (e.g. '**/*.py').",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, relative to the repo root."}
            },
            "required": ["pattern"],
        },
    },
}

GREP_FILES = {
    "type": "function",
    "function": {
        "name": "grep_files",
        "description": "Search file contents for a regular expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression to search for."},
                "path": {
                    "type": "string",
                    "description": "File or directory to search, relative to the repo root. Defaults to '.'.",
                },
            },
            "required": ["pattern"],
        },
    },
}

WRITE_FILE = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Create a file or fully overwrite it. Prefer edit_file for small changes.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file, relative to the repo root."},
                "content": {"type": "string", "description": "The full content to write."},
            },
            "required": ["path", "content"],
        },
    },
}

EDIT_FILE = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": (
            "Replace one exact, unique occurrence of old_str with new_str in an existing file. "
            "Include enough surrounding context in old_str to make it unique."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file, relative to the repo root."},
                "old_str": {
                    "type": "string",
                    "description": "The exact text to replace (must be unique in the file).",
                },
                "new_str": {"type": "string", "description": "The replacement text."},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
}

BASH = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command in the repository root — e.g. to run tests, lint, or build.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run."},
                "timeout_seconds": {"type": "integer", "description": "Timeout in seconds. Defaults to 120."},
            },
            "required": ["command"],
        },
    },
}

GIT_STATUS = {
    "type": "function",
    "function": {
        "name": "git_status",
        "description": "Show the working tree status (uncommitted changes).",
        "parameters": {"type": "object", "properties": {}},
    },
}

GIT_DIFF = {
    "type": "function",
    "function": {
        "name": "git_diff",
        "description": "Show the diff of uncommitted changes.",
        "parameters": {"type": "object", "properties": {}},
    },
}

SUBMIT_FOR_TEST = {
    "type": "function",
    "function": {
        "name": "submit_for_test",
        "description": (
            "Declare the implementation complete and ready for the Tester to verify against the "
            "acceptance criteria. This ends your turn — call it only once every acceptance criterion "
            "is implemented and committed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A short summary of what changed and why, for the Tester and the audit.",
                }
            },
            "required": ["summary"],
        },
    },
}

DEVELOPER_TOOLS = [
    READ_FILE,
    LIST_FILES,
    GLOB_FILES,
    GREP_FILES,
    WRITE_FILE,
    EDIT_FILE,
    BASH,
    GIT_STATUS,
    GIT_DIFF,
    SUBMIT_FOR_TEST,
]

DEVELOPER_TERMINAL_TOOL = "submit_for_test"

SUBMIT_SPEC = {
    "type": "function",
    "function": {
        "name": "submit_spec",
        "description": (
            "Finalize the spec and acceptance criteria for the Developer to implement against. "
            "This ends your turn — call it only once you're confident the criteria are concrete "
            "and independently verifiable."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "spec": {"type": "string", "description": "The implementation spec, in prose."},
                "acceptance_criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of concrete, individually checkable acceptance criteria.",
                },
                "summary": {"type": "string", "description": "A one-line summary for the audit log."},
            },
            "required": ["spec", "acceptance_criteria", "summary"],
        },
    },
}

PM_TOOLS = [READ_FILE, LIST_FILES, GLOB_FILES, GREP_FILES, SUBMIT_SPEC]
PM_TERMINAL_TOOL = "submit_spec"

MAX_PROPOSED_TASKS = 5

PROPOSE_TASKS = {
    "type": "function",
    "function": {
        "name": "propose_tasks",
        "description": (
            "Propose new cards for the backlog based on gaps, bugs, rough edges, or genuine "
            "opportunities you found in the repository. Each becomes a new card, worked exactly like "
            "a human-submitted request — starting in the PM column. This ends your turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_PROPOSED_TASKS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "A short title for the card."},
                            "raw_request": {
                                "type": "string",
                                "description": (
                                    "The gap/bug/opportunity and what should be done about it, "
                                    "written as if a human requested it."
                                ),
                            },
                        },
                        "required": ["title", "raw_request"],
                    },
                }
            },
            "required": ["tasks"],
        },
    },
}

DISCOVERY_TOOLS = [READ_FILE, LIST_FILES, GLOB_FILES, GREP_FILES, PROPOSE_TASKS]
DISCOVERY_TERMINAL_TOOL = "propose_tasks"

APPROVE = {
    "type": "function",
    "function": {
        "name": "approve",
        "description": (
            "Approve the implementation and advance the card to Deployer. You must have run the "
            "test suite via bash and gotten exit code 0 first — this is checked server-side, so "
            "claiming success without actually running it will be rejected."
        ),
        "parameters": {
            "type": "object",
            "properties": {"notes": {"type": "string", "description": "Brief notes for the audit log."}},
            "required": ["notes"],
        },
    },
}

REQUEST_CHANGES = {
    "type": "function",
    "function": {
        "name": "request_changes",
        "description": "Send the card back to the Developer with feedback on what needs to change.",
        "parameters": {
            "type": "object",
            "properties": {
                "feedback": {
                    "type": "string",
                    "description": "Specific, actionable feedback for the Developer — what failed and why.",
                },
                "summary": {"type": "string", "description": "A one-line summary for the audit log."},
            },
            "required": ["feedback", "summary"],
        },
    },
}

TESTER_TOOLS = [READ_FILE, GREP_FILES, BASH, WRITE_FILE, EDIT_FILE, APPROVE, REQUEST_CHANGES]
TESTER_TERMINAL_TOOLS = ("approve", "request_changes")

RUN_DEPLOY = {
    "type": "function",
    "function": {
        "name": "run_deploy",
        "description": (
            "Merge this card's branch into the default branch, push it, and run the project's "
            "configured deploy command. Takes no arguments — the merge, push, and deploy command are "
            "fixed by project configuration, not by you. If it reports a merge conflict, this does "
            "NOT end your turn: resolve the listed files with read_file/write_file/edit_file, then "
            "call run_deploy() again to pick up where it left off and complete the merge. Only ends "
            "your turn on an actual outcome — success, a non-conflict failure, or you completing the "
            "merge — there is no confirmation step after a real completion."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}
DEPLOYER_AUTO_MAIN_TERMINAL_TOOL = "run_deploy"

ABANDON_DEPLOY = {
    "type": "function",
    "function": {
        "name": "abandon_deploy",
        "description": (
            "Give up on this deploy attempt and leave it for a human, instead of guessing. Use this "
            "for a merge conflict you can't reasonably resolve yourself — e.g. the two sides make "
            "genuinely conflicting product decisions about the same content, not just an overlapping "
            "file — or any other situation your tools can't fix. This ends your turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "A specific, actionable explanation of why this needs a human.",
                }
            },
            "required": ["reason"],
        },
    },
}
DEPLOYER_ABANDON_TERMINAL_TOOL = "abandon_deploy"

OPEN_PULL_REQUEST = {
    "type": "function",
    "function": {
        "name": "open_pull_request",
        "description": (
            "Push this card's branch and open a GitHub pull request against the default branch for "
            "a human to review and merge. This ends your turn — nothing merges or deploys "
            "automatically after this; a human takes over from the PR."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A description of the change, used as the PR body.",
                }
            },
            "required": ["summary"],
        },
    },
}
DEPLOYER_PR_TERMINAL_TOOL = "open_pull_request"

_DEPLOYER_READ_TOOLS = [READ_FILE, LIST_FILES, GLOB_FILES, GREP_FILES]
# auto_main only: real editing tools, scoped by the caller to the Deployer's merge
# worktree (not the card's own worktree) — this is what actually lets the agent fix
# a merge conflict instead of just detecting and reporting one.
_DEPLOYER_CONFLICT_TOOLS = [WRITE_FILE, EDIT_FILE, GIT_STATUS, GIT_DIFF]


def deployer_tools(mode: DeployMode) -> list[dict]:
    """auto_main gets read tools, conflict-resolution editing tools, and both
    terminal tools (run_deploy, abandon_deploy). pr_to_operator never merges, so it
    only needs read tools plus its own single terminal tool — the model never sees a
    tool it can't use."""
    if mode == DeployMode.AUTO_MAIN:
        return [*_DEPLOYER_READ_TOOLS, *_DEPLOYER_CONFLICT_TOOLS, RUN_DEPLOY, ABANDON_DEPLOY]
    return [*_DEPLOYER_READ_TOOLS, OPEN_PULL_REQUEST]


LIST_STUCK_CARDS = {
    "type": "function",
    "function": {
        "name": "list_stuck_cards",
        "description": (
            "List cards currently blocked or failed, longest-stuck first: id, title, project, column, "
            "how many times the Reviver has already retried it, and the reason it's stuck."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

READ_CARD_HISTORY = {
    "type": "function",
    "function": {
        "name": "read_card_history",
        "description": (
            "Get a stuck card's full context: its spec, and a recap of its most recent attempt — how "
            "it ended and its last several tool calls. Use this before deciding whether and how to "
            "revive a card whose reason for being stuck isn't already obvious."
        ),
        "parameters": {
            "type": "object",
            "properties": {"card_id": {"type": "string", "description": "The card to inspect."}},
            "required": ["card_id"],
        },
    },
}

REVIVE_CARD = {
    "type": "function",
    "function": {
        "name": "revive_card",
        "description": (
            "Retry a stuck card, optionally with a note for whichever column runs next. Use this for "
            "cards stuck on something retryable: a transient infrastructure failure (timeout, "
            "connection error) needs no note; a genuine issue you can diagnose from its history "
            "(e.g. a merge conflict, a fixable misunderstanding) should get a specific, actionable "
            "note. Does not end your turn — keep reviewing other stuck cards after this."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "card_id": {"type": "string", "description": "The card to retry."},
                "note": {
                    "type": "string",
                    "description": "Optional instruction for the next attempt. Omit for a plain retry.",
                },
            },
            "required": ["card_id"],
        },
    },
}

LEAVE_BLOCKED = {
    "type": "function",
    "function": {
        "name": "leave_blocked",
        "description": (
            "Explicitly leave a stuck card alone for a human — use this for anything retrying can't "
            "fix: missing configuration (no deploy config, no credentials), a decision only a human "
            "can make, or a card that has already exhausted its automatic-retry budget. Records your "
            "reasoning so a human reviewing later can see you looked at it. Does not end your turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "card_id": {"type": "string", "description": "The card to leave blocked."},
                "reason": {"type": "string", "description": "Why retrying wouldn't help."},
            },
            "required": ["card_id", "reason"],
        },
    },
}

DONE_FOR_NOW = {
    "type": "function",
    "function": {
        "name": "done_for_now",
        "description": (
            "Call this once you've reviewed every stuck card worth reviewing this pass (or there were "
            "none). Ends your turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string", "description": "Brief summary of what you did."}},
            "required": ["summary"],
        },
    },
}

REVIVER_TOOLS = [LIST_STUCK_CARDS, READ_CARD_HISTORY, REVIVE_CARD, LEAVE_BLOCKED, DONE_FOR_NOW]
REVIVER_TERMINAL_TOOL = "done_for_now"
REVIVER_ACTION_TOOLS = ("revive_card", "leave_blocked")

LIST_RECENT_VISIT_OUTCOMES = {
    "type": "function",
    "function": {
        "name": "list_recent_visit_outcomes",
        "description": (
            "List this project's recently closed column visits — card title, column, outcome, and "
            "summary — since your last pass (or the last 30 if this is your first). Your raw material "
            "for deciding whether a real, recurring pattern is worth documenting."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

TENDER_DONE_FOR_NOW = {
    "type": "function",
    "function": {
        "name": "done_for_now",
        "description": (
            "Call this once you've decided what (if anything) to write, and made any edit. Ends your "
            "turn — fine to call immediately if recent activity had nothing new worth recording."
        ),
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string", "description": "Brief summary of what you did."}},
            "required": ["summary"],
        },
    },
}

# Reuses the general-purpose file tools already built for Developer — AGENTS.md is
# just a file in the repo, no bespoke read/write tools needed for it.
TENDER_TOOLS = [
    READ_FILE,
    LIST_FILES,
    GLOB_FILES,
    GREP_FILES,
    WRITE_FILE,
    EDIT_FILE,
    LIST_RECENT_VISIT_OUTCOMES,
    TENDER_DONE_FOR_NOW,
]
TENDER_TERMINAL_TOOL = "done_for_now"
