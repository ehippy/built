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
            "fixed by project configuration, not by you. This ends your turn and cannot be undone: "
            "there is no confirmation step after this."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}
DEPLOYER_AUTO_MAIN_TERMINAL_TOOL = "run_deploy"

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


def deployer_tools(mode: DeployMode) -> list[dict]:
    """Read tools plus only the one terminal tool matching this project's configured
    deploy mode — the model never sees a tool it can't use."""
    terminal = RUN_DEPLOY if mode == DeployMode.AUTO_MAIN else OPEN_PULL_REQUEST
    return [*_DEPLOYER_READ_TOOLS, terminal]
