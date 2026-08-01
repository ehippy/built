"""OpenAI-compatible function-calling tool schemas, one list per column role."""

from built.domain.enums import Column, DeployMode, Severity

READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read a file's contents, prefixed with line numbers (format: a right-aligned number, a "
            "tab, then the actual line — never include that prefix in old_str if you use this output "
            "with edit_file). Defaults to the first 2000 lines; if the response says more lines "
            "follow, either pass limit to pull more at once, or better, pass offset to jump straight "
            "to the section you actually need. For a large or unfamiliar file, grep_files first to "
            "find which lines matter, then read_file with offset near that line — cheaper and more "
            "precise than reading start-to-finish when you already know what you're looking for. "
            "Reach for this once you know which file and roughly which lines; use grep_files/"
            "glob_files instead when you're still figuring out which file that is."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file, relative to the repo root."},
                "offset": {
                    "type": "integer",
                    "description": (
                        "1-indexed line number to start reading from. Defaults to 1 (the start of the file)."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to return. Defaults to 2000.",
                },
            },
            "required": ["path"],
        },
    },
}

LIST_FILES = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": (
            "List the files and subdirectories directly inside one directory (not recursive). Use "
            "this for a quick look at what's in a specific folder; use glob_files instead when "
            "you're searching by name/extension pattern across the whole tree."
        ),
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
        "description": (
            "Find files by name or path pattern (e.g. '**/*.py', 'src/**/test_*.js') without reading "
            "any file contents — essentially free. Use this to answer 'does this file exist' or "
            "'what's the naming convention here' before you commit to reading or writing anything; "
            "reach for grep_files instead when what you actually know is a symbol or string you're "
            "looking for, not a filename shape."
        ),
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
        "description": (
            "Search file contents for a regular expression; returns 'path:line_number:line text' for "
            "every match. This is the cheap way to find where a symbol, string, or pattern lives — "
            "prefer it over read_file whenever you're looking for something specific rather than "
            "trying to understand a whole file's structure. Once a hit tells you which file and line, "
            "follow up with read_file(path, offset=<that line>) instead of reading the file "
            "start-to-finish."
        ),
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

FETCH_DOCS = {
    "type": "function",
    "function": {
        "name": "fetch_docs",
        "description": (
            "Fetch a page from the public internet and return its text content — use this to check "
            "the real, current documentation for a third-party API, library, or framework before "
            "writing or judging code against it, instead of relying on memory of its shape, which "
            "can be outdated or wrong. Only http/https URLs that resolve to a public address are "
            "allowed; internal, private, and loopback addresses are rejected, as are more than a "
            "handful of redirects. Large pages are truncated."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The full URL to fetch, including scheme."}
            },
            "required": ["url"],
        },
    },
}

WRITE_FILE = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "Create a new file, or fully overwrite an existing one. Prefer edit_file whenever a file "
            "already exists and you're only changing part of it — write_file makes you (and the "
            "model reading the diff later) pay for the whole file every time, even the 99% that "
            "didn't change. Reach for it for genuinely new files, or when a rewrite is actually "
            "clearer than a series of edits. If the same change applies to several existing files "
            "at once, neither write_file nor edit_file is the right tool — that's a one-shot "
            "sed/grep job via bash (see its description) instead of N calls here. "
            "Whatever you write should be complete and correct as committed, not a draft: match this "
            "repo's existing conventions (naming, formatting, import style) rather than inventing "
            "your own, don't leave placeholder or TODO content standing in for real implementation, "
            "and don't pad it with speculative options or abstractions the task doesn't need — write "
            "the thing that's actually being asked for, done properly, and nothing else."
        ),
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
            "Replace old_str with new_str in an existing file — the primary way to change code here: "
            "it touches only what's actually changing instead of making you reproduce the whole file "
            "the way write_file would. By default old_str must be unique in the file and exactly one "
            "occurrence is replaced; include just enough surrounding context (usually a line or two "
            "is plenty) to pin down one location — you don't need to quote a large block. old_str "
            "must match the file's actual bytes exactly: if you copied it from read_file's output, "
            "strip the leading line-number-and-tab from every line first, that prefix isn't part of "
            "the file. If old_str isn't unique, this fails and tells you which lines it hit — narrow "
            "old_str with more context to pick one of them, or set replace_all when you deliberately "
            "mean to change all of them (e.g. renaming a variable throughout a file) — that's for "
            "repeats within one file; when the same mechanical change spans multiple files, prefer "
            "bash (sed/grep) in a single call over calling edit_file once per file. "
            "Make each call one coherent, self-contained change — prefer several precise edit_file "
            "calls over a single old_str that spans unrelated lines just to make one big diff, and "
            "prefer editing over commenting out or duplicating code you're replacing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file, relative to the repo root."},
                "old_str": {
                    "type": "string",
                    "description": (
                        "The exact text to replace (must be unique in the file unless replace_all is set)."
                    ),
                },
                "new_str": {"type": "string", "description": "The replacement text."},
                "replace_all": {
                    "type": "boolean",
                    "description": (
                        "Replace every occurrence of old_str instead of requiring exactly one. Use "
                        "this deliberately for a genuine global change (e.g. a rename) — not as a "
                        "shortcut to avoid picking context that uniquely identifies one location."
                    ),
                },
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
}

BASH = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run a shell command in the repository root, inside a Linux container with standard "
            "GNU tools (sed, grep, find, awk — not macOS/BSD, so `sed -i 's/x/y/g' file` works "
            "directly with no backup-suffix argument needed). Use it to run tests, lint, or build — "
            "and just as importantly, use it for mechanical text changes that are the same across "
            "multiple files. That's almost always cheaper and more reliable than repeating "
            "write_file or edit_file once per file: find the affected files with grep -rl or find, "
            "then apply the change to all of them in one sed/awk call, e.g. "
            "`grep -rl '.site-footer' games/ | xargs sed -i '/\\.site-footer {/,/^}/d'` to strip the "
            "same duplicated CSS block out of every game page in one shot. Reach for edit_file "
            "instead when the change needs judgment about surrounding context that a regex can't "
            "safely express, or only touches one place. Whichever you use, verify a bulk edit "
            "actually did what you intended — grep_files for the old pattern, or git_diff — before "
            "moving on; a sed pattern that's slightly too greedy is a classic way to silently "
            "mangle more than you meant to."
        ),
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
        "description": (
            "Show the diff of uncommitted changes. Very large diffs are truncated — if that "
            "happens, use bash to diff a specific path instead of the whole tree."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

UPDATE_PLAN = {
    "type": "function",
    "function": {
        "name": "update_plan",
        "description": (
            "Write or update your implementation plan: the full, ordered list of concrete steps "
            "needed to satisfy every acceptance criterion. Call this first, before exploring or "
            "changing anything, to lay out your plan — then call it again, with the complete current "
            "list, whenever a step's status changes (pending -> in_progress when you start it, "
            "in_progress -> done once it's actually implemented and committed). Always pass every "
            "step, not just the one that changed — each call replaces the whole plan. "
            "submit_for_test is rejected server-side unless your most recent call here has at least "
            "one step and every step marked done."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {
                                "type": "string",
                                "description": "A concrete, individually checkable unit of work.",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "done"],
                            },
                        },
                        "required": ["step", "status"],
                    },
                }
            },
            "required": ["steps"],
        },
    },
}

SUBMIT_FOR_TEST = {
    "type": "function",
    "function": {
        "name": "submit_for_test",
        "description": (
            "Declare the implementation complete and ready for the Tester to verify against the "
            "acceptance criteria. This ends your turn — call it only once every acceptance criterion "
            "is implemented and committed, AND you've run the project's test command via bash and "
            "gotten exit code 0 yourself. This is checked server-side: your most recent update_plan "
            "call must have every step marked done, this visit must have actually changed a file, and "
            "the test run must be that exact command, must have just passed, and no file may have "
            "changed since — claiming completion without real, planned, verified work will be "
            "rejected."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "A short summary of what changed and why, for the Tester and the audit. Also "
                        "note how it went if it's worth knowing — dead ends, backtracking, anything "
                        "that made this harder or easier than a normal card — not just the final "
                        "result; a later pass mines these summaries for recurring friction."
                    ),
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
    FETCH_DOCS,
    WRITE_FILE,
    EDIT_FILE,
    BASH,
    GIT_STATUS,
    GIT_DIFF,
    UPDATE_PLAN,
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

MAX_SPLIT_TASKS = 8

SPLIT_INTO_SUBTASKS = {
    "type": "function",
    "function": {
        "name": "split_into_subtasks",
        "description": (
            "Use instead of submit_spec when this card is too big for one Developer visit to "
            "implement and one Tester visit to verify as a single coherent pass — a whole-repo "
            "refactor, the same mechanical change repeated across many independent files, or "
            "several genuinely unrelated asks bundled into one request. A long acceptance-criteria "
            "list is not by itself a reason to split — most real cards have several criteria that "
            "all belong to one unit of work; split when the criteria don't actually belong "
            "together, not just because there are many of them. "
            "This retires the current card (archived, not claimable again) and creates each task "
            "below as its own new card, which goes through the normal PM -> Developer -> Tester -> "
            "Deployer pipeline independently — you're only naming and scoping the pieces here, not "
            "writing their specs; each gets its own future PM visit for that. Write each "
            "raw_request as self-contained: whoever specs it later won't have this card's original "
            "request in front of them, so restate whatever context they'd need (file paths, names, "
            "the specific slice of the original ask). List foundational pieces first — e.g. 'add "
            "the shared layout' before 'convert page X to use it' — since there's no enforced "
            "dependency between the resulting cards, only the order they're likely to be picked up "
            f"in. At most {MAX_SPLIT_TASKS} tasks; if it genuinely needs more than that, group "
            "related pieces into batches rather than one task per file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "minItems": 2,
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Short title for the new card."},
                            "raw_request": {
                                "type": "string",
                                "description": (
                                    "Self-contained description of this piece — written so it makes "
                                    "sense without the original card's request in front of you."
                                ),
                            },
                        },
                        "required": ["title", "raw_request"],
                    },
                },
                "summary": {
                    "type": "string",
                    "description": "A one-line summary of how and why this was split, for the audit log.",
                },
            },
            "required": ["tasks", "summary"],
        },
    },
}

MAX_EPIC_TASKS = 8  # own constant, not reused from MAX_SPLIT_TASKS — independently tunable later

DEFINE_EPIC = {
    "type": "function",
    "function": {
        "name": "define_epic",
        "description": (
            "Use instead of submit_spec when this request is a genuine multi-piece initiative worth "
            "tracking as a whole — several cards that together deliver one larger thing, where you "
            "want the parent visible on the board until every piece is done. Unlike split_into_subtasks "
            "(which retires this card and cuts the new ones loose with no enforced order), this card "
            "stays alive, tracks its children, and automatically reaches done once every child does. "
            "Set up real dependency ordering between the children with depends_on — e.g. 'add the "
            "shared layout' before 'convert page X to use it' — this IS enforced: the orchestrator will "
            "not claim a dependent card until everything it depends on is fully done. "
            f"At most {MAX_EPIC_TASKS} child tasks; if it genuinely needs more, group related pieces "
            "into batches rather than one task per file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "spec": {
                    "type": "string",
                    "description": (
                        "The epic's own overview — the shared context and goal tying its children "
                        "together. Nothing implements this card directly, only its children, so this "
                        "isn't itself an implementable spec the way submit_spec's is."
                    ),
                },
                "tasks": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": MAX_EPIC_TASKS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Short title for the child card."},
                            "raw_request": {
                                "type": "string",
                                "description": (
                                    "Self-contained description of this piece — written so it makes "
                                    "sense without this card's original request in front of you."
                                ),
                            },
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": (
                                    "0-based indices into this same tasks array — every earlier task "
                                    "that must fully reach done before this one can be claimed. Only "
                                    "indices strictly less than this task's own position are valid (no "
                                    "forward or self references) — list every direct prerequisite, not "
                                    "just the immediately preceding one. Omit or leave empty if this "
                                    "task has no prerequisites among the others."
                                ),
                            },
                        },
                        "required": ["title", "raw_request"],
                    },
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "A one-line summary of the epic and why it's structured this way, for the "
                        "audit log."
                    ),
                },
            },
            "required": ["spec", "tasks", "summary"],
        },
    },
}

PM_TOOLS = [READ_FILE, LIST_FILES, GLOB_FILES, GREP_FILES, SUBMIT_SPEC, SPLIT_INTO_SUBTASKS, DEFINE_EPIC]
PM_TERMINAL_TOOL = "submit_spec"
PM_SPLIT_TERMINAL_TOOL = "split_into_subtasks"
PM_EPIC_TERMINAL_TOOL = "define_epic"

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
                            "severity": {
                                "type": "string",
                                "enum": [s.value for s in Severity],
                                "description": (
                                    "How much this matters if left unaddressed — rate honestly, not "
                                    "inflated. critical: actively broken/exploitable/data-loss risk. "
                                    "high: clearly worth doing soon. medium: real but not urgent. "
                                    "low: minor/cosmetic. A server-side policy uses this to decide "
                                    "which candidates actually become cards, so an inflated rating "
                                    "doesn't help this one get filed and just makes future ratings "
                                    "less trustworthy."
                                ),
                            },
                            "raw_request": {
                                "type": "string",
                                "description": (
                                    "The gap/bug/opportunity and what should be done about it, "
                                    "written as if a human requested it. One issue per task — if you "
                                    "found several distinct problems, that's several array entries, "
                                    "not one raw_request that lists them all."
                                ),
                            },
                        },
                        "required": ["title", "severity", "raw_request"],
                    },
                }
            },
            "required": ["tasks"],
        },
    },
}

# Shared by every curation kind (agent/curation.py) — bug_sweep, opportunity_brainstorm,
# polish_review, stay_dry, security_sweep, coverage_sweep, refactor_sweep, and agents_md
# all explore read-only and end with propose_tasks. None of them can write to the repo;
# the only thing any curation pass can do is create new cards, exactly like a human PM
# filing a ticket.
CURATION_TOOLS = [READ_FILE, LIST_FILES, GLOB_FILES, GREP_FILES, PROPOSE_TASKS]
CURATION_TERMINAL_TOOL = "propose_tasks"

# A separate, narrow tool for agent/curation.py's live-DB duplicate check — deliberately
# not part of CURATION_TOOLS/the explore-then-propose loop above: this is a single
# judgment call made server-side, after propose_tasks has already returned, comparing
# severity-surviving candidates against the project's live open cards.
REPORT_DUPLICATE_CANDIDATES = {
    "type": "function",
    "function": {
        "name": "report_duplicate_candidates",
        "description": (
            "Report which numbered candidates duplicate an already-open card on the board — the "
            "same underlying request, not just a similar topic. Call this exactly once; omit a "
            "candidate entirely if it's genuinely new."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "duplicates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_index": {
                                "type": "integer",
                                "description": "0-based index of the duplicate candidate, as numbered above.",
                            },
                            "duplicate_of_card_id": {
                                "type": "string",
                                "description": "id of the existing open card it duplicates.",
                            },
                        },
                        "required": ["candidate_index", "duplicate_of_card_id"],
                    },
                }
            },
            "required": ["duplicates"],
        },
    },
}
CURATION_DEDUP_TOOLS = [REPORT_DUPLICATE_CANDIDATES]
CURATION_DEDUP_TERMINAL_TOOL = "report_duplicate_candidates"

# agent/summarizer.py's postmortem call: the card's column-visit map is handed in
# directly, and get_visit_detail is the one narrow "explore" tool available — no
# read/grep/bash, since there's nothing in the repo itself to look at here, just
# more detail on a visit the map already summarized.
GET_VISIT_DETAIL = {
    "type": "function",
    "function": {
        "name": "get_visit_detail",
        "description": (
            "Look up the full detail behind one line of the column-visit map above — the complete "
            "feedback text (if this visit was a rejection) and what tools were actually called during "
            "it. Use this when a visit's one-line summary hints at real trouble and the specifics "
            "matter for an honest postmortem; skip it for visits that were routine — most are. You "
            "can call this more than once, then call submit_postmortem when you're done."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "enum": [c.value for c in Column],
                    "description": "The column, exactly as labeled in the map above (e.g. 'developer').",
                },
                "attempt_number": {
                    "type": "integer",
                    "description": "The attempt number, exactly as labeled in the map above.",
                },
            },
            "required": ["column", "attempt_number"],
        },
    },
}

SUBMIT_POSTMORTEM = {
    "type": "function",
    "function": {
        "name": "submit_postmortem",
        "description": (
            "Submit your postmortem for this now-closed card. Ends your turn — call this exactly "
            "once."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "went_well": {
                    "type": "string",
                    "description": (
                        "What worked smoothly, if anything specific stands out — a few sentences. "
                        "'Nothing notable' is a completely fine answer; don't invent texture that "
                        "wasn't there."
                    ),
                },
                "struggles": {
                    "type": "string",
                    "description": (
                        "What was a genuine struggle — retries, back-and-forth between columns, dead "
                        "ends, confusing feedback loops — a few sentences. 'Nothing notable' is a "
                        "completely fine answer."
                    ),
                },
            },
            "required": ["went_well", "struggles"],
        },
    },
}
SUMMARIZER_TOOLS = [SUBMIT_POSTMORTEM, GET_VISIT_DETAIL]
SUMMARIZER_TERMINAL_TOOL = "submit_postmortem"
SUMMARIZER_DETAIL_TOOL = "get_visit_detail"

MAX_TICKETS_PER_CHAT_TURN = 5

CREATE_TICKET = {
    "type": "function",
    "function": {
        "name": "create_ticket",
        "description": (
            "Create a new card on this project's board, in the PM column — exactly like a human "
            "filing a request. Use this once you and the human have actually converged on a "
            "concrete, well-scoped idea during this conversation, not speculatively for every idea "
            "mentioned in passing. Does not end the conversation — keep chatting afterward. At most "
            f"{MAX_TICKETS_PER_CHAT_TURN} new tickets per turn; if you're proposing more, check in "
            "with the human before filing the rest as a follow-up."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "A short title for the card."},
                "raw_request": {
                    "type": "string",
                    "description": (
                        "Self-contained description of the work, written as if a human requested it "
                        "directly — whoever specs it later won't have this conversation in front of "
                        "them."
                    ),
                },
            },
            "required": ["title", "raw_request"],
        },
    },
}

UPDATE_TICKET = {
    "type": "function",
    "function": {
        "name": "update_ticket",
        "description": (
            "Edit an existing card's title and/or request text — e.g. after the human clarifies or "
            "changes their mind about something already filed. Only title and raw_request are "
            "editable here; a card's spec/acceptance criteria are written later by the PM column, "
            "not by this conversation, and won't retroactively change if the card has already moved "
            "past PM."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "card_id": {"type": "string", "description": "The card to edit."},
                "title": {"type": "string", "description": "New title."},
                "raw_request": {"type": "string", "description": "New request text."},
            },
            "required": ["card_id", "title", "raw_request"],
        },
    },
}

SEARCH_TICKETS = {
    "type": "function",
    "function": {
        "name": "search_tickets",
        "description": (
            "Search or list cards already on this project's board — use this to find the card a "
            "human is referring to (by title or description keywords) before discussing or editing "
            "it, rather than guessing its id. Matches against title and request text, "
            "case-insensitive. Leave query empty to list the most recently created cards instead of "
            "searching. Returns a short line per match (id, title, column, priority, status) — call "
            "get_ticket with a specific id for the full request text before editing it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search text to match against title/request, empty to list recent cards.",
                },
                "include_archived": {
                    "type": "boolean",
                    "description": "Include archived cards in results. Defaults to false.",
                },
            },
            "required": [],
        },
    },
}

GET_TICKET = {
    "type": "function",
    "function": {
        "name": "get_ticket",
        "description": (
            "Fetch one card's full detail by id — title, request text, spec (if PM has already "
            "written one), column, priority, and status. Use this after search_tickets to see "
            "exactly what a card currently says before discussing or editing it."
        ),
        "parameters": {
            "type": "object",
            "properties": {"card_id": {"type": "string", "description": "The card to fetch."}},
            "required": ["card_id"],
        },
    },
}

# Project chat (agent/chat.py) — read-only browsing plus search_tickets/get_ticket/
# create_ticket/update_ticket, all ordinary (non-terminal) tools: a chat turn ends when the
# model replies with no further tool calls, not by calling a designated "done" tool, so the
# conversation stays open after a card is looked up, filed, or edited.
CHAT_TOOLS = [
    READ_FILE,
    LIST_FILES,
    GLOB_FILES,
    GREP_FILES,
    SEARCH_TICKETS,
    GET_TICKET,
    CREATE_TICKET,
    UPDATE_TICKET,
]

APPROVE = {
    "type": "function",
    "function": {
        "name": "approve",
        "description": (
            "Approve the implementation and advance the card to Deployer. Your most recent bash "
            "call must be exactly the project's configured test command, must have exited 0, and "
            "no file may have changed since (write_file/edit_file/bash) — this is checked "
            "server-side, so claiming success without a fresh, exact, passing run will be rejected."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "string",
                    "description": (
                        "Brief notes for the audit log — not just that it passed, but how it went: "
                        "clean, or did something about this card make it slower or trickier to verify "
                        "than usual? A later pass mines these for recurring friction."
                    ),
                }
            },
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
                    "description": (
                        "Detailed, itemized feedback for the Developer — this is the only record of "
                        "your review they'll see, not your exploration, so it has to stand on its own. "
                        "One item per problem: the exact file(s) affected, what's wrong or missing "
                        "there specifically, and — for test failures — the actual failure output, not "
                        "just a count. 'Old test files not deleted, 4 tests not converted, missing "
                        "README, 4 test failures' is a compressed inventory, not feedback — it leaves "
                        "the Developer re-discovering everything you already found. Write what you'd "
                        "want handed to you if you were about to fix this yourself."
                    ),
                },
                "summary": {"type": "string", "description": "A one-line summary for the audit log."},
            },
            "required": ["feedback", "summary"],
        },
    },
}

TESTER_TOOLS = [READ_FILE, GREP_FILES, FETCH_DOCS, BASH, WRITE_FILE, EDIT_FILE, APPROVE, REQUEST_CHANGES]
TESTER_TERMINAL_TOOLS = ("approve", "request_changes")

REVIEW_DIFF = {
    "type": "function",
    "function": {
        "name": "review_diff",
        "description": (
            "Show the diff of everything this card's branch has changed relative to the "
            "project's default branch — the actual change under review. Unlike git_diff, this is not "
            "limited to uncommitted changes. Very large diffs are truncated — if that happens, use "
            "bash to diff a specific path (e.g. `git diff <ref>...HEAD -- path/to/file`) instead."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

REVIEWER_APPROVE = {
    "type": "function",
    "function": {
        "name": "approve",
        "description": (
            "Approve this implementation from a code-review perspective and advance the card to "
            "Deployer. The Tester has already verified the tests pass — your job is a second, "
            "independent opinion on the diff itself: design, security, maintainability, and whether "
            "it actually satisfies the spec and acceptance criteria. Call this only once you've "
            "actually read the diff and have no unresolved concerns."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "string",
                    "description": (
                        "Brief notes for the audit log — not just that it passed, but how it went: "
                        "clean, or did something about this card make it slower or trickier to verify "
                        "than usual? A later pass mines these for recurring friction."
                    ),
                }
            },
            "required": ["notes"],
        },
    },
}

REVIEWER_REQUEST_CHANGES = {
    "type": "function",
    "function": {
        "name": "request_changes",
        "description": (
            "Send the card back to the Developer with specific, actionable review feedback — a design "
            "problem, a security issue, unmaintainable code, or a gap versus the spec/acceptance "
            "criteria that passing tests didn't catch."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "feedback": {
                    "type": "string",
                    "description": "Specific, actionable feedback for the Developer — what's wrong and why.",
                },
                "summary": {"type": "string", "description": "A one-line summary for the audit log."},
            },
            "required": ["feedback", "summary"],
        },
    },
}

# Read-only on purpose: Reviewer can inspect the diff and the surrounding code but
# cannot edit anything or run commands — unlike Tester (which can strengthen tests)
# its whole job is to judge the diff as it stands, not to fix it. That's what makes
# it a genuine independent check rather than the same actor wearing a different hat.
# FETCH_DOCS doesn't break that: it can't touch the repo, only ground a judgment
# (e.g. "this doesn't match the real API contract") in the actual current docs.
REVIEWER_TOOLS = [
    READ_FILE,
    LIST_FILES,
    GLOB_FILES,
    GREP_FILES,
    FETCH_DOCS,
    REVIEW_DIFF,
    REVIEWER_APPROVE,
    REVIEWER_REQUEST_CHANGES,
]
REVIEWER_TERMINAL_TOOLS = ("approve", "request_changes")

RUN_DEPLOY = {
    "type": "function",
    "function": {
        "name": "run_deploy",
        "description": (
            "Merge this card's branch into the default branch, push it, and run the project's "
            "configured deploy command. Takes no arguments — the merge, push, and deploy command are "
            "fixed by project configuration, not by you. This always ends your turn: on success, on "
            "an ordinary failure (push rejected, deploy command failed), and on a merge conflict too "
            "— a conflict is not yours to resolve, the card is automatically sent back to Developer "
            "to reconcile against the current default branch."
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
            "Give up on this deploy attempt and leave it for a human, instead of guessing — e.g. the "
            "project's deploy configuration itself looks wrong, or something about this specific "
            "deploy genuinely shouldn't be attempted unsupervised. (A merge conflict is handled for "
            "you automatically — you don't need this tool for that.) This ends your turn."
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


def deployer_tools(mode: DeployMode) -> list[dict]:
    """auto_main gets read tools and both terminal tools (run_deploy, abandon_deploy)
    — no editing tools, since a merge conflict is no longer resolved in-place here
    (see sandbox/deploy_runner.py). pr_to_operator never merges, so it only needs
    read tools plus its own single terminal tool — the model never sees a tool it
    can't use."""
    if mode == DeployMode.AUTO_MAIN:
        return [*_DEPLOYER_READ_TOOLS, RUN_DEPLOY, ABANDON_DEPLOY]
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
