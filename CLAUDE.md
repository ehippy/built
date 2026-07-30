# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`built` is an agentic software factory: an LLM-staffed Kanban pipeline that turns feature
requests/bug reports into shipped code with no human in the loop by default. Every card moves
through five columns — `PM → Developer → Tester → Reviewer → Deployer` — each staffed by an LLM
agent with its own tools and system prompt. A handful of background loops (Curator, Reviver, CI
watcher, Archiver) keep the board healthy without a human babysitting it.

## Commands

```bash
pip install -e ".[dev]"        # install (Python 3.11+)
cp .env.example .env           # then set BUILT_API_KEY — mutating endpoints 503 without it

uvicorn built.main:app --reload   # run the server (needs a reachable Docker daemon for real card work)

pytest                          # full suite — 200+ tests, real git repos + real sqlite, LLM/Docker faked
pytest tests/unit/test_foo.py::test_bar -v   # single test
pytest tests/unit                            # unit only (pure logic, no I/O)
pytest tests/integration                     # integration only (real git + sqlite, still no real Docker/LLM)

ruff check .                    # lint (this is the only lint/format step; CI runs the same two commands)
```

Tests never need a reachable Docker daemon or LLM endpoint — `tests/unit/fakes.py`
(`ScriptedLLMClient`, `FakeCommandExecutor`) fakes the two things `LLMClient` and
`CommandExecutor` abstract away as Protocols specifically so the agent loop and tool dispatcher
are fully testable without either. `tests/conftest.py` points the app at an isolated temp
sqlite file and wipes every table before each test (autouse `_clean_database` fixture) — tests
never touch `./data/built.db`.

## Architecture

### The pipeline: agent loop + state machine

`agent/loop.py`'s `run_column_visit` is the one agentic loop shared by every column: build
context → call the LLM (`llm/client.py`'s `FallbackLLMClient`, which retries down an ordered
chain of `EndpointConfig`s) → dispatch tool calls (`tools/dispatcher.py`) → repeat until a
*terminal* tool is called or the iteration cap is hit. Commits to `CardEvent` happen after every
LLM response and every tool call, not just at the end, so the dashboard can show a card's
progress mid-run.

Terminal tools (`submit_spec`, `split_into_subtasks`, `submit_for_test`, `approve`,
`request_changes`, `run_deploy`, ...) are *not* handled by the dispatcher — `agent/loop.py`
intercepts them directly and hands off to `domain/transitions.py`, the pure, ORM-only state
machine that owns every column transition and every safety valve (revision cap, deploy-attempt
cap, iteration cap, run-error handling). This split matters: dispatcher tools produce a
`ToolResult` fed back to the model; terminal tools end the run and mutate `Card.column` /
`Card.lifecycle_state`. When adding a new terminal action for a column, you touch three places:
the tool schema (`llm/tool_schemas.py`), the interception + handler wiring (`agent/loop.py`),
and the actual transition (`domain/transitions.py`).

Developer/Tester/Deployer terminal transitions are server-verified, not taken on the model's
say-so: `domain/run_attempts.py` is the source of truth that a claimed "tests pass" or "deploy
succeeded" corresponds to an actual, exact, most-recent, non-stale command run recorded via the
`bash` tool.

### Orchestrator: claiming and background loops

`orchestrator/worker.py` is a single-process asyncio pool (no Celery/Redis at this scale) that
polls for claimable cards and runs one column visit each. `claim_next_card` is an atomic
conditional `UPDATE` checked by rowcount — a drop-in seam for a future
`SELECT ... FOR UPDATE SKIP LOCKED` if this ever needs multi-process scaling. Claim order:
a human's manual `Priority` bless first, then "stop starting, start finishing" (cards closer to
Deployer before cards closer to PM), then recency. Only one card per project may be claimed at
a time — two cards in the same repo would otherwise step on each other's worktree/branch.

Alongside the worker pool, `orchestrator/` runs independently-timed loops, each individually
toggleable via `BUILT_*_ENABLED` (`config.py`):
- **Curator** (`agent/curation.py` + `orchestrator/curator.py`) — periodic read-only passes per
  project that end in `propose_tasks`, filing new cards exactly like a human PM. The category
  registry is `domain/enums.py`'s `ActivityKind`; adding a new curation category means touching
  four places: the enum member (`enums.py`), its focus prompt (`agent/context.py`'s
  `_CURATION_FOCUS`), its UI label (`ui/routers/board.py`'s `_CURATION_LABELS`), and its button
  (`ui/templates/board.html.j2`). The shared prompt template and the `propose_tasks` tool schema
  both enforce that each proposed card is one atomic, independently-workable unit — never a
  bundle of unrelated findings.
- **Reviver** (`agent/reviver.py`) — decides whether to retry blocked/failed cards or leave them
  for a human, with its own bounded retry budget.
- **CI watcher** — polls GitHub's Checks API for `auto_main` deploys that pushed but haven't been
  confirmed green yet; opens a follow-up card on red rather than touching the shipped commit.
- **Archiver** — deterministic (no LLM) sweep that archives `DONE` cards after they've sat idle.

### Sandboxing and trust boundaries

Two independent confinement layers protect the rest of the filesystem/host from an autonomous
tool-calling LLM:
- `tools/base.py`'s `ToolContext.resolve()` confines every read/write tool path argument to one
  card's git worktree — rejects absolute paths, `..` traversal, and symlink escapes. This is the
  *only* protection for tools that run in-process (read/write/grep/glob).
- `sandbox/container.py` runs the `bash` tool inside an ephemeral, locked-down Docker container —
  the one place containerization matters most, since arbitrary shell trivially escapes path
  confinement. `CommandExecutor` is a Protocol for exactly this reason (fakeable in tests).

`sandbox/deploy_runner.py` is a third, separate trust boundary: Deployer's `run_deploy` executes
in the *orchestrator's own process*, never inside the LLM-accessible sandbox, because it's the
only place real credentials (deploy secrets, GitHub PAT) get injected. The LLM never gets shell
access to anything that can see them.

`sandbox/worktree.py`: one bare, service-owned git clone per project; one worktree per card,
created once and reused across every column that card visits.

### Layers and layout

```
src/built/
  domain/        card state machine (transitions.py), enums, run-attempt verification — pure, no I/O
  agent/         the shared agentic loop (loop.py) + per-column system prompts (context.py)
  llm/           LLM client fallback chain + tool schemas (one schema set per column/curation kind)
  tools/         read/write/bash/git tool implementations, path-confined via ToolContext
  sandbox/       Docker execution, git worktrees, the deploy trust boundary
  orchestrator/  worker pool (claiming) + curator/reviver/archiver/ci_watcher background loops
  services/      the DB-facing layer both api/ and ui/ build on — no business logic lives twice
  api/           JSON API (X-API-Key auth via api/deps.py's RequireApiKey)
  ui/            server-rendered dashboard — routers call services/ directly, no self-HTTP-calling
  db/            SQLAlchemy models (db/models.py) and schema setup (create_all + ADDITIVE_COLUMNS)
tests/
  unit/          pure logic, no I/O
  integration/   real git repos + real sqlite; LLM and Docker always faked (tests/unit/fakes.py)
```

No Alembic, but not fully migration-free either — `db/base.py`'s `create_all()` creates any
missing table from current model metadata *and* runs `_add_missing_columns()`, which
`ALTER TABLE ADD COLUMN`s anything listed in `ADDITIVE_COLUMNS` that isn't already present. A
new nullable column on an existing table needs zero manual steps: add the `mapped_column` and
one `(table, column, sql_type)` tuple to `ADDITIVE_COLUMNS`, and it's picked up automatically on
next restart against an already-deployed database — same effort as a brand-new table, which
`create_all()` always handles on its own. What this mechanism can't do: change an existing
column's type, drop a column, or alter constraints on a column that's already there — those
still need a manual one-off script (see `data/migrate_context_window.py` for the pattern, from
before `ADDITIVE_COLUMNS` existed).

### A few conventions worth knowing before editing

- Every enum lives in `domain/enums.py` as a `StrEnum` with a docstring explaining what each
  value means and where it's consumed — read it before adding or reinterpreting a value.
  `Column`/`LifecycleState`/`ActivityKind` specifically are exhaustively iterated in several
  places (`card_service.get_board`'s `{column: [] for column in Column}`,
  `_board_fragment.html.j2`'s hardcoded column loop, `orchestrator/curator.py`'s
  `for kind in ActivityKind`) — a new *value* on one of those three ripples through every such
  loop. `VisitOutcome` has no such pattern anywhere and is cheap to extend. When something needs
  a new "kind of card" that doesn't fit the existing 5-column/4-state shape (e.g. an epic-tracking
  card), prefer a separate table (`EpicLink`) or a plain new column over a new enum value.
- `services/` is the only layer that should touch the DB from `api/`/`ui/` — routers don't run
  raw queries.
- UI templates are `*.html.j2` (not `*.html`), because Starlette/Jinja2 autoescape only fires on
  `.html`/`.htm`/`.xml` by default; `ui/templates.py` force-enables it — don't add a template
  extension that would silently disable escaping.
- A project's GitHub token is stored as an env var *name* (`github_token_ref`), never a raw
  secret in the database.
