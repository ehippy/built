# built

Agentic software factory — an LLM-staffed Kanban pipeline that ships code autonomously.

You give it a repo and a project goal. It turns feature requests and bug reports into
cards, and pushes each card through a five-stage pipeline — spec it, build it, test it,
review it, ship it — with an LLM agent staffing every stage. A handful of background
loops keep the board healthy without a human babysitting it: one grooms the backlog,
one retries stuck cards, one confirms CI actually went green, one archives old work.

## How it works

Every card moves through five columns, each staffed by its own agent role with its own
tools and its own system prompt:

```
PM → Developer → Tester → Reviewer → Deployer
```

- **PM** turns a raw request into a concrete spec with independently checkable
  acceptance criteria.
- **Developer** implements it against a git worktree on the card's own branch, and
  can't hand off to Tester without a real, server-verified passing run of the
  project's test command — not just its own say-so.
- **Tester** verifies the implementation against the acceptance criteria and extends
  the project's standing test suite for anything not already covered. Same
  server-side test-gate as Developer: approving requires a fresh, exact, passing run.
- **Reviewer** is a second, independent opinion — design, security, maintainability,
  real fit to the spec. It has no bash/write/edit tools, so it can only judge the
  diff, not fix it itself.
- **Deployer** merges to the default branch (or opens a PR for a human, depending on
  project config), runs the deploy command, and hands off to the CI watcher if there's
  CI to wait on.

Every column shares the same safety valves: a capped number of Tester/Reviewer
bounce-backs before a card blocks for a human, a capped number of deploy retries, and
a capped number of agent iterations per visit. Nothing is fully unsupervised —
anything that can't converge stops and waits for a person.

Alongside the pipeline, a few deterministic and LLM-driven background loops run on
their own timers:

- **Curator** periodically explores each project (bug sweeps, opportunity
  brainstorms, polish passes) and files new cards, exactly like a human PM would —
  read-only, it never edits the repo directly.
- **Reviver** looks at blocked/failed cards and decides whether to retry them or
  leave them for a human, with its own bounded retry budget.
- **CI watcher** polls GitHub's Checks API for cards whose deploy pushed successfully
  but hasn't been confirmed yet, and opens a follow-up card if CI comes back red
  rather than touching the already-shipped commit.
- **Archiver** clears finished cards off the board after they've sat done for a
  while.

A priority field (high/normal/low) lets a human bless a card as more important than
the rest of the backlog — it's the first thing the orchestrator sorts on when
deciding what to work on next.

## Requirements

- Python 3.11+
- A reachable Docker daemon (the sandboxed `bash` tool runs every shell command in a
  locked-down container — this is the one place containerization matters most, since
  arbitrary shell trivially escapes the path confinement protecting the read/write
  tools)
- At least one OpenAI-tool-calling-compatible LLM endpoint (a hosted API, or a local
  server like vLLM/llama.cpp/Ollama that speaks the OpenAI chat-completions format)

## Quick start

```bash
pip install -e ".[dev]"
cp .env.example .env   # if you don't have one yet — see Configuration below
```

At minimum, set an API key that the mutating endpoints will require (there's no
human-approval gate anywhere in the pipeline, so this is refused by default rather
than silently left open):

```bash
# .env
BUILT_API_KEY=some-long-random-string
```

Start it:

```bash
uvicorn built.main:app --reload
```

Then, from the UI (or the API, with `X-API-Key` set to the value above):

1. **Add a global LLM endpoint** at `/ui/endpoint-configs` — base URL, model name, and
   whether it supports tool calling. Every project falls back to this unless it
   defines its own per-role chain.
2. **Create a project** at `/ui/projects` — a name, an overarching goal (what the
   agents are trying to build), and the git remote to work against.
3. **Configure a test command** in the project's settings (e.g. `pytest -q`,
   `npm test`) — Developer and Tester can't hand off work without one.
4. **Set a deploy config** — `auto_main` merges straight to the default branch and
   runs your deploy command with zero human gate; `pr_to_operator` pushes the
   card's branch and opens a PR for you to merge yourself instead.
5. **File a card** and watch the board. Cards claim automatically once the
   background worker pool is running (it is, by default, as soon as the app starts).

## Configuration

Everything is a `BUILT_`-prefixed environment variable (or a `.env` file in the repo
root); see [`src/built/config.py`](src/built/config.py) for the full list and the
reasoning behind each default. The ones you'll actually touch:

| Variable | Default | Purpose |
| --- | --- | --- |
| `BUILT_API_KEY` | unset | Required for every mutating endpoint. Left unset, they 503 rather than run open. |
| `BUILT_DATA_DIR` | `./data` | Where the sqlite db, managed clones, and worktrees live. |
| `BUILT_DATABASE_URL` | derived from `BUILT_DATA_DIR` | Set this explicitly if you want the database somewhere `BUILT_DATA_DIR` doesn't also govern (e.g. pointing a scratch run at an isolated db without moving clones/worktrees too). |
| `BUILT_LOG_LEVEL` | `INFO` | `WARNING` quiets normal background-loop chatter down to just problems. |
| `BUILT_ORCHESTRATOR_CONCURRENCY` | `1` | How many cards can be worked at once. Raise this only alongside more/faster LLM capacity — a single local model instance rarely usefully serves more than one call at a time. |
| `BUILT_LLM_TIMEOUT_SECONDS` | `300` | Per-call timeout to the LLM. Local single-model servers can be slow under queued load. |
| `BUILT_REVIVER_ENABLED` / `BUILT_CURATOR_ENABLED` / `BUILT_ARCHIVER_ENABLED` / `BUILT_CI_WATCHER_ENABLED` | `true` | Turn off any background loop you don't want running. |

A project's GitHub token (for PR creation and CI-status polling) is configured as an
env var *name* on its deploy config, not a raw secret in the database — e.g. set
`GH_PAT` in the environment and point the project's `github_token_ref` at `"GH_PAT"`.

## Development

```bash
pytest              # 200+ tests, real git repos and a real sqlite db, LLM/Docker faked
ruff check .         # lint
```

CI (`.github/workflows/ci.yml`) runs both on every push and PR.

Optional but recommended: a `pre-push` hook that runs the same two commands before a push
leaves your machine, so a broken push fails locally instead of turning CI red. Enable it once
per clone:

```bash
git config core.hooksPath .githooks
```

## Project layout

```
src/built/
  domain/        card state machine, enums, safety-valve logic — pure, no I/O
  agent/         the shared agentic loop + per-column system prompts
  llm/           LLM client (fallback chains) and tool schemas
  tools/         the actual read/write/bash/git tools an agent can call
  sandbox/       Docker execution, git worktrees, the trusted deploy path
  orchestrator/  worker pool, curator, reviver, archiver, CI watcher
  services/      the DB-facing layer everything above builds on
  api/           JSON API (used by anything scripting against `built`)
  ui/            server-rendered dashboard (the board, card detail, logs)
  db/            SQLAlchemy models and schema setup
tests/
  unit/          pure logic, no I/O
  integration/   real git repos + real sqlite, LLM and Docker faked
```
