---
name: built-api
description: Drive a running `built` server over its JSON HTTP API with curl — create/inspect projects, file and manage cards, read the board, watch a card's event transcript, manage LLM endpoint configs. Use whenever asked to interact with `built` itself over HTTP rather than by reading/editing its source — e.g. "create a project in built", "file a card for X", "check the board", "retry that blocked card", "add an endpoint config", "watch this card's progress".
---

# Driving `built` over HTTP

`built` exposes a JSON API under `/api/v1` (FastAPI, see `src/built/api/`). This skill is
a curl cookbook for driving it directly — creating projects, filing cards, polling the
board, retrying stuck cards, managing LLM endpoint configs — without touching Python.

This is about operating a *running instance* of built. If the task is instead to modify
built's own source code, ignore this skill and just edit the code as usual.

## Prerequisites

- The server must be running and reachable: `uvicorn built.main:app --reload` (default
  `http://127.0.0.1:8000`). If a request fails to connect, check the server is up before
  debugging further.
- All **mutating** endpoints (POST/PATCH/PUT/DELETE) require an `X-API-Key` header
  matching the server's `BUILT_API_KEY` env var. If that var isn't set on the server,
  mutating endpoints return `503` unconditionally — a wrong or missing key returns `401`.
- **Read** endpoints (GET) take no auth.
- Ask the user for the base URL and API key if not already known — don't guess a
  deployed URL, and don't print the key back verbatim in output.

Set these once per session:

```bash
BUILT_URL="http://127.0.0.1:8000"
BUILT_KEY="$BUILT_API_KEY"   # or whatever the user gives you
```

Every example below assumes those two vars. Pipe responses through `jq` for readability.

## Health

```bash
curl -s "$BUILT_URL/healthz"                # process is up
curl -s "$BUILT_URL/readyz"                 # DB is reachable too
```

## Enums you'll need

- `Priority`: `high` | `normal` | `low`
- `Column`: `pm` | `developer` | `tester` | `reviewer` | `deployer`
- `LifecycleState` (read-only, on `CardOut.lifecycle_state`): `active` | `blocked` | `done` | `failed`
- `DeployMode`: `auto_main` | `pr_to_operator`
- `DeployKind`: `none` | `script` | `command` | `webhook`
- `ActivityKind` (curation passes): `bug_sweep` | `opportunity_brainstorm` | `polish_review` | `stay_dry` | `agents_md`

## Projects

Create a project (the repo `built` will work in — must be a URL it can `git clone`):

```bash
curl -s -X POST "$BUILT_URL/api/v1/projects" \
  -H "X-API-Key: $BUILT_KEY" -H "Content-Type: application/json" \
  -d '{
    "name": "My Project",
    "overarching_goal": "Short description of what this project is for",
    "repo_remote_url": "git@github.com:org/repo.git",
    "default_branch": "main"
  }' | jq
```

Optional fields on create/update: `sandbox_image`, `test_command`, `max_revisions`,
`max_deploy_attempts`, `max_iterations_per_run`, `max_tokens`, and per-role prompt guidance
(`pm_guidance`, `developer_guidance`, `tester_guidance`, `reviewer_guidance`, `deployer_guidance`)
— free text appended to that role's system prompt for this project only; `pm_guidance` also
applies to curation passes and project chat.

```bash
curl -s "$BUILT_URL/api/v1/projects" | jq                       # list (add ?include_archived=true)
curl -s "$BUILT_URL/api/v1/projects/$PROJECT_ID" | jq            # get one

curl -s -X PATCH "$BUILT_URL/api/v1/projects/$PROJECT_ID" \
  -H "X-API-Key: $BUILT_KEY" -H "Content-Type: application/json" \
  -d '{"test_command": "pytest"}' | jq

curl -s -X DELETE "$BUILT_URL/api/v1/projects/$PROJECT_ID" -H "X-API-Key: $BUILT_KEY"   # archive (204)
curl -s -X POST "$BUILT_URL/api/v1/projects/$PROJECT_ID/pause"  -H "X-API-Key: $BUILT_KEY" | jq
curl -s -X POST "$BUILT_URL/api/v1/projects/$PROJECT_ID/resume" -H "X-API-Key: $BUILT_KEY" | jq
```

Deploy config (how Deployer ships an approved card):

```bash
curl -s -X PUT "$BUILT_URL/api/v1/projects/$PROJECT_ID/deploy-config" \
  -H "X-API-Key: $BUILT_KEY" -H "Content-Type: application/json" \
  -d '{
    "mode": "pr_to_operator",
    "kind": "command",
    "command": "npm run deploy",
    "timeout_seconds": 600,
    "github_token_ref": "GH_PAT"
  }' | jq
```

`github_token_ref` / `env_var_refs` are env var *names* the deploy runner reads at
deploy time, never raw secrets — the value must already be set in built's own
environment, not sent through this API.

Trigger a curation pass (fire-and-forget; new cards, if any, show up on the board):

```bash
curl -s -X POST "$BUILT_URL/api/v1/projects/$PROJECT_ID/curate/bug_sweep" -H "X-API-Key: $BUILT_KEY" | jq
```
Returns `202 {"status": "started"}` immediately, or `409` if that kind is already running
for the project. Poll the board to see results.

## Board

```bash
curl -s "$BUILT_URL/api/v1/projects/$PROJECT_ID/board" | jq
```
Returns `{pm: [...], developer: [...], tester: [...], reviewer: [...], deployer: [...]}`,
each a list of `CardOut`. Add `?include_archived=true` to include archived cards.

## Cards

File a new card (this is how you hand `built` a feature request or bug report — it
enters the PM column):

```bash
curl -s -X POST "$BUILT_URL/api/v1/projects/$PROJECT_ID/cards" \
  -H "X-API-Key: $BUILT_KEY" -H "Content-Type: application/json" \
  -d '{
    "title": "Short title",
    "raw_request": "Full description of the request, as a human would write it",
    "priority": "normal"
  }' | jq
```

```bash
curl -s "$BUILT_URL/api/v1/projects/$PROJECT_ID/cards" | jq       # list (add ?include_archived=true)
curl -s "$BUILT_URL/api/v1/cards/$CARD_ID" | jq                   # get one
```

Mutate a card:

```bash
# Un-stick a blocked/failed card with a fresh safety-valve budget; note (optional) is
# surfaced to the next column visit, then cleared.
curl -s -X POST "$BUILT_URL/api/v1/cards/$CARD_ID/retry" \
  -H "X-API-Key: $BUILT_KEY" -H "Content-Type: application/json" \
  -d '{"note": "Try again, the flaky test was unrelated"}' | jq

# Drop a note in for whichever agent is (or will next be) working this card — no
# state restriction like retry has, and doesn't touch any safety-valve counter.
# Reaches an actively-running visit within one iteration; otherwise it's picked
# up at the start of the next one.
curl -s -X POST "$BUILT_URL/api/v1/cards/$CARD_ID/nudge" \
  -H "X-API-Key: $BUILT_KEY" -H "Content-Type: application/json" \
  -d '{"note": "Skip the payments module, that one is handled elsewhere"}' | jq

curl -s -X POST "$BUILT_URL/api/v1/cards/$CARD_ID/cancel"    -H "X-API-Key: $BUILT_KEY" | jq
curl -s -X POST "$BUILT_URL/api/v1/cards/$CARD_ID/archive"   -H "X-API-Key: $BUILT_KEY" | jq
curl -s -X POST "$BUILT_URL/api/v1/cards/$CARD_ID/unarchive" -H "X-API-Key: $BUILT_KEY" | jq

curl -s -X POST "$BUILT_URL/api/v1/cards/$CARD_ID/priority" \
  -H "X-API-Key: $BUILT_KEY" -H "Content-Type: application/json" \
  -d '{"priority": "high"}' | jq

curl -s -X PATCH "$BUILT_URL/api/v1/cards/$CARD_ID" \
  -H "X-API-Key: $BUILT_KEY" -H "Content-Type: application/json" \
  -d '{"title": "New title", "raw_request": "Updated request text"}' | jq
```

`retry`/`cancel` return `409` if the card isn't in a state that allows it (e.g.
cancelling a card that's already `done`).

## Watching a card work

`column-visits` gives one row per column attempt (with `outcome`); `events` gives the
full LLM/tool transcript, paginated by `seq`:

```bash
curl -s "$BUILT_URL/api/v1/cards/$CARD_ID/column-visits" | jq

# First page
curl -s "$BUILT_URL/api/v1/cards/$CARD_ID/events?limit=200" | jq

# Poll for new events since the last-seen seq (events are append-only)
curl -s "$BUILT_URL/api/v1/cards/$CARD_ID/events?since_seq=$LAST_SEQ&limit=200" | jq
```

To watch a card live, poll `events` on an interval and track the highest `seq` seen;
`type` is one of `llm_request`/`llm_response`/`tool_call`/`tool_result`/`transition`/
`system_note`/`error`/`compaction`.

## Endpoint configs (LLM backends)

```bash
curl -s -X POST "$BUILT_URL/api/v1/endpoint-configs" \
  -H "X-API-Key: $BUILT_KEY" -H "Content-Type: application/json" \
  -d '{
    "base_url": "https://api.example.com/v1",
    "model": "some-model",
    "project_id": null,
    "role": null,
    "priority": 0,
    "api_key_ref": "SOME_PROVIDER_API_KEY",
    "supports_tool_calling": true,
    "max_concurrency": 1
  }' | jq
```

`project_id: null` + `role: null` makes it a global fallback used by every
project/column that has no more specific match; scope it by setting either or both.
`api_key_ref` is an env var *name* built reads at call time, never a raw key.

```bash
curl -s "$BUILT_URL/api/v1/endpoint-configs" | jq                                  # all
curl -s "$BUILT_URL/api/v1/endpoint-configs?scope=global" | jq                     # only global
curl -s "$BUILT_URL/api/v1/endpoint-configs?project_id=$PROJECT_ID" | jq

curl -s -X PATCH "$BUILT_URL/api/v1/endpoint-configs/$ENDPOINT_ID" \
  -H "X-API-Key: $BUILT_KEY" -H "Content-Type: application/json" \
  -d '{"enabled": false}' | jq

curl -s -X DELETE "$BUILT_URL/api/v1/endpoint-configs/$ENDPOINT_ID" -H "X-API-Key: $BUILT_KEY"  # 204

# Debug: the effective resolved fallback chain for a (project, column) pair
curl -s "$BUILT_URL/api/v1/projects/$PROJECT_ID/endpoint-chain/developer" | jq
```

## Common recipes

**File a card and watch it move through the board:**
```bash
CARD_ID=$(curl -s -X POST "$BUILT_URL/api/v1/projects/$PROJECT_ID/cards" \
  -H "X-API-Key: $BUILT_KEY" -H "Content-Type: application/json" \
  -d '{"title": "...", "raw_request": "..."}' | jq -r .id)

watch -n 5 "curl -s $BUILT_URL/api/v1/cards/$CARD_ID | jq '{column, lifecycle_state}'"
```

**Find and retry every blocked card in a project:**
```bash
curl -s "$BUILT_URL/api/v1/projects/$PROJECT_ID/board" \
  | jq -r '.[][] | select(.lifecycle_state == "blocked") | .id' \
  | while read -r id; do
      curl -s -X POST "$BUILT_URL/api/v1/cards/$id/retry" -H "X-API-Key: $BUILT_KEY" | jq -c '{id, column, lifecycle_state}'
    done
```

## Notes

- This is a JSON API only — the server-rendered dashboard (`ui/` routers, e.g. project
  chat) is HTML for browsers, not a stable API surface, and isn't covered here.
- All list/board responses embed full `CardOut`/`ProjectOut` objects — no separate
  "get by id after list" round trip needed in most cases.
- `NotFoundError` → `404`; state-machine conflicts (retry/cancel on a terminal or
  already-active card, duplicate curation kind) → `409`; missing/wrong API key on a
  mutating call → `503`/`401` respectively.
