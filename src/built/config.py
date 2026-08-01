from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BUILT_", env_file=".env", extra="ignore")

    data_dir: Path = DEFAULT_DATA_DIR
    # Derived from data_dir in _default_database_url below unless BUILT_DATABASE_URL
    # is set explicitly. A bare f-string default here would bake in DEFAULT_DATA_DIR
    # at class-definition time and silently ignore BUILT_DATA_DIR overrides at
    # runtime — confirmed the hard way: pointing BUILT_DATA_DIR at a scratch
    # directory for an isolated test run still wrote straight into the real
    # data/built.db, because this field's old hardcoded default never looked at
    # data_dir at all.
    database_url: str = ""

    # Level for the unified "built" logger (see logging_config.py) — every
    # built.* module logger is a child of it. "INFO" surfaces normal background-
    # loop activity (worker claims, Reviver/Curator/archiver passes); "WARNING"
    # quiets that down to just problems.
    log_level: str = "INFO"

    # v1 security floor: a single shared API key required on mutating endpoints.
    api_key: str | None = None

    # Safety-valve caps (overridable per project).
    default_max_revisions: int = 3
    default_max_deploy_attempts: int = 2
    default_max_iterations_per_run: int = 25
    default_max_tokens: int = 128_000
    default_keep_messages: int = 10

    orchestrator_enabled: bool = True
    # Matched to the single local LLM endpoint's real serial capacity (max_concurrency=1
    # today) — a higher value just lets more projects' cards get claimed than the
    # backend can actually get to before their lease/timeout, piling up queued LLM
    # calls that time out instead of completing. Raise this only alongside more (or
    # faster) LLM capacity.
    orchestrator_concurrency: int = 1
    orchestrator_poll_interval_seconds: float = 1.5

    # How long a claim is valid without being renewed — the safety window that lets
    # a *different* worker recognize a crashed one and take over its card (see
    # orchestrator/worker.claim_next_card's staleness check). agent/loop.run_column_visit
    # renews this every iteration while a visit is genuinely still running, so it
    # only actually expires on a real crash — not on an ordinary long visit. Must
    # comfortably exceed the time a single iteration can take (one llm_timeout_seconds
    # call plus tool execution), or a slow-but-healthy iteration would look expired
    # before it ever gets to renew.
    claim_lease_seconds: int = 600

    # Per-call timeout passed to litellm.acompletion. Local single-model servers can be
    # slow, especially under queued load from the per-endpoint concurrency semaphore —
    # too tight a timeout fails calls that just needed more time, not calls that are
    # actually stuck.
    llm_timeout_seconds: float = 300

    # litellm's own per-call retry (see llm/client.py's module docstring for how this
    # is split from FallbackLLMClient's own next-endpoint fallback). Was fixed at 1 —
    # too low for a single-endpoint setup (the common case: one local LLM server, no
    # fallback chain to fall through to), where a transient blip (connection reset,
    # momentary overload) had nowhere to go but straight to blocking the card. Higher
    # rides out more transient failures at the cost of a genuinely-down endpoint
    # taking longer to finally give up and fail over (or block).
    llm_num_retries: int = 3

    # The Reviver (agent/reviver.py): an autonomous background pass over blocked/failed
    # cards that decides whether to retry them (with a diagnostic note) or leave them
    # for a human. Wakes on its own on a timer — no manual trigger.
    reviver_enabled: bool = True
    reviver_poll_interval_seconds: float = 600
    reviver_max_cards_per_pass: int = 5
    reviver_max_auto_revives: int = 3
    reviver_max_iterations: int = 20

    # The curator (agent/curation.py, orchestrator/curator.py): autonomous
    # background passes, one project at a time, that propose cards — never edit the
    # repo directly. Each ActivityKind is paced by orchestrator/curator.py's
    # _CURATION_INTERVALS (per-kind, in code) — curator_poll_interval_seconds below is
    # only the fallback default for a kind with no entry there. agents_md reviews
    # recently closed work and proposes AGENTS.md updates, gated by "anything closed
    # since last run" *in addition* to its interval. Each kind is also manually
    # triggerable per project, which ignores cadence entirely.
    curator_enabled: bool = True
    curator_poll_interval_seconds: float = 1800
    curator_max_iterations: int = 15
    # WIP limit: no kind proposes new cards while the PM column already has this
    # many (or more) sitting in it — with no per-kind cooldown, curation could
    # otherwise pile up backlog far faster than a concurrency-capped orchestrator
    # can ever work through it.
    curator_max_pm_backlog: int = 15

    # Project chat (agent/chat.py + ui/routers/chat.py): a human brainstorming
    # tickets with an LLM for one project. Purely request-driven — invoked
    # synchronously per human message, no scheduler loop — so no *_enabled/
    # poll_interval pair the way curator/reviver have. Small relative to
    # curator/reviver's iteration caps: a chat turn should resolve in a couple of
    # tool-call exchanges, not an exhaustive unsupervised pass.
    chat_max_tool_iterations: int = 6

    # Auto-archiver: a deterministic sweep (no LLM, unlike Reviver/Curator) that
    # archives DONE cards once they've sat idle past auto_archive_done_after_days —
    # keeps the board from accumulating finished work forever without a human
    # remembering to archive it by hand. Hourly polling is plenty for a day-scale
    # threshold.
    archiver_enabled: bool = True
    archiver_poll_interval_seconds: float = 3600
    auto_archive_done_after_days: int = 7

    # CI watcher: a deterministic sweep (no LLM) that polls GitHub's Checks API for
    # cards whose auto_main deploy pushed successfully but hasn't been confirmed by
    # CI yet (Card.deploying_commit_sha) — see orchestrator/ci_watcher.py. Fast
    # polling relative to the archiver: CI resolution timing is user-visible in a
    # way "eventually archive an old done card" isn't.
    ci_watcher_enabled: bool = True
    ci_watcher_poll_interval_seconds: float = 60
    # How long a commit with zero reported check-runs is treated as "CI hasn't
    # registered the push yet" rather than "this repo genuinely has no CI" —
    # GitHub Actions usually creates check-run objects within seconds, but not
    # always instantly.
    ci_watcher_grace_period_seconds: int = 180
    # How long to keep polling a commit whose checks are still in progress before
    # giving up and blocking the card for a human.
    ci_watcher_timeout_seconds: int = 3600

    @model_validator(mode="after")
    def _default_database_url(self) -> "Settings":
        """Only fills in database_url when it wasn't explicitly provided (env var
        or constructor kwarg) — model_fields_set excludes fields still on their
        class default, so an explicit BUILT_DATABASE_URL always wins over this."""
        if "database_url" not in self.model_fields_set:
            self.database_url = f"sqlite+aiosqlite:///{self.data_dir / 'built.db'}"
        return self


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
