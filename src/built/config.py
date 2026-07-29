from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BUILT_", env_file=".env", extra="ignore")

    data_dir: Path = DEFAULT_DATA_DIR
    database_url: str = f"sqlite+aiosqlite:///{DEFAULT_DATA_DIR / 'built.db'}"

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

    # Per-call timeout passed to litellm.acompletion. Local single-model servers can be
    # slow, especially under queued load from the per-endpoint concurrency semaphore —
    # too tight a timeout fails calls that just needed more time, not calls that are
    # actually stuck.
    llm_timeout_seconds: float = 300

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
    # repo directly. Four kinds (ActivityKind): bug_sweep/opportunity_brainstorm/
    # polish_review have no cooldown — every wake is another chance to propose new
    # work, so curator_poll_interval_seconds is the only pacing. agents_md reviews
    # recently closed work and proposes AGENTS.md updates, gated by "anything closed
    # since last run" instead. Each kind is also manually triggerable per project.
    curator_enabled: bool = True
    curator_poll_interval_seconds: float = 1800
    curator_max_iterations: int = 15
    # WIP limit: no kind proposes new cards while the PM column already has this
    # many (or more) sitting in it — with no per-kind cooldown, curation could
    # otherwise pile up backlog far faster than a concurrency-capped orchestrator
    # can ever work through it.
    curator_max_pm_backlog: int = 15


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
