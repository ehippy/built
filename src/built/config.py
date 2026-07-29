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

    # Default safety-valve caps (overridable per project).
    default_max_revisions: int = 3
    default_max_deploy_attempts: int = 2
    default_max_iterations_per_run: int = 25

    orchestrator_enabled: bool = True
    orchestrator_concurrency: int = 4
    orchestrator_poll_interval_seconds: float = 1.5


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
