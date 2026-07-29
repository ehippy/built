from datetime import datetime

from pydantic import BaseModel, ConfigDict

from built.domain.enums import Column, DeployKind, DeployMode, EventType, LifecycleState, VisitOutcome


class ProjectCreate(BaseModel):
    name: str
    overarching_goal: str
    repo_remote_url: str
    default_branch: str = "main"
    sandbox_image: str | None = None
    max_revisions: int | None = None
    max_deploy_attempts: int | None = None
    max_iterations_per_run: int | None = None
    max_tokens: int | None = None


class ProjectUpdate(BaseModel):
    name: str | None = None
    overarching_goal: str | None = None
    default_branch: str | None = None
    sandbox_image: str | None = None
    max_revisions: int | None = None
    max_deploy_attempts: int | None = None
    max_iterations_per_run: int | None = None
    max_tokens: int | None = None


class DeployConfigIn(BaseModel):
    mode: DeployMode = DeployMode.PR_TO_OPERATOR
    kind: DeployKind
    command: str | None = None
    script_path: str | None = None
    webhook_url: str | None = None
    env_var_refs: list[str] = []
    timeout_seconds: int = 600
    github_token_ref: str | None = None


class DeployConfigOut(DeployConfigIn):
    model_config = ConfigDict(from_attributes=True)
    id: str
    project_id: str


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    slug: str
    overarching_goal: str
    repo_remote_url: str
    default_branch: str
    sandbox_image: str | None
    max_revisions: int
    max_deploy_attempts: int
    max_iterations_per_run: int
    max_tokens: int | None
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None
    deploy_config: DeployConfigOut | None = None


class EndpointConfigCreate(BaseModel):
    base_url: str
    model: str
    project_id: str | None = None
    role: Column | None = None
    priority: int = 0
    api_key_ref: str | None = None
    supports_tool_calling: bool = True
    extra_params: dict = {}
    enabled: bool = True
    max_concurrency: int = 1
    context_window: int | None = None


class EndpointConfigUpdate(BaseModel):
    base_url: str | None = None
    model: str | None = None
    role: Column | None = None
    priority: int | None = None
    api_key_ref: str | None = None
    supports_tool_calling: bool | None = None
    extra_params: dict | None = None
    enabled: bool | None = None
    max_concurrency: int | None = None
    context_window: int | None = None


class EndpointConfigOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str | None
    role: Column | None
    priority: int
    base_url: str
    model: str
    api_key_ref: str | None
    supports_tool_calling: bool
    extra_params: dict
    enabled: bool
    max_concurrency: int
    context_window: int | None
    created_at: datetime


class CardCreate(BaseModel):
    title: str
    raw_request: str


class CardRetryIn(BaseModel):
    note: str | None = None


class CardOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    title: str
    raw_request: str
    column: Column
    lifecycle_state: LifecycleState
    branch_name: str | None
    worktree_path: str | None
    spec: str | None
    acceptance_criteria: list[str]
    revision_count: int
    retry_note: str | None
    latest_feedback: str | None
    deploy_attempt_count: int
    deploy_url: str | None
    created_at: datetime
    updated_at: datetime


class BoardOut(BaseModel):
    pm: list[CardOut]
    developer: list[CardOut]
    tester: list[CardOut]
    deployer: list[CardOut]


class CardColumnVisitOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    card_id: str
    column: Column
    attempt_number: int
    started_at: datetime
    ended_at: datetime | None
    outcome: VisitOutcome | None
    endpoint_used: str | None
    summary: str | None


class CardEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    card_id: str
    column_visit_id: str | None
    seq: int
    type: EventType
    payload: dict
    tokens_in: int | None
    tokens_out: int | None
    latency_ms: int | None
    cost_estimate: float | None
    created_at: datetime
