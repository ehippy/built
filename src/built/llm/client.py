"""Wraps litellm behind an internal LLMClient protocol so nothing else in the
codebase imports litellm directly — it stays swappable for a hand-rolled client later
without touching call sites.

Endpoint fallback is a hand-rolled ordered loop, not litellm.Router: retry ownership
is split so each piece has exactly one job. litellm's own per-call retry
(`num_retries`, kept low) owns *transient* failures — timeouts, 5xx, rate limits — on
a single endpoint. This loop owns *falling through* to the next configured endpoint
once litellm gives up on the current one. Router's own fallback configuration would
do something similar, but through a config surface this loop makes explicit instead.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Protocol

import litellm

from built.db.models import EndpointConfig

DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_NUM_RETRIES = 1

# Keyed by (base_url, model) — the physical backend, not the EndpointConfig row —
# so a project-specific and a global EndpointConfig that happen to point at the same
# server share one cap instead of each getting their own and doubling the load it
# actually sees. Module-level and never cleared: a handful of long-lived semaphores
# for the lifetime of the process is negligible, and there's no natural point at
# which an endpoint is known to be permanently gone.
_endpoint_semaphores: dict[tuple[str, str], asyncio.Semaphore] = {}


def _semaphore_for(endpoint: EndpointConfig) -> asyncio.Semaphore:
    key = (endpoint.base_url, endpoint.model)
    semaphore = _endpoint_semaphores.get(key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(max(1, endpoint.max_concurrency))
        _endpoint_semaphores[key] = semaphore
    return semaphore


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResult:
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    endpoint_used: str = ""
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: int | None = None


class LLMClient(Protocol):
    async def complete(self, *, messages: list[dict], tools: list[dict]) -> LLMResult: ...


class AllEndpointsFailedError(Exception):
    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__(f"every endpoint in the fallback chain failed: {errors}")


class FallbackLLMClient:
    """Tries each EndpointConfig in priority order; the first one that responds wins.
    Endpoints not flagged `supports_tool_calling` are dropped at construction time —
    every column relies on tool calls for its terminal actions, so such an endpoint
    can never usefully serve this client regardless of how the call goes."""

    def __init__(self, chain: list[EndpointConfig]):
        usable = [e for e in chain if e.supports_tool_calling]
        if not usable:
            raise ValueError("no usable (tool-calling) endpoints in the resolved fallback chain")
        self._chain = usable

    async def complete(self, *, messages: list[dict], tools: list[dict]) -> LLMResult:
        errors: dict[str, str] = {}
        for endpoint in self._chain:
            try:
                return await self._complete_one(endpoint, messages=messages, tools=tools)
            except Exception as exc:  # noqa: BLE001 — deliberate: fall through to the next endpoint
                errors[f"{endpoint.base_url}::{endpoint.model}"] = repr(exc)
        raise AllEndpointsFailedError(errors)

    async def _complete_one(
        self, endpoint: EndpointConfig, *, messages: list[dict], tools: list[dict]
    ) -> LLMResult:
        # Never let more than endpoint.max_concurrency requests reach this physical
        # backend at once — a local single-model server doesn't parallelize well, and
        # letting concurrent visits pile requests onto it is what actually caused the
        # request-timeout failures this was added to fix. Queued time isn't counted
        # in latency_ms below — that measures the call itself, not time spent waiting.
        async with _semaphore_for(endpoint):
            start = time.monotonic()
            response = await litellm.acompletion(
                model=f"openai/{endpoint.model}",
                api_base=endpoint.base_url,
                api_key=_resolve_api_key(endpoint.api_key_ref),
                messages=messages,
                tools=tools or None,
                num_retries=DEFAULT_NUM_RETRIES,
                timeout=DEFAULT_TIMEOUT_SECONDS,
            )
            latency_ms = int((time.monotonic() - start) * 1000)

        message = response.choices[0].message
        tool_calls = [
            ToolCallRequest(
                id=tc.id, name=tc.function.name, arguments=_parse_arguments(tc.function.arguments)
            )
            for tc in (message.tool_calls or [])
        ]
        usage = getattr(response, "usage", None)
        return LLMResult(
            content=message.content,
            tool_calls=tool_calls,
            endpoint_used=f"{endpoint.base_url}::{endpoint.model}",
            tokens_in=getattr(usage, "prompt_tokens", None),
            tokens_out=getattr(usage, "completion_tokens", None),
            latency_ms=latency_ms,
        )


def _resolve_api_key(api_key_ref: str | None) -> str:
    """`api_key_ref` is an env var *name*, never a raw key — see EndpointConfig.
    litellm treats an empty string as "no credentials" and refuses to even send the
    request (`OpenAIException - Missing credentials`), which breaks self-hosted
    OpenAI-compatible servers (vLLM, llama.cpp-based servers, ...) that don't check
    the key at all — they just need *something* non-empty in the Authorization header."""
    if not api_key_ref:
        return "not-needed"
    return os.environ.get(api_key_ref) or "not-needed"


def _parse_arguments(raw: str) -> dict:
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
