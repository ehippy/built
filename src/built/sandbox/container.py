"""Executes the `bash` tool inside an ephemeral, locked-down Docker container — the
one place containerization matters most, since arbitrary shell trivially escapes the
path confinement that protects the read/write tools.

CommandExecutor is a Protocol specifically so the dispatcher and agent loop can be
unit-tested against a fake, independent of whether a Docker daemon is reachable.

Known gap: the plan calls for allowing general network egress while specifically
blocking the cloud metadata address (169.254.169.254). Docker has no per-container
"block this one IP, allow the rest" knob without a custom bridge network and
iptables/nftables rules; that isn't implemented yet. `network_disabled` here is only
an all-or-nothing switch — leave it False (egress allowed, for pip/npm installs)
unless a project needs full network isolation instead."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

DEFAULT_IMAGE = "python:3.12-slim"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MEM_LIMIT = "1g"
DEFAULT_CPU_PERIOD = 100_000
DEFAULT_CPU_QUOTA = 100_000  # 1 CPU
DEFAULT_PIDS_LIMIT = 256


class DockerDaemonAccessError(RuntimeError):
    """Docker is unavailable to the account running the application."""


@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class CommandExecutor(Protocol):
    async def run(self, *, worktree: Path, command: str, timeout_seconds: int) -> CommandResult: ...


class DockerCommandExecutor:
    """Runs `command` inside a fresh container per call: worktree bind-mounted
    read-write at /workspace, everything else read-only, no Linux capabilities, no
    root, bounded CPU/memory/PIDs, `--rm` after the call.

    Requires the `docker` package and a reachable Docker daemon at runtime — neither
    is available in every environment (including the one this was developed in), so
    this class is structurally complete per docker-py's documented API but has not
    been exercised against a live daemon. Verify it in an environment with Docker
    before relying on it."""

    def __init__(self, image: str = DEFAULT_IMAGE, *, network_disabled: bool = False):
        self.image = image
        self.network_disabled = network_disabled

    async def run(
        self, *, worktree: Path, command: str, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    ) -> CommandResult:
        # docker-py's client is synchronous/blocking; run it off the event loop
        # thread so one slow container doesn't stall every other card's asyncio tasks.
        return await asyncio.to_thread(self._run_sync, worktree, command, timeout_seconds)

    def _run_sync(self, worktree: Path, command: str, timeout_seconds: int) -> CommandResult:
        import docker

        try:
            client = docker.from_env()
        except docker.errors.DockerException as exc:
            raise DockerDaemonAccessError(
                "Cannot access the Docker daemon. The account running Built/Uvicorn "
                "must be allowed to use Docker (usually by belonging to the docker group); "
                "restart the service after changing its group membership. "
                f"Docker reported: {exc}"
            ) from exc
        container = None
        try:
            container = client.containers.run(
                self.image,
                ["bash", "-lc", command],
                working_dir="/workspace",
                volumes={str(worktree): {"bind": "/workspace", "mode": "rw"}},
                # HOME defaults to a path under the read-only root FS (e.g. /home/node) —
                # package managers that cache there (npm, pip, etc.) fail or silently
                # corrupt writes. Point HOME at the writable /tmp tmpfs instead.
                environment={"HOME": "/tmp"},
                mem_limit=DEFAULT_MEM_LIMIT,
                cpu_period=DEFAULT_CPU_PERIOD,
                cpu_quota=DEFAULT_CPU_QUOTA,
                pids_limit=DEFAULT_PIDS_LIMIT,
                cap_drop=["ALL"],
                network_disabled=self.network_disabled,
                read_only=True,
                tmpfs={"/tmp": ""},
                user="1000:1000",
                detach=True,
            )
            try:
                wait_result = container.wait(timeout=timeout_seconds)
                exit_code = wait_result.get("StatusCode", -1)
                stdout = container.logs(stdout=True, stderr=False).decode(errors="replace")
                stderr = container.logs(stdout=False, stderr=True).decode(errors="replace")
                return CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)
            except Exception:
                container.kill()
                return CommandResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"command timed out after {timeout_seconds}s",
                    timed_out=True,
                )
        finally:
            if container is not None:
                container.remove(force=True)
            client.close()
