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
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

DEFAULT_IMAGE = "python:3.12-slim"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MEM_LIMIT = "1g"
DEFAULT_CPU_PERIOD = 100_000
DEFAULT_CPU_QUOTA = 100_000  # 1 CPU
DEFAULT_PIDS_LIMIT = 256
# Docker's tmpfs default is "rw,noexec,nosuid,nodev,size=65536k" — fine for package
# managers that only cache files there, but toolchains that compile-and-execute a
# helper into a temp dir (e.g. Swift Package Manager's manifest compiler) fail with
# "posix_spawn: Permission denied" under noexec. "exec" opts back into execution; the
# larger size gives those toolchains room to actually build, still well under
# DEFAULT_MEM_LIMIT since tmpfs usage counts against the container's memory cgroup.
TMPFS_MOUNT_OPTIONS = "exec,size=512m"


class DockerDaemonAccessError(RuntimeError):
    """Docker is unavailable to the account running the application."""


@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    # bash's own $PIPESTATUS right after `command` ran — one exit code per top-level
    # pipeline stage (e.g. [1, 0] for "npm test | grep ..." where the suite actually
    # failed but grep still matched and exited 0). None if the command exited before
    # reaching the capture (an explicit `exit` mid-script, a timeout) — never guess in
    # that case, since domain/run_attempts.py treats None as "can't verify this way."
    pipestatus: list[int] | None = None


class CommandExecutor(Protocol):
    async def run(self, *, worktree: Path, command: str, timeout_seconds: int) -> CommandResult: ...

    async def build_sandbox(self, *, worktree: Path) -> str: ...


# Appended to stderr (never stdout, so it can't get mixed into output a command
# pipes/redirects) by _wrap_for_pipestatus, then stripped back out by
# _extract_pipestatus before the model ever sees it. Always the very last thing
# printed — the wrapper's `exit "$__built_rc"` immediately follows it and nothing in
# the wrapped command can run after that within the same invocation — so even if the
# command's own output happens to contain this exact text earlier, anchoring the
# regex on end-of-string (see _extract_pipestatus) means only the genuine trailing
# one is ever trusted.
_PIPESTATUS_MARKER = "__BUILT_PIPESTATUS__:"
_PIPESTATUS_TRAILER_RE = re.compile(re.escape(_PIPESTATUS_MARKER) + r"([0-9 ]*)\n?\Z")


def _wrap_for_pipestatus(command: str) -> str:
    """Runs `command` exactly as given, then reports bash's $PIPESTATUS for
    whatever pipeline it last ran, without changing the invocation's own exit code
    or its stdout. This only works because these extra lines are appended to the
    *same* script rather than run via a separate `bash file.sh` — PIPESTATUS is
    shell-instance-scoped and wouldn't survive a fork into a child process.

    The `set -- "$?" "${PIPESTATUS[@]}"` line is doing something specific and not
    obvious: capturing $? into its own variable first (`__built_rc=$?`, on its own
    line, then reading ${PIPESTATUS[@]} separately afterward) was the first thing
    tried here, and is wrong — confirmed empirically against both this project's
    actual sandbox base image (python:3.12-slim's bash 5.2) and a local bash: a
    bare assignment is itself a simple command, and bash updates $PIPESTATUS the
    moment *any* simple command finishes (collapsing it to that command's own
    one-element status) — so by the time the following line reads
    ${PIPESTATUS[@]}, `command`'s real per-stage array is already gone, replaced
    by the assignment's own trivial [0]. `set --` sidesteps this because both
    parameter expansions — $? and ${PIPESTATUS[@]} — are evaluated as `set`'s
    arguments *before* `set` itself runs and does its own PIPESTATUS update, so
    both land safely in $1.. as plain positional parameters, immune to whatever
    PIPESTATUS becomes afterward."""
    return (
        f"{command}\n"
        'set -- "$?" "${PIPESTATUS[@]}"\n'
        '__built_rc="$1"; shift\n'
        f'printf "\\n{_PIPESTATUS_MARKER}%s\\n" "$*" 1>&2\n'
        'exit "$__built_rc"\n'
    )


def _extract_pipestatus(stderr: str) -> tuple[str, list[int] | None]:
    """Reverses _wrap_for_pipestatus's stderr trailer. Returns (stderr, None) if the
    marker isn't there at all (the wrapped command exited before reaching it — an
    explicit `exit` mid-script, a timeout) or isn't parseable — never fabricate a
    pipestatus the command didn't actually report."""
    match = _PIPESTATUS_TRAILER_RE.search(stderr)
    if match is None:
        return stderr, None
    codes_text = match.group(1).strip()
    if not codes_text:
        return stderr, None
    try:
        codes = [int(part) for part in codes_text.split()]
    except ValueError:
        return stderr, None
    return stderr[: match.start()].rstrip("\n"), codes


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
        self._built_sandbox_fingerprint: tuple[str, str] | None = None

    async def run(
        self, *, worktree: Path, command: str, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    ) -> CommandResult:
        # docker-py's client is synchronous/blocking; run it off the event loop
        # thread so one slow container doesn't stall every other card's asyncio tasks.
        return await asyncio.to_thread(self._run_sync, worktree, command, timeout_seconds)

    async def build_sandbox(self, *, worktree: Path) -> str:
        return await asyncio.to_thread(self._build_sandbox_sync, worktree)

    def _build_sandbox_sync(self, worktree: Path) -> str:
        dockerfile = worktree / "Dockerfile.built-sandbox"
        if not dockerfile.is_file():
            raise ValueError("Dockerfile.built-sandbox does not exist in the repository root")
        import docker
        tag = f"built-sandbox:{hashlib.sha256(str(worktree.resolve()).encode()).hexdigest()[:16]}"
        try:
            client = docker.from_env()
            try:
                client.images.build(
                    path=str(worktree), dockerfile=dockerfile.name, tag=tag, rm=True, forcerm=True
                )
            finally:
                client.close()
        except docker.errors.DockerException as exc:
            raise DockerDaemonAccessError(f"Unable to build Dockerfile.built-sandbox: {exc}") from exc
        self.image = tag
        self._built_sandbox_fingerprint = (
            str(worktree.resolve()),
            hashlib.sha256(dockerfile.read_bytes()).hexdigest(),
        )
        return tag

    def _ensure_worktree_sandbox(self, worktree: Path) -> None:
        """Prefer a repository's opt-in sandbox image over the configured default."""
        dockerfile = worktree / "Dockerfile.built-sandbox"
        if not dockerfile.is_file():
            return
        fingerprint = (str(worktree.resolve()), hashlib.sha256(dockerfile.read_bytes()).hexdigest())
        if fingerprint != self._built_sandbox_fingerprint:
            self._build_sandbox_sync(worktree)

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
        try:
            self._ensure_worktree_sandbox(worktree)
        except DockerDaemonAccessError as exc:
            # Unlike the docker.from_env() failure above (an ops problem no bash call
            # can work around), a broken Dockerfile.built-sandbox is something the
            # agent itself can fix — editing it and retrying. Reaching this point
            # already proved the daemon itself is reachable (the from_env() call
            # above succeeded), so this is the sandbox build failing, not the
            # daemon being down. Returning an ordinary failed CommandResult instead
            # of letting this propagate keeps it a recoverable tool-level failure,
            # not one that blocks the whole card for a human via
            # run_column_visit's outer exception handler.
            client.close()
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=(
                    f"{exc}\n\nThis command never ran because the project's sandbox image failed to "
                    "build. Fix Dockerfile.built-sandbox (call build_sandbox after editing it to "
                    "verify the fix) before retrying."
                ),
            )
        container = None
        try:
            container = client.containers.run(
                self.image,
                ["bash", "-lc", _wrap_for_pipestatus(command)],
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
                tmpfs={"/tmp": TMPFS_MOUNT_OPTIONS},
                user="1000:1000",
                detach=True,
            )
            try:
                wait_result = container.wait(timeout=timeout_seconds)
                exit_code = wait_result.get("StatusCode", -1)
                stdout = container.logs(stdout=True, stderr=False).decode(errors="replace")
                raw_stderr = container.logs(stdout=False, stderr=True).decode(errors="replace")
                stderr, pipestatus = _extract_pipestatus(raw_stderr)
                return CommandResult(
                    exit_code=exit_code, stdout=stdout, stderr=stderr, pipestatus=pipestatus
                )
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
