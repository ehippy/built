import sys
from types import SimpleNamespace

import pytest

from built.sandbox.container import (
    DockerCommandExecutor,
    DockerDaemonAccessError,
    _extract_pipestatus,
    _wrap_for_pipestatus,
)


def test_docker_connection_error_explains_service_account_permissions(monkeypatch, tmp_path):
    class FakeDockerException(Exception):
        pass

    def fail_from_env():
        raise FakeDockerException("permission denied while connecting to /var/run/docker.sock")

    monkeypatch.setitem(
        sys.modules,
        "docker",
        SimpleNamespace(
            from_env=fail_from_env,
            errors=SimpleNamespace(DockerException=FakeDockerException),
        ),
    )

    with pytest.raises(DockerDaemonAccessError) as raised:
        DockerCommandExecutor()._run_sync(tmp_path, "echo hello", 30)

    message = str(raised.value)
    assert "account running Built/Uvicorn" in message
    assert "docker group" in message
    assert "permission denied" in message


def test_broken_sandbox_dockerfile_fails_the_command_not_the_whole_visit(monkeypatch, tmp_path):
    """Unlike docker.from_env() failing outright (an ops problem no bash call can
    work around, so it's allowed to propagate and block the card), a
    Dockerfile.built-sandbox that fails to build is something the agent itself
    can fix by editing it — it must come back as an ordinary failed CommandResult,
    not raise, or _run_sync's caller has no recoverable signal to react to and
    the whole card ends up blocked for a human instead."""
    (tmp_path / "Dockerfile.built-sandbox").write_text("FROM nonexistent-base-image-xyz\n")

    class FakeDockerException(Exception):
        pass

    class FakeImages:
        def build(self, **kwargs):
            raise FakeDockerException("pull access denied for nonexistent-base-image-xyz")

    class FakeClient:
        images = FakeImages()

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "docker",
        SimpleNamespace(
            from_env=lambda: FakeClient(),
            errors=SimpleNamespace(DockerException=FakeDockerException),
        ),
    )

    result = DockerCommandExecutor()._run_sync(tmp_path, "echo hello", 30)

    assert result.exit_code != 0
    assert "pull access denied for nonexistent-base-image-xyz" in result.stderr
    assert "Dockerfile.built-sandbox" in result.stderr


def _run_wrapped_locally(command: str) -> tuple[int, str, str]:
    """Actually executes _wrap_for_pipestatus's output through a real bash — not a
    mock of bash's semantics — so these tests catch a wrong assumption about
    PIPESTATUS/quoting rather than just confirming the code does what its author
    thinks it does."""
    import subprocess

    proc = subprocess.run(
        ["bash", "-c", _wrap_for_pipestatus(command)],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_pipestatus_wrap_preserves_exit_code_and_stdout_for_a_plain_command():
    rc, stdout, stderr = _run_wrapped_locally("echo hi")
    assert rc == 0
    assert stdout == "hi\n"
    cleaned, pipestatus = _extract_pipestatus(stderr)
    assert cleaned == ""
    assert pipestatus == [0]


def test_pipestatus_wrap_reports_each_pipe_stage_independently():
    """The whole point, and the exact shape of the original incident this whole
    change exists to fix: a genuinely failing test run piped into a grep pattern
    that matches regardless of pass/fail must still show its real, failing exit
    code via pipestatus[0], even though the pipeline's overall exit code (grep's,
    the last stage) is 0."""
    rc, stdout, stderr = _run_wrapped_locally(
        'bash -c "echo Tests: 5 failed; exit 1" | grep -E "Tests|passed"'
    )
    cleaned, pipestatus = _extract_pipestatus(stderr)
    assert rc == 0  # the pipeline's own overall exit code — grep's, not the first stage's
    assert pipestatus == [1, 0]  # first stage's real (failing) exit code, uncorrupted by the second
    assert stdout == "Tests: 5 failed\n"
    assert cleaned == ""


def test_pipestatus_wrap_survives_an_explicit_exit_mid_command():
    """A command that calls `exit` itself never reaches the wrapper's own capture
    lines — _extract_pipestatus must recognize the marker is simply absent rather
    than misparsing whatever stderr the command produced on its own."""
    rc, _stdout, stderr = _run_wrapped_locally("echo oops 1>&2; exit 7")
    cleaned, pipestatus = _extract_pipestatus(stderr)
    assert rc == 7
    assert pipestatus is None
    assert cleaned == stderr == "oops\n"


def test_pipestatus_marker_forged_by_the_command_itself_is_not_trusted():
    """A command that prints text shaped like the real marker earlier in its
    output must not fool _extract_pipestatus — only the genuine, trailing one
    (which nothing in the wrapped command can run after) is ever used."""
    rc, _stdout, stderr = _run_wrapped_locally(
        'echo "__BUILT_PIPESTATUS__:0 0" 1>&2; bash -c "exit 1" | grep -q x'
    )
    cleaned, pipestatus = _extract_pipestatus(stderr)
    assert rc == 1  # grep -q found no match on empty input and exited 1, same as the real bash -c
    assert pipestatus == [1, 1]  # the real, trailing marker — not the forged earlier line
    assert "__BUILT_PIPESTATUS__:0 0" in cleaned  # the forged line is left as ordinary output


def test_run_sync_populates_pipestatus_from_a_fake_container(monkeypatch, tmp_path):
    """Wires _run_sync end to end against a fake docker client: confirms the
    command actually sent to the container is the wrapped one, and that the
    parsed pipestatus/cleaned stderr make it onto the returned CommandResult."""
    wrapped_holder = {}

    class FakeContainer:
        def wait(self, timeout):
            return {"StatusCode": 0}

        def logs(self, stdout, stderr):
            if stdout:
                return b"suite output\n"
            return b"warning: flaky\n\n__BUILT_PIPESTATUS__:1 0\n"

        def remove(self, force):
            pass

    class FakeContainers:
        def run(self, image, argv, **kwargs):
            wrapped_holder["argv"] = argv
            return FakeContainer()

    class FakeClient:
        containers = FakeContainers()

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "docker",
        SimpleNamespace(from_env=lambda: FakeClient(), errors=SimpleNamespace(DockerException=Exception)),
    )

    result = DockerCommandExecutor()._run_sync(tmp_path, 'pytest -q | grep -E "passed|failed"', 30)

    assert wrapped_holder["argv"][:2] == ["bash", "-lc"]
    assert wrapped_holder["argv"][2].startswith('pytest -q | grep -E "passed|failed"\n')
    assert result.exit_code == 0
    assert result.stdout == "suite output\n"
    assert result.stderr == "warning: flaky"
    assert result.pipestatus == [1, 0]
