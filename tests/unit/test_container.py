import sys
from types import SimpleNamespace

import pytest

from built.sandbox.container import DockerCommandExecutor, DockerDaemonAccessError


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
