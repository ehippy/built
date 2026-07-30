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
