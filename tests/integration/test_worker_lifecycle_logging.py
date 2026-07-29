"""Task-lifecycle log lines on worker.run_one_card: "claimed" (pickup) and
"finished" (stage completion + outcome) — see logging_config.py. Deliberately
avoids needing a live LLM or real Docker daemon: the "claimed" log fires before
any LLM call, and a connection failure to an unreachable endpoint is caught
inside agent/loop.py's own run_column_visit (fail_visit_with_error), so the visit
still completes its lifecycle normally — it just ends up blocked, which is
exactly the outcome this test asserts on."""

from built.domain.enums import Column, LifecycleState
from built.logging_config import get_logs
from built.orchestrator.worker import claim_next_card, run_one_card
from built.services import card_service, endpoint_service, project_service


async def test_run_one_card_logs_the_claim_and_the_finish(db_session, toy_repo_remote):
    project = await project_service.create_project(
        db_session,
        name="worker-logging",
        overarching_goal="goal",
        repo_remote_url=str(toy_repo_remote),
    )
    await endpoint_service.create_endpoint_config(
        db_session,
        base_url="http://127.0.0.1:1",  # unreachable — connection refused, fails fast
        model="fake-model",
        project_id=project.id,
        role=Column.PM,
    )
    card = await card_service.create_card(db_session, project.id, title="Add a widget", raw_request="r")
    await db_session.commit()

    claimed = await claim_next_card(db_session, "worker-test-1")
    assert claimed is not None
    await db_session.commit()

    prior_logs = get_logs()
    cutoff = prior_logs[-1].seq if prior_logs else 0

    await run_one_card(db_session, claimed)

    new_logs = get_logs(since_seq=cutoff)
    messages = [e.message for e in new_logs]

    claim_lines = [m for m in messages if "claimed card" in m]
    assert len(claim_lines) == 1
    assert card.id in claim_lines[0]
    assert "Add a widget" in claim_lines[0]
    assert "column=pm" in claim_lines[0]

    finish_lines = [m for m in messages if "finished card" in m]
    assert len(finish_lines) == 1
    assert card.id in finish_lines[0]
    # The unreachable endpoint fails the visit — still a real, logged completion
    # with a real outcome, not a silent crash.
    assert claimed.lifecycle_state == LifecycleState.BLOCKED
    assert "lifecycle=blocked" in finish_lines[0]
