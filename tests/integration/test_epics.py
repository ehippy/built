"""Epics (a Card that stays alive tracking child cards, created via the PM's
define_epic — see tests/integration/test_pm_loop.py for that agent-facing path) and
dependency edges — drives domain.transitions/services.card_service directly, no
agents or HTTP, mirroring test_transitions.py's style."""

from built.db.models import Project
from built.domain import transitions
from built.domain.enums import LifecycleState
from built.services import card_service, project_service


async def _make_project(session, **overrides) -> Project:
    defaults = {
        "name": f"epics-{overrides.get('name', 'x')}",
        "overarching_goal": "Ship a thing.",
        "repo_remote_url": "https://example.invalid/repo.git",
    }
    defaults.update(overrides)
    return await project_service.create_project(session, **defaults)


# --- Completion propagation: domain/transitions.py's _maybe_complete_epic --------


async def test_epic_completes_once_every_child_is_done(db_session):
    project = await _make_project(db_session, name="propagation")
    epic = await card_service.create_card(db_session, project.id, title="Epic", raw_request="r")
    c1 = await card_service.create_card(db_session, project.id, title="C1", raw_request="r")
    c2 = await card_service.create_card(db_session, project.id, title="C2", raw_request="r")
    await db_session.commit()
    await card_service.link_epic_child(db_session, parent_card_id=epic.id, child_card_id=c1.id)
    await card_service.link_epic_child(db_session, parent_card_id=epic.id, child_card_id=c2.id)
    await db_session.commit()

    visit1 = await transitions.start_visit(db_session, c1)
    await transitions.complete_deployer_visit(db_session, c1, visit1, success=True, summary="s")
    assert epic.lifecycle_state == LifecycleState.ACTIVE

    visit2 = await transitions.start_visit(db_session, c2)
    await transitions.complete_deployer_visit(db_session, c2, visit2, success=True, summary="s")

    assert epic.lifecycle_state == LifecycleState.DONE
    events = await card_service.list_events(db_session, epic.id)
    auto_completed = [e for e in events if e.payload.get("action") == "epic_auto_completed"]
    assert len(auto_completed) == 1
    assert auto_completed[0].payload["completed_via_child"] == c2.id


async def test_an_archived_child_does_not_block_epic_completion(db_session):
    project = await _make_project(db_session, name="archived-child")
    epic = await card_service.create_card(db_session, project.id, title="Epic", raw_request="r")
    c1 = await card_service.create_card(db_session, project.id, title="C1", raw_request="r")
    c2 = await card_service.create_card(db_session, project.id, title="C2", raw_request="r")
    await db_session.commit()
    await card_service.link_epic_child(db_session, parent_card_id=epic.id, child_card_id=c1.id)
    await card_service.link_epic_child(db_session, parent_card_id=epic.id, child_card_id=c2.id)
    await card_service.archive_card(db_session, c2.id)
    await db_session.commit()

    visit1 = await transitions.start_visit(db_session, c1)
    await transitions.complete_deployer_visit(db_session, c1, visit1, success=True, summary="s")

    assert epic.lifecycle_state == LifecycleState.DONE


async def test_a_failed_child_means_the_epic_never_auto_completes(db_session):
    project = await _make_project(db_session, name="failed-child")
    epic = await card_service.create_card(db_session, project.id, title="Epic", raw_request="r")
    c1 = await card_service.create_card(db_session, project.id, title="C1", raw_request="r")
    c2 = await card_service.create_card(db_session, project.id, title="C2", raw_request="r")
    await db_session.commit()
    await card_service.link_epic_child(db_session, parent_card_id=epic.id, child_card_id=c1.id)
    await card_service.link_epic_child(db_session, parent_card_id=epic.id, child_card_id=c2.id)
    project_row = await project_service.get_project(db_session, project.id)
    await db_session.commit()

    for _ in range(project_row.max_deploy_attempts):
        visit = await transitions.start_visit(db_session, c1)
        await transitions.complete_deployer_visit(db_session, c1, visit, success=False, summary="fail")
    assert c1.lifecycle_state == LifecycleState.FAILED

    visit2 = await transitions.start_visit(db_session, c2)
    await transitions.complete_deployer_visit(db_session, c2, visit2, success=True, summary="s")

    assert epic.lifecycle_state == LifecycleState.ACTIVE


async def test_confirm_ci_passed_also_propagates_to_the_epic(db_session):
    """complete_deployer_visit and confirm_ci_passed are two independent
    lifecycle_state=DONE sites, not routed through one shared function — each
    needs its own _maybe_complete_epic call, verified separately."""
    project = await _make_project(db_session, name="ci-passed")
    epic = await card_service.create_card(db_session, project.id, title="Epic", raw_request="r")
    c1 = await card_service.create_card(db_session, project.id, title="C1", raw_request="r")
    await db_session.commit()
    await card_service.link_epic_child(db_session, parent_card_id=epic.id, child_card_id=c1.id)
    await db_session.commit()

    await transitions.confirm_ci_passed(db_session, c1, note="green")

    assert epic.lifecycle_state == LifecycleState.DONE


async def test_ci_failure_reopens_the_child_without_completing_the_epic(db_session):
    """Unlike confirm_ci_passed, a CI failure reopens the child card back to
    Developer (ACTIVE) rather than DONE — the epic must not auto-complete on a
    child whose shipped work just turned out to be broken."""
    project = await _make_project(db_session, name="ci-failed")
    epic = await card_service.create_card(db_session, project.id, title="Epic", raw_request="r")
    c1 = await card_service.create_card(db_session, project.id, title="C1", raw_request="r")
    await db_session.commit()
    await card_service.link_epic_child(db_session, parent_card_id=epic.id, child_card_id=c1.id)
    await db_session.commit()

    await transitions.reopen_after_ci_failure(db_session, c1, project, feedback="red")

    assert c1.lifecycle_state == LifecycleState.ACTIVE
    assert epic.lifecycle_state == LifecycleState.ACTIVE


async def test_maybe_complete_epic_guards_against_zero_non_archived_siblings(db_session):
    """_maybe_complete_epic's own docstring: zero non-archived siblings left
    doesn't count as "all done" — nothing to judge by. Unreachable through the
    three real DONE call sites (the triggering card itself is always in that set,
    and it isn't archived at the moment it goes DONE), so this calls the helper
    directly to verify the guard holds regardless."""
    project = await _make_project(db_session, name="zero-siblings")
    epic = await card_service.create_card(db_session, project.id, title="Epic", raw_request="r")
    c1 = await card_service.create_card(db_session, project.id, title="C1", raw_request="r")
    await db_session.commit()
    await card_service.link_epic_child(db_session, parent_card_id=epic.id, child_card_id=c1.id)
    await card_service.archive_card(db_session, c1.id)
    await db_session.commit()

    await transitions._maybe_complete_epic(db_session, c1)

    assert epic.lifecycle_state == LifecycleState.ACTIVE


# --- Dependency edges: services/card_service.py's add_dependency -----------------


async def test_add_dependency_rejects_a_cycle(db_session):
    project = await _make_project(db_session, name="cycle")
    a = await card_service.create_card(db_session, project.id, title="A", raw_request="r")
    b = await card_service.create_card(db_session, project.id, title="B", raw_request="r")
    c = await card_service.create_card(db_session, project.id, title="C", raw_request="r")
    await db_session.commit()
    await card_service.add_dependency(db_session, card_id=a.id, depends_on_card_id=b.id)
    await card_service.add_dependency(db_session, card_id=b.id, depends_on_card_id=c.id)

    try:
        await card_service.add_dependency(db_session, card_id=c.id, depends_on_card_id=a.id)
        raise AssertionError("expected a cycle ValueError")
    except ValueError as exc:
        assert "cycle" in str(exc)


async def test_add_dependency_rejects_self_dependency(db_session):
    project = await _make_project(db_session, name="self-dep")
    a = await card_service.create_card(db_session, project.id, title="A", raw_request="r")
    await db_session.commit()

    try:
        await card_service.add_dependency(db_session, card_id=a.id, depends_on_card_id=a.id)
        raise AssertionError("expected a ValueError")
    except ValueError as exc:
        assert "itself" in str(exc)


async def test_add_dependency_rejects_cross_project_edges(db_session):
    project_a = await _make_project(db_session, name="cross-a")
    project_b = await _make_project(db_session, name="cross-b")
    a = await card_service.create_card(db_session, project_a.id, title="A", raw_request="r")
    b = await card_service.create_card(db_session, project_b.id, title="B", raw_request="r")
    await db_session.commit()

    try:
        await card_service.add_dependency(db_session, card_id=a.id, depends_on_card_id=b.id)
        raise AssertionError("expected a ValueError")
    except ValueError as exc:
        assert "same project" in str(exc)


async def test_add_dependency_is_idempotent(db_session):
    project = await _make_project(db_session, name="idempotent")
    a = await card_service.create_card(db_session, project.id, title="A", raw_request="r")
    b = await card_service.create_card(db_session, project.id, title="B", raw_request="r")
    await db_session.commit()

    first = await card_service.add_dependency(db_session, card_id=a.id, depends_on_card_id=b.id)
    second = await card_service.add_dependency(db_session, card_id=a.id, depends_on_card_id=b.id)

    assert first.id == second.id
    deps = await card_service.list_dependencies(db_session, a.id)
    assert [d.id for d in deps] == [b.id]


async def test_remove_dependency(db_session):
    project = await _make_project(db_session, name="remove-dep")
    a = await card_service.create_card(db_session, project.id, title="A", raw_request="r")
    b = await card_service.create_card(db_session, project.id, title="B", raw_request="r")
    await db_session.commit()
    await card_service.add_dependency(db_session, card_id=a.id, depends_on_card_id=b.id)

    await card_service.remove_dependency(db_session, card_id=a.id, depends_on_card_id=b.id)

    assert await card_service.list_dependencies(db_session, a.id) == []
    # No-op, not an error, if it's already gone.
    await card_service.remove_dependency(db_session, card_id=a.id, depends_on_card_id=b.id)


# --- Epic nesting guards: services/card_service.py's link_epic_child -------------


async def test_link_epic_child_rejects_an_epic_parent_as_a_child(db_session):
    project = await _make_project(db_session, name="nest-1")
    epic = await card_service.create_card(db_session, project.id, title="Epic", raw_request="r")
    child = await card_service.create_card(db_session, project.id, title="C", raw_request="r")
    other = await card_service.create_card(db_session, project.id, title="O", raw_request="r")
    await db_session.commit()
    await card_service.link_epic_child(db_session, parent_card_id=epic.id, child_card_id=child.id)

    try:
        await card_service.link_epic_child(db_session, parent_card_id=other.id, child_card_id=epic.id)
        raise AssertionError("expected a ValueError")
    except ValueError as exc:
        assert "already an epic parent" in str(exc)


async def test_link_epic_child_rejects_an_epic_child_as_a_parent(db_session):
    project = await _make_project(db_session, name="nest-2")
    epic = await card_service.create_card(db_session, project.id, title="Epic", raw_request="r")
    child = await card_service.create_card(db_session, project.id, title="C", raw_request="r")
    grandchild = await card_service.create_card(db_session, project.id, title="G", raw_request="r")
    await db_session.commit()
    await card_service.link_epic_child(db_session, parent_card_id=epic.id, child_card_id=child.id)

    try:
        await card_service.link_epic_child(
            db_session, parent_card_id=child.id, child_card_id=grandchild.id
        )
        raise AssertionError("expected a ValueError")
    except ValueError as exc:
        assert "already an epic child" in str(exc)


async def test_link_epic_child_rejects_a_child_belonging_to_two_epics(db_session):
    project = await _make_project(db_session, name="nest-3")
    epic_a = await card_service.create_card(db_session, project.id, title="EpicA", raw_request="r")
    epic_b = await card_service.create_card(db_session, project.id, title="EpicB", raw_request="r")
    child = await card_service.create_card(db_session, project.id, title="C", raw_request="r")
    await db_session.commit()
    await card_service.link_epic_child(db_session, parent_card_id=epic_a.id, child_card_id=child.id)

    try:
        await card_service.link_epic_child(db_session, parent_card_id=epic_b.id, child_card_id=child.id)
        raise AssertionError("expected a ValueError")
    except ValueError as exc:
        assert "already belongs to epic" in str(exc)
