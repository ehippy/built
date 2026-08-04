"""Which roles get which tools is a real safety boundary (Reviewer/Deployer stay
unable to fix things themselves) — worth a direct regression test rather than only
ever being exercised incidentally through a full agent-loop test."""

from built.domain.enums import DeployMode
from built.llm.tool_schemas import DEVELOPER_TOOLS, REVIEWER_TOOLS, TESTER_TOOLS, deployer_tools


def _names(tools: list[dict]) -> set[str]:
    return {t["function"]["name"] for t in tools}


def test_reviewer_has_run_check_but_no_write_or_edit():
    names = _names(REVIEWER_TOOLS)
    assert "run_check" in names
    assert "write_file" not in names
    assert "edit_file" not in names
    assert "bash" not in names


def test_developer_and_tester_have_bash_not_run_check():
    """run_check is specifically for roles that can't otherwise execute anything —
    Developer/Tester already have real bash, so there's no reason to also hand
    them a tool whose whole point is discarding what it does."""
    assert "run_check" not in _names(DEVELOPER_TOOLS)
    assert "run_check" not in _names(TESTER_TOOLS)
    assert "bash" in _names(DEVELOPER_TOOLS)
    assert "bash" in _names(TESTER_TOOLS)


def test_deployer_auto_main_has_no_run_check():
    """In AUTO_MAIN mode the worktree is still just the default branch until
    run_deploy itself merges the card in — there's nothing belonging to this card
    to check yet, so run_check has no honest use here (see deployer_tools)."""
    names = _names(deployer_tools(DeployMode.AUTO_MAIN))
    assert "run_check" not in names
    assert "run_deploy" in names


def test_deployer_pr_to_operator_has_run_check():
    """pr_to_operator's worktree already reflects the card's real branch, so a
    pre-PR sanity check is genuinely meaningful there."""
    names = _names(deployer_tools(DeployMode.PR_TO_OPERATOR))
    assert "run_check" in names
    assert "open_pull_request" in names
