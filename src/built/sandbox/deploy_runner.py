"""Deployer's trusted execution path — runs in the orchestrator's own process, never
inside the LLM-accessible Docker sandbox. Real credentials (deploy secrets, GitHub
PAT) are injected only here, exactly like the plan's original design for run_deploy:
the LLM never gets shell access to anything that can see them.

Two independent flows, selected by the project's DeployConfig.mode:
  - auto_main: merge the card's branch into default_branch, push, run the configured
    deploy command. Zero human gate — if CI comes back red on the pushed commit,
    orchestrator/ci_watcher.py does not attempt to revert it (too risky to do
    unsupervised against shared history); it opens a follow-up card instead.
  - pr_to_operator: push the card's branch as-is and open a GitHub PR. Nothing
    merges from the agent's side — orchestrator/pr_watcher.py polls the PR: an
    approving review gets it merged via the merge API (and the card marked done),
    a "changes requested" review bounces the card back to Developer. No deploy
    command runs."""

import asyncio
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from built.db.models import Card, DeployConfig, Project
from built.domain.enums import DeployKind
from built.sandbox.worktree import bare_repo_path
from built.tools import git_tools
from built.tools.git_tools import GitCommandError, run_git

GITHUB_API_BASE = "https://api.github.com"


@dataclass
class DeployRunResult:
    success: bool
    message: str
    url: str | None = None
    # True only for a merge conflict. Unlike the old behavior, the Deployer agent
    # never sees or fixes this itself — it has no bash/test tools to re-verify a
    # resolution, and fixing it here would ship straight to production without ever
    # going back through Tester or Reviewer. The caller (agent/loop.py) bounces the
    # card back to Developer instead (domain/transitions.complete_deployer_visit_conflict).
    # Every other failure (push rejected, deploy command failed) is an ordinary
    # terminal failure, counted against the project's max_deploy_attempts.
    conflict: bool = False
    conflicted_paths: list[str] = field(default_factory=list)
    # The commit actually pushed to default_branch (auto_main only) — what
    # orchestrator/ci_watcher.py polls GitHub's Checks API for afterward.
    commit_sha: str | None = None
    # The GitHub PR number (pr_to_operator only) — what orchestrator/pr_watcher.py
    # polls reviews on and ultimately merges.
    pr_number: int | None = None


async def run_auto_main_deploy(project: Project, card: Card, wt_path: Path) -> DeployRunResult:
    """Merge the card's branch into default_branch, push, then run the configured
    deploy command.

    wt_path is the Deployer's dedicated worktree, freshly reset to the tip of
    default_branch by the caller before this runs. A merge conflict aborts the
    merge and reports conflicted_paths — see DeployRunResult.conflict."""
    # Discard anything sitting in the worktree first — the agent has no file tools
    # of its own here anymore, but this guards the same edge case that motivated
    # it originally: a stray untracked file left over from a previous, unrelated
    # visit blocking the merge with an unrelated git error ("untracked working
    # tree file would be overwritten"), not a real conflict.
    await run_git("reset", "--hard", "HEAD", cwd=wt_path)
    await run_git("clean", "-fd", cwd=wt_path)
    conflicted = await git_tools.attempt_merge(wt_path, card.branch_name, abort_on_conflict=True)
    if conflicted:
        return DeployRunResult(
            success=False,
            conflict=True,
            conflicted_paths=conflicted,
            message=(
                f"Merge conflict in: {', '.join(conflicted)} — sent back to Developer to resolve "
                f"against the current {project.default_branch}."
            ),
        )

    deploy_config = project.deploy_config
    assert deploy_config is not None
    token = os.environ.get(deploy_config.github_token_ref or "") if deploy_config.github_token_ref else None

    try:
        await run_git("push", "origin", f"HEAD:{project.default_branch}", cwd=wt_path, token=token)
    except GitCommandError as exc:
        return DeployRunResult(success=False, message=f"push failed: {exc.stderr.strip()}")

    # Best-effort: if this fails for some reason, the caller just won't have a
    # commit to watch CI on — better to fall through to an ordinary immediate
    # "done" than to fail an otherwise-successful deploy over it.
    try:
        commit_sha = (await run_git("rev-parse", "HEAD", cwd=wt_path)).strip()
    except GitCommandError:
        commit_sha = None

    result = await _run_deploy_command(deploy_config, cwd=wt_path)
    result.commit_sha = commit_sha if result.success else None
    return result


async def _run_deploy_command(deploy_config: DeployConfig, *, cwd) -> DeployRunResult:
    if deploy_config.kind == DeployKind.NONE:
        return DeployRunResult(
            success=True, message="merged and pushed to the default branch — no deploy step configured"
        )
    if deploy_config.kind == DeployKind.COMMAND:
        command = deploy_config.command or ""
    elif deploy_config.kind == DeployKind.SCRIPT:
        command = deploy_config.script_path or ""
    else:
        return await _call_webhook(deploy_config)

    if not command:
        return DeployRunResult(success=False, message=f"no {deploy_config.kind.value} configured")

    env = os.environ.copy()
    for ref in deploy_config.env_var_refs:
        if ref in os.environ:
            env[ref] = os.environ[ref]

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=deploy_config.timeout_seconds)
    except TimeoutError:
        return DeployRunResult(
            success=False, message=f"deploy command timed out after {deploy_config.timeout_seconds}s"
        )

    output = (stdout + stderr).decode(errors="replace")[-4000:]
    if process.returncode == 0:
        return DeployRunResult(success=True, message=f"deploy command succeeded:\n{output}")
    return DeployRunResult(success=False, message=f"deploy command exited {process.returncode}:\n{output}")


async def _call_webhook(deploy_config: DeployConfig) -> DeployRunResult:
    if not deploy_config.webhook_url:
        return DeployRunResult(success=False, message="no webhook_url configured")
    async with httpx.AsyncClient(timeout=deploy_config.timeout_seconds) as client:
        try:
            response = await client.post(deploy_config.webhook_url)
        except httpx.HTTPError as exc:
            return DeployRunResult(success=False, message=f"webhook request failed: {exc}")
    if response.is_success:
        return DeployRunResult(success=True, message=f"webhook returned {response.status_code}")
    return DeployRunResult(
        success=False, message=f"webhook returned {response.status_code}: {response.text[:2000]}"
    )


_GITHUB_URL_PATTERNS = [
    re.compile(r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(\.git)?/?$"),
    re.compile(r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(\.git)?$"),
]


def parse_github_owner_repo(remote_url: str) -> tuple[str, str] | None:
    for pattern in _GITHUB_URL_PATTERNS:
        match = pattern.match(remote_url.strip())
        if match:
            return match.group("owner"), match.group("repo")
    return None


@dataclass
class CheckRun:
    name: str
    status: str  # "queued" | "in_progress" | "completed"
    conclusion: str | None  # only set once status == "completed"
    html_url: str | None = None


FAILING_CONCLUSIONS = frozenset({"failure", "timed_out", "cancelled", "action_required", "stale"})


class CIStatusUnavailableError(Exception):
    """A transient failure fetching check-run status — network blip, GitHub rate
    limit, or the commit not replicated to GitHub's API yet (briefly common right
    after a push). Distinct from fetch_check_runs returning None, which means CI
    status can *never* be determined for this project (no token, not a GitHub
    remote) — a caller should retry later on this error, not treat it the same as
    "confirmed, nothing to wait for"."""


async def fetch_check_runs(project: Project, commit_sha: str) -> list[CheckRun] | None:
    """GitHub's Checks API results for a commit — what GitHub Actions (and most
    other GitHub-integrated CI) reports against. Returns None (not an empty list —
    that's a real, meaningful "zero checks reported" result) when CI status can
    never be determined for this project: not a github.com remote, or no token
    configured. Raises CIStatusUnavailableError for anything that looks transient
    instead, so a caller doesn't mistake "couldn't reach GitHub just now" for "this
    repo has no CI"."""
    owner_repo = parse_github_owner_repo(project.repo_remote_url)
    if owner_repo is None:
        return None
    owner, repo = owner_repo

    deploy_config = project.deploy_config
    token_ref = deploy_config.github_token_ref if deploy_config else None
    token = os.environ.get(token_ref or "") if token_ref else None
    if not token:
        return None

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.get(
                f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{commit_sha}/check-runs",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                },
            )
        except httpx.HTTPError as exc:
            raise CIStatusUnavailableError(f"request failed: {exc}") from exc
    if not response.is_success:
        raise CIStatusUnavailableError(f"GitHub API returned {response.status_code}: {response.text[:500]}")

    return [
        CheckRun(
            name=run["name"],
            status=run["status"],
            conclusion=run.get("conclusion"),
            html_url=run.get("html_url"),
        )
        for run in response.json().get("check_runs", [])
    ]


def _github_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}


def _github_token(project: Project) -> str | None:
    """The project's GitHub PAT, resolved from its env-var *name* (github_token_ref)
    — never a raw secret stored in the database."""
    deploy_config = project.deploy_config
    token_ref = deploy_config.github_token_ref if deploy_config else None
    return os.environ.get(token_ref or "") if token_ref else None


_ACTIONS_JOB_URL_RE = re.compile(r"/actions/runs/\d+/job/(?P<job_id>\d+)")


async def fetch_job_error_lines(project: Project, html_url: str | None, *, max_lines: int = 20) -> str | None:
    """Best-effort: pull the `##[error]` lines out of a failed GitHub Actions job's
    log, the same signal `gh run view --log-failed` surfaces. A CheckRun's
    name/conclusion/html_url alone tells a follow-up card *that* something failed,
    not *why* — without the actual error text, whoever picks up the follow-up card
    has to re-discover the failure from scratch, which is exactly what let a single
    root cause (e.g. one broken workflow file) spawn a long chain of follow-up cards
    that each misdiagnosed it as an unrelated app regression. Returns None (never
    raises) for anything that doesn't pan out — a missing/malformed URL, no token,
    a network error, or a log with no `##[error]` lines — since this is pure
    enrichment and should never block filing the follow-up card it's for."""
    if not html_url:
        return None
    match = _ACTIONS_JOB_URL_RE.search(html_url)
    if not match:
        return None
    owner_repo = parse_github_owner_repo(project.repo_remote_url)
    if owner_repo is None:
        return None
    owner, repo = owner_repo

    token = _github_token(project)
    if not token:
        return None

    job_id = match.group("job_id")
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        try:
            response = await client.get(
                f"{GITHUB_API_BASE}/repos/{owner}/{repo}/actions/jobs/{job_id}/logs",
                headers=_github_headers(token),
            )
        except httpx.HTTPError:
            return None
    if not response.is_success:
        return None

    error_lines = [
        # Strip GitHub's leading ISO-timestamp log prefix — pure noise for a card
        # that's meant to be read and acted on by an LLM.
        re.sub(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z ", "", line)
        for line in response.text.splitlines()
        if "##[error]" in line
    ]
    if not error_lines:
        return None
    return "\n".join(error_lines[:max_lines])


async def open_pull_request(project: Project, card: Card, *, summary: str) -> DeployRunResult:
    """Push the card's branch as-is (no merge) and open a GitHub PR against
    default_branch. If the branch already has an open PR (a card that bounced back
    to Developer on review feedback and flowed through the pipeline again), re-use
    it — re-push the branch and refresh its body rather than opening a duplicate.
    The pipeline waits on the PR from here (orchestrator/pr_watcher.py); no deploy
    command runs."""
    owner_repo = parse_github_owner_repo(project.repo_remote_url)
    if owner_repo is None:
        return DeployRunResult(
            success=False,
            message=f"repo_remote_url {project.repo_remote_url!r} is not a github.com URL",
        )
    owner, repo = owner_repo

    token = _github_token(project)
    if not token:
        return DeployRunResult(
            success=False,
            message="no GitHub token configured — set github_token_ref in Project Settings "
            "to the name of an env var holding a repo-scoped PAT",
        )

    bare_path = bare_repo_path(project)
    try:
        await run_git("push", "origin", f"{card.branch_name}:{card.branch_name}", cwd=bare_path, token=token)
    except GitCommandError as exc:
        return DeployRunResult(success=False, message=f"push failed: {exc.stderr.strip()}")

    body = summary or f"Automated PR for card {card.id}."
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            existing = await client.get(
                f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls",
                headers=_github_headers(token),
                params={"head": f"{owner}:{card.branch_name}", "state": "open"},
            )
        except httpx.HTTPError as exc:
            return DeployRunResult(success=False, message=f"GitHub API request failed: {exc}")
        if not existing.is_success:
            return DeployRunResult(
                success=False, message=f"GitHub API returned {existing.status_code}: {existing.text[:2000]}"
            )
        pulls = existing.json()
        if pulls:
            pull = pulls[0]
            pr_number = pull.get("number")
            if pull.get("body") != body:
                try:
                    updated = await client.patch(
                        f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}",
                        headers=_github_headers(token),
                        json={"body": body},
                    )
                except httpx.HTTPError as exc:
                    return DeployRunResult(success=False, message=f"GitHub API request failed: {exc}")
                if not updated.is_success:
                    return DeployRunResult(
                        success=False,
                        message=f"GitHub API returned {updated.status_code}: {updated.text[:2000]}",
                    )
            return DeployRunResult(
                success=True,
                message="re-used existing PR (branch re-pushed with the new changes)",
                url=pull.get("html_url") or f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}",
                pr_number=pr_number,
            )

        try:
            response = await client.post(
                f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls",
                headers=_github_headers(token),
                json={
                    "title": card.title,
                    "head": card.branch_name,
                    "base": project.default_branch,
                    "body": body,
                },
            )
        except httpx.HTTPError as exc:
            return DeployRunResult(success=False, message=f"GitHub API request failed: {exc}")

    if response.status_code == 201:
        data = response.json()
        return DeployRunResult(
            success=True,
            message="PR opened",
            url=data.get("html_url"),
            pr_number=data.get("number"),
        )
    return DeployRunResult(
        success=False, message=f"GitHub API returned {response.status_code}: {response.text[:2000]}"
    )


@dataclass
class PullRequestStatus:
    merged: bool
    state: str  # "open" | "closed"
    review_decision: str | None = None  # "approved" | "changes_requested" | None (no substantive review yet)
    feedback: str | None = None  # body of the deciding changes-requested review


class PrStatusUnavailableError(Exception):
    """A transient failure fetching PR status — network blip, GitHub rate limit,
    or the PR not replicated to GitHub's API yet (briefly common right after
    opening). A caller should retry later on this error, not treat it as a
    settled state."""


async def fetch_pr_status(project: Project, card: Card) -> PullRequestStatus:
    """GitHub's current state for the PR this card opened (card.pr_number) — merged
    vs open, and the review decision among the substantive reviews: the most recent
    approve / changes-requested review decides. Raises PrStatusUnavailableError for
    anything transient so a caller retries next pass instead of misreading it."""
    owner_repo = parse_github_owner_repo(project.repo_remote_url)
    if owner_repo is None:
        raise PrStatusUnavailableError("not a github.com remote")
    owner, repo = owner_repo
    token = _github_token(project)
    if not token or card.pr_number is None:
        raise PrStatusUnavailableError("no GitHub token configured, or no pr_number set")

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.get(
                f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{card.pr_number}",
                headers=_github_headers(token),
            )
        except httpx.HTTPError as exc:
            raise PrStatusUnavailableError(f"request failed: {exc}") from exc
        if not response.is_success:
            raise PrStatusUnavailableError(
                f"GitHub API returned {response.status_code}: {response.text[:500]}"
            )
        pull = response.json()
        status = PullRequestStatus(merged=bool(pull.get("merged")), state=pull.get("state", "open"))

        if status.merged or status.state != "open":
            return status

        try:
            reviews_response = await client.get(
                f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{card.pr_number}/reviews",
                headers=_github_headers(token),
            )
        except httpx.HTTPError as exc:
            raise PrStatusUnavailableError(f"request failed: {exc}") from exc
        if not reviews_response.is_success:
            raise PrStatusUnavailableError(
                f"GitHub API returned {reviews_response.status_code}: {reviews_response.text[:500]}"
            )

    substantive = [
        review
        for review in reviews_response.json()
        if review.get("state") in ("APPROVED", "CHANGES_REQUESTED")
    ]
    substantive.sort(key=lambda review: review.get("submitted_at") or "")
    if substantive:
        deciding = substantive[-1]
        status.review_decision = (
            "approved" if deciding["state"] == "APPROVED" else "changes_requested"
        )
        if status.review_decision == "changes_requested":
            status.feedback = deciding.get("body")
    return status


async def merge_pull_request(project: Project, card: Card) -> DeployRunResult:
    """Merge the open PR this card tracks (card.pr_number). Runs only after an
    approving review — the pr_watcher's decision, never at the model's say-so."""
    owner_repo = parse_github_owner_repo(project.repo_remote_url)
    if owner_repo is None:
        return DeployRunResult(success=False, message="not a github.com remote")
    owner, repo = owner_repo
    token = _github_token(project)
    if not token or card.pr_number is None:
        return DeployRunResult(success=False, message="no GitHub token configured, or no pr_number set")

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.put(
                f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{card.pr_number}/merge",
                headers=_github_headers(token),
                json={"commit_title": f"Merge {card.branch_name} (card {card.id})"},
            )
        except httpx.HTTPError as exc:
            return DeployRunResult(success=False, message=f"GitHub API request failed: {exc}")

    if response.status_code in (200, 201):
        return DeployRunResult(
            success=True,
            message=f"PR #{card.pr_number} merged",
            url=card.deploy_url,
            commit_sha=response.json().get("sha"),
        )
    if response.status_code == 405:
        # Pull request is not mergeable — default_branch has advanced into a real
        # conflict with the card's branch. An autonomous rebase of shared history is
        # a bigger, riskier action than this pipeline should take unsupervised (the
        # same call ci_watcher makes about reverting) — the watcher blocks the card
        # for a human.
        return DeployRunResult(
            success=False,
            message=f"PR #{card.pr_number} is not mergeable (conflicts with {project.default_branch}): "
            f"{response.text[:2000]}",
        )
    return DeployRunResult(
        success=False, message=f"GitHub API returned {response.status_code}: {response.text[:2000]}"
    )


__all__ = [
    "CheckRun",
    "CIStatusUnavailableError",
    "DeployRunResult",
    "FAILING_CONCLUSIONS",
    "PrStatusUnavailableError",
    "PullRequestStatus",
    "fetch_check_runs",
    "fetch_pr_status",
    "merge_pull_request",
    "open_pull_request",
    "parse_github_owner_repo",
    "run_auto_main_deploy",
]
