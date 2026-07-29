"""Deployer's trusted execution path — runs in the orchestrator's own process, never
inside the LLM-accessible Docker sandbox. Real credentials (deploy secrets, GitHub
PAT) are injected only here, exactly like the plan's original design for run_deploy:
the LLM never gets shell access to anything that can see them.

Two independent flows, selected by the project's DeployConfig.mode:
  - auto_main: merge the card's branch into default_branch, push, run the configured
    deploy command. Zero human gate.
  - pr_to_operator: push the card's branch as-is and open a GitHub PR. No merge, no
    deploy command — a human takes over from the PR onward."""

import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from built.db.models import Card, DeployConfig, Project
from built.domain.enums import DeployKind
from built.sandbox.worktree import bare_repo_path
from built.tools import git_tools
from built.tools.git_tools import GitCommandError, run_git

GITHUB_API_BASE = "https://api.github.com"

_MERGE_AUTHOR_NAME = "built-deployer"
_MERGE_AUTHOR_EMAIL = "deployer@built.local"


@dataclass
class DeployRunResult:
    success: bool
    message: str
    url: str | None = None
    # True only for a merge conflict — the one failure mode the Deployer agent can
    # actually act on (it has file tools scoped to this same worktree). Every other
    # failure (push rejected, deploy command failed) is reported the same way but
    # left as an ordinary terminal failure, since no amount of file editing fixes those.
    conflict: bool = False


async def run_auto_main_deploy(project: Project, card: Card, wt_path: Path) -> DeployRunResult:
    """Merge the card's branch into default_branch, push, then run the configured
    deploy command.

    wt_path is the Deployer's dedicated worktree, already ensured/reset exactly once
    at visit start by the caller — this function never re-resolves it, so an
    in-progress conflict resolution from an earlier tool call in the *same* visit
    survives across repeated calls here instead of being wiped by a fresh reset.

    A merge conflict is not auto-aborted: the conflicted worktree is left as-is and
    reported back with conflict=True so the Deployer agent's file tools (read_file /
    write_file / edit_file, scoped to this same worktree) can see the actual
    conflict markers and fix them, then call run_deploy() again. That second call
    re-checks every conflicted path's actual content for leftover markers (not just
    git's index state, which nothing but `git add` would clear) and, only once none
    remain, completes the merge commit itself — the per-tool auto-commit is off for
    this worktree specifically so a partial fix can never silently bake leftover
    markers from an untouched file into the merge commit."""
    if await git_tools.merge_in_progress(wt_path):
        conflicted = await git_tools.conflicted_paths(wt_path)
        if conflicted:
            return DeployRunResult(
                success=False,
                conflict=True,
                message=(
                    f"Still conflicted: {', '.join(conflicted)}. Resolve the remaining "
                    "<<<<<<<' / '=======' / '>>>>>>>' markers with write_file or edit_file, then "
                    "call run_deploy() again."
                ),
            )
        await git_tools.complete_merge(
            wt_path,
            message=f"Merge {card.branch_name}",
            author_name=_MERGE_AUTHOR_NAME,
            author_email=_MERGE_AUTHOR_EMAIL,
        )
    else:
        try:
            await run_git(
                "-c",
                f"user.name={_MERGE_AUTHOR_NAME}",
                "-c",
                f"user.email={_MERGE_AUTHOR_EMAIL}",
                "merge",
                "--no-ff",
                "-m",
                f"Merge {card.branch_name}",
                card.branch_name,
                cwd=wt_path,
            )
        except GitCommandError as exc:
            conflicted = await git_tools.conflicted_paths(wt_path)
            detail = (exc.stdout.strip() + "\n" + exc.stderr.strip()).strip()
            return DeployRunResult(
                success=False,
                conflict=True,
                message=(
                    f"Merge conflict in: {', '.join(conflicted) or 'unknown files'}.\n{detail}\n\n"
                    "Use read_file to see the conflict markers in each listed file, resolve them by "
                    "combining both sides sensibly (no '<<<<<<<', '=======', or '>>>>>>>' left "
                    "behind), save the fix with write_file or edit_file, then call run_deploy() "
                    "again to complete the merge. If you can't reasonably resolve it yourself — the "
                    "two sides represent a genuine product conflict, not just a mechanical one — "
                    "call abandon_deploy with a clear reason instead of guessing."
                ),
            )

    try:
        await run_git("push", "origin", f"HEAD:{project.default_branch}", cwd=wt_path)
    except GitCommandError as exc:
        return DeployRunResult(success=False, message=f"push failed: {exc.stderr.strip()}")

    deploy_config = project.deploy_config
    assert deploy_config is not None
    return await _run_deploy_command(deploy_config, cwd=wt_path)


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


def _parse_github_owner_repo(remote_url: str) -> tuple[str, str] | None:
    for pattern in _GITHUB_URL_PATTERNS:
        match = pattern.match(remote_url.strip())
        if match:
            return match.group("owner"), match.group("repo")
    return None


async def open_pull_request(project: Project, card: Card, *, summary: str) -> DeployRunResult:
    """Push the card's branch as-is (no merge) and open a GitHub PR against
    default_branch. A human takes over from here — no deploy command runs."""
    owner_repo = _parse_github_owner_repo(project.repo_remote_url)
    if owner_repo is None:
        return DeployRunResult(
            success=False,
            message=f"repo_remote_url {project.repo_remote_url!r} is not a github.com URL",
        )
    owner, repo = owner_repo

    deploy_config = project.deploy_config
    assert deploy_config is not None
    token = os.environ.get(deploy_config.github_token_ref or "") if deploy_config.github_token_ref else None
    if not token:
        return DeployRunResult(
            success=False,
            message="no GitHub token configured — set github_token_ref in Project Settings "
            "to the name of an env var holding a repo-scoped PAT",
        )

    bare_path = bare_repo_path(project)
    try:
        await run_git("push", "origin", f"{card.branch_name}:{card.branch_name}", cwd=bare_path)
    except GitCommandError as exc:
        return DeployRunResult(success=False, message=f"push failed: {exc.stderr.strip()}")

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                },
                json={
                    "title": card.title,
                    "head": card.branch_name,
                    "base": project.default_branch,
                    "body": summary or f"Automated PR for card {card.id}.",
                },
            )
        except httpx.HTTPError as exc:
            return DeployRunResult(success=False, message=f"GitHub API request failed: {exc}")

    if response.status_code == 201:
        data = response.json()
        return DeployRunResult(success=True, message="PR opened", url=data.get("html_url"))
    return DeployRunResult(
        success=False, message=f"GitHub API returned {response.status_code}: {response.text[:2000]}"
    )


__all__ = ["DeployRunResult", "run_auto_main_deploy", "open_pull_request"]
