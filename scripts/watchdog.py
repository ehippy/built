#!/usr/bin/env python3
"""
built watchdog — pull latest and restart the service if repo changed.

Called by built-watchdog.timer every 15 minutes as a oneshot systemd service.
Logs results to stdout (captured by systemd journal).
"""

import logging
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
REMOTE_URL = "https://github.com/Ehippy/built"
LOG_FORMAT = "%(asctime)s [built-watchdog] %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
log = logging.getLogger(__name__)


def git_head(ref: str) -> str | None:
    """Return short SHA for a git ref, or None on failure."""
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO_DIR), "rev-parse", "--short", ref],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def main():
    before_head = git_head("HEAD")
    if before_head is None:
        log.error("Could not read local HEAD — is %s a git repo?", REPO_DIR)
        sys.exit(1)

    # --- pull ---
    log.info("Pulling latest from %s", REMOTE_URL)
    r = subprocess.run(
        ["git", "-C", str(REPO_DIR), "pull", "--ff-only"],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        # A non-zero return here is a real failure: diverged branch, auth error,
        # network error, etc. "Already up to date" exits 0, so this is never the
        # normal case — don't let it look like a routine no-op.
        log.error("git pull failed (exit %d): %s", r.returncode, r.stderr.strip() or r.stdout.strip())
        sys.exit(1)

    after_head = git_head("HEAD")
    if after_head is None:
        log.error("Could not read local HEAD after pull")
        sys.exit(1)

    # --- did we change? ---
    if before_head == after_head:
        log.info("No changes — HEAD still %s", before_head)
        return

    log.info("Repo changed: %s → %s", before_head, after_head)
    log.info("Restarting built service…")

    r = subprocess.run(
        ["sudo", "-n", "systemctl", "restart", "built.service"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode == 0:
        log.info("built.service restarted successfully")
    else:
        log.error("Failed to restart built.service: %s", r.stderr.strip() or r.stdout.strip())
        sys.exit(1)


if __name__ == "__main__":
    main()
