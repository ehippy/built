#!/usr/bin/env python3
"""
built watchdog — pull latest and restart the service if repo changed.

Called by built-watchdog.timer every 15 minutes as a oneshot systemd service.
Logs results to stdout (captured by systemd journal).
"""

import logging
import subprocess
import sys
from datetime import datetime, timezone
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
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    before_head = git_head("HEAD")

    # --- pull ---
    log.info("Pulling latest from %s", REMOTE_URL)
    r = subprocess.run(
        ["git", "-C", str(REPO_DIR), "pull", "--ff-only"],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        # Non-zero return is normal when already up to date; stderr may just be "Already up to date."
        log.info("Pull result: %s", r.stdout.strip())
        if r.stderr.strip():
            log.debug("Pull stderr: %s", r.stderr.strip())

    after_head = git_head("HEAD")

    # --- did we change? ---
    if before_head == after_head:
        log.info("No changes — HEAD still %s", before_head)
        return

    log.info("Repo changed: %s → %s", before_head, after_head)
    log.info("Restarting built service…")

    r = subprocess.run(
        ["systemctl", "restart", "built.service"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode == 0:
        log.info("built.service restarted successfully")
    else:
        log.error("Failed to restart built.service: %s", r.stderr.strip())
        sys.exit(1)


if __name__ == "__main__":
    main()
