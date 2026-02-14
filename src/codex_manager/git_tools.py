"""Git helper utilities for branch management, diffs, commits, and reverts."""

from __future__ import annotations

import contextlib
import datetime as dt
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GitError(RuntimeError):
    """Raised when a git command fails unexpectedly."""


def _run_git(
    *args: str,
    cwd: Path,
    check: bool = True,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    """Run a git command and return the CompletedProcess."""
    cmd = ["git", *args]
    logger.debug("git %s (cwd=%s)", " ".join(args), cwd)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise GitError(
            f"`git {' '.join(args)}` failed (rc={result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def diff_stat(repo: str | Path, revspec: str | None = None) -> str:
    """Return ``git diff --stat`` output.

    When ``revspec`` is provided, runs ``git diff --stat <revspec>``.
    """
    args = ["diff", "--stat"]
    if revspec:
        args.append(revspec)
    return _run_git(*args, cwd=Path(repo)).stdout.strip()


def diff_numstat_entries(
    repo: str | Path, revspec: str | None = None
) -> list[dict[str, Any]]:
    """Return file-level diff entries from ``git diff --numstat``.

    Each entry contains:
    - ``path``: file path
    - ``insertions``: integer or ``None`` for non-text/binary entries
    - ``deletions``: integer or ``None`` for non-text/binary entries
    """
    args = ["diff", "--numstat"]
    if revspec:
        args.append(revspec)
    out = _run_git(*args, cwd=Path(repo)).stdout.strip()
    if not out:
        return []

    entries: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        ins_raw, del_raw, path = parts

        ins_val: int | None = None
        del_val: int | None = None
        with contextlib.suppress(ValueError):
            ins_val = int(ins_raw)
        with contextlib.suppress(ValueError):
            del_val = int(del_raw)

        entries.append(
            {
                "path": path,
                "insertions": ins_val,
                "deletions": del_val,
            }
        )
    return entries


def diff_numstat(
    repo: str | Path, revspec: str | None = None
) -> tuple[int, int, int]:
    """Return (files_changed, insertions, deletions) from ``git diff --numstat``."""
    entries = diff_numstat_entries(repo, revspec=revspec)
    if not entries:
        return 0, 0, 0
    files = insertions = deletions = 0
    for entry in entries:
        files += 1
        if isinstance(entry.get("insertions"), int):
            insertions += int(entry["insertions"])
        if isinstance(entry.get("deletions"), int):
            deletions += int(entry["deletions"])
    return files, insertions, deletions


def net_lines_changed(repo: str | Path, revspec: str | None = None) -> int:
    """Net lines changed (insertions - deletions)."""
    _, ins, dels = diff_numstat(repo, revspec=revspec)
    return ins - dels


def status_porcelain(repo: str | Path) -> str:
    """Return ``git status --porcelain`` output."""
    return _run_git("status", "--porcelain", cwd=Path(repo)).stdout.strip()


def current_branch(repo: str | Path) -> str:
    """Return the name of the current branch."""
    return _run_git("rev-parse", "--abbrev-ref", "HEAD", cwd=Path(repo)).stdout.strip()


def head_sha(repo: str | Path) -> str:
    """Return the short SHA of HEAD."""
    return _run_git("rev-parse", "--short", "HEAD", cwd=Path(repo)).stdout.strip()


def is_clean(repo: str | Path) -> bool:
    """Return True when the working tree is clean."""
    return status_porcelain(repo) == ""


# ---------------------------------------------------------------------------
# Mutation helpers
# ---------------------------------------------------------------------------

def ensure_git_identity(repo: str | Path) -> None:
    """Ensure the repo has a git identity configured for commits.

    Checks ``user.name`` and ``user.email`` in the repo-local config.
    If either is missing, sets a default so ``git commit`` won't fail.
    """
    cwd = Path(repo)
    for key, fallback in [
        ("user.name", "Codex Manager"),
        ("user.email", "codex-manager@localhost"),
    ]:
        result = _run_git("config", key, cwd=cwd, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            _run_git("config", key, fallback, cwd=cwd)
            logger.info("Set %s = %s in %s", key, fallback, cwd)


def create_branch(repo: str | Path, branch_name: str | None = None) -> str:
    """Create and checkout a new branch; return its name.

    If *branch_name* is None a timestamped name is generated:
    ``codex-manager/20260206T153012``.
    """
    if branch_name is None:
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
        branch_name = f"codex-manager/{ts}"
    _run_git("checkout", "-b", branch_name, cwd=Path(repo))
    logger.info("Created branch %s", branch_name)
    return branch_name


def commit_all(repo: str | Path, message: str) -> str:
    """Stage everything and commit.  Return the new commit SHA."""
    cwd = Path(repo)
    _run_git("add", "-A", cwd=cwd)
    _run_git("commit", "-m", message, "--allow-empty", cwd=cwd)
    return head_sha(repo)


def revert_all(repo: str | Path) -> None:
    """Reset the working tree to HEAD while preserving tool runtime artifacts."""
    cwd = Path(repo)
    _run_git("checkout", "--", ".", cwd=cwd, check=False)
    # Preserve codex-manager runtime artifacts (logs, step outputs).
    _run_git("clean", "-fd", "-e", ".codex_manager/", cwd=cwd, check=False)
    logger.info("Reverted working tree to HEAD in %s", cwd)


def generate_commit_message(round_number: int, prompt: str, eval_summary: str) -> str:
    """Build a structured commit message for a Codex-manager round."""
    # Sanitise the prompt to one line, max 72 chars for the subject
    subject = re.sub(r"\s+", " ", prompt).strip()
    if len(subject) > 60:
        subject = subject[:57] + "..."
    return (
        f"[codex-manager] round {round_number}: {subject}\n\n"
        f"Eval: {eval_summary}\n"
    )
