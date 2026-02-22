"""Git helper utilities for branch management, diffs, commits, and reverts."""

from __future__ import annotations

import contextlib
import datetime as dt
import logging
import os
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _git_subprocess_isolation_kwargs() -> dict[str, object]:
    """Return kwargs that prevent child console events from reaching the parent on Windows."""
    if os.name != "nt":
        return {}
    new_pg = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    no_win = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    flags = new_pg | no_win
    return {"creationflags": flags} if flags else {}


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
        **_git_subprocess_isolation_kwargs(),
    )
    if check and result.returncode != 0:
        raise GitError(
            f"`git {' '.join(args)}` failed (rc={result.returncode}): {result.stderr.strip()}"
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


def _parse_numstat_output(raw: str) -> list[dict[str, Any]]:
    """Parse ``git diff --numstat`` output into structured entries."""
    out = str(raw or "").strip()
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


def diff_numstat_entries(
    repo: str | Path,
    revspec: str | None = None,
    *,
    cached: bool = False,
) -> list[dict[str, Any]]:
    """Return file-level diff entries from ``git diff --numstat``.

    Each entry contains:
    - ``path``: file path
    - ``insertions``: integer or ``None`` for non-text/binary entries
    - ``deletions``: integer or ``None`` for non-text/binary entries
    """
    args = ["diff", "--numstat"]
    if cached and not revspec:
        args.append("--cached")
    if revspec:
        args.append(revspec)
    out = _run_git(*args, cwd=Path(repo)).stdout
    return _parse_numstat_output(out)


def _count_text_file_lines(path: Path) -> int | None:
    """Return line count for a likely-text file, ``None`` for binary/unreadable."""
    try:
        with path.open("rb") as handle:
            newline_count = 0
            saw_any = False
            last_byte = b""
            while True:
                chunk = handle.read(8192)
                if not chunk:
                    break
                saw_any = True
                if b"\x00" in chunk:
                    return None
                newline_count += chunk.count(b"\n")
                last_byte = chunk[-1:]
            if not saw_any:
                return 0
            return newline_count if last_byte == b"\n" else newline_count + 1
    except OSError:
        return None


def _untracked_numstat_entries(repo: str | Path) -> list[dict[str, Any]]:
    """Return synthetic numstat entries for untracked files."""
    raw = _run_git("ls-files", "--others", "--exclude-standard", "-z", cwd=Path(repo)).stdout
    if not raw:
        return []

    entries: list[dict[str, Any]] = []
    for rel in (part for part in raw.split("\x00") if part):
        rel_path = Path(rel)
        abs_path = Path(repo) / rel_path
        if not abs_path.is_file():
            continue
        lines = _count_text_file_lines(abs_path)
        entries.append(
            {
                "path": rel_path.as_posix(),
                "insertions": lines,
                "deletions": 0 if isinstance(lines, int) else None,
            }
        )
    return entries


def pending_numstat_entries(repo: str | Path) -> list[dict[str, Any]]:
    """Return pending repo deltas (staged + unstaged + untracked) vs current HEAD."""
    cwd = Path(repo)
    entries_by_path: dict[str, dict[str, Any]] = {}

    def _merge(entries: Sequence[dict[str, Any]]) -> None:
        for entry in entries:
            path = str(entry.get("path") or "").strip()
            if not path:
                continue
            existing = entries_by_path.get(path)
            ins = entry.get("insertions")
            dels = entry.get("deletions")
            if existing is None:
                entries_by_path[path] = {
                    "path": path,
                    "insertions": int(ins) if isinstance(ins, int) else None,
                    "deletions": int(dels) if isinstance(dels, int) else None,
                }
                continue
            prev_ins = existing.get("insertions")
            prev_dels = existing.get("deletions")
            existing["insertions"] = (
                int(prev_ins) + int(ins) if isinstance(prev_ins, int) and isinstance(ins, int) else None
            )
            existing["deletions"] = (
                int(prev_dels) + int(dels) if isinstance(prev_dels, int) and isinstance(dels, int) else None
            )

    # Preferred path: one tracked diff against HEAD captures staged + unstaged.
    tracked_entries: list[dict[str, Any]]
    try:
        tracked_entries = diff_numstat_entries(cwd, revspec="HEAD")
    except GitError:
        # Unborn HEAD fallback (new repo before first commit).
        tracked_entries = diff_numstat_entries(cwd)
        tracked_entries.extend(diff_numstat_entries(cwd, cached=True))

    _merge(tracked_entries)
    _merge(_untracked_numstat_entries(cwd))
    return list(entries_by_path.values())


def pending_numstat(repo: str | Path) -> tuple[int, int, int]:
    """Return ``(files_changed, insertions, deletions)`` for pending repo deltas."""
    return summarize_numstat_entries(pending_numstat_entries(repo))


def summarize_numstat_entries(entries: Sequence[dict[str, Any]]) -> tuple[int, int, int]:
    """Return (files_changed, insertions, deletions) for parsed numstat entries."""
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


def diff_numstat(repo: str | Path, revspec: str | None = None) -> tuple[int, int, int]:
    """Return (files_changed, insertions, deletions) from ``git diff --numstat``."""
    entries = diff_numstat_entries(repo, revspec=revspec)
    return summarize_numstat_entries(entries)


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
        ("user.name", "WarpFoundry"),
        ("user.email", "warpfoundry@localhost"),
    ]:
        result = _run_git("config", key, cwd=cwd, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            _run_git("config", key, fallback, cwd=cwd)
            logger.info("Set %s = %s in %s", key, fallback, cwd)


def create_branch(repo: str | Path, branch_name: str | None = None) -> str:
    """Create and checkout a new branch; return its name.

    If *branch_name* is None a timestamped name is generated:
    ``warpfoundry/20260206T153012``.
    """
    if branch_name is None:
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
        branch_name = f"warpfoundry/{ts}"
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
    # Preserve WarpFoundry runtime artifacts (logs, step outputs).
    _run_git("clean", "-fd", "-e", ".codex_manager/", cwd=cwd, check=False)
    logger.info("Reverted working tree to HEAD in %s", cwd)


def reset_to_ref(repo: str | Path, ref: str) -> None:
    """Hard-reset repository to *ref* while preserving manager artifacts."""
    cwd = Path(repo)
    _run_git("reset", "--hard", ref, cwd=cwd)
    _run_git("clean", "-fd", "-e", ".codex_manager/", cwd=cwd, check=False)
    logger.info("Reset repository to %s in %s", ref, cwd)


def generate_commit_message(round_number: int, prompt: str, eval_summary: str) -> str:
    """Build a structured commit message for a WarpFoundry round."""
    # Sanitise the prompt to one line, max 72 chars for the subject
    subject = re.sub(r"\s+", " ", prompt).strip()
    if len(subject) > 60:
        subject = subject[:57] + "..."
    return f"[warpfoundry] round {round_number}: {subject}\n\nEval: {eval_summary}\n"
