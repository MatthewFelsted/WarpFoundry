"""Evaluation utilities: run tests and gather repo diagnostics."""

from __future__ import annotations

import contextlib
import logging
import os
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

from codex_manager.git_tools import (
    diff_stat,
    pending_numstat_entries,
    status_porcelain,
    summarize_numstat_entries,
)
from codex_manager.schemas import EvalResult, TestOutcome

logger = logging.getLogger(__name__)

# Default test command executed when none is configured.
DEFAULT_TEST_CMD = ["python", "-m", "pytest", "-q"]


def _is_windows_platform() -> bool:
    """Return ``True`` when command parsing should follow Windows rules."""
    return os.name == "nt"


def _strip_wrapping_quotes(token: str) -> str:
    """Remove one pair of matching wrapping quotes from *token* when present."""
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
        return token[1:-1]
    return token


def parse_test_command(command: str | Sequence[str] | None) -> list[str] | None:
    """Parse test command input into argv-style tokens.

    Accepts either a shell-like string (supports quoted arguments) or a
    pre-tokenized sequence. Returns ``None`` for empty/blank commands.
    """
    if command is None:
        return None

    if isinstance(command, str):
        raw = command.strip()
        if not raw:
            return None
        try:
            if _is_windows_platform():
                # ``posix=True`` treats backslashes as escapes and mangles
                # unquoted Windows paths like ``tests\test_app.py``.
                parts = [_strip_wrapping_quotes(part) for part in shlex.split(raw, posix=False)]
            else:
                parts = shlex.split(raw, posix=True)
        except ValueError:
            logger.warning(
                "Could not parse test command %r with shell quoting; falling back to whitespace split.",
                raw,
            )
            parts = raw.split()
        cleaned = [part for part in parts if part]
        return cleaned or None

    cleaned: list[str] = []
    for part in command:
        if part is None:
            continue
        token = str(part).strip()
        if token:
            cleaned.append(token)
    return cleaned or None


class RepoEvaluator:
    """Run a test suite and gather diff statistics for a repository.

    Parameters
    ----------
    test_cmd:
        Shell command list used to run the test suite.
    timeout:
        Maximum seconds to wait for the test command.
    """

    def __init__(
        self,
        test_cmd: list[str] | None = None,
        timeout: int = 300,
        skip_tests: bool = False,
    ) -> None:
        self.test_cmd = list(test_cmd) if test_cmd else list(DEFAULT_TEST_CMD)
        self.timeout = timeout
        self.skip_tests = skip_tests

    def evaluate(self, repo_path: str | Path) -> EvalResult:
        """Run evaluation and return an :class:`EvalResult`."""
        repo_path = Path(repo_path).resolve()
        test_outcome, test_summary, test_exit = self._run_tests(repo_path)
        changed_files = pending_numstat_entries(repo_path)
        files_changed, ins, dels = summarize_numstat_entries(changed_files)
        stat = ""
        if files_changed > 0:
            with contextlib.suppress(Exception):
                stat = diff_stat(repo_path, revspec="HEAD")
            if not stat:
                with contextlib.suppress(Exception):
                    stat = diff_stat(repo_path)
            if not stat:
                stat = f"{files_changed} files changed, {ins} insertions(+), {dels} deletions(-)"
        porcelain = status_porcelain(repo_path)

        return EvalResult(
            test_outcome=test_outcome,
            test_summary=test_summary,
            test_exit_code=test_exit,
            diff_stat=stat,
            status_porcelain=porcelain,
            net_lines_changed=ins - dels,
            files_changed=files_changed,
            changed_files=changed_files,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_tests(self, cwd: Path) -> tuple[TestOutcome, str, int]:
        """Execute the test command and return (outcome, summary, exit_code)."""
        if self.skip_tests:
            return TestOutcome.SKIPPED, "Tests skipped (no test command configured)", 0
        if not self.test_cmd:
            return TestOutcome.SKIPPED, "Tests skipped (empty test command)", 0

        logger.info("Running tests: %s (cwd=%s)", " ".join(self.test_cmd), cwd)
        try:
            proc = subprocess.run(
                self.test_cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except FileNotFoundError as exc:
            return TestOutcome.ERROR, f"Test command not found: {exc}", -1
        except subprocess.TimeoutExpired:
            return TestOutcome.ERROR, f"Test command timed out after {self.timeout}s", -1
        except ValueError as exc:
            return TestOutcome.ERROR, f"Invalid test command configuration: {exc}", -1

        combined = (proc.stdout + "\n" + proc.stderr).strip()
        summary = _summarise_output(combined)

        if proc.returncode == 0:
            return TestOutcome.PASSED, summary, 0
        if proc.returncode == 5:
            # pytest exit code 5 = no tests collected
            return TestOutcome.SKIPPED, summary, 5
        return TestOutcome.FAILED, summary, proc.returncode


def _summarise_output(text: str, max_lines: int = 30) -> str:
    """Truncate and clean test output to a manageable summary."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text

    # Keep first 10 and last 20 lines for context
    head = lines[:10]
    tail = lines[-20:]
    skipped = len(lines) - 30
    return "\n".join([*head, f"  ... ({skipped} lines omitted) ...", *tail])
