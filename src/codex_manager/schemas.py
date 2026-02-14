"""Pydantic models for structured data throughout the manager."""

from __future__ import annotations

import datetime as dt
import logging
import os
import tempfile
import time
from contextlib import suppress
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)
_ATOMIC_REPLACE_MAX_RETRIES = 8
_ATOMIC_REPLACE_RETRY_SECONDS = 0.01

# ---------------------------------------------------------------------------
# Codex CLI run results
# ---------------------------------------------------------------------------

class EventKind(str, Enum):
    """Known JSONL event types emitted by ``codex exec --json``."""

    AGENT_MESSAGE = "agent_message"
    FILE_CHANGE = "file_change"
    COMMAND_EXEC = "command_exec"
    TURN_COMPLETED = "turn.completed"
    ERROR = "error"
    UNKNOWN = "unknown"


class CodexEvent(BaseModel):
    """A single parsed JSONL event from a Codex CLI run."""

    kind: EventKind = EventKind.UNKNOWN
    raw: dict[str, Any] = Field(default_factory=dict)
    text: str | None = None


class UsageInfo(BaseModel):
    """Token / cost usage reported by Codex."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str | None = None


class RunResult(BaseModel):
    """Aggregated result of a single ``codex exec`` invocation."""

    success: bool = False
    exit_code: int = -1
    final_message: str = ""
    events: list[CodexEvent] = Field(default_factory=list)
    file_changes: list[dict[str, Any]] = Field(default_factory=list)
    command_executions: list[dict[str, Any]] = Field(default_factory=list)
    usage: UsageInfo = Field(default_factory=UsageInfo)
    errors: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------

class TestOutcome(str, Enum):
    """Outcome of test execution during repository evaluation."""
    __test__ = False  # Prevent pytest from collecting this enum as a test class.

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class EvalResult(BaseModel):
    """Result of evaluating the repo after a Codex run."""

    test_outcome: TestOutcome = TestOutcome.ERROR
    test_summary: str = ""
    test_exit_code: int = -1
    diff_stat: str = ""
    status_porcelain: str = ""
    net_lines_changed: int = 0
    files_changed: int = 0
    changed_files: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Loop / orchestration state
# ---------------------------------------------------------------------------

class RoundRecord(BaseModel):
    """Persisted record of a single improvement round."""

    round_number: int
    timestamp: str = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    prompt: str = ""
    run_result: RunResult = Field(default_factory=RunResult)
    eval_result: EvalResult = Field(default_factory=EvalResult)
    commit_sha: str | None = None
    skipped: bool = False
    skip_reason: str | None = None


class StopReason(str, Enum):
    """Reason an improvement loop stopped."""
    MAX_ROUNDS = "max_rounds"
    TESTS_PASS_LOW_IMPACT = "tests_pass_low_impact"
    BUDGET_EXHAUSTED = "budget_exhausted"
    USER_ABORT = "user_abort"
    CONSECUTIVE_FAILURES = "consecutive_failures"
    GOAL_MET = "goal_met"


class LoopState(BaseModel):
    """Full persisted state of an improvement loop - written to .codex_manager/state.json."""

    repo_path: str
    goal: str
    mode: str = "dry-run"
    branch_name: str | None = None
    max_rounds: int = 10
    current_round: int = 0
    rounds: list[RoundRecord] = Field(default_factory=list)
    stop_reason: StopReason | None = None
    started_at: str = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    finished_at: str | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # -- helpers --

    def state_path(self) -> Path:
        """Return the persisted loop-state file path."""
        return Path(self.repo_path) / ".codex_manager" / "state.json"

    def save(self) -> None:
        """Persist state to disk."""
        path = self.state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump_json(indent=2)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f"{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            _replace_file_with_retry(tmp_path, path)
        finally:
            with suppress(OSError):
                tmp_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, repo_path: str | Path) -> LoopState | None:
        """Load persisted state from disk, or return ``None`` when absent."""
        path = Path(repo_path) / ".codex_manager" / "state.json"
        if path.exists():
            try:
                raw = path.read_text(encoding="utf-8")
                if not raw.strip():
                    logger.warning("State file is empty; ignoring: %s", path)
                    return None
                return cls.model_validate_json(raw)
            except (OSError, ValidationError) as exc:
                logger.warning("Could not load state file %s: %s", path, exc)
                return None
        return None


def _replace_file_with_retry(src: Path, dst: Path) -> None:
    """Replace *dst* with *src*, retrying on transient Windows file-lock races."""
    last_error: OSError | None = None
    for attempt in range(_ATOMIC_REPLACE_MAX_RETRIES):
        try:
            src.replace(dst)
            return
        except PermissionError as exc:
            last_error = exc
        except OSError as exc:
            if exc.errno != 13:
                raise
            last_error = exc
        if attempt < _ATOMIC_REPLACE_MAX_RETRIES - 1:
            time.sleep(_ATOMIC_REPLACE_RETRY_SECONDS * (attempt + 1))
    if last_error is not None:
        raise last_error
