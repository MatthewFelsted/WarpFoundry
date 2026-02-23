"""Task-chain executor -- runs a multi-step improvement loop in a background thread."""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import logging
import os
import queue
import re
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from codex_manager.agent_runner import AgentRunner
from codex_manager.agent_signals import (
    TERMINATE_STEP_TAG,
    contains_terminate_step_signal,
    terminate_step_instruction,
)
from codex_manager.artifact_retention import RetentionPolicy, cleanup_runtime_artifacts
from codex_manager.brain.logbook import BrainLogbook
from codex_manager.brain.manager import BrainConfig, BrainManager
from codex_manager.claude_code import ClaudeCodeRunner
from codex_manager.codex_cli import CodexRunner
from codex_manager.eval_tools import RepoEvaluator, parse_test_command
from codex_manager.git_tools import (
    commit_all,
    create_branch,
    diff_numstat_entries,
    diff_stat,
    ensure_git_identity,
    generate_commit_message,
    head_sha,
    is_clean,
    reset_to_ref,
    revert_all,
    summarize_numstat_entries,
)
from codex_manager.gui.models import ChainConfig, ChainState, StepResult, TaskStep
from codex_manager.gui.presets import get_prompt
from codex_manager.history_log import HistoryLogbook
from codex_manager.ledger import KnowledgeLedger
from codex_manager.managed_artifacts import (
    capture_artifact_snapshot,
    merge_eval_result_with_artifact_delta,
    summarize_artifact_delta,
)
from codex_manager.memory.vector_store import ProjectVectorMemory
from codex_manager.pipeline.tracker import LogTracker
from codex_manager.prompt_logging import (
    format_prompt_log_line,
    format_prompt_preview,
    prompt_metadata,
)

logger = logging.getLogger(__name__)

# #region agent log
_DEBUG_LOG_ENABLED_ENV = "CODEX_MANAGER_DEBUG_ENABLED"
_DEBUG_LOG_PATH_ENV = "CODEX_MANAGER_DEBUG_LOG_PATH"
_DEFAULT_DEBUG_FILE_NAME = "CHAIN_AGENT_DEBUG.jsonl"
_STEP_MEMORY_MAX_ENTRIES = 240
_STEP_MEMORY_CONTEXT_ITEMS = 6
_STEP_MEMORY_CONTEXT_CHARS = 2_800
_STEP_MEMORY_EXCERPT_CHARS = 900
_NO_PROGRESS_STREAK_RESULTS = 4
_EVIDENCE_SUMMARY_CHARS = 1_200
_MUTATING_JOB_TYPES = {
    "implementation",
    "bug_hunting",
    "refactoring",
    "performance",
    "security_audit",
    "strategic_product_maximization",
    "visual_asset_generation",
}
_HISTORY_ERROR_CONTEXT_CHARS = 2000
_CHAIN_MANAGED_ARTIFACT_GLOBS: tuple[str, ...] = (
    ".codex_manager/logs/WISHLIST.md",
    ".codex_manager/logs/TESTPLAN.md",
    ".codex_manager/logs/EXPERIMENTS.md",
    ".codex_manager/logs/RESEARCH.md",
    ".codex_manager/logs/SCIENTIST_REPORT.md",
)


def _env_flag(value: str | None) -> bool:
    raw = str(value or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


# #endregion


class ChainExecutor:
    """Manages the lifecycle of a task-chain execution.

    * ``start(config)`` -- kicks off a daemon thread that iterates through
      the chain's steps in a loop.
    * ``stop()`` / ``pause()`` / ``resume()`` -- thread-safe controls.
    * ``log_queue`` -- a :class:`queue.Queue` of ``{time, level, message}``
      dicts consumed by the SSE endpoint.
    """

    def __init__(
        self,
        *,
        on_run_finalized: Callable[[ChainState], None] | None = None,
    ) -> None:
        self.config: ChainConfig | None = None
        self.state = ChainState()
        self._stop_event = threading.Event()
        self._stop_after_step_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused
        self._thread: threading.Thread | None = None
        self.log_queue: queue.Queue[dict] = queue.Queue(maxsize=10_000)
        self.output_dir: Path | None = None
        self._brain_logbook: BrainLogbook | None = None
        self._history_logbook: HistoryLogbook | None = None
        self._log_tracker: LogTracker | None = None
        self._step_memory_path: Path | None = None
        self._step_memory_entries: list[dict[str, Any]] = []
        self.vector_memory: ProjectVectorMemory | None = None
        self._debug_enabled = False
        self._debug_log_path: Path | None = None
        self._debug_log_failure_reported = False
        self._error_log_failure_reported = False
        self._next_log_event_id = 0
        self._on_run_finalized = on_run_finalized

    # ------------------------------------------------------------------
    # Public controls
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @staticmethod
    def _brain_goal(config: ChainConfig, repo: Path) -> str:
        """Return a repo-anchored goal string for the brain layer.

        Chain names are user labels and may not be real repository names.
        """
        repo_name = repo.name or "repository"
        return f"Improve repository '{repo_name}' at path: {repo}"

    def start(self, config: ChainConfig) -> None:
        if self.is_running:
            self._log("error", "Chain is already running")
            return

        self.config = config
        repo = Path(config.repo_path).resolve()
        self.output_dir = repo / ".codex_manager" / "outputs"
        self._configure_debug_logging(repo, config)
        self._error_log_failure_reported = False
        self._next_log_event_id = 0
        self._log_tracker = LogTracker(repo)
        try:
            self._log_tracker.initialize()
        except Exception as exc:
            logger.warning("Could not initialize log tracker for chain runtime errors: %s", exc)
            self._log_tracker = None
        self.state = ChainState(
            running=True,
            started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            run_max_loops=max(1, int(config.max_loops)),
            run_unlimited=bool(config.unlimited),
        )
        self._stop_event.clear()
        self._stop_after_step_event.clear()
        self.state.stop_after_current_step = False
        self._pause_event.set()

        # Drain any stale log entries
        while not self.log_queue.empty():
            try:
                self.log_queue.get_nowait()
            except queue.Empty:
                break

        self._prepare_output_dir()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._log("info", f"Chain started: {config.name} ({config.mode} mode)")
        self._log_execution_mode_warnings(config)

    def set_on_run_finalized(
        self,
        callback: Callable[[ChainState], None] | None,
    ) -> None:
        """Install or clear the callback invoked after run finalization."""
        self._on_run_finalized = callback

    def _configure_debug_logging(self, repo: Path, config: ChainConfig) -> None:
        """Resolve debug logging controls from config/environment."""
        config_enabled = bool(getattr(config, "debug_logging_enabled", False))
        env_enabled = _env_flag(os.getenv(_DEBUG_LOG_ENABLED_ENV))
        self._debug_enabled = bool(config_enabled or env_enabled)
        self._debug_log_failure_reported = False

        if not self._debug_enabled:
            self._debug_log_path = None
            return

        config_path = str(getattr(config, "debug_log_path", "") or "").strip()
        env_path = os.getenv(_DEBUG_LOG_PATH_ENV, "").strip()
        raw_path = config_path or env_path
        if raw_path:
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = repo / candidate
            self._debug_log_path = candidate
            return
        self._debug_log_path = repo / ".codex_manager" / "logs" / _DEFAULT_DEBUG_FILE_NAME

    def _prepare_output_dir(self) -> None:
        """Create a clean per-run output directory under .codex_manager/outputs.

        Uses best-effort in-place cleanup to avoid Windows directory-lock
        errors (for example WinError 32 on OneDrive-managed folders).
        """
        if self.output_dir is None:
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._archive_previous_outputs()
            cleanup_failures: list[str] = []
            for entry in self.output_dir.iterdir():
                try:
                    if entry.is_dir():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink(missing_ok=True)
                except Exception as exc:
                    cleanup_failures.append(f"{entry.name}: {exc}")

            if cleanup_failures:
                preview = "; ".join(cleanup_failures[:3])
                if len(cleanup_failures) > 3:
                    preview += "; ..."
                self._log(
                    "warn",
                    f"Could not fully clean output directory ({len(cleanup_failures)} item(s)): {preview}",
                )
            self._log("info", f"Run outputs: {self.output_dir}")
        except Exception as exc:
            self._log("warn", f"Could not prepare output directory: {exc}")

    def stop(self) -> None:
        self._stop_after_step_event.clear()
        self.state.stop_after_current_step = False
        self._stop_event.set()
        self._pause_event.set()  # unblock if paused
        self._log("warn", "Stop requested")

    def set_stop_after_current_step(self, enabled: bool) -> None:
        enabled_flag = bool(enabled)
        current = bool(self.state.stop_after_current_step)
        if enabled_flag == current:
            return
        self.state.stop_after_current_step = enabled_flag
        if enabled_flag:
            self._stop_after_step_event.set()
            self._log(
                "warn",
                "Stop-after-step armed: run will stop after the active step completes.",
            )
        else:
            self._stop_after_step_event.clear()
            self._log("info", "Stop-after-step cleared: run will continue normally.")

    def pause(self) -> None:
        self._pause_event.clear()
        self.state.paused = True
        self._log("info", "Paused")

    def resume(self) -> None:
        self._pause_event.set()
        self.state.paused = False
        self._log("info", "Resumed")

    def get_state(self) -> dict:
        return self.state.model_dump()

    def get_state_summary(self, *, since_results: int | None = None) -> dict[str, Any]:
        """Return polling-friendly state, optionally including only new results."""
        payload: dict[str, Any] = self.state.model_dump(exclude={"results"})
        results = self.state.results
        total_results = len(results)
        payload["total_results"] = total_results

        if since_results is None:
            payload["results"] = [r.model_dump() for r in results]
            return payload

        offset = min(max(0, since_results), total_results)
        payload["results_delta"] = [r.model_dump() for r in results[offset:]]
        return payload

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _debug_log(self, message: str) -> None:
        """Emit a verbose debug line only when debug logging is enabled."""
        if not self._debug_enabled:
            return
        self._log("info", f"[DEBUG] {message}")

    def _write_agent_debug_event(
        self,
        location: str,
        message: str,
        data: dict[str, Any] | None = None,
        hypothesis_id: str = "",
    ) -> None:
        """Persist one structured debug event when debug logging is enabled."""
        if not self._debug_enabled or self._debug_log_path is None:
            return
        payload: dict[str, Any] = {
            "id": f"log_{int(time.time() * 1000)}",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
        }
        if data:
            payload["data"] = data
        if hypothesis_id:
            payload["hypothesisId"] = hypothesis_id
        try:
            self._debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._debug_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            if not self._debug_log_failure_reported:
                self._debug_log_failure_reported = True
                logger.warning("Could not write chain debug event log: %s", exc)

    def _log(self, level: str, message: str) -> None:
        log_epoch_ms = int(time.time() * 1000)
        self._next_log_event_id += 1
        entry = {
            "id": self._next_log_event_id,
            "time": dt.datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "message": message,
        }
        try:
            self.log_queue.put_nowait(entry)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                self.log_queue.get_nowait()
            self.log_queue.put_nowait(entry)
        getattr(logger, level if level != "warn" else "warning", logger.info)("[chain] %s", message)
        self.state.last_log_epoch_ms = log_epoch_ms
        self.state.last_log_level = level
        self.state.last_log_message = message[:500]

        # Persist errors and warnings to ERRORS.md in the repo
        if level in ("error", "warn") and self.config:
            self._append_error_log(entry["time"], level, message)

    def get_log_events_since(
        self,
        after_id: int,
        *,
        limit: int = 500,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return non-destructive log replay entries newer than ``after_id``.

        Returns ``(events, replay_gap_detected)`` where replay_gap_detected is
        true when the in-memory queue has already dropped entries newer than
        the caller's cursor.
        """
        with self.log_queue.mutex:
            snapshot = list(self.log_queue.queue)
        if not snapshot:
            return [], False
        oldest_id = int(snapshot[0].get("id", 0) or 0)
        replay_gap = bool(after_id > 0 and oldest_id > after_id + 1)
        events = [entry for entry in snapshot if int(entry.get("id", 0) or 0) > after_id]
        if limit > 0 and len(events) > limit:
            events = events[-limit:]
        return events, replay_gap

    def _append_error_log(self, timestamp: str, level: str, message: str) -> None:
        """Append an error/warning entry to the canonical tracker ERRORS.md."""
        try:
            if self._log_tracker is None and self.config is not None:
                self._log_tracker = LogTracker(Path(self.config.repo_path).resolve())
                self._log_tracker.initialize()
            if self._log_tracker is None:
                return
            tag = "ERROR" if level == "error" else "WARN"
            date_str = dt.datetime.now().strftime("%Y-%m-%d")
            self._log_tracker.append(
                "ERRORS.md",
                f"- **[{tag}]** `{date_str} {timestamp}` - {message}",
            )
        except Exception as exc:
            if not self._error_log_failure_reported:
                self._error_log_failure_reported = True
                logger.warning("Could not append runtime error to ERRORS.md: %s", exc)

    def _record_brain_note(
        self,
        event: str,
        summary: str,
        *,
        level: str = "info",
        context: dict | None = None,
    ) -> None:
        """Persist a brain observation note for debugging and analysis."""
        if self._brain_logbook is None:
            return
        self._brain_logbook.record(
            scope="chain",
            event=event,
            summary=summary,
            level=level,
            context=context or {},
        )

    def _record_history_note(
        self,
        event: str,
        summary: str,
        *,
        level: str = "info",
        context: dict | None = None,
    ) -> None:
        """Persist a run-history note for user-visible auditing."""
        if self._history_logbook is None:
            return
        self._history_logbook.record(
            scope="chain",
            event=event,
            summary=summary,
            level=level,
            context=context or {},
        )

    def _append_chain_debug_event(self, repo: Path, payload: dict[str, Any]) -> None:
        """Append one structured debug event for chain step troubleshooting."""
        try:
            path = repo / ".codex_manager" / "logs" / "CHAIN_DEBUG.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Could not append chain debug event: %s", exc)

    def _finalize_run(
        self,
        *,
        start_time: float,
        history_level: str = "info",
        extra_history_context: dict[str, Any] | None = None,
    ) -> None:
        """Finalize state/logging for any chain terminal path."""
        self._stop_after_step_event.clear()
        self.state.stop_after_current_step = False
        self.state.running = False
        self.state.current_step_name = ""
        self.state.current_step_started_at_epoch_ms = 0
        self.state.finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
        self.state.elapsed_seconds = time.monotonic() - start_time
        self._log(
            "info",
            f"Chain finished -- {self.state.stop_reason} "
            f"({self.state.total_loops_completed} loops, "
            f"{self.state.total_steps_completed} steps)",
        )
        history_context = {
            "stop_reason": self.state.stop_reason,
            "total_loops_completed": self.state.total_loops_completed,
            "total_steps_completed": self.state.total_steps_completed,
            "total_tokens": self.state.total_tokens,
            "elapsed_seconds": round(self.state.elapsed_seconds, 1),
        }
        if extra_history_context:
            history_context.update(extra_history_context)
        self._record_history_note(
            "run_finished",
            f"Chain finished with stop_reason='{self.state.stop_reason}'.",
            level=history_level,
            context=history_context,
        )
        if self._on_run_finalized is not None:
            try:
                self._on_run_finalized(self.state)
            except Exception as exc:
                logger.warning("Chain run-finalized callback failed: %s", exc)

    def _consume_stop_after_step_request(self, *, context: str = "step") -> bool:
        """Consume a pending stop-after-step request and set terminal state."""
        if not self._stop_after_step_event.is_set():
            return False
        self._stop_after_step_event.clear()
        self.state.stop_after_current_step = False
        self.state.stop_reason = "user_stopped_after_step"
        self._log(
            "warn",
            f"Stop-after-step request honored; stopping before next {context}.",
        )
        return True

    @staticmethod
    def _memory_log_path(repo: Path) -> Path:
        """Return the persistent chain memory log file path."""
        return repo / ".codex_manager" / "memory" / "chain_step_memory.jsonl"

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        """Trim text to a bounded size for prompts/log persistence."""
        clean = re.sub(r"\s+", " ", (text or "")).strip()
        if len(clean) <= max_chars:
            return clean
        return clean[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _looks_like_placeholder_agent_output(text: str) -> bool:
        """Detect generic status replies that are not meaningful step output."""
        normalized = re.sub(r"\s+", " ", (text or "")).strip().lower().replace("\u2019", "'")
        if not normalized:
            return True

        if (
            normalized.startswith("working in `")
            and "share the task you want implemented" in normalized
        ):
            return True
        if normalized.startswith("working directory set to `"):
            return True
        if normalized.startswith("using `") and "as the working directory" in normalized:
            return True

        placeholders = (
            "share the task you want implemented",
            "share the task you'd like implemented",
            "tell me what you want implemented",
            "provide the task you want implemented",
            "share the task and i'll execute from there",
            "send the task you want implemented",
            "send the task you want me to run",
            "send the specific change you want",
        )
        if any(p in normalized for p in placeholders):
            return True
        return ChainExecutor._looks_like_human_input_request(normalized)

    @staticmethod
    def _looks_like_human_input_request(text: str) -> bool:
        """Detect prompts/responses that ask the human for missing context."""
        normalized = re.sub(r"\s+", " ", (text or "")).strip().lower().replace("\u2019", "'")
        if not normalized:
            return False
        cues = (
            "please provide",
            "confirm the target",
            "if you want me to proceed",
            "if you want me to continue",
            "which command",
            "what specific",
            "share current ui screenshots",
            "share the task",
            "send the task",
        )
        cue_hits = sum(1 for cue in cues if cue in normalized)
        if cue_hits >= 2:
            return True
        if "please provide" in normalized and any(
            tok in normalized for tok in ("(1)", "1)", "1.", "2)", "2.")
        ):
            return True
        return "confirm" in normalized and "to proceed" in normalized

    def _select_agent_output(self, run_result) -> str:
        """Choose the most meaningful text output from a run result."""
        candidates: list[str] = []
        if run_result.final_message:
            candidates.append(run_result.final_message)

        for ev in run_result.events:
            if not ev.text:
                continue
            if ev.kind.value == "agent_message":
                candidates.append(ev.text)

        # Last-resort fallback for runners that do not classify events well.
        for ev in run_result.events:
            if ev.text:
                candidates.append(ev.text)

        seen: set[str] = set()
        for raw in reversed(candidates):
            cleaned = self._truncate_text(raw, 20_000)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            if self._looks_like_placeholder_agent_output(cleaned):
                continue
            return cleaned
        return ""

    def _has_no_progress_streak(self, *, min_results: int = _NO_PROGRESS_STREAK_RESULTS) -> bool:
        """Return True when the latest N attempts produced zero repo deltas."""
        if min_results <= 0:
            return False
        if len(self.state.results) < min_results:
            return False
        recent = self.state.results[-min_results:]
        return all(r.files_changed <= 0 and r.net_lines_changed == 0 for r in recent)

    @staticmethod
    def _step_requires_repo_delta(step: TaskStep) -> bool:
        """Return True when the step should normally produce repository edits."""
        return (step.job_type or "").strip().lower() in _MUTATING_JOB_TYPES

    def _append_execution_evidence(
        self,
        *,
        out_file: Path,
        loop_num: int,
        step_idx: int,
        runner_name: str,
        commit_sha: str | None,
        files_changed: int,
        net_lines_changed: int,
        changed_files: list[dict[str, Any]],
        agent_output: str,
    ) -> None:
        """Append a manager-generated evidence block to the step output file."""
        if files_changed <= 0:
            return

        lines = [
            "",
            "---",
            f"## Execution Evidence (Loop {loop_num}, Step {step_idx + 1})",
            f"- Agent: {runner_name}",
            f"- Files changed: {files_changed}",
            f"- Net lines changed: {net_lines_changed:+d}",
            f"- Commit: {commit_sha or 'not committed'}",
            "- Changed files:",
        ]

        if changed_files:
            for item in changed_files[:40]:
                path = str(item.get("path", "(unknown)"))
                ins = item.get("insertions")
                dels = item.get("deletions")
                if isinstance(ins, int) and isinstance(dels, int):
                    lines.append(f"  - `{path}` (+{ins} / -{dels})")
                else:
                    lines.append(f"  - `{path}` (binary/non-text)")
        else:
            lines.append("  - (details unavailable)")

        summary = self._truncate_text(agent_output, _EVIDENCE_SUMMARY_CHARS)
        if summary:
            lines.append("- Improvement Notes:")
            lines.append("```text")
            lines.append(summary)
            lines.append("```")

        try:
            with out_file.open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            self._log("info", f"Execution evidence saved to {out_file}")
        except Exception as exc:
            self._log("warn", f"Could not append execution evidence to {out_file}: {exc}")

    def _initialize_step_memory(self, repo: Path) -> None:
        """Load persisted cross-run memory used for step-to-step handoff."""
        path = self._memory_log_path(repo)
        self._step_memory_path = path
        self._step_memory_entries = []

        if not path.exists():
            return

        loaded: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        loaded.append(payload)
        except Exception as exc:
            self._log("warn", f"Could not read step memory log: {exc}")
            return

        if loaded:
            self._step_memory_entries = loaded[-_STEP_MEMORY_MAX_ENTRIES:]
            self._log(
                "info",
                f"Loaded {len(self._step_memory_entries)} persisted step memory entries",
            )

    def _record_step_memory(
        self,
        *,
        loop_num: int,
        step_idx: int,
        step: TaskStep,
        step_result: StepResult,
        output_text: str,
    ) -> None:
        """Persist compact step memory for future steps and reruns."""
        excerpt = self._truncate_text(output_text, _STEP_MEMORY_EXCERPT_CHARS)
        if (
            not excerpt
            and step_result.files_changed <= 0
            and step_result.net_lines_changed == 0
            and not step_result.error_message
        ):
            return

        entry: dict[str, Any] = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "loop_number": loop_num,
            "step_index": step_idx + 1,
            "step_name": step.name or step.job_type,
            "job_type": step.job_type,
            "output_file": str(self._step_output_relpath(step)).replace("\\", "/"),
            "test_outcome": step_result.test_outcome,
            "files_changed": step_result.files_changed,
            "net_lines_changed": step_result.net_lines_changed,
            "success": bool(step_result.success),
            "output_excerpt": excerpt,
        }
        if step_result.error_message:
            entry["error_message"] = self._truncate_text(step_result.error_message, 280)

        self._step_memory_entries.append(entry)
        self._step_memory_entries = self._step_memory_entries[-_STEP_MEMORY_MAX_ENTRIES:]

        if self._step_memory_path is None:
            return
        try:
            self._step_memory_path.parent.mkdir(parents=True, exist_ok=True)
            with self._step_memory_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            self._log("warn", f"Could not persist step memory entry: {exc}")

    def _build_step_memory_context(self) -> str:
        """Build compact prompt context from recent persisted step memory."""
        if not self._step_memory_entries:
            return ""

        lines: list[str] = []
        budget = _STEP_MEMORY_CONTEXT_CHARS
        added = 0
        for entry in reversed(self._step_memory_entries):
            if added >= _STEP_MEMORY_CONTEXT_ITEMS or budget <= 0:
                break
            loop_ref = entry.get("loop_number", "?")
            step_ref = entry.get("step_index", "?")
            step_name = str(entry.get("step_name") or "step")
            output_file = str(entry.get("output_file") or "").strip()
            tests = str(entry.get("test_outcome") or "unknown")
            files = int(entry.get("files_changed") or 0)
            net = int(entry.get("net_lines_changed") or 0)
            header = (
                f"- Loop {loop_ref}, Step {step_ref} ({step_name}): "
                f"tests={tests}, files={files}, net_lines={net:+d}"
            )
            if output_file:
                header += f", output_file={output_file}"

            excerpt = str(entry.get("output_excerpt") or "").strip()
            body = f"{header}\n  Output excerpt: {excerpt}" if excerpt else header
            clipped = self._truncate_text(body, min(900, budget))
            if not clipped:
                continue
            lines.append(clipped)
            budget -= len(clipped)
            added += 1

        if not lines:
            return ""
        return "\n".join(lines)

    def _build_vector_memory_context(self, *, step: TaskStep, query_hint: str) -> str:
        """Return long-term memory context for the current step when enabled."""
        vm = self.vector_memory
        if vm is None or not vm.enabled:
            return ""
        query = "\n".join(
            [
                f"step: {step.name or step.job_type}",
                f"job_type: {step.job_type}",
                query_hint[:1200],
            ]
        ).strip()
        if not query:
            return ""
        hits = vm.search(
            query,
            top_k=int(getattr(self.config, "vector_memory_top_k", 8) or 8),
            categories=[
                "chain_step",
                "pipeline_phase",
                "deep_research",
                "scientist_report",
                "recovered_backlog",
            ],
        )
        if not hits:
            return ""
        lines = ["## Similar Past Context (Vector Memory)", ""]
        for hit in hits:
            meta = hit.metadata or {}
            category = str(meta.get("category") or "note").strip() or "note"
            source = str(meta.get("source") or "").strip()
            snippet = re.sub(r"\s+", " ", hit.document).strip()
            if len(snippet) > 260:
                snippet = snippet[:257].rstrip() + "..."
            prefix = f"- [{category}] score={hit.score:.2f}"
            if source:
                prefix += f" source={source}"
            lines.append(f"{prefix}: {snippet}")
        return "\n".join(lines)

    def _record_vector_memory_step(
        self,
        *,
        loop_num: int,
        step_idx: int,
        step: TaskStep,
        step_result: StepResult,
        output_text: str,
    ) -> None:
        """Persist chain step outcomes into vector memory for cross-run reuse."""
        vm = self.vector_memory
        if vm is None or not vm.enabled:
            return
        summary = (
            f"Chain step loop={loop_num} step={step_idx + 1} "
            f"name={step.name or step.job_type} job_type={step.job_type} "
            f"success={step_result.success} tests={step_result.test_outcome} "
            f"files={step_result.files_changed} lines={step_result.net_lines_changed:+d}. "
            f"Output: {self._truncate_text(output_text or step_result.error_message, 1600)}"
        )
        vm.add_note(
            summary[:3500],
            category="chain_step",
            source="chain:step",
            metadata={
                "loop_number": loop_num,
                "step_index": step_idx + 1,
                "step_name": step.name or step.job_type,
                "job_type": step.job_type,
                "success": bool(step_result.success),
                "test_outcome": step_result.test_outcome,
                "files_changed": int(step_result.files_changed),
                "net_lines_changed": int(step_result.net_lines_changed),
            },
        )

    def _archive_previous_outputs(self) -> None:
        """Move existing run outputs into timestamped history folders."""
        if self.output_dir is None or not self.output_dir.exists():
            return
        try:
            existing = list(self.output_dir.iterdir())
        except Exception:
            return
        if not existing:
            return

        archive_root = self.output_dir.parent / "output_history"
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_dir = archive_root / ts
        archive_dir.mkdir(parents=True, exist_ok=True)

        moved = 0
        failures: list[str] = []
        for entry in existing:
            target = archive_dir / entry.name
            try:
                shutil.move(str(entry), str(target))
                moved += 1
                continue
            except Exception:
                pass
            try:
                if entry.is_dir():
                    shutil.copytree(entry, target, dirs_exist_ok=True)
                    shutil.rmtree(entry)
                else:
                    shutil.copy2(entry, target)
                    entry.unlink(missing_ok=True)
                moved += 1
            except Exception as exc:
                failures.append(f"{entry.name}: {exc}")

        if moved:
            self._log("info", f"Archived previous outputs to {archive_dir}")
            self._record_history_note(
                "outputs_archived",
                f"Archived {moved} prior output item(s).",
                context={
                    "archive_dir": str(archive_dir),
                    "moved_items": moved,
                    "failed_items": failures[:5],
                },
            )
        if failures:
            preview = "; ".join(failures[:3])
            if len(failures) > 3:
                preview += "; ..."
            self._log("warn", f"Could not archive some previous outputs: {preview}")

        # Keep most recent output-history archives only.
        try:
            max_archives = 30
            if self.config is not None:
                max_archives = max(
                    1,
                    int(getattr(self.config, "artifact_retention_max_output_runs", 30)),
                )
            archives = sorted(
                [p for p in archive_root.iterdir() if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old in archives[max_archives:]:
                shutil.rmtree(old, ignore_errors=True)
        except Exception:
            pass

    @staticmethod
    def _retention_policy(config: ChainConfig) -> RetentionPolicy:
        """Build runtime-artifact retention policy from chain configuration."""
        return RetentionPolicy(
            enabled=bool(getattr(config, "artifact_retention_enabled", True)),
            max_age_days=max(1, int(getattr(config, "artifact_retention_max_age_days", 30))),
            max_files=max(1, int(getattr(config, "artifact_retention_max_files", 5000))),
            max_bytes=max(1, int(getattr(config, "artifact_retention_max_bytes", 2_000_000_000))),
            max_output_history_runs=max(
                1, int(getattr(config, "artifact_retention_max_output_runs", 30))
            ),
        )

    def _run_retention_cleanup(self, repo: Path, config: ChainConfig, *, reason: str) -> None:
        """Apply retention cleanup for managed runtime artifacts."""
        try:
            cleanup = cleanup_runtime_artifacts(
                repo,
                policy=self._retention_policy(config),
                active_paths=[self.output_dir] if self.output_dir is not None else [],
            )
        except Exception as exc:
            self._log("warn", f"Retention cleanup skipped ({reason}): {exc}")
            return
        removed_total = (
            int(cleanup.get("removed_files", 0))
            + int(cleanup.get("removed_dirs", 0))
            + int(cleanup.get("removed_runs", 0))
        )
        if removed_total <= 0:
            return
        freed_bytes = int(cleanup.get("freed_bytes", 0))
        self._log(
            "info",
            (
                f"Retention cleanup ({reason}): removed "
                f"{cleanup['removed_files']} files, {cleanup['removed_dirs']} dirs, "
                f"{cleanup['removed_runs']} archived runs; freed {freed_bytes} bytes."
            ),
        )

    def _log_execution_mode_warnings(self, config: ChainConfig) -> None:
        """Log prominent warnings when execution is in a constrained safety mode."""
        if config.mode == "dry-run":
            self._log(
                "warn",
                "SAFE MODE ACTIVE: dry-run mode is enabled. Any file edits will be reverted.",
            )

        uses_codex = any(
            step.enabled and (step.agent or "auto") != "claude_code" for step in config.steps
        )
        if uses_codex and config.codex_sandbox_mode == "read-only":
            self._log(
                "warn",
                "READ-ONLY SANDBOX ACTIVE: Codex can inspect files but cannot write changes.",
            )

    # ------------------------------------------------------------------
    # Main loop (runs in a daemon thread)
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        start_time = time.monotonic()
        config = self.config
        if config is None:
            self._log("error", "Chain started without a config")
            self.state.stop_reason = "no_config"
            self._finalize_run(start_time=start_time, history_level="error")
            return

        repo = Path(config.repo_path).resolve()
        self.state.run_max_loops = max(1, int(config.max_loops))
        self.state.run_unlimited = bool(config.unlimited)
        brain_goal = self._brain_goal(config, repo)
        self.ledger = KnowledgeLedger(repo)
        self._history_logbook = None
        try:
            history = HistoryLogbook(repo)
            history.initialize()
            self._history_logbook = history
        except Exception as exc:
            self._log("warn", f"History logbook disabled: {exc}")
        self._record_history_note(
            "run_started",
            f"Chain run started in {config.mode} mode.",
            context={
                "repo": str(repo),
                "chain_name": config.name,
                "mode": config.mode,
                "max_loops": config.max_loops,
                "unlimited": bool(config.unlimited),
                "steps": [s.name or s.job_type for s in config.steps if s.enabled],
                "vector_memory_enabled": bool(config.vector_memory_enabled),
                "vector_memory_backend": config.vector_memory_backend,
            },
        )
        self._initialize_step_memory(repo)
        self.vector_memory = ProjectVectorMemory(
            repo,
            enabled=bool(getattr(config, "vector_memory_enabled", False)),
            backend=str(getattr(config, "vector_memory_backend", "chroma") or "chroma"),
            collection_name=str(getattr(config, "vector_memory_collection", "") or ""),
            default_top_k=int(getattr(config, "vector_memory_top_k", 8) or 8),
        )
        if bool(config.vector_memory_enabled):
            if self.vector_memory.available:
                self._log(
                    "info",
                    "Vector memory enabled: "
                    f"backend={self.vector_memory.backend}, "
                    f"collection={self.vector_memory.collection_name}",
                )
            else:
                self._log("warn", f"Vector memory unavailable: {self.vector_memory.reason}")
        # #region agent log
        self._write_agent_debug_event(
            "chain.py:_run_loop",
            "run_loop started",
            {
                "repo_path": str(config.repo_path),
                "repo_resolved": str(repo),
                "steps_count": len(config.steps),
                "enabled_steps": [s.name or s.job_type for s in config.steps if s.enabled],
            },
            "H1",
        )
        # #endregion
        self._run_retention_cleanup(repo, config, reason="startup")

        # Build agent runners
        runners: dict[str, AgentRunner] = {
            "codex": CodexRunner(
                codex_binary=config.codex_binary,
                timeout=config.timeout_per_step,
                sandbox_mode=config.codex_sandbox_mode,
                approval_policy=config.codex_approval_policy,
                reasoning_effort=config.codex_reasoning_effort,
                bypass_approvals_and_sandbox=config.codex_bypass_approvals_and_sandbox,
            ),
            "claude_code": ClaudeCodeRunner(
                claude_binary=config.claude_binary,
                timeout=config.timeout_per_step,
            ),
        }
        # "auto" maps to codex by default; the brain may override
        runners["auto"] = runners["codex"]

        # Log which agents are configured
        agents_in_use = sorted({s.agent for s in config.steps if s.enabled})
        self._log("info", f"Agents: {', '.join(agents_in_use)}")

        # Verify agent binaries exist on PATH
        import shutil

        binary_map = {
            "codex": config.codex_binary,
            "claude_code": config.claude_binary,
            "auto": config.codex_binary,
        }
        for agent_key in agents_in_use:
            binary = binary_map.get(agent_key, "")
            if binary and not shutil.which(binary):
                self._log(
                    "error",
                    f"Agent binary not found: '{binary}' -- "
                    f"install it or update the binary path in settings. "
                    f"Steps using [{agent_key}] will fail.",
                )

        test_cmd = parse_test_command(config.test_cmd)
        evaluator = RepoEvaluator(test_cmd=test_cmd, skip_tests=(test_cmd is None))

        # Initialise the brain (thinking layer)
        brain = BrainManager(
            BrainConfig(
                enabled=config.brain_enabled,
                model=config.brain_model,
                local_only=getattr(config, "local_only", False),
            )
        )
        self._brain_logbook = None
        if brain.enabled:
            try:
                logbook = BrainLogbook(repo)
                logbook.initialize()
                self._brain_logbook = logbook
            except Exception as exc:
                self._brain_logbook = None
                self._log("warn", f"Brain logbook disabled: {exc}")

            model_tag = brain.config.model
            if brain.config.local_only:
                self._log("info", f"Brain enabled (LOCAL ONLY): {model_tag}")
            else:
                self._log("info", f"Brain enabled: {model_tag}")
            self._record_brain_note(
                "brain_enabled",
                f"Brain enabled with model '{model_tag}'",
                context={
                    "repo": str(repo),
                    "chain_name": config.name,
                    "local_only": bool(brain.config.local_only),
                },
            )

        # Ensure git identity is set so commits don't fail
        try:
            ensure_git_identity(repo)
        except Exception as exc:
            self._log("warn", f"Could not set git identity: {exc}")

        # In apply mode, create a working branch
        if config.mode == "apply":
            try:
                branch = create_branch(repo)
                self._log("info", f"Created branch: {branch}")
            except Exception as exc:
                self._log("error", f"Failed to create branch: {exc}")
                if getattr(self, "ledger", None):
                    self.ledger.add(
                        category="error",
                        title=f"Branch creation failed: {str(exc)[:60]}",
                        detail=str(exc),
                        severity="critical",
                        source="chain:startup",
                        step_ref="branch_creation",
                    )
                self.state.stop_reason = "branch_creation_failed"
                self._finalize_run(
                    start_time=start_time,
                    history_level="error",
                    extra_history_context={"error": str(exc)},
                )
                return

        # Unlimited mode uses a very high ceiling; the improvement-threshold
        # check (inside _check_stop_conditions) is what actually ends the run.
        effective_max = 999_999 if config.unlimited else config.max_loops

        try:
            for loop_num in range(1, effective_max + 1):
                if self._stop_event.is_set():
                    self.state.stop_reason = "user_stopped"
                    break
                if self._consume_stop_after_step_request(context="loop"):
                    break

                self.state.current_loop = loop_num
                enabled_steps = [s for s in config.steps if s.enabled]

                loop_label = (
                    f"\u2501\u2501\u2501 Loop {loop_num} / \u221e "
                    if config.unlimited
                    else f"\u2501\u2501\u2501 Loop {loop_num} / {config.max_loops} "
                )
                self._log(
                    "info",
                    f"{loop_label}({len(enabled_steps)} steps) \u2501\u2501\u2501",
                )

                loop_aborted = False

                # â”€â”€ Parallel execution mode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                if config.parallel_execution and len(enabled_steps) > 1:
                    loop_aborted = self._run_steps_parallel(
                        runners,
                        evaluator,
                        repo,
                        config,
                        loop_num,
                        enabled_steps,
                        brain,
                        start_time,
                    )
                else:
                    # â”€â”€ Sequential execution (default) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                    for step_idx, step in enumerate(config.steps):
                        self._pause_event.wait()
                        if self._stop_event.is_set():
                            self.state.stop_reason = "user_stopped"
                            loop_aborted = True
                            break
                        if self._consume_stop_after_step_request(context="step"):
                            loop_aborted = True
                            break

                        if not step.enabled:
                            continue

                        agent_key = step.agent or "codex"
                        runner = runners.get(agent_key, runners["codex"])
                        self.state.current_step = step_idx
                        step_loops = max(1, step.loop_count)
                        step_label = step.name or step.job_type

                        for step_rep in range(1, step_loops + 1):
                            self._pause_event.wait()
                            if self._stop_event.is_set():
                                self.state.stop_reason = "user_stopped"
                                loop_aborted = True
                                break
                            if self._consume_stop_after_step_request(context="step repetition"):
                                loop_aborted = True
                                break

                            rep_tag = f" (rep {step_rep}/{step_loops})" if step_loops > 1 else ""
                            self.state.current_step_name = f"{step_label}{rep_tag}"
                            self.state.current_step_started_at_epoch_ms = int(time.time() * 1000)
                            self._log(
                                "info",
                                f"Step {step_idx + 1}/{len(config.steps)}: "
                                f"{step_label} [{runner.name}]{rep_tag}",
                            )

                            result = self._execute_step(
                                runner,
                                evaluator,
                                repo,
                                config,
                                loop_num,
                                step_idx,
                                step,
                                brain,
                            )
                            self.state.results.append(result)
                            if (
                                not result.success
                                and result.error_message
                                and getattr(self, "ledger", None)
                            ):
                                self.ledger.add(
                                    category="error",
                                    title=result.error_message[:80] or "Step failed",
                                    detail=result.error_message,
                                    severity="major",
                                    source=f"chain:{result.step_name}",
                                    step_ref=f"loop{loop_num}:step{step_idx}:{step.job_type}",
                                )
                            self.state.total_steps_completed += 1
                            self.state.total_tokens += result.input_tokens + result.output_tokens
                            self.state.elapsed_seconds = time.monotonic() - start_time
                            if self._check_strict_token_budget(config):
                                loop_aborted = True
                                break
                            if self._consume_stop_after_step_request(context="step"):
                                loop_aborted = True
                                break

                            # --- Brain post-evaluation ---
                            if brain.enabled:
                                ledger_ctx = ""
                                if getattr(self, "ledger", None):
                                    ledger_ctx = self.ledger.get_context_for_prompt(
                                        categories=[
                                            "error",
                                            "bug",
                                            "observation",
                                            "suggestion",
                                            "wishlist",
                                            "todo",
                                            "feature",
                                        ],
                                        max_items=15,
                                    )
                                decision = brain.evaluate_step(
                                    step_name=step_label,
                                    success=result.success,
                                    test_outcome=result.test_outcome,
                                    files_changed=result.files_changed,
                                    net_lines=result.net_lines_changed,
                                    errors=([result.error_message] if result.error_message else []),
                                    goal=brain_goal,
                                    ledger_context=ledger_ctx,
                                )
                                self._record_brain_note(
                                    "evaluate_step",
                                    f"Brain selected action '{decision.action}' for step '{step_label}'",
                                    context={
                                        "loop": loop_num,
                                        "step_index": step_idx,
                                        "step_rep": step_rep,
                                        "step_loops": step_loops,
                                        "action": decision.action,
                                        "severity": decision.severity,
                                        "reasoning": decision.reasoning[:400],
                                        "success": result.success,
                                        "test_outcome": result.test_outcome,
                                        "files_changed": result.files_changed,
                                        "net_lines_changed": result.net_lines_changed,
                                    },
                                )
                                if decision.reasoning:
                                    self._log(
                                        "info",
                                        f"Brain: {decision.reasoning[:200]}",
                                    )

                                if decision.action in ("follow_up", "retry"):
                                    if result.terminate_repeats:
                                        self._log(
                                            "info",
                                            f"{TERMINATE_STEP_TAG} detected; skipping brain follow-up for this repetition.",
                                        )
                                        self._record_brain_note(
                                            "follow_up_skipped",
                                            "Skipped brain follow-up because step emitted terminate signal.",
                                            context={
                                                "loop": loop_num,
                                                "step_index": step_idx,
                                                "step_rep": step_rep,
                                                "action": decision.action,
                                            },
                                        )
                                    else:
                                        followup_prompt = (
                                            decision.follow_up_prompt or ""
                                        ).strip() or (
                                            f"Retry step '{step_label}' to resolve remaining issues. "
                                            f"Prior attempt test_outcome={result.test_outcome}, "
                                            f"files_changed={result.files_changed}, "
                                            f"net_lines={result.net_lines_changed:+d}."
                                        )
                                        if self._looks_like_human_input_request(followup_prompt):
                                            self._log(
                                                "warn",
                                                "Brain follow-up requires human input; stopping run to avoid non-productive retries.",
                                            )
                                            self._record_brain_note(
                                                "follow_up_requires_input",
                                                "Brain follow-up requested user input; run stopped.",
                                                level="warn",
                                                context={
                                                    "loop": loop_num,
                                                    "step_index": step_idx,
                                                    "step_rep": step_rep,
                                                    "action": decision.action,
                                                    "prompt_preview": format_prompt_preview(
                                                        followup_prompt
                                                    ),
                                                    "prompt_metadata": prompt_metadata(
                                                        followup_prompt
                                                    ),
                                                },
                                            )
                                            self.state.stop_reason = "brain_needs_input"
                                            loop_aborted = True
                                            break
                                        self._log(
                                            "info",
                                            f"Brain recommends {decision.action} action",
                                        )
                                        self._record_brain_note(
                                            "follow_up_started",
                                            f"Running brain {decision.action} before resuming the normal loop.",
                                            context={
                                                "loop": loop_num,
                                                "step_index": step_idx,
                                                "step_rep": step_rep,
                                                "action": decision.action,
                                                "prompt_preview": format_prompt_preview(
                                                    followup_prompt
                                                ),
                                                "prompt_metadata": prompt_metadata(
                                                    followup_prompt
                                                ),
                                            },
                                        )
                                        self.state.current_step_name = (
                                            f"{step_label} (brain follow-up)"
                                        )
                                        self.state.current_step_started_at_epoch_ms = int(
                                            time.time() * 1000
                                        )
                                        followup = self._execute_step(
                                            runner,
                                            evaluator,
                                            repo,
                                            config,
                                            loop_num,
                                            step_idx,
                                            step,
                                            brain,
                                            override_prompt=followup_prompt,
                                        )
                                        self.state.results.append(followup)
                                        self.state.total_steps_completed += 1
                                        self.state.total_tokens += (
                                            followup.input_tokens + followup.output_tokens
                                        )
                                        self.state.elapsed_seconds = time.monotonic() - start_time
                                        if self._check_strict_token_budget(config):
                                            loop_aborted = True
                                            break
                                        if self._consume_stop_after_step_request(context="step"):
                                            loop_aborted = True
                                            break
                                        self._record_brain_note(
                                            "follow_up_finished",
                                            f"Brain {decision.action} completed; resuming normal loop flow.",
                                            context={
                                                "loop": loop_num,
                                                "step_index": step_idx,
                                                "step_rep": step_rep,
                                                "success": followup.success,
                                                "test_outcome": followup.test_outcome,
                                                "files_changed": followup.files_changed,
                                                "net_lines_changed": followup.net_lines_changed,
                                                "terminate_repeats": followup.terminate_repeats,
                                            },
                                        )
                                        result = followup

                                elif decision.action == "escalate":
                                    self._log(
                                        "error",
                                        f"Brain escalation: {decision.human_message[:300]}",
                                    )
                                    self._record_brain_note(
                                        "escalation",
                                        "Brain escalated and paused the chain.",
                                        level="error",
                                        context={
                                            "loop": loop_num,
                                            "step_index": step_idx,
                                            "step_rep": step_rep,
                                            "human_message": decision.human_message[:400],
                                            "reasoning": decision.reasoning[:400],
                                        },
                                    )
                                    self.state.stop_reason = "brain_escalation"
                                    self.pause()
                                    loop_aborted = True
                                    break
                                elif decision.action == "stop":
                                    self._log(
                                        "info",
                                        "Brain requested stopping this chain run.",
                                    )
                                    self._record_brain_note(
                                        "brain_stop",
                                        "Brain requested stop after post-step evaluation.",
                                        context={
                                            "loop": loop_num,
                                            "step_index": step_idx,
                                            "step_rep": step_rep,
                                            "reasoning": decision.reasoning[:400],
                                        },
                                    )
                                    self.state.stop_reason = "brain_requested_stop"
                                    loop_aborted = True
                                    break

                            if not result.success and step.on_failure == "abort":
                                if brain.enabled and brain.config.auto_fix_errors:
                                    err_decision = brain.handle_error(
                                        result.error_message,
                                        step_label,
                                    )
                                    self._record_brain_note(
                                        "handle_error",
                                        f"Brain selected error action '{err_decision.action}'",
                                        context={
                                            "loop": loop_num,
                                            "step_index": step_idx,
                                            "step_rep": step_rep,
                                            "action": err_decision.action,
                                            "severity": err_decision.severity,
                                            "reasoning": err_decision.reasoning[:400],
                                        },
                                    )
                                    if err_decision.action == "escalate":
                                        self._log(
                                            "error",
                                            f"Brain cannot fix: {err_decision.human_message[:300]}",
                                        )
                                        self._record_brain_note(
                                            "error_escalation",
                                            "Brain could not auto-fix and escalated.",
                                            level="error",
                                            context={
                                                "loop": loop_num,
                                                "step_index": step_idx,
                                                "step_rep": step_rep,
                                                "human_message": err_decision.human_message[:400],
                                            },
                                        )
                                        self.state.stop_reason = "brain_escalation"
                                        self.pause()
                                        loop_aborted = True
                                        break
                                    elif err_decision.action in (
                                        "follow_up",
                                        "retry",
                                    ):
                                        self._log(
                                            "info",
                                            "Brain attempting error recovery...",
                                        )
                                        self._record_brain_note(
                                            "error_recovery_started",
                                            "Brain started an error-recovery step.",
                                            context={
                                                "loop": loop_num,
                                                "step_index": step_idx,
                                                "step_rep": step_rep,
                                                "action": err_decision.action,
                                            },
                                        )
                                        self.state.current_step_name = (
                                            f"{step_label} (brain recovery)"
                                        )
                                        self.state.current_step_started_at_epoch_ms = int(
                                            time.time() * 1000
                                        )
                                        fix_prompt = (
                                            err_decision.follow_up_prompt
                                            or f"Fix this error: {result.error_message[:500]}"
                                        )
                                        fix_result = self._execute_step(
                                            runner,
                                            evaluator,
                                            repo,
                                            config,
                                            loop_num,
                                            step_idx,
                                            step,
                                            brain,
                                            override_prompt=fix_prompt,
                                        )
                                        self.state.results.append(fix_result)
                                        self.state.total_steps_completed += 1
                                        self.state.total_tokens += (
                                            fix_result.input_tokens + fix_result.output_tokens
                                        )
                                        self.state.elapsed_seconds = time.monotonic() - start_time
                                        if self._check_strict_token_budget(config):
                                            loop_aborted = True
                                            break
                                        if self._consume_stop_after_step_request(context="step"):
                                            loop_aborted = True
                                            break
                                        if fix_result.success:
                                            self._log(
                                                "info",
                                                "Brain error recovery succeeded!",
                                            )
                                            self._record_brain_note(
                                                "error_recovery_finished",
                                                "Brain error-recovery succeeded; resuming loop.",
                                                context={
                                                    "loop": loop_num,
                                                    "step_index": step_idx,
                                                    "step_rep": step_rep,
                                                    "success": True,
                                                    "test_outcome": fix_result.test_outcome,
                                                },
                                            )
                                            continue
                                        self._record_brain_note(
                                            "error_recovery_finished",
                                            "Brain error-recovery failed.",
                                            level="warn",
                                            context={
                                                "loop": loop_num,
                                                "step_index": step_idx,
                                                "step_rep": step_rep,
                                                "success": False,
                                                "test_outcome": fix_result.test_outcome,
                                                "error": fix_result.error_message[:400],
                                            },
                                        )
                                self._log(
                                    "error",
                                    "Step failed -- aborting chain (on_failure=abort)",
                                )
                                self.state.stop_reason = "step_failed_abort"
                                loop_aborted = True
                                break

                            if config.stop_on_convergence and self._has_no_progress_streak():
                                self._log(
                                    "info",
                                    f"No repository changes across the latest {_NO_PROGRESS_STREAK_RESULTS} step attempts - stopping early.",
                                )
                                self.state.stop_reason = "no_progress_detected"
                                loop_aborted = True
                                break

                            if result.terminate_repeats and step_rep < step_loops:
                                remaining = step_loops - step_rep
                                self._log(
                                    "info",
                                    f"Step '{step_label}' emitted {TERMINATE_STEP_TAG}; "
                                    f"skipping {remaining} remaining repeat(s).",
                                )
                                break

                        if loop_aborted:
                            break

                if loop_aborted or self.state.stop_reason:
                    break

                self.state.total_loops_completed = loop_num
                self._run_retention_cleanup(repo, config, reason=f"loop-{loop_num}")

                # Brain progress assessment (between loops)
                if brain.enabled and loop_num >= 2:
                    progress = brain.assess_progress(
                        goal=brain_goal,
                        total_loops=loop_num,
                        history_summary=self._build_history_summary(max_entries=20),
                    )
                    self._record_brain_note(
                        "progress_assessment",
                        f"Brain progress assessment action='{progress.action}'",
                        context={
                            "loop": loop_num,
                            "action": progress.action,
                            "reasoning": progress.reasoning[:500],
                        },
                    )
                    if progress.reasoning:
                        self._log("info", f"Brain assessment: {progress.reasoning[:200]}")
                    if progress.action == "stop":
                        self.state.stop_reason = "brain_converged"
                        self._log("info", "Brain recommends stopping -- goal achieved")
                        break

                # Loop-level stop conditions
                stop = self._check_stop_conditions(config, start_time)
                if stop:
                    self.state.stop_reason = stop
                    self._log("info", f"Stopping: {stop}")
                    break
            else:
                self.state.stop_reason = "max_loops_reached"

        except Exception as exc:
            # #region agent log
            self._write_agent_debug_event(
                "chain.py:_run_loop",
                "loop exception",
                {"error": str(exc), "error_type": type(exc).__name__},
                "H5",
            )
            # #endregion
            self._log("error", f"Unexpected error: {exc}")
            if getattr(self, "ledger", None):
                self.ledger.add(
                    category="error",
                    title=f"Chain error: {str(exc)[:60]}",
                    detail=str(exc),
                    severity="critical",
                    source="chain:runtime",
                    step_ref="",
                )
            self.state.stop_reason = f"error: {exc}"

        finally:
            self._finalize_run(start_time=start_time)

    # ------------------------------------------------------------------
    # Single step execution
    # ------------------------------------------------------------------

    def _run_agent_with_keepalive(
        self,
        runner: AgentRunner,
        repo: Path,
        prompt: str,
        *,
        activity_label: str,
        timeout_seconds: int,
        full_auto: bool,
    ):
        """Run an agent and emit periodic keepalive logs while it is busy."""
        started = time.monotonic()
        done = threading.Event()
        holder: dict[str, object] = {}

        def _worker() -> None:
            try:
                holder["result"] = runner.run(repo, prompt, full_auto=full_auto)
            except Exception as exc:  # pragma: no cover - defensive passthrough
                holder["error"] = exc
            finally:
                done.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        while not done.wait(timeout=20.0):
            elapsed = int(time.monotonic() - started)
            if timeout_seconds > 0:
                remaining = max(0, timeout_seconds - elapsed)
                self._log(
                    "info",
                    f"{activity_label} still running ({elapsed}s elapsed, inactivity timeout {timeout_seconds}s, about {remaining}s remaining)",
                )
            else:
                self._log(
                    "info",
                    f"{activity_label} still running ({elapsed}s elapsed, inactivity timeout disabled)",
                )

        t.join()
        err = holder.get("error")
        if err is not None:
            raise err
        if "result" not in holder:
            raise RuntimeError("Agent run finished without returning a result")
        return holder["result"]

    def _execute_step(
        self,
        runner: AgentRunner,
        evaluator: RepoEvaluator,
        repo: Path,
        config: ChainConfig,
        loop_num: int,
        step_idx: int,
        step: TaskStep,
        brain: BrainManager | None = None,
        override_prompt: str | None = None,
    ) -> StepResult:
        # #region agent log
        self._write_agent_debug_event(
            "chain.py:_execute_step",
            "execute_step entry",
            {
                "step_idx": step_idx,
                "step_name": step.name or step.job_type,
                "repo": str(repo),
                "loop_num": loop_num,
            },
            "H1",
        )
        # #endregion

        # â”€â”€ CUA visual test step â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if step.job_type == "visual_test" and not override_prompt:
            cua_result = self._execute_cua_step(config, loop_num, step_idx, step)
            self._record_history_note(
                "step_result",
                (
                    f"Loop {loop_num}, Step {step_idx + 1} '{cua_result.step_name}' "
                    f"finished with status={'ok' if cua_result.success else 'failed'} "
                    f"and tests={cua_result.test_outcome}."
                ),
                level="info" if cua_result.success else "warn",
                context={
                    "loop_number": loop_num,
                    "step_index": step_idx,
                    "step_name": cua_result.step_name,
                    "job_type": cua_result.job_type,
                    "agent_used": cua_result.agent_used,
                    "mode": config.mode,
                    "agent_success": cua_result.agent_success,
                    "validation_success": cua_result.validation_success,
                    "tests_passed": cua_result.tests_passed,
                    "success": cua_result.success,
                    "test_outcome": cua_result.test_outcome,
                    "files_changed": cua_result.files_changed,
                    "net_lines_changed": cua_result.net_lines_changed,
                    "duration_seconds": cua_result.duration_seconds,
                    "error_message": self._truncate_text(
                        cua_result.error_message, _HISTORY_ERROR_CONTEXT_CHARS
                    ),
                },
            )
            return cua_result

        if override_prompt:
            prompt = override_prompt
        else:
            prompt = self._resolve_prompt(step, loop_num, step_idx)
            # Inject knowledge ledger context (open errors, observations, etc.)
            ledger_ctx = self._get_ledger_context_for_step(step)
            if ledger_ctx:
                prompt = prompt + "\n\n" + ledger_ctx

        # Let the brain refine the prompt if enabled
        if brain and brain.enabled and not override_prompt:
            original_prompt = prompt
            history_summary = self._build_history_summary()
            ledger_ctx = ""
            if getattr(self, "ledger", None):
                ledger_ctx = self.ledger.get_context_for_prompt(
                    categories=[
                        "error",
                        "bug",
                        "observation",
                        "suggestion",
                        "wishlist",
                        "todo",
                        "feature",
                    ],
                    max_items=15,
                )
            brain_goal = self._brain_goal(config, repo)
            prompt = brain.plan_step(
                goal=brain_goal,
                step_name=step.name or step.job_type,
                base_prompt=prompt,
                history_summary=history_summary,
                ledger_context=ledger_ctx,
            )
            self._record_brain_note(
                "plan_step",
                "Brain refined a step prompt before execution.",
                context={
                    "loop": loop_num,
                    "step_index": step_idx,
                    "step_name": step.name or step.job_type,
                    "prompt_changed": prompt != original_prompt,
                    "original_length": len(original_prompt),
                    "refined_length": len(prompt),
                },
            )

        # Append file I/O instructions so the agent knows where to write
        # and what files from prior steps are available to read.
        prompt = self._append_file_instructions(prompt, repo, config, step, step_idx)

        self._log("info", format_prompt_log_line(prompt))

        step_start = time.monotonic()
        start_head_sha = ""
        try:
            start_head_sha = head_sha(repo)
        except Exception:
            start_head_sha = ""
        artifact_snapshot = capture_artifact_snapshot(
            repo,
            extra_globs=_CHAIN_MANAGED_ARTIFACT_GLOBS,
        )
        max_attempts = (step.max_retries + 1) if step.on_failure == "retry" else 1

        # Debug: log the working directory and command details
        self._debug_log(f"Agent working dir: {repo}")
        self._debug_log(f"Agent: {runner.name}, full_auto=True")
        self._debug_log(f"Prompt length: {len(prompt)} chars")

        run_result = None
        for attempt in range(1, max_attempts + 1):
            if self._stop_event.is_set():
                break
            self._pause_event.wait()
            # Always pass full_auto=True -- agents run non-interactively and
            # can't prompt for approval.  Dry-run vs apply controls whether
            # changes are committed or reverted AFTER the agent finishes.
            run_result = self._run_agent_with_keepalive(
                runner,
                repo,
                prompt,
                activity_label=f"Step '{step.name or step.job_type}'",
                timeout_seconds=config.timeout_per_step,
                full_auto=True,
            )

            # Debug: log full result details
            self._debug_log(
                f"Agent result: success={run_result.success}, "
                f"exit={run_result.exit_code}, "
                f"events={len(run_result.events)}, "
                f"duration={run_result.duration_seconds:.1f}s, "
                f"final_msg_len={len(run_result.final_message)}, "
                f"file_changes={len(run_result.file_changes)}, "
                f"errors={run_result.errors[:2] if run_result.errors else '[]'}"
            )
            # #region agent log
            self._write_agent_debug_event(
                "chain.py:_execute_step",
                "after runner.run",
                {
                    "success": run_result.success,
                    "file_changes_count": len(run_result.file_changes),
                    "final_message_len": len(run_result.final_message),
                    "events_count": len(run_result.events),
                    "errors": run_result.errors[:3] if run_result.errors else [],
                    "step_idx": step_idx,
                },
                "H2,H3,H4",
            )
            # #endregion
            # Debug: log event types seen
            evt_summary = {}
            for ev in run_result.events:
                k = ev.kind.value
                evt_summary[k] = evt_summary.get(k, 0) + 1
            self._debug_log(f"Event types: {evt_summary}")

            if run_result.success:
                break
            # Log agent failure details so the user can see what went wrong
            err_detail = "; ".join(run_result.errors) if run_result.errors else "unknown error"
            duration_tag = (
                f" ({run_result.duration_seconds:.1f}s)" if run_result.duration_seconds else ""
            )
            self._log(
                "error",
                f"Agent failed (exit {run_result.exit_code}){duration_tag}: {err_detail[:300]}",
            )
            if attempt < max_attempts:
                self._log("warn", f"Attempt {attempt}/{max_attempts} failed, retrying...")

        if run_result is None:
            from codex_manager.schemas import RunResult

            run_result = RunResult(success=False, errors=["Execution cancelled"])

        # Keep only meaningful output. Ignore boilerplate status lines such as
        # "Working in <repo> now. Share the task..." that waste step memory.
        all_text_parts = [ev.text for ev in run_result.events if ev.text]
        agent_message_parts = [
            ev.text for ev in run_result.events if ev.text and ev.kind.value == "agent_message"
        ]
        agent_output = self._select_agent_output(run_result)
        terminate_repeats = contains_terminate_step_signal(agent_output)
        if terminate_repeats:
            self._log(
                "info",
                f"Agent emitted {TERMINATE_STEP_TAG}; remaining repeats for this step can be skipped.",
            )

        self._debug_log(
            f"Output capture: final_msg={len(run_result.final_message)} chars, "
            f"all_events_text={len(all_text_parts)} parts ({sum(len(t) for t in all_text_parts)} chars), "
            f"agent_event_text={len(agent_message_parts)} parts, "
            f"agent_output={len(agent_output)} chars, "
            f"file_changes={len(run_result.file_changes)}"
        )

        # If the agent didn't create/edit files itself, save its text output
        # to a markdown file so work isn't lost.
        out_file = self._step_output_path(repo, step)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        # #region agent log
        self._write_agent_debug_event(
            "chain.py:_execute_step",
            "save decision",
            {
                "agent_output_len": len(agent_output),
                "file_changes_count": len(run_result.file_changes),
                "will_save": bool(agent_output and not run_result.file_changes),
                "out_file": str(out_file),
                "step_idx": step_idx,
            },
            "H3,H4",
        )
        # #endregion
        if agent_output and not run_result.file_changes:
            self._debug_log(f"Saving output to: {out_file}")
            try:
                with open(out_file, "a", encoding="utf-8") as f:
                    header = f"\n\n---\n## Loop {loop_num}, Step {step_idx + 1}\n\n"
                    f.write(header + agent_output + "\n")
                self._log("info", f"Agent output saved to {out_file} ({len(agent_output)} chars)")
                # #region agent log
                self._write_agent_debug_event(
                    "chain.py:_execute_step",
                    "saved to file",
                    {"out_file": str(out_file), "chars": len(agent_output)},
                    "H5",
                )
                # #endregion
            except Exception as exc:
                self._log("error", f"Could not save agent output to {out_file}: {exc}")
                # #region agent log
                self._write_agent_debug_event(
                    "chain.py:_execute_step",
                    "save failed",
                    {"out_file": str(out_file), "error": str(exc)},
                    "H5",
                )
                # #endregion
        elif agent_output and run_result.file_changes:
            # Preserve step output even when the agent also changed code files.
            if out_file.exists():
                self._debug_log(f"Agent created output file directly: {out_file}")
            else:
                try:
                    with open(out_file, "a", encoding="utf-8") as f:
                        header = (
                            f"\n\n---\n## Loop {loop_num}, Step {step_idx + 1} "
                            "(captured fallback)\n\n"
                        )
                        f.write(header + agent_output + "\n")
                    self._log(
                        "info",
                        f"Captured fallback output to {out_file} ({len(agent_output)} chars)",
                    )
                except Exception as exc:
                    self._log("warn", f"Could not capture fallback output: {exc}")
        elif run_result.file_changes:
            self._log(
                "info",
                "Agent changed files but produced no meaningful text output to persist.",
            )
        else:
            self._log(
                "warn",
                f"{runner.name} produced no text output and no file changes. "
                f"Events: {len(run_result.events)}, "
                f"Exit: {run_result.exit_code}, "
                f"Errors: {len(run_result.errors)}",
            )
            # Debug: dump first few events raw
            for i, ev in enumerate(run_result.events[:5]):
                import json as _json

                raw_str = _json.dumps(ev.raw)[:200]
                self._debug_log(f"Event[{i}]: kind={ev.kind.value} raw={raw_str}")

        # Evaluate
        eval_result = evaluator.evaluate(repo)
        artifact_entries, artifact_insertions, artifact_deletions = summarize_artifact_delta(
            repo,
            artifact_snapshot,
            extra_globs=_CHAIN_MANAGED_ARTIFACT_GLOBS,
        )
        artifact_merge = merge_eval_result_with_artifact_delta(eval_result, artifact_entries)
        artifact_net = artifact_merge["insertions"] - artifact_merge["deletions"]
        if artifact_merge["files_added"] > 0:
            self._log(
                "info",
                (
                    "Managed artifacts: "
                    f"{artifact_merge['files_added']} files, net {artifact_net:+d} "
                    "(counted outside git numstat)."
                ),
            )
        outcome_level = (
            "info" if eval_result.test_outcome.value in ("passed", "skipped") else "warn"
        )
        self._log(
            outcome_level,
            f"Tests: {eval_result.test_outcome.value} | "
            f"Files: {eval_result.files_changed} | "
            f"Net \u0394: {eval_result.net_lines_changed:+d}",
        )
        repo_dirty = bool((eval_result.status_porcelain or "").strip())
        agent_authored_commit_sha: str | None = None
        end_head_sha = ""
        head_advanced = False
        if start_head_sha:
            try:
                end_head_sha = head_sha(repo)
            except Exception:
                end_head_sha = ""
            head_advanced = bool(end_head_sha and end_head_sha != start_head_sha)

        if eval_result.files_changed <= 0 and head_advanced and start_head_sha and end_head_sha:
            revspec = f"{start_head_sha}..{end_head_sha}"
            changed_entries = diff_numstat_entries(repo, revspec=revspec)
            files_changed, ins, dels = summarize_numstat_entries(changed_entries)
            if files_changed > 0:
                eval_result.files_changed = files_changed
                eval_result.net_lines_changed = ins - dels
                eval_result.changed_files = changed_entries
                eval_result.diff_stat = diff_stat(repo, revspec=revspec)
                if config.mode == "apply":
                    agent_authored_commit_sha = end_head_sha
                self._log(
                    "info",
                    (
                        f"Detected agent-authored commit {end_head_sha} "
                        f"for this step ({files_changed} files, net {eval_result.net_lines_changed:+d})."
                    ),
                )
        if eval_result.files_changed > 0:
            changed_preview = ", ".join(
                str(item.get("path", "(unknown)")) for item in eval_result.changed_files[:8]
            )
            if changed_preview:
                self._log(
                    "info",
                    f"Changed files ({eval_result.files_changed}) via {runner.name}: {changed_preview}",
                )

        # Commit or revert
        commit_sha = None
        if config.mode == "apply" and repo_dirty:
            try:
                msg = generate_commit_message(
                    loop_num * 100 + step_idx,
                    step.name or step.job_type,
                    eval_result.test_outcome.value,
                )
                commit_sha = commit_all(repo, msg)
                self._log("info", f"Committed: {commit_sha}")
            except Exception as exc:
                self._log("error", f"Commit failed: {exc}")
                if getattr(self, "ledger", None):
                    self.ledger.add(
                        category="error",
                        title=f"Commit failed: {str(exc)[:60]}",
                        detail=str(exc),
                        severity="major",
                        source="chain:commit",
                        step_ref=f"loop{loop_num}:step{step_idx}",
                    )
        elif config.mode == "apply" and agent_authored_commit_sha:
            commit_sha = agent_authored_commit_sha
            self._log("info", f"Using agent-authored commit: {commit_sha}")
        elif config.mode == "dry-run":
            if head_advanced and start_head_sha:
                try:
                    reset_to_ref(repo, start_head_sha)
                    self._log(
                        "info",
                        (
                            "Dry-run rollback restored repository to pre-step HEAD "
                            f"({start_head_sha})."
                        ),
                    )
                except Exception as exc:
                    self._log("warn", f"Could not reset dry-run commit(s): {exc}")
                    if not is_clean(repo):
                        revert_all(repo)
                        self._log("info", "Changes reverted (dry-run)")
            elif repo_dirty:
                revert_all(repo)
                self._log("info", "Changes reverted (dry-run)")

        self._append_execution_evidence(
            out_file=out_file,
            loop_num=loop_num,
            step_idx=step_idx,
            runner_name=runner.name,
            commit_sha=commit_sha,
            files_changed=eval_result.files_changed,
            net_lines_changed=eval_result.net_lines_changed,
            changed_files=eval_result.changed_files,
            agent_output=agent_output,
        )

        tests_outcome = eval_result.test_outcome.value
        tests_passed = tests_outcome == "passed"
        tests_validation_success = tests_outcome in ("passed", "skipped")
        requires_repo_delta = self._step_requires_repo_delta(step)
        has_repo_delta = (
            eval_result.files_changed > 0 or eval_result.net_lines_changed != 0 or bool(commit_sha)
        )
        repo_delta_success = (not requires_repo_delta) or has_repo_delta or terminate_repeats
        validation_success = tests_validation_success and repo_delta_success
        final_success = bool(run_result.success) and validation_success

        validation_failures: list[str] = []
        if not tests_validation_success:
            validation_failures.append(f"tests={tests_outcome}")
        if requires_repo_delta and not has_repo_delta and not terminate_repeats:
            validation_failures.append("no repository changes detected")
        if validation_failures and run_result.success:
            self._log(
                "warn",
                "Validation marked step as failed despite agent exit success: "
                + ", ".join(validation_failures),
            )

        error_message = "; ".join(run_result.errors) if run_result.errors else ""
        if validation_failures:
            validation_msg = "Validation failed: " + ", ".join(validation_failures)
            error_message = (
                f"{error_message}; {validation_msg}" if error_message else validation_msg
            )

        duration = time.monotonic() - step_start
        step_result = StepResult(
            loop_number=loop_num,
            step_index=step_idx,
            step_name=step.name or step.job_type,
            job_type=step.job_type,
            agent_used=runner.name,
            prompt_used=prompt,
            terminate_repeats=terminate_repeats,
            agent_success=bool(run_result.success),
            validation_success=validation_success,
            tests_passed=tests_passed,
            success=final_success,
            test_outcome=tests_outcome,
            files_changed=eval_result.files_changed,
            net_lines_changed=eval_result.net_lines_changed,
            changed_files=eval_result.changed_files,
            commit_sha=commit_sha,
            error_message=error_message,
            duration_seconds=round(duration, 1),
            output_chars=len(agent_output),
            input_tokens=run_result.usage.input_tokens,
            output_tokens=run_result.usage.output_tokens,
        )
        self._record_step_memory(
            loop_num=loop_num,
            step_idx=step_idx,
            step=step,
            step_result=step_result,
            output_text=agent_output,
        )
        self._record_vector_memory_step(
            loop_num=loop_num,
            step_idx=step_idx,
            step=step,
            step_result=step_result,
            output_text=agent_output,
        )
        self._record_history_note(
            "step_result",
            (
                f"Loop {loop_num}, Step {step_idx + 1} '{step_result.step_name}' "
                f"finished with status={'ok' if step_result.success else 'failed'} "
                f"and tests={step_result.test_outcome}."
            ),
            level="info" if step_result.success else "warn",
            context={
                "loop_number": loop_num,
                "step_index": step_idx,
                "step_name": step_result.step_name,
                "job_type": step_result.job_type,
                "agent_used": step_result.agent_used,
                "mode": config.mode,
                "agent_success": step_result.agent_success,
                "validation_success": step_result.validation_success,
                "tests_passed": step_result.tests_passed,
                "success": step_result.success,
                "test_outcome": step_result.test_outcome,
                "files_changed": step_result.files_changed,
                "net_lines_changed": step_result.net_lines_changed,
                "changed_files": step_result.changed_files,
                "duration_seconds": step_result.duration_seconds,
                "output_chars": step_result.output_chars,
                "input_tokens": step_result.input_tokens,
                "output_tokens": step_result.output_tokens,
                "total_tokens": step_result.input_tokens + step_result.output_tokens,
                "model": str(run_result.usage.model or "").strip(),
                "commit_sha": step_result.commit_sha,
                "terminate_repeats": step_result.terminate_repeats,
                "error_message": self._truncate_text(
                    step_result.error_message, _HISTORY_ERROR_CONTEXT_CHARS
                ),
                "output_file": str(out_file),
            },
        )
        changed_preview = []
        for entry in eval_result.changed_files[:40]:
            changed_preview.append(
                {
                    "path": str(entry.get("path") or ""),
                    "insertions": entry.get("insertions"),
                    "deletions": entry.get("deletions"),
                    "source": entry.get("source", "git"),
                }
            )
        prompt_meta = prompt_metadata(prompt)
        self._append_chain_debug_event(
            repo,
            {
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "loop_number": loop_num,
                "step_index": step_idx,
                "step_name": step.name or step.job_type,
                "job_type": step.job_type,
                "mode": config.mode,
                "runner": runner.name,
                "prompt_length": prompt_meta["length_chars"],
                "prompt_sha256": prompt_meta["sha256"],
                "prompt_redaction_hits": prompt_meta["redaction_hits"],
                "prompt_preview": format_prompt_preview(prompt),
                "prompt_metadata": prompt_meta,
                "run": {
                    "success": bool(run_result.success),
                    "exit_code": run_result.exit_code,
                    "duration_seconds": run_result.duration_seconds,
                    "events_count": len(run_result.events),
                    "file_changes_count": len(run_result.file_changes),
                    "errors": [self._truncate_text(err, 4000) for err in run_result.errors],
                    "usage": {
                        "input_tokens": run_result.usage.input_tokens,
                        "output_tokens": run_result.usage.output_tokens,
                        "total_tokens": run_result.usage.total_tokens,
                        "model": run_result.usage.model,
                    },
                },
                "git": {
                    "start_head_sha": start_head_sha,
                    "end_head_sha": end_head_sha,
                    "head_advanced": head_advanced,
                    "repo_dirty": repo_dirty,
                    "commit_sha": commit_sha,
                },
                "artifact_delta": {
                    "snapshot_files": len(artifact_snapshot),
                    "changed_files": len(artifact_entries),
                    "insertions": artifact_insertions,
                    "deletions": artifact_deletions,
                    "merged_files": artifact_merge["files_added"],
                    "merged_net_lines": artifact_net,
                },
                "metrics": {
                    "test_outcome": eval_result.test_outcome.value,
                    "test_exit_code": eval_result.test_exit_code,
                    "files_changed": eval_result.files_changed,
                    "net_lines_changed": eval_result.net_lines_changed,
                    "changed_files_total": len(eval_result.changed_files),
                    "changed_files_preview": changed_preview,
                },
                "validation": {
                    "agent_success": step_result.agent_success,
                    "tests_validation_success": tests_validation_success,
                    "requires_repo_delta": requires_repo_delta,
                    "has_repo_delta": has_repo_delta,
                    "terminate_repeats": terminate_repeats,
                    "validation_failures": validation_failures,
                    "validation_success": step_result.validation_success,
                    "final_success": step_result.success,
                },
                "messages": {
                    "test_summary": self._truncate_text(eval_result.test_summary, 2000),
                    "error_message": self._truncate_text(error_message, 4000),
                    "agent_output_excerpt": self._truncate_text(agent_output, 2000),
                },
            },
        )
        return step_result

    # ------------------------------------------------------------------
    # File I/O instructions injected into prompts
    # ------------------------------------------------------------------

    @staticmethod
    def _step_output_filename(step: TaskStep) -> str:
        """Derive an output filename from the step name."""
        raw = (step.name or step.job_type or "step").strip()
        slug = re.sub(r"[^\w\-]+", "-", raw).strip("-")
        return f"{slug or 'step'}.md"

    @classmethod
    def _step_output_relpath(cls, step: TaskStep) -> Path:
        """Path to a step output file relative to repo root."""
        return Path(".codex_manager") / "outputs" / cls._step_output_filename(step)

    @classmethod
    def _step_output_path(cls, repo: Path, step: TaskStep) -> Path:
        """Absolute path to a step output file under repo/.codex_manager/outputs."""
        return repo / cls._step_output_relpath(step)

    def _append_file_instructions(
        self,
        prompt: str,
        repo: Path,
        config: ChainConfig,
        step: TaskStep,
        step_idx: int,
    ) -> str:
        """Append instructions telling the agent where to write and what to read."""
        out_file = str(self._step_output_relpath(step)).replace("\\", "/")

        # Discover output files from OTHER steps that already exist
        available_files: list[str] = []
        for other in config.steps:
            if other.id == step.id or not other.enabled:
                continue
            other_file = self._step_output_relpath(other)
            if (repo / other_file).exists():
                available_files.append(str(other_file).replace("\\", "/"))

        handoff_file = ""
        for prev_idx in range(step_idx - 1, -1, -1):
            prev_step = config.steps[prev_idx]
            if not prev_step.enabled or prev_step.id == step.id:
                continue
            prev_file = self._step_output_relpath(prev_step)
            if (repo / prev_file).exists():
                handoff_file = str(prev_file).replace("\\", "/")
                break

        lines = [
            "",
            "---",
            "## Repository Scope (Strict)",
            f"You are operating inside repository `{repo}`.",
            "Use this repository as the single source of truth for this run.",
            "Treat chain/config labels as task labels only, not repository identifiers.",
            "Do not search for, switch to, or request any other repository/project.",
            "Do not read or modify files outside this repository root.",
            "",
            "---",
            "## Output Instructions",
            f"Write all of your output to the file `{out_file}`.",
            f"If `{out_file}` already exists, APPEND your new content to the end - do not overwrite previous content.",
        ]

        allow_path_creation = bool(getattr(config, "allow_path_creation", True))
        dep_policy = str(
            getattr(config, "dependency_install_policy", "project_only") or "project_only"
        ).strip().lower()
        if dep_policy not in {"disallow", "project_only", "allow_system"}:
            dep_policy = "project_only"

        image_enabled = bool(getattr(config, "image_generation_enabled", False))
        image_provider = str(getattr(config, "image_provider", "openai") or "openai").strip().lower()
        if image_provider not in {"openai", "google"}:
            image_provider = "openai"
        image_model = str(getattr(config, "image_model", "") or "").strip()
        if not image_model:
            image_model = "gpt-image-1" if image_provider == "openai" else "nano-banana"

        lines.append("")
        lines.append("---")
        lines.append("## Capability Contract")
        lines.append(
            "- File and directory creation inside repository: "
            + ("allowed." if allow_path_creation else "not allowed; edit existing paths only.")
        )

        if dep_policy == "disallow":
            lines.append(
                "- Dependency installation: not allowed. Do not run pip/npm/brew/apt/choco install commands."
            )
        elif dep_policy == "project_only":
            lines.append(
                "- Dependency installation: allowed only for project-scoped environments "
                "(for example venv, uv, poetry, npm/pnpm/yarn in this repo)."
            )
            lines.append(
                "- Do not install global/system-wide dependencies. Prefer pinned versions and minimal additions."
            )
        else:
            lines.append(
                "- Dependency installation: system-wide and project-scoped installs are allowed when required."
            )
            lines.append(
                "- Keep changes minimal, document why the install is needed, and include rollback notes."
            )

        if image_enabled:
            lines.append(
                f"- Image generation: enabled using provider `{image_provider}` and model `{image_model}`."
            )
            if image_provider == "openai":
                lines.append(
                    "- Requires OPENAI_API_KEY or CODEX_API_KEY (or Codex CLI auth) in environment."
                )
            else:
                lines.append("- Requires GOOGLE_API_KEY or GEMINI_API_KEY in environment.")
            lines.append(
                "- If you generate assets, save files under the repository "
                "(for example assets/icons, docs/images)."
            )
        else:
            lines.append("- Image generation: disabled for this run.")

        if handoff_file:
            lines.append("")
            lines.append("## Step Handoff")
            lines.append(f"Primary handoff file from the previous step: `{handoff_file}`")
            lines.append(
                "Read it before making changes, then carry forward concrete findings into this step."
            )

        if available_files:
            lines.append("")
            lines.append("## Available Input Files")
            lines.append(
                "The following files contain output from previous steps. Read them if relevant:"
            )
            for f in available_files:
                lines.append(f"- `{f}`")

        lines.append("")
        lines.append("## Repeat Control")
        lines.append(terminate_step_instruction("step repeat"))

        return prompt + "\n" + "\n".join(lines) + "\n"
    # ------------------------------------------------------------------
    # CUA visual test step execution
    # ------------------------------------------------------------------

    def _execute_cua_step(
        self,
        config: ChainConfig,
        loop_num: int,
        step_idx: int,
        step: TaskStep,
    ) -> StepResult:
        """Execute a CUA visual test step within the chain."""
        step_start = time.monotonic()
        step_label = step.name or "Visual Test (CUA)"

        try:
            from codex_manager.cua.actions import CUAProvider, CUASessionConfig
            from codex_manager.cua.session import run_cua_session_sync
        except ImportError as exc:
            self._log("error", f"CUA dependencies not installed: {exc}")
            return StepResult(
                loop_number=loop_num,
                step_index=step_idx,
                step_name=step_label,
                job_type="visual_test",
                agent_used="cua",
                agent_success=False,
                validation_success=False,
                tests_passed=False,
                success=False,
                error_message=f"CUA not installed: {exc}",
                duration_seconds=round(time.monotonic() - step_start, 1),
            )

        # Determine provider: step override > chain config > default
        provider_str = step.cua_provider or config.cua_provider or "anthropic"
        provider = CUAProvider.ANTHROPIC if provider_str == "anthropic" else CUAProvider.OPENAI

        # Determine target URL: step override > chain config
        target_url = step.cua_target_url or config.cua_target_url or ""

        # Build the task prompt
        task = (
            config.cua_task
            or step.custom_prompt
            or (
                "Visually inspect the application UI. Navigate through the main views, "
                "test interactive elements (buttons, forms, dropdowns), and report any "
                "visual bugs, broken layouts, or usability issues you find."
            )
        )

        cua_config = CUASessionConfig(
            provider=provider,
            target_url=target_url,
            task=task,
            headless=getattr(config, "cua_headless", True),
            max_steps=30,
            timeout_seconds=config.timeout_per_step,
            save_screenshots=True,
        )

        self._log("info", f"CUA: {provider.value} -> {target_url or '(no URL)'}")
        self._log("info", f"CUA task: {task[:120]}...")

        step_ref = f"loop{loop_num}:step{step_idx}:visual_test"
        ledger = getattr(self, "ledger", None)
        try:
            result = run_cua_session_sync(cua_config, ledger=ledger, step_ref=step_ref)

            self._log(
                "info" if result.success else "warn",
                f"CUA finished: {result.total_steps} steps, "
                f"{result.duration_seconds}s, success={result.success}",
            )

            # Log observations
            if result.observations:
                self._log("info", f"CUA found {len(result.observations)} observations:")
                for obs in result.observations[:10]:
                    icon = {
                        "critical": "[CRIT]",
                        "major": "[MAJOR]",
                        "minor": "[MINOR]",
                        "cosmetic": "[COSMETIC]",
                        "positive": "[POSITIVE]",
                    }.get(obs.severity, "*")
                    self._log("info", f"  {icon} [{obs.severity}] {obs.element}: {obs.actual[:80]}")

            if result.summary:
                # Clean summary (remove raw OBSERVATION lines)
                clean = "\n".join(
                    summary_line
                    for summary_line in result.summary.splitlines()
                    if not summary_line.strip().upper().startswith("OBSERVATION|")
                ).strip()
                if clean:
                    self._log("info", f"CUA summary: {clean[:300]}")

            if result.error:
                self._log("error", f"CUA error: {result.error}")

            # Store the observation report as the "prompt used" so it feeds
            # into the brain and history for subsequent steps
            obs_report = result.observations_markdown() if result.observations else ""
            prompt_record = f"[CUA Visual Test]\n{task}"
            if obs_report:
                prompt_record += f"\n\n--- CUA Findings ---\n{obs_report}"

            return StepResult(
                loop_number=loop_num,
                step_index=step_idx,
                step_name=step_label,
                job_type="visual_test",
                agent_used=f"cua:{provider.value}",
                prompt_used=prompt_record,
                agent_success=result.success,
                validation_success=result.success,
                tests_passed=result.success,
                success=result.success,
                test_outcome="passed" if result.success else "failed",
                files_changed=0,
                net_lines_changed=0,
                error_message=result.error,
                duration_seconds=round(time.monotonic() - step_start, 1),
            )

        except Exception as exc:
            self._log("error", f"CUA step failed: {exc}")
            if ledger:
                ledger.add(
                    category="error",
                    title=f"CUA failed: {str(exc)[:60]}",
                    detail=str(exc),
                    severity="major",
                    source="chain:visual_test",
                    step_ref=step_ref,
                )
            return StepResult(
                loop_number=loop_num,
                step_index=step_idx,
                step_name=step_label,
                job_type="visual_test",
                agent_used="cua",
                agent_success=False,
                validation_success=False,
                tests_passed=False,
                success=False,
                error_message=str(exc),
                duration_seconds=round(time.monotonic() - step_start, 1),
            )

    # ------------------------------------------------------------------
    # Parallel step execution
    # ------------------------------------------------------------------

    def _run_steps_parallel(
        self,
        runners: dict[str, AgentRunner],
        evaluator: RepoEvaluator,
        repo: Path,
        config: ChainConfig,
        loop_num: int,
        enabled_steps: list[TaskStep],
        brain: BrainManager,
        start_time: float,
    ) -> bool:
        """Execute enabled steps for a shared repository safely.

        Parallel execution against a single working tree is unsafe because
        each step may independently mutate files, commit, or reset state.
        This method therefore falls back to deterministic sequential
        execution while preserving the same stop/token accounting behavior.
        """
        if self._consume_stop_after_step_request(context="parallel batch"):
            return True

        self._log(
            "warn",
            "Parallel mode requested, but shared-repo step execution is race-prone. "
            "Running this batch sequentially for correctness.",
        )
        self.state.current_step_name = f"Parallel batch ({len(enabled_steps)} steps)"
        self.state.current_step_started_at_epoch_ms = int(time.time() * 1000)

        # Map step id -> original index in config.steps
        step_index_map = {s.id: i for i, s in enumerate(config.steps)}
        ordered_steps = sorted(enabled_steps, key=lambda step: step_index_map.get(step.id, 0))
        for step in ordered_steps:
            step_idx = step_index_map.get(step.id, 0)
            agent_key = step.agent or "codex"
            runner = runners.get(agent_key, runners["codex"])
            step_label = step.name or step.job_type
            step_loops = max(1, step.loop_count)
            self.state.current_step = step_idx
            for step_rep in range(1, step_loops + 1):
                self._pause_event.wait()
                if self._stop_event.is_set():
                    self.state.stop_reason = "user_stopped"
                    return True
                if self._consume_stop_after_step_request(context="parallel batch repetition"):
                    return True

                rep_tag = f" (rep {step_rep}/{step_loops})" if step_loops > 1 else ""
                self.state.current_step_name = f"{step_label}{rep_tag}"
                self.state.current_step_started_at_epoch_ms = int(time.time() * 1000)
                self._log(
                    "info",
                    f"  Running: {step_label} [{runner.name}]{rep_tag}",
                )

                try:
                    result = self._execute_step(
                        runner,
                        evaluator,
                        repo,
                        config,
                        loop_num,
                        step_idx,
                        step,
                        brain,
                    )
                except Exception as exc:
                    result = StepResult(
                        loop_number=loop_num,
                        step_index=step_idx,
                        step_name=step_label,
                        job_type=step.job_type,
                        agent_used=step.agent,
                        agent_success=False,
                        validation_success=False,
                        tests_passed=False,
                        success=False,
                        error_message=str(exc),
                    )

                self._log(
                    "info" if result.success else "warn",
                    f"  Completed: {result.step_name} [{result.agent_used}] - "
                    f"{'OK' if result.success else 'FAIL'}",
                )
                self.state.results.append(result)
                if not result.success and result.error_message and getattr(self, "ledger", None):
                    self.ledger.add(
                        category="error",
                        title=result.error_message[:80] or "Step failed",
                        detail=result.error_message,
                        severity="major",
                        source=f"chain:{result.step_name}",
                        step_ref=f"loop{loop_num}:step{step_idx}:{result.job_type}",
                    )
                self.state.total_steps_completed += 1
                self.state.total_tokens += result.input_tokens + result.output_tokens
                self.state.elapsed_seconds = time.monotonic() - start_time
                if self._check_strict_token_budget(config):
                    return True
                if result.terminate_repeats and step_rep < step_loops:
                    self._log(
                        "info",
                        (
                            f"Skipping remaining repetitions for {step_label} "
                            f"after {TERMINATE_STEP_TAG}."
                        ),
                    )
                    break
        return self._consume_stop_after_step_request(context="parallel batch")

    # ------------------------------------------------------------------
    # Ledger context for steps
    # ------------------------------------------------------------------

    def _get_ledger_context_for_step(self, step: TaskStep) -> str:
        """Return formatted ledger context relevant to this step type."""
        ledger = getattr(self, "ledger", None)
        if ledger is None:
            return ""
        categories: list[str] | None = None
        if step.job_type in ("bug_hunting", "testing"):
            categories = ["error", "bug", "observation"]
        elif step.job_type in (
            "feature_discovery",
            "implementation",
            "strategic_product_maximization",
        ):
            categories = ["suggestion", "wishlist", "feature", "todo"]
        elif step.job_type == "visual_test":
            categories = ["observation", "bug"]
        if not categories:
            return ""
        return ledger.get_context_for_prompt(categories=categories, max_items=20)

    # ------------------------------------------------------------------
    # Prompt resolution
    # ------------------------------------------------------------------

    def _resolve_prompt(self, step: TaskStep, loop_num: int, step_idx: int) -> str:
        if step.prompt_mode == "custom":
            base = step.custom_prompt
        else:
            ai_decides = step.prompt_mode == "ai_decides"
            base = get_prompt(step.job_type, ai_decides=ai_decides)
            if not base:
                base = step.custom_prompt or (f"Improve this repository. Focus on: {step.name}")

        if not (base or "").strip():
            step_label = step.name or step.job_type or f"step-{step_idx + 1}"
            base = (
                f"Execute the '{step_label}' step for this repository. "
                "Use existing files and previous step memory to make concrete progress."
            )

        # Inject context from previous step
        context_parts = [f"\n\n[Loop {loop_num}, Step {step_idx + 1}]"]
        if self.state.results:
            last = self.state.results[-1]
            context_parts.append(
                f"Previous step ({last.step_name}): "
                f"tests={last.test_outcome}, "
                f"files_changed={last.files_changed}, "
                f"net_lines={last.net_lines_changed:+d}"
            )
        memory_context = self._build_step_memory_context()
        if memory_context:
            context_parts.append("\n--- Recent Chain Memory ---")
            context_parts.append(memory_context)
        vector_context = self._build_vector_memory_context(
            step=step,
            query_hint=base + "\n" + (memory_context or ""),
        )
        if vector_context:
            context_parts.append("\n--- Long-Term Memory Context ---")
            context_parts.append(vector_context)
        return base + "\n".join(context_parts)

    # ------------------------------------------------------------------
    # History summary for the brain
    # ------------------------------------------------------------------

    def _build_history_summary(self, max_entries: int = 10) -> str:
        """Build a concise summary of recent results for the brain.

        Includes CUA observation reports so the brain can suggest fixes
        for visually-detected issues.
        """
        if not self.state.results:
            return "No previous results."
        recent = self.state.results[-max_entries:]
        lines = []
        for r in recent:
            status = "OK" if r.success else "FAIL"
            lines.append(
                f"  Loop {r.loop_number}, Step {r.step_index} ({r.step_name}): "
                f"{status}, tests={r.test_outcome}, "
                f"files={r.files_changed}, net_lines={r.net_lines_changed:+d}"
            )
            # Include CUA findings in history so the brain can act on them
            if r.job_type == "visual_test" and r.prompt_used and "CUA Findings" in r.prompt_used:
                # Extract the findings section
                findings_start = r.prompt_used.find("--- CUA Findings ---")
                if findings_start != -1:
                    findings_text = r.prompt_used[findings_start:]
                    # Truncate to keep history manageable
                    lines.append(f"    {findings_text[:500]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stop-condition checks
    # ------------------------------------------------------------------

    def _check_strict_token_budget(self, config: ChainConfig) -> bool:
        """Stop immediately when strict token-budget mode is enabled."""
        if not getattr(config, "strict_token_budget", False):
            return False
        if config.max_total_tokens <= 0:
            return False
        if self.state.total_tokens < config.max_total_tokens:
            return False
        if self.state.stop_reason != "budget_exhausted":
            self.state.stop_reason = "budget_exhausted"
            self._log(
                "warn",
                f"Token budget reached ({self.state.total_tokens:,}/{config.max_total_tokens:,}) -- strict budget mode stopping run now",
            )
        return True

    def _check_stop_conditions(self, config: ChainConfig, start_time: float) -> str | None:
        elapsed_min = (time.monotonic() - start_time) / 60

        if config.max_time_minutes > 0 and elapsed_min >= config.max_time_minutes:
            return "max_time_reached"

        if config.max_total_tokens > 0 and self.state.total_tokens >= config.max_total_tokens:
            return "budget_exhausted"

        latest_loop = [r for r in self.state.results if r.loop_number == self.state.current_loop]
        if config.stop_on_convergence and latest_loop:
            no_repo_deltas = all(
                r.files_changed <= 0 and r.net_lines_changed == 0 for r in latest_loop
            )
            total_output_chars = sum(
                max(0, int(getattr(r, "output_chars", 0))) for r in latest_loop
            )
            if no_repo_deltas and total_output_chars < 200:
                self._log(
                    "info",
                    "No meaningful progress detected in the latest loop "
                    "(no file deltas and minimal output) - stopping early.",
                )
                return "no_progress_detected"

        # â”€â”€ Improvement-threshold check (powers the "unlimited" mode) â”€â”€
        # Compare the latest full loop to the previous one.  "Improvement"
        # is measured as a composite of:
        #   â€¢ files changed  (are we still touching things?)
        #   â€¢ net lines changed  (is the delta shrinking?)
        #   â€¢ success rate  (are steps still succeeding?)
        # When the % improvement drops below the configured threshold the
        # chain is considered converged.
        n_enabled = max(1, sum(1 for s in config.steps if s.enabled))
        results = self.state.results

        if len(results) >= n_enabled * 2:
            prev_loop = results[-(n_enabled * 2) : -n_enabled]
            curr_loop = results[-n_enabled:]

            imp_pct = self._compute_improvement(prev_loop, curr_loop)
            self.state.improvement_pct = round(imp_pct, 2)

            threshold = config.improvement_threshold
            if (config.unlimited or config.stop_on_convergence) and imp_pct < threshold:
                self._log(
                    "info",
                    f"Improvement dropped to {imp_pct:.2f}% "
                    f"(threshold {threshold}%) -- diminishing returns",
                )
                return "diminishing_returns"

        # Legacy simple convergence (still useful as a fallback)
        if config.stop_on_convergence and len(results) >= 4:
            last_4 = results[-4:]
            all_pass = all(r.test_outcome in ("passed", "skipped") for r in last_4)
            all_low = all(abs(r.net_lines_changed) < 20 and r.files_changed <= 2 for r in last_4)
            if all_pass and all_low:
                return "convergence_detected"

        return None

    # ------------------------------------------------------------------
    # Improvement metric
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_improvement(prev: list[StepResult], curr: list[StepResult]) -> float:
        """Return a 0-100 improvement percentage for *curr* vs *prev*.

        The metric blends three signals:
        1. **Activity** - total files changed (are we still doing work?)
        2. **Magnitude** - total absolute net lines changed
        3. **Success delta** - did the success rate go up, down, or stay?

        A score of 0% means "this loop did nothing new compared to the
        last one".  100% is a strong improvement.
        """

        def _activity(results: list[StepResult]) -> int:
            return sum(r.files_changed for r in results)

        def _magnitude(results: list[StepResult]) -> int:
            return sum(abs(r.net_lines_changed) for r in results)

        def _success_rate(results: list[StepResult]) -> float:
            if not results:
                return 0.0
            return sum(1 for r in results if r.success) / len(results) * 100

        prev_act = max(_activity(prev), 1)
        curr_act = _activity(curr)

        prev_mag = max(_magnitude(prev), 1)
        curr_mag = _magnitude(curr)

        prev_sr = _success_rate(prev)
        curr_sr = _success_rate(curr)

        # Activity ratio: how much work this loop did relative to last
        activity_ratio = curr_act / prev_act * 100  # >100 = more active

        # Magnitude ratio
        magnitude_ratio = curr_mag / prev_mag * 100

        # Success delta (clamp to 0-100 range contribution)
        success_delta = max(0, curr_sr - prev_sr + 50)  # 50 = no change

        # Weighted blend (activity matters most, then magnitude, then success)
        score = activity_ratio * 0.45 + magnitude_ratio * 0.35 + success_delta * 0.20

        # Normalise: 100 means "same as before", below 100 = declining
        # We report how much of the previous loop's impact was retained
        return min(score, 200.0)  # cap at 200% (big jump)


