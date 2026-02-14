"""Pipeline orchestrator - drives the autonomous improvement pipeline.

The orchestrator runs multiple *cycles*, where each cycle executes all
enabled phases in order:

    Ideation -> Prioritization -> Implementation -> Testing -> Debugging
    -> Commit -> Progress Review -> (Theorize -> Experiment -> Skeptic -> Analyze)

Each phase can iterate multiple times within a cycle. The orchestrator
integrates with:
- Agent runners (Codex / Claude Code) for code execution
- The prompt catalog for phase-specific prompts
- The log tracker for structured markdown file management
- The brain layer for prompt refinement and evaluation
- Git tools for commits and branching
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from codex_manager.agent_runner import AgentRunner
from codex_manager.agent_signals import (
    TERMINATE_STEP_TAG,
    contains_terminate_step_signal,
    terminate_step_instruction,
)
from codex_manager.brain.logbook import BrainLogbook
from codex_manager.brain.manager import BrainConfig, BrainManager
from codex_manager.codex_cli import CodexRunner
from codex_manager.eval_tools import RepoEvaluator, parse_test_command
from codex_manager.git_tools import (
    commit_all,
    create_branch,
    diff_numstat_entries,
    diff_stat,
    generate_commit_message,
    head_sha,
    is_clean,
    reset_to_ref,
    revert_all,
    summarize_numstat_entries,
)
from codex_manager.history_log import HistoryLogbook
from codex_manager.ledger import KnowledgeLedger
from codex_manager.pipeline.phases import (
    PHASE_LOG_FILES,
    PhaseResult,
    PipelineConfig,
    PipelinePhase,
    PipelineState,
)
from codex_manager.pipeline.tracker import LogTracker
from codex_manager.preflight import binary_exists as shared_binary_exists
from codex_manager.prompts.catalog import PromptCatalog, get_catalog
from codex_manager.schemas import EvalResult

logger = logging.getLogger(__name__)

_MUTATING_PHASES = {
    PipelinePhase.IMPLEMENTATION,
    PipelinePhase.DEBUGGING,
}


class PipelineOrchestrator:
    """Drives the full autonomous improvement pipeline.

    Parameters
    ----------
    repo_path:
        Path to the target git repository.
    config:
        Pipeline configuration (phases, budgets, agent selection, etc.).
    catalog:
        Prompt catalog instance (uses global singleton if omitted).
    log_callback:
        Optional callback ``(level, message)`` for real-time log streaming
        (used by the GUI).
    """

    def __init__(
        self,
        repo_path: str | Path,
        config: PipelineConfig | None = None,
        catalog: PromptCatalog | None = None,
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.config = config or PipelineConfig()
        self.catalog = catalog or get_catalog()
        self.tracker = LogTracker(self.repo_path)
        self.ledger = KnowledgeLedger(self.repo_path)

        self.state = PipelineState()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused initially
        self._thread: threading.Thread | None = None
        self.log_queue: queue.Queue[dict] = queue.Queue()
        self._log_callback = log_callback
        self._science_experiment_by_hypothesis: dict[str, str] = {}
        self._brain_logbook: BrainLogbook | None = None
        self._history_logbook: HistoryLogbook | None = None

    # ------------------------------------------------------------------
    # Public controls
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the pipeline in a background thread."""
        if self.is_running:
            self._log("error", "Pipeline is already running")
            return

        self.state = PipelineState(
            running=True,
            started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        self._stop_event.clear()
        self._pause_event.set()

        # Drain stale log entries
        while not self.log_queue.empty():
            try:
                self.log_queue.get_nowait()
            except queue.Empty:
                break

        self._thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self._thread.start()
        self._log("info", f"Pipeline started ({self.config.mode} mode)")
        self._log_execution_mode_warnings()

    def run(self) -> PipelineState:
        """Run the pipeline synchronously (for CLI usage)."""
        self.state = PipelineState(
            running=True,
            started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        self._stop_event.clear()
        self._pause_event.set()
        self._log("info", f"Pipeline started ({self.config.mode} mode)")
        self._log_execution_mode_warnings()
        self._run_pipeline()
        return self.state

    def stop(self) -> None:
        """Request the pipeline to stop."""
        self._stop_event.set()
        self._pause_event.set()
        self._log("warn", "Stop requested")

    def pause(self) -> None:
        """Pause the pipeline."""
        self._pause_event.clear()
        self.state.paused = True
        self._log("info", "Paused")

    def resume(self) -> None:
        """Resume the pipeline."""
        self._pause_event.set()
        self.state.paused = False
        self._log("info", "Resumed")

    def get_state(self) -> dict:
        """Return the current state as a dict."""
        return self.state.model_dump()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, level: str, message: str) -> None:
        log_epoch_ms = int(time.time() * 1000)
        entry = {
            "time": dt.datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "message": message,
        }
        self.log_queue.put(entry)
        if self._log_callback:
            self._log_callback(level, message)
        getattr(logger, level if level != "warn" else "warning", logger.info)(
            "[pipeline] %s", message
        )
        self.state.last_log_epoch_ms = log_epoch_ms
        self.state.last_log_level = level
        self.state.last_log_message = message[:500]

    def _log_execution_mode_warnings(self) -> None:
        """Log prominent warnings when execution is in a constrained safety mode."""
        config = self.config
        if config.mode == "dry-run":
            self._log(
                "warn",
                "SAFE MODE ACTIVE: dry-run mode is enabled. Any file edits will be reverted.",
            )

        phase_order = config.get_phase_order()
        uses_codex = (
            any((phase.agent or config.agent) != "claude_code" for phase in phase_order)
            if phase_order
            else (config.agent != "claude_code")
        )
        if uses_codex and config.codex_sandbox_mode == "read-only":
            self._log(
                "warn",
                "READ-ONLY SANDBOX ACTIVE: Codex can inspect files but cannot write changes.",
            )

    def _record_brain_note(
        self,
        event: str,
        summary: str,
        *,
        level: str = "info",
        context: dict[str, Any] | None = None,
    ) -> None:
        """Persist a brain observation note for debugging and analysis."""
        if self._brain_logbook is None:
            return
        self._brain_logbook.record(
            scope="pipeline",
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
        context: dict[str, Any] | None = None,
    ) -> None:
        """Persist a run-history note for user-visible auditing."""
        if self._history_logbook is None:
            return
        self._history_logbook.record(
            scope="pipeline",
            event=event,
            summary=summary,
            level=level,
            context=context or {},
        )

    def _finalize_run(
        self,
        *,
        start_time: float,
        history_level: str = "info",
        extra_history_context: dict[str, Any] | None = None,
    ) -> None:
        """Finalize state/logging for any pipeline terminal path."""
        self.state.running = False
        self.state.current_phase = ""
        self.state.current_iteration = 0
        self.state.current_phase_started_at_epoch_ms = 0
        self.state.finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
        self.state.elapsed_seconds = time.monotonic() - start_time
        self._log(
            "info",
            f"Pipeline finished - {self.state.stop_reason} "
            f"({self.state.total_cycles_completed} cycles, "
            f"{self.state.total_phases_completed} phases)",
        )
        history_context = {
            "stop_reason": self.state.stop_reason,
            "total_cycles_completed": self.state.total_cycles_completed,
            "total_phases_completed": self.state.total_phases_completed,
            "total_tokens": self.state.total_tokens,
            "elapsed_seconds": round(self.state.elapsed_seconds, 1),
        }
        if extra_history_context:
            history_context.update(extra_history_context)
        self._record_history_note(
            "run_finished",
            f"Pipeline finished with stop_reason='{self.state.stop_reason}'.",
            level=history_level,
            context=history_context,
        )

    @staticmethod
    def _is_science_phase(phase: PipelinePhase) -> bool:
        return phase in (
            PipelinePhase.THEORIZE,
            PipelinePhase.EXPERIMENT,
            PipelinePhase.SKEPTIC,
            PipelinePhase.ANALYZE,
        )

    def _brain_goal(self, phase: str) -> str:
        """Return a repo-anchored goal string for brain prompts."""
        repo_name = self.repo_path.name or "repository"
        return f"Pipeline phase '{phase}' for repository '{repo_name}' at path: {self.repo_path}"

    @staticmethod
    def _test_outcome_rank(outcome: str) -> int:
        ranking = {
            "error": 0,
            "failed": 1,
            "skipped": 2,
            "passed": 3,
        }
        return ranking.get((outcome or "").strip().lower(), 0)

    @staticmethod
    def _eval_snapshot(eval_result: EvalResult) -> dict[str, object]:
        return {
            "test_outcome": eval_result.test_outcome.value,
            "test_exit_code": eval_result.test_exit_code,
            "test_summary": (eval_result.test_summary or "")[:1500],
            "files_changed": eval_result.files_changed,
            "net_lines_changed": eval_result.net_lines_changed,
            "diff_stat": (eval_result.diff_stat or "")[:1500],
            "changed_files": eval_result.changed_files[:200],
        }

    @classmethod
    def _science_tradeoff_deltas(
        cls,
        baseline_eval: EvalResult,
        result: PhaseResult,
    ) -> dict[str, int]:
        return {
            "delta_test_rank": cls._test_outcome_rank(result.test_outcome)
            - cls._test_outcome_rank(baseline_eval.test_outcome.value),
            "delta_test_exit_code": int(result.test_exit_code) - int(baseline_eval.test_exit_code),
            "delta_files_changed": int(result.files_changed) - int(baseline_eval.files_changed),
            "delta_net_lines_changed": int(result.net_lines_changed)
            - int(baseline_eval.net_lines_changed),
        }

    @classmethod
    def _science_experiment_verdict(
        cls,
        baseline_eval: EvalResult,
        result: PhaseResult,
    ) -> tuple[str, str]:
        if not result.agent_success:
            return "refuted", "agent execution failed"

        before = baseline_eval.test_outcome.value
        after = result.test_outcome
        before_rank = cls._test_outcome_rank(before)
        after_rank = cls._test_outcome_rank(after)
        if after_rank < before_rank:
            return "refuted", f"test outcome regressed ({before} -> {after})"

        delta_files = result.files_changed - baseline_eval.files_changed
        delta_lines = result.net_lines_changed - baseline_eval.net_lines_changed
        if delta_files == 0 and delta_lines == 0:
            return "inconclusive", "no measurable repo delta versus baseline"

        return "supported", "no regression and measurable repo delta observed"

    @classmethod
    def _science_skeptic_verdict(
        cls,
        baseline_eval: EvalResult,
        result: PhaseResult,
    ) -> tuple[str, str, str]:
        if not result.agent_success:
            return "refuted", "agent execution failed", "low"

        output = result.agent_final_message or ""
        verdict_match = re.search(
            r"^\s*SKEPTIC_VERDICT:\s*(supported|refuted|inconclusive)\s*$",
            output,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        confidence_match = re.search(
            r"^\s*SKEPTIC_CONFIDENCE:\s*(low|medium|high)\s*$",
            output,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        rationale_match = re.search(
            r"^\s*SKEPTIC_RATIONALE:\s*(.+)\s*$",
            output,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        parsed_verdict = verdict_match.group(1).lower() if verdict_match else ""
        confidence = confidence_match.group(1).lower() if confidence_match else "medium"
        parsed_rationale = rationale_match.group(1).strip() if rationale_match else ""

        before = baseline_eval.test_outcome.value
        after = result.test_outcome
        before_rank = cls._test_outcome_rank(before)
        after_rank = cls._test_outcome_rank(after)
        if after_rank < before_rank:
            return (
                "refuted",
                f"skeptic validation regressed tests ({before} -> {after})",
                confidence,
            )

        if parsed_verdict:
            if parsed_verdict == "supported":
                rationale = parsed_rationale or "skeptic replication supported the experiment"
                return "supported", rationale, confidence
            if parsed_verdict == "refuted":
                rationale = parsed_rationale or "skeptic challenge refuted the experiment"
                return "refuted", rationale, confidence
            rationale = parsed_rationale or "skeptic could not reach a decisive conclusion"
            return "inconclusive", rationale, confidence

        return "inconclusive", "missing skeptic verdict footer in output", confidence

    @staticmethod
    def _science_threshold_met(
        phase: PipelinePhase,
        verdict: str,
        tradeoff_deltas: dict[str, int],
    ) -> bool:
        if verdict != "supported":
            return False
        if phase == PipelinePhase.EXPERIMENT:
            return tradeoff_deltas.get("delta_test_rank", 0) >= 0 and (
                tradeoff_deltas.get("delta_files_changed", 0) != 0
                or tradeoff_deltas.get("delta_net_lines_changed", 0) != 0
            )
        if phase == PipelinePhase.SKEPTIC:
            return tradeoff_deltas.get("delta_test_rank", 0) >= 0
        return True

    @staticmethod
    def _parse_science_hypotheses(markdown: str) -> list[dict[str, str]]:
        if not markdown.strip():
            return []
        pattern = re.compile(
            r"^### \[(EXP-[0-9]{3,})\]\s*(.*?)(?:\r?\n)(.*?)(?=^### \[(?:EXP-[0-9]{3,})\]|\Z)",
            flags=re.MULTILINE | re.DOTALL,
        )
        items: list[dict[str, str]] = []
        for match in pattern.finditer(markdown):
            hyp_id = match.group(1).strip()
            title = match.group(2).strip()
            body = match.group(3).strip()
            block = f"### [{hyp_id}] {title}\n{body}".strip()

            def _field(label: str, block_text: str = block) -> str:
                field_match = re.search(
                    rf"^- \*\*{re.escape(label)}\*\*:\s*(.+)$",
                    block_text,
                    flags=re.IGNORECASE | re.MULTILINE,
                )
                return field_match.group(1).strip() if field_match else ""

            items.append(
                {
                    "id": hyp_id,
                    "title": title,
                    "status": _field("Status").lower(),
                    "hypothesis": _field("Hypothesis"),
                    "success_criteria": _field("Success Criteria"),
                    "confidence": _field("Confidence").lower(),
                    "block": block,
                }
            )
        return items

    def _select_science_hypothesis(
        self,
        markdown: str,
        preferred_hypothesis_id: str = "",
    ) -> dict[str, str]:
        hypotheses = self._parse_science_hypotheses(markdown)
        if not hypotheses:
            return {
                "id": "",
                "title": "",
                "status": "",
                "hypothesis": "",
                "success_criteria": "",
                "confidence": "medium",
                "block": "",
            }

        preferred = (preferred_hypothesis_id or "").strip()
        if preferred:
            for hyp in hypotheses:
                if hyp.get("id", "").strip().upper() == preferred.upper():
                    return hyp

        for status in ("testing", "proposed", "pending"):
            for hyp in hypotheses:
                if hyp.get("status", "").startswith(status):
                    return hyp

        return hypotheses[0]

    @staticmethod
    def _extract_hypothesis_blocks(markdown: str) -> str:
        hypotheses = PipelineOrchestrator._parse_science_hypotheses(markdown)
        return "\n\n".join(h["block"] for h in hypotheses if h.get("block"))

    def _build_science_prompt(
        self,
        *,
        prompt: str,
        phase: PipelinePhase,
        cycle: int,
        iteration: int,
        baseline_eval: EvalResult,
        selected_hypothesis: dict[str, str] | None,
        preferred_experiment_id: str = "",
    ) -> str:
        hypothesis = selected_hypothesis or {}
        baseline_snapshot = json.dumps(
            self._eval_snapshot(baseline_eval),
            indent=2,
            sort_keys=True,
        )

        lines = [
            "",
            "--- SCIENCE CONTRACT ---",
            f"Phase: {phase.value} | Cycle: {cycle} | Iteration: {iteration}",
        ]

        if phase in (PipelinePhase.EXPERIMENT, PipelinePhase.SKEPTIC):
            hyp_id = hypothesis.get("id", "").strip() or "(unresolved)"
            lines.extend(
                [
                    "Controlled-experiment requirements (enforced in runtime):",
                    "- Operate on one hypothesis only for this trial.",
                    "- Keep scope narrow and avoid unrelated refactors.",
                    "- Use baseline metrics below as control.",
                    "",
                    f"Hypothesis ID: {hyp_id}",
                ]
            )
            if hypothesis.get("title"):
                lines.append(f"Hypothesis Title: {hypothesis['title']}")
            if hypothesis.get("hypothesis"):
                lines.append(f"Hypothesis Statement: {hypothesis['hypothesis']}")
            if hypothesis.get("success_criteria"):
                lines.append(f"Success Criteria: {hypothesis['success_criteria']}")
            if preferred_experiment_id:
                lines.append(f"Experiment ID: {preferred_experiment_id}")
            lines.extend(
                [
                    "",
                    "Baseline Snapshot (control):",
                    baseline_snapshot,
                ]
            )
            if phase == PipelinePhase.SKEPTIC:
                lines.extend(
                    [
                        "",
                        "Skeptic requirements:",
                        "- Independently challenge the prior conclusion.",
                        "- Attempt replication before accepting a claim.",
                        "- End output with SKEPTIC_VERDICT/SKEPTIC_CONFIDENCE/SKEPTIC_RATIONALE.",
                    ]
                )
        elif phase == PipelinePhase.THEORIZE:
            lines.extend(
                [
                    "Use scientific hypothesis format with unique EXP IDs and measurable criteria.",
                    "Do not reuse IDs from existing experiments.",
                ]
            )

        return f"{prompt}\n" + "\n".join(lines)

    @staticmethod
    def _next_science_id(prefix: str) -> str:
        clean_prefix = re.sub(r"[^A-Za-z0-9_-]+", "-", prefix).strip("-") or "SCI"
        return f"{clean_prefix}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"

    def _record_science_trial(
        self,
        *,
        cycle: int,
        phase: PipelinePhase,
        iteration: int,
        prompt: str,
        result: PhaseResult,
        baseline_eval: EvalResult,
        repo: Path,
        baseline_repo_clean: bool,
        selected_hypothesis: dict[str, str] | None = None,
        preferred_experiment_id: str = "",
    ) -> dict[str, Any]:
        self.tracker.initialize_science()
        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()

        experiments_content = self.tracker.read("EXPERIMENTS.md")
        hypothesis = selected_hypothesis or self._select_science_hypothesis(experiments_content)
        hypothesis_id = (hypothesis.get("id", "") or "").strip()
        hypothesis_title = (hypothesis.get("title", "") or "").strip()
        confidence = (hypothesis.get("confidence", "") or "").strip() or "medium"

        trial_id = self._next_science_id("SCI-TRIAL")
        experiment_id = ""
        if phase == PipelinePhase.EXPERIMENT:
            id_hint = hypothesis_id if hypothesis_id else "UNSPECIFIED"
            experiment_id = self._next_science_id(f"SCI-EXP-{id_hint}")
            if hypothesis_id:
                self._science_experiment_by_hypothesis[hypothesis_id] = experiment_id
        elif phase == PipelinePhase.SKEPTIC:
            experiment_id = (
                preferred_experiment_id
                or (
                    self._science_experiment_by_hypothesis.get(hypothesis_id)
                    if hypothesis_id
                    else ""
                )
                or self._next_science_id("SCI-EXP-UNLINKED")
            )

        verdict = "n/a"
        rationale = "verdict applies to experiment/skeptic phases only"
        if phase == PipelinePhase.EXPERIMENT:
            verdict, rationale = self._science_experiment_verdict(baseline_eval, result)
        elif phase == PipelinePhase.SKEPTIC:
            verdict, rationale, skeptic_confidence = self._science_skeptic_verdict(
                baseline_eval,
                result,
            )
            confidence = skeptic_confidence or confidence

        tradeoff_deltas = self._science_tradeoff_deltas(baseline_eval, result)
        threshold_met = self._science_threshold_met(phase, verdict, tradeoff_deltas)

        rollback_action = "not_applicable"
        if phase in (PipelinePhase.EXPERIMENT, PipelinePhase.SKEPTIC):
            if verdict != "supported":
                result.validation_success = False
                result.success = False
                evidence_reason = f"Science verdict={verdict}: {rationale}"
                result.error_message = (
                    f"{result.error_message}; {evidence_reason}"
                    if result.error_message
                    else evidence_reason
                )
                if baseline_repo_clean:
                    if is_clean(repo):
                        rollback_action = "already_clean"
                    else:
                        revert_all(repo)
                        rollback_action = "reverted"
                        result.science_rolled_back = True
                        self._log(
                            "warn",
                            f"  Science rollback applied after {phase.value} verdict={verdict}",
                        )
                else:
                    rollback_action = "skipped_baseline_dirty"
                    result.error_message = f"{result.error_message}; rollback skipped because baseline repo was not clean"
                    self._log(
                        "warn",
                        "  Science rollback skipped because baseline repo was already dirty",
                    )
            else:
                rollback_action = "kept"

        result.science_trial_id = trial_id
        result.science_experiment_id = experiment_id
        result.science_hypothesis_id = hypothesis_id
        result.science_hypothesis_title = hypothesis_title
        result.science_confidence = confidence
        result.science_verdict = verdict
        result.science_verdict_rationale = rationale
        result.science_tradeoff_deltas = dict(tradeoff_deltas)

        stem = f"cycle-{cycle:03d}-{phase.value}-iter-{iteration:02d}-{trial_id.lower()}"
        prompt_path = self.tracker.save_science_artifact(
            "prompts",
            stem,
            prompt,
            suffix=".txt",
        )
        output_text = result.agent_final_message.strip() or (
            result.error_message.strip() if result.error_message else "(no output captured)"
        )
        output_path = self.tracker.save_science_artifact(
            "outputs",
            stem,
            output_text,
            suffix=".md",
        )
        snapshot_path = self.tracker.save_science_artifact(
            "snapshots",
            f"{stem}-experiments",
            experiments_content or "_(empty)_",
            suffix=".md",
        )
        self.tracker.write_science(
            "EXPERIMENTS_LATEST.md",
            experiments_content or "# EXPERIMENTS\n\n(no content captured)\n",
        )

        if phase == PipelinePhase.THEORIZE:
            blocks = self._extract_hypothesis_blocks(experiments_content)
            if blocks:
                self.tracker.append_science(
                    "HYPOTHESES.md",
                    (f"\n## Cycle {cycle}, Iteration {iteration} - {timestamp}\n\n{blocks}\n"),
                )
        elif phase == PipelinePhase.ANALYZE:
            self.tracker.append_science(
                "ANALYSIS.md",
                (f"\n## Cycle {cycle}, Iteration {iteration} - {timestamp}\n\n{output_text}\n"),
            )

        payload = {
            "timestamp": timestamp,
            "trial_id": trial_id,
            "experiment_id": experiment_id,
            "cycle": cycle,
            "phase": phase.value,
            "iteration": iteration,
            "agent": result.agent_used,
            "agent_success": result.agent_success,
            "validation_success": result.validation_success,
            "tests_passed": result.tests_passed,
            "success": result.success,
            "verdict": verdict,
            "verdict_rationale": rationale,
            "confidence": confidence,
            "threshold_met": threshold_met,
            "baseline_repo_clean": baseline_repo_clean,
            "rollback_action": rollback_action,
            "hypothesis": {
                "id": hypothesis_id,
                "title": hypothesis_title,
                "status": hypothesis.get("status", ""),
                "statement": hypothesis.get("hypothesis", ""),
                "success_criteria": hypothesis.get("success_criteria", ""),
            },
            "baseline": self._eval_snapshot(baseline_eval),
            "post": {
                "test_outcome": result.test_outcome,
                "test_exit_code": result.test_exit_code,
                "test_summary": (result.test_summary or "")[:1500],
                "files_changed": result.files_changed,
                "net_lines_changed": result.net_lines_changed,
            },
            "tradeoff_deltas": tradeoff_deltas,
            "usage": {
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.input_tokens + result.output_tokens,
            },
            "error_message": (result.error_message or "")[:1000],
            "prompt_file": str(prompt_path),
            "output_file": str(output_path),
            "experiments_snapshot_file": str(snapshot_path),
            "commit_sha": result.commit_sha,
            "duration_seconds": result.duration_seconds,
        }
        self.tracker.append_science_jsonl("TRIALS.jsonl", payload)
        self.tracker.append_science(
            "EVIDENCE.md",
            (
                f"\n## {timestamp} - Cycle {cycle}, {phase.value}, Iteration {iteration}\n"
                f"- **Trial ID**: `{trial_id}`\n"
                f"- **Experiment ID**: `{experiment_id or 'n/a'}`\n"
                f"- **Hypothesis**: `{hypothesis_id or 'n/a'}` {hypothesis_title}\n"
                f"- **Agent**: {result.agent_used}\n"
                f"- **Success**: {result.success}\n"
                f"- **Verdict**: {verdict} ({rationale})\n"
                f"- **Confidence**: {confidence}\n"
                f"- **Threshold Met**: {threshold_met}\n"
                f"- **Rollback**: {rollback_action}\n"
                f"- **Baseline Tests**: {baseline_eval.test_outcome.value} "
                f"(exit {baseline_eval.test_exit_code})\n"
                f"- **Post Tests**: {result.test_outcome} (exit {result.test_exit_code})\n"
                f"- **Tradeoff Deltas**: {tradeoff_deltas}\n"
                f"- **Repo Delta**: files={result.files_changed}, lines={result.net_lines_changed:+d}\n"
                f"- **Tokens**: in={result.input_tokens:,}, out={result.output_tokens:,}\n"
                f"- **Prompt File**: `{prompt_path}`\n"
                f"- **Output File**: `{output_path}`\n"
                f"- **Experiments Snapshot**: `{snapshot_path}`\n"
            ),
        )

        return {
            "trial_id": trial_id,
            "experiment_id": experiment_id,
            "hypothesis_id": hypothesis_id,
            "verdict": verdict,
            "rationale": rationale,
            "confidence": confidence,
            "threshold_met": threshold_met,
            "rollback_action": rollback_action,
            "tradeoff_deltas": tradeoff_deltas,
        }

    @staticmethod
    def _science_commit_gate(
        experiment_trials: list[dict[str, Any]],
        skeptic_trials: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        if not experiment_trials:
            return False, "no experiment trials recorded"
        if any(t.get("verdict") != "supported" for t in experiment_trials):
            return False, "experiment verdict was not supported"
        if any(not t.get("threshold_met", False) for t in experiment_trials):
            return False, "experiment threshold was not met"

        if not skeptic_trials:
            return False, "skeptic validation did not run"
        if any(t.get("verdict") != "supported" for t in skeptic_trials):
            return False, "skeptic verdict was not supported"
        if any(not t.get("threshold_met", False) for t in skeptic_trials):
            return False, "skeptic threshold was not met"

        return True, ""

    @staticmethod
    def _is_auto_commit_candidate_phase(phase: PipelinePhase) -> bool:
        return phase in (
            PipelinePhase.IMPLEMENTATION,
            PipelinePhase.DEBUGGING,
            PipelinePhase.SKEPTIC,
        )

    @staticmethod
    def _phase_requires_repo_delta(phase: PipelinePhase) -> bool:
        return phase in _MUTATING_PHASES

    def _auto_commit_repo(
        self,
        *,
        repo: Path,
        cycle_num: int,
        commit_scope: str,
    ) -> bool:
        """Commit repository changes for an auto-commit checkpoint."""
        if is_clean(repo):
            return False
        try:
            msg = generate_commit_message(
                cycle_num * 100,
                commit_scope,
                "pipeline",
            )
            sha = commit_all(repo, msg)
            self._log("info", f"  Auto-committed: {sha}")
            return True
        except Exception as exc:
            self._log("warn", f"  Auto-commit failed: {exc}")
            return False

    def _check_phase_boundary_token_budget(self, config: PipelineConfig) -> bool:
        """Stop at phase boundary once token budget is exhausted in non-strict mode."""
        if getattr(config, "strict_token_budget", False):
            return False
        if config.max_total_tokens <= 0:
            return False
        if self.state.total_tokens < config.max_total_tokens:
            return False
        if self.state.stop_reason != "budget_exhausted":
            self.state.stop_reason = "budget_exhausted"
            self._log(
                "warn",
                f"Token budget reached ({self.state.total_tokens:,}/{config.max_total_tokens:,}) - stopping after current phase",
            )
        return True

    @staticmethod
    def _binary_exists(binary: str) -> bool:
        return shared_binary_exists(binary)

    @staticmethod
    def _has_codex_auth() -> bool:
        if os.getenv("CODEX_API_KEY") or os.getenv("OPENAI_API_KEY"):
            return True
        home = Path.home()
        for path in (
            home / ".codex" / "auth.json",
            home / ".config" / "codex" / "auth.json",
        ):
            if path.exists():
                return True
        return False

    @staticmethod
    def _has_claude_auth() -> bool:
        if os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY"):
            return True
        home = Path.home()
        for path in (
            home / ".claude.json",
            home / ".claude" / "auth.json",
            home / ".config" / "claude" / "auth.json",
            home / ".config" / "claude-code" / "auth.json",
        ):
            if path.exists():
                return True
        return False

    @staticmethod
    def _repo_write_error(repo: Path) -> str | None:
        try:
            probe_dir = repo / ".codex_manager"
            probe_dir.mkdir(parents=True, exist_ok=True)
            probe_path = probe_dir / f".pipeline-preflight-{dt.datetime.now().timestamp():.6f}.tmp"
            probe_path.write_text("ok", encoding="utf-8")
            probe_path.unlink(missing_ok=True)
            return None
        except Exception as exc:
            return f"Repository is not writable: {exc}"

    def _collect_required_agents(self, config: PipelineConfig) -> set[str]:
        phase_order = config.get_phase_order()
        agents = {
            (p.agent or config.agent or "codex").strip().lower() for p in phase_order if p.enabled
        }
        if not agents:
            agents = {(config.agent or "codex").strip().lower()}
        normalized = set()
        for agent in agents:
            normalized.add("codex" if agent in {"", "auto"} else agent)
        return normalized

    def _preflight_issues(self, config: PipelineConfig, repo: Path) -> list[str]:
        issues: list[str] = []
        phase_order = config.get_phase_order()
        if not (repo / ".git").exists():
            issues.append(f"Not a git repository: {repo}")

        write_error = self._repo_write_error(repo)
        if write_error:
            issues.append(write_error)

        for agent in sorted(self._collect_required_agents(config)):
            if agent == "codex":
                if not self._binary_exists(config.codex_binary):
                    issues.append(f"Codex binary not found: '{config.codex_binary}'")
                if not self._has_codex_auth():
                    issues.append(
                        "Codex auth not detected. Set CODEX_API_KEY or OPENAI_API_KEY, "
                        "or run 'codex login' first."
                    )
            elif agent == "claude_code":
                if not self._binary_exists(config.claude_binary):
                    issues.append(f"Claude Code binary not found: '{config.claude_binary}'")
                if not self._has_claude_auth():
                    issues.append(
                        "Claude auth not detected. Set ANTHROPIC_API_KEY (or CLAUDE_API_KEY), "
                        "or log in with the Claude CLI first."
                    )
            else:
                issues.append(f"Unknown agent '{agent}'. Supported: codex, claude_code, auto")

        uses_cua = any(p.enabled and p.phase == PipelinePhase.VISUAL_TEST for p in phase_order)
        if uses_cua:
            provider = (config.cua_provider or "openai").strip().lower()
            if provider == "openai":
                try:
                    import openai  # noqa: F401
                except Exception:
                    issues.append(
                        "CUA visual test requires the OpenAI SDK. Install with: pip install openai"
                    )
                if not os.getenv("OPENAI_API_KEY"):
                    issues.append("CUA visual test (openai provider) requires OPENAI_API_KEY.")
            elif provider == "anthropic":
                try:
                    import anthropic  # noqa: F401
                except Exception:
                    issues.append(
                        "CUA visual test requires the Anthropic SDK. Install with: pip install anthropic"
                    )
                if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")):
                    issues.append(
                        "CUA visual test (anthropic provider) requires ANTHROPIC_API_KEY or CLAUDE_API_KEY."
                    )

            try:
                from playwright.async_api import async_playwright  # noqa: F401
            except Exception:
                issues.append(
                    "CUA visual test requires Playwright. Install with: pip install playwright && python -m playwright install"
                )

        return issues

    # ------------------------------------------------------------------
    # Main pipeline loop
    # ------------------------------------------------------------------

    def _run_pipeline(self) -> None:
        config = self.config
        repo = self.repo_path
        start_time = time.monotonic()
        self._history_logbook = None
        try:
            history = HistoryLogbook(repo)
            history.initialize()
            self._history_logbook = history
        except Exception as exc:
            self._log("warn", f"History logbook disabled: {exc}")

        self._record_history_note(
            "run_started",
            f"Pipeline run started in {config.mode} mode.",
            context={
                "repo": str(repo),
                "mode": config.mode,
                "max_cycles": config.max_cycles,
                "unlimited": bool(config.unlimited),
                "phase_order": [p.phase.value for p in config.get_phase_order()],
                "science_enabled": bool(config.science_enabled),
                "brain_enabled": bool(config.brain_enabled),
            },
        )

        issues = self._preflight_issues(config, repo)
        if issues:
            for issue in issues:
                self._log("error", f"Preflight check failed: {issue}")
            self.state.stop_reason = "preflight_failed"
            self._finalize_run(
                start_time=start_time,
                history_level="error",
                extra_history_context={"preflight_issues": issues},
            )
            return

        # Build agent runners
        try:
            import codex_manager.claude_code as claude_code_module
        except ImportError:
            claude_code_module = None

        runners: dict[str, AgentRunner] = {
            "codex": CodexRunner(
                codex_binary=config.codex_binary,
                timeout=config.timeout_per_phase,
                sandbox_mode=config.codex_sandbox_mode,
                approval_policy=config.codex_approval_policy,
                reasoning_effort=config.codex_reasoning_effort,
                bypass_approvals_and_sandbox=config.codex_bypass_approvals_and_sandbox,
            ),
        }
        if claude_code_module is not None:
            runners["claude_code"] = claude_code_module.ClaudeCodeRunner(
                claude_binary=config.claude_binary,
                timeout=config.timeout_per_phase,
            )
        runners["auto"] = runners["codex"]

        test_cmd = parse_test_command(config.test_cmd)
        evaluator = RepoEvaluator(test_cmd=test_cmd, skip_tests=(test_cmd is None))

        # Initialize brain
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
                    "local_only": bool(brain.config.local_only),
                },
            )

        # Initialize log files
        try:
            self.tracker.initialize()
            self._log("info", "Log files initialized")
            if config.science_enabled:
                self.tracker.initialize_science()
                self._log(
                    "info",
                    f"Scientist evidence directory: {self.tracker.science_dir()}",
                )
        except Exception as exc:
            self._log("error", f"Failed to initialize pipeline logs: {exc}")
            self.ledger.add(
                category="error",
                title=f"Log initialization failed: {str(exc)[:60]}",
                detail=str(exc),
                severity="critical",
                source="pipeline:startup",
                step_ref="log_init",
            )
            self.state.stop_reason = f"error: {exc}"
            self._finalize_run(
                start_time=start_time,
                history_level="error",
                extra_history_context={"error": str(exc)},
            )
            return

        # Create branch in apply mode
        if config.mode == "apply":
            try:
                branch = create_branch(repo)
                self._log("info", f"Created branch: {branch}")
            except Exception as exc:
                self._log("error", f"Failed to create branch: {exc}")
                self.ledger.add(
                    category="error",
                    title=f"Branch creation failed: {str(exc)[:60]}",
                    detail=str(exc),
                    severity="critical",
                    source="pipeline:startup",
                    step_ref="branch_creation",
                )
                self.state.stop_reason = "branch_creation_failed"
                self._finalize_run(
                    start_time=start_time,
                    history_level="error",
                    extra_history_context={"error": str(exc)},
                )
                return

        phase_order = config.get_phase_order()
        self._log("info", f"Pipeline phases: {[p.phase.value for p in phase_order]}")
        commit_frequency = (config.commit_frequency or "per_phase").strip().lower()
        if config.auto_commit and config.mode == "apply":
            self._log("info", f"Auto-commit policy: {commit_frequency}")

        # Unlimited mode uses a very high ceiling; the improvement-threshold
        # check (inside _check_stop_conditions) is what actually ends the run.
        effective_max = 999_999 if config.unlimited else config.max_cycles

        try:
            for cycle_num in range(1, effective_max + 1):
                if self._stop_event.is_set():
                    self.state.stop_reason = "user_stopped"
                    break

                self.state.current_cycle = cycle_num
                cycle_label = (
                    f"{'=' * 20} Cycle {cycle_num} / inf {'=' * 20}"
                    if config.unlimited
                    else f"{'=' * 20} Cycle {cycle_num} / {config.max_cycles} {'=' * 20}"
                )
                self._log("info", cycle_label)

                cycle_aborted = False
                science_cycle_trials: dict[str, list[dict[str, Any]]] = {
                    "experiment": [],
                    "skeptic": [],
                }
                science_latest_hypothesis_id = ""
                science_latest_experiment_id = ""
                cycle_has_auto_commit_candidate = False
                cycle_science_commit_ok = True
                cycle_science_commit_reason = ""

                for phase_cfg in phase_order:
                    self._pause_event.wait()
                    if self._stop_event.is_set():
                        self.state.stop_reason = "user_stopped"
                        cycle_aborted = True
                        break

                    phase = phase_cfg.phase
                    self.state.current_phase = phase.value

                    # CUA visual test phase - runs separately
                    if phase == PipelinePhase.VISUAL_TEST:
                        self._log("info", f"Phase: {phase.value} (CUA Visual Test)")
                        self.state.current_iteration = 1
                        self.state.current_phase_started_at_epoch_ms = int(time.time() * 1000)
                        cua_result = self._run_cua_phase(config, cycle_num)
                        self._record_history_note(
                            "phase_result",
                            (
                                f"Cycle {cycle_num}, phase '{phase.value}' "
                                f"finished with status={'ok' if cua_result else 'failed'}."
                            ),
                            level="info" if cua_result else "warn",
                            context={
                                "cycle": cycle_num,
                                "phase": phase.value,
                                "iteration": 1,
                                "mode": config.mode,
                                "agent_success": bool(cua_result),
                                "validation_success": bool(cua_result),
                                "tests_passed": bool(cua_result),
                                "success": bool(cua_result),
                                "test_outcome": "passed" if cua_result else "failed",
                                "files_changed": 0,
                                "net_lines_changed": 0,
                                "changed_files": [],
                                "cua": True,
                            },
                        )
                        if cua_result:
                            self.state.total_phases_completed += 1
                            self.state.elapsed_seconds = time.monotonic() - start_time
                        continue

                    # Determine prompt source: custom override > catalog
                    custom = getattr(phase_cfg, "custom_prompt", "")
                    if custom and custom.strip():
                        base_prompt = custom.strip()
                    elif phase.value in ("theorize", "experiment", "skeptic", "analyze"):
                        base_prompt = self.catalog.scientist(phase.value)
                    else:
                        base_prompt = self.catalog.pipeline(phase.value)

                    if not base_prompt:
                        self._log("warn", f"No prompt found for phase: {phase.value}")
                        continue

                    self._log(
                        "info",
                        f"Phase: {phase.value} "
                        f"({phase_cfg.iterations} iteration{'s' if phase_cfg.iterations > 1 else ''})",
                    )

                    for iteration in range(1, phase_cfg.iterations + 1):
                        self._pause_event.wait()
                        if self._stop_event.is_set():
                            self.state.stop_reason = "user_stopped"
                            cycle_aborted = True
                            break

                        self.state.current_iteration = iteration
                        self.state.current_phase_started_at_epoch_ms = int(time.time() * 1000)

                        if phase_cfg.iterations > 1:
                            self._log("info", f"  Iteration {iteration}/{phase_cfg.iterations}")

                        science_baseline_eval: EvalResult | None = None
                        science_baseline_clean = False
                        science_hypothesis: dict[str, str] | None = None
                        if self._is_science_phase(phase):
                            science_baseline_eval = evaluator.evaluate(repo)
                            science_baseline_clean = is_clean(repo)
                            preferred_hypothesis_id = (
                                science_latest_hypothesis_id
                                if phase == PipelinePhase.SKEPTIC
                                else ""
                            )
                            science_hypothesis = self._select_science_hypothesis(
                                self.tracker.read("EXPERIMENTS.md"),
                                preferred_hypothesis_id=preferred_hypothesis_id,
                            )

                        # Build the full prompt with log context
                        context = self.tracker.get_context_for_phase(
                            phase.value, ledger=self.ledger
                        )
                        full_prompt = self._build_phase_prompt(
                            base_prompt,
                            phase,
                            context,
                            cycle_num,
                            iteration,
                        )

                        # Brain refinement
                        if brain.enabled:
                            original_prompt = full_prompt
                            history = self._build_history_summary()
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
                            full_prompt = brain.plan_step(
                                goal=self._brain_goal(phase.value),
                                step_name=phase.value,
                                base_prompt=full_prompt,
                                history_summary=history,
                                ledger_context=ledger_ctx,
                            )
                            self._record_brain_note(
                                "plan_phase",
                                f"Brain refined prompt for phase '{phase.value}'",
                                context={
                                    "cycle": cycle_num,
                                    "phase": phase.value,
                                    "iteration": iteration,
                                    "prompt_changed": full_prompt != original_prompt,
                                    "original_length": len(original_prompt),
                                    "refined_length": len(full_prompt),
                                },
                            )

                        if science_baseline_eval is not None:
                            full_prompt = self._build_science_prompt(
                                prompt=full_prompt,
                                phase=phase,
                                cycle=cycle_num,
                                iteration=iteration,
                                baseline_eval=science_baseline_eval,
                                selected_hypothesis=science_hypothesis,
                                preferred_experiment_id=(
                                    science_latest_experiment_id
                                    if phase == PipelinePhase.SKEPTIC
                                    else ""
                                ),
                            )

                        self._log("info", f"  Prompt: {full_prompt[:120]}...")

                        # Execute
                        agent_key = (phase_cfg.agent or "").strip().lower()
                        if agent_key in {"", "auto"}:
                            agent_key = (config.agent or "codex").strip().lower()
                        runner = runners.get(agent_key, runners["codex"])

                        result = self._execute_phase(
                            runner,
                            evaluator,
                            repo,
                            config,
                            phase,
                            cycle_num,
                            iteration,
                            full_prompt,
                        )

                        if science_baseline_eval is not None:
                            trial = self._record_science_trial(
                                cycle=cycle_num,
                                phase=phase,
                                iteration=iteration,
                                prompt=full_prompt,
                                result=result,
                                baseline_eval=science_baseline_eval,
                                repo=repo,
                                baseline_repo_clean=science_baseline_clean,
                                selected_hypothesis=science_hypothesis,
                                preferred_experiment_id=(
                                    science_latest_experiment_id
                                    if phase == PipelinePhase.SKEPTIC
                                    else ""
                                ),
                            )
                            if phase in (PipelinePhase.EXPERIMENT, PipelinePhase.SKEPTIC):
                                science_cycle_trials[phase.value].append(trial)
                            if phase == PipelinePhase.EXPERIMENT:
                                if trial.get("hypothesis_id"):
                                    science_latest_hypothesis_id = str(trial["hypothesis_id"])
                                if trial.get("experiment_id"):
                                    science_latest_experiment_id = str(trial["experiment_id"])
                            if phase in (PipelinePhase.EXPERIMENT, PipelinePhase.SKEPTIC):
                                self._log(
                                    "info",
                                    f"  Scientist verdict: {trial.get('verdict', 'n/a')} "
                                    f"({trial.get('rationale', '')}) | "
                                    f"threshold={trial.get('threshold_met', False)} | "
                                    f"rollback={trial.get('rollback_action', 'n/a')}",
                                )

                        self.state.results.append(result)
                        if result.success:
                            self.state.successes += 1
                        else:
                            self.state.failures += 1
                        self.state.total_phases_completed += 1
                        self.state.total_tokens += result.input_tokens + result.output_tokens
                        self.state.elapsed_seconds = time.monotonic() - start_time
                        if self._check_strict_token_budget(config):
                            cycle_aborted = True
                            break

                        # Log failures to knowledge ledger
                        if not result.success and result.error_message:
                            self.ledger.add(
                                category="error",
                                title=result.error_message[:80] or "Phase failed",
                                detail=result.error_message,
                                severity="major",
                                source=f"pipeline:{phase.value}",
                                step_ref=f"cycle{cycle_num}:{phase.value}:iter{iteration}",
                            )

                        # Log to tracker
                        self.tracker.log_phase_result(
                            phase.value,
                            iteration,
                            result.success,
                            f"Files: {result.files_changed}, "
                            f"Lines: {result.net_lines_changed:+d}, "
                            f"Tests: {result.test_outcome}",
                        )
                        self._record_history_note(
                            "phase_result",
                            (
                                f"Cycle {cycle_num}, phase '{phase.value}' "
                                f"(iteration {iteration}) finished with "
                                f"status={'ok' if result.success else 'failed'} "
                                f"and tests={result.test_outcome}."
                            ),
                            level="info" if result.success else "warn",
                            context={
                                "cycle": cycle_num,
                                "phase": phase.value,
                                "iteration": iteration,
                                "mode": config.mode,
                                "agent_success": result.agent_success,
                                "validation_success": result.validation_success,
                                "tests_passed": result.tests_passed,
                                "success": result.success,
                                "test_outcome": result.test_outcome,
                                "files_changed": result.files_changed,
                                "net_lines_changed": result.net_lines_changed,
                                "changed_files": result.changed_files,
                                "duration_seconds": result.duration_seconds,
                                "commit_sha": result.commit_sha,
                                "terminate_repeats": result.terminate_repeats,
                                "error_message": result.error_message[:500],
                            },
                        )

                        # Brain post-evaluation
                        control_result = result
                        if brain.enabled:
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
                                step_name=phase.value,
                                success=result.success,
                                test_outcome=result.test_outcome,
                                files_changed=result.files_changed,
                                net_lines=result.net_lines_changed,
                                errors=[result.error_message] if result.error_message else [],
                                goal=self._brain_goal(phase.value),
                                ledger_context=ledger_ctx,
                            )
                            self._record_brain_note(
                                "evaluate_phase",
                                f"Brain selected action '{decision.action}' for phase '{phase.value}'",
                                context={
                                    "cycle": cycle_num,
                                    "phase": phase.value,
                                    "iteration": iteration,
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
                                self._log("info", f"  Brain: {decision.reasoning[:200]}")
                            if decision.action in ("follow_up", "retry"):
                                if result.terminate_repeats:
                                    self._log(
                                        "info",
                                        f"  {TERMINATE_STEP_TAG} detected; skipping brain follow-up for this phase iteration.",
                                    )
                                    self._record_brain_note(
                                        "follow_up_skipped",
                                        "Skipped brain follow-up because phase emitted terminate signal.",
                                        context={
                                            "cycle": cycle_num,
                                            "phase": phase.value,
                                            "iteration": iteration,
                                            "action": decision.action,
                                        },
                                    )
                                else:
                                    followup_prompt = (decision.follow_up_prompt or "").strip() or (
                                        f"Retry phase '{phase.value}' to resolve remaining issues. "
                                        f"Prior attempt test_outcome={result.test_outcome}, "
                                        f"files_changed={result.files_changed}, "
                                        f"net_lines={result.net_lines_changed:+d}."
                                    )
                                    self._log(
                                        "info",
                                        f"  Brain requested {decision.action}; running one follow-up before resuming pipeline flow.",
                                    )
                                    self._record_brain_note(
                                        "follow_up_started",
                                        f"Running brain {decision.action} follow-up before returning to pipeline order.",
                                        context={
                                            "cycle": cycle_num,
                                            "phase": phase.value,
                                            "iteration": iteration,
                                            "action": decision.action,
                                            "prompt_preview": followup_prompt[:400],
                                        },
                                    )
                                    followup_result = self._execute_phase(
                                        runner,
                                        evaluator,
                                        repo,
                                        config,
                                        phase,
                                        cycle_num,
                                        iteration,
                                        followup_prompt,
                                    )
                                    self.state.results.append(followup_result)
                                    if followup_result.success:
                                        self.state.successes += 1
                                    else:
                                        self.state.failures += 1
                                    self.state.total_phases_completed += 1
                                    self.state.total_tokens += (
                                        followup_result.input_tokens + followup_result.output_tokens
                                    )
                                    self.state.elapsed_seconds = time.monotonic() - start_time
                                    if self._check_strict_token_budget(config):
                                        cycle_aborted = True
                                        break

                                    if (
                                        not followup_result.success
                                        and followup_result.error_message
                                    ):
                                        self.ledger.add(
                                            category="error",
                                            title=followup_result.error_message[:80]
                                            or "Brain follow-up failed",
                                            detail=followup_result.error_message,
                                            severity="major",
                                            source=f"pipeline:{phase.value}:brain_followup",
                                            step_ref=f"cycle{cycle_num}:{phase.value}:iter{iteration}:brain_followup",
                                        )

                                    self.tracker.log_phase_result(
                                        phase.value,
                                        iteration,
                                        followup_result.success,
                                        "[Brain follow-up] "
                                        f"Files: {followup_result.files_changed}, "
                                        f"Lines: {followup_result.net_lines_changed:+d}, "
                                        f"Tests: {followup_result.test_outcome}",
                                    )
                                    self._record_history_note(
                                        "phase_result",
                                        (
                                            f"Cycle {cycle_num}, phase '{phase.value}' "
                                            f"(iteration {iteration}, brain follow-up) finished with "
                                            f"status={'ok' if followup_result.success else 'failed'} "
                                            f"and tests={followup_result.test_outcome}."
                                        ),
                                        level="info" if followup_result.success else "warn",
                                        context={
                                            "cycle": cycle_num,
                                            "phase": phase.value,
                                            "iteration": iteration,
                                            "mode": config.mode,
                                            "agent_success": followup_result.agent_success,
                                            "validation_success": followup_result.validation_success,
                                            "tests_passed": followup_result.tests_passed,
                                            "success": followup_result.success,
                                            "test_outcome": followup_result.test_outcome,
                                            "files_changed": followup_result.files_changed,
                                            "net_lines_changed": followup_result.net_lines_changed,
                                            "changed_files": followup_result.changed_files,
                                            "duration_seconds": followup_result.duration_seconds,
                                            "commit_sha": followup_result.commit_sha,
                                            "terminate_repeats": followup_result.terminate_repeats,
                                            "error_message": followup_result.error_message[:500],
                                            "brain_follow_up": True,
                                        },
                                    )
                                    self._record_brain_note(
                                        "follow_up_finished",
                                        "Brain follow-up completed; resuming normal pipeline sequence.",
                                        context={
                                            "cycle": cycle_num,
                                            "phase": phase.value,
                                            "iteration": iteration,
                                            "success": followup_result.success,
                                            "test_outcome": followup_result.test_outcome,
                                            "files_changed": followup_result.files_changed,
                                            "net_lines_changed": followup_result.net_lines_changed,
                                            "terminate_repeats": followup_result.terminate_repeats,
                                        },
                                    )
                                    control_result = followup_result
                            elif decision.action == "escalate":
                                self._log(
                                    "error", f"Brain escalation: {decision.human_message[:300]}"
                                )
                                self._record_brain_note(
                                    "escalation",
                                    "Brain escalated and paused the pipeline.",
                                    level="error",
                                    context={
                                        "cycle": cycle_num,
                                        "phase": phase.value,
                                        "iteration": iteration,
                                        "human_message": decision.human_message[:400],
                                        "reasoning": decision.reasoning[:400],
                                    },
                                )
                                self.state.stop_reason = "brain_escalation"
                                self.pause()
                                cycle_aborted = True
                                break
                            elif decision.action == "stop":
                                self._log("info", "Brain requested stopping this pipeline run.")
                                self._record_brain_note(
                                    "brain_stop",
                                    "Brain requested stop after phase evaluation.",
                                    context={
                                        "cycle": cycle_num,
                                        "phase": phase.value,
                                        "iteration": iteration,
                                        "reasoning": decision.reasoning[:400],
                                    },
                                )
                                self.state.stop_reason = "brain_requested_stop"
                                cycle_aborted = True
                                break

                        # Handle failures
                        if not control_result.success and phase_cfg.on_failure == "abort":
                            self._log("error", f"Phase {phase.value} failed - aborting")
                            self.state.stop_reason = "phase_failed_abort"
                            cycle_aborted = True
                            break

                        terminate_signal = (
                            result.terminate_repeats or control_result.terminate_repeats
                        )
                        if terminate_signal and iteration < phase_cfg.iterations:
                            remaining = phase_cfg.iterations - iteration
                            self._log(
                                "info",
                                f"  Phase {phase.value} emitted {TERMINATE_STEP_TAG}; "
                                f"skipping {remaining} remaining iteration(s).",
                            )
                            break

                    if cycle_aborted:
                        break

                    science_commit_ok = True
                    science_commit_reason = ""
                    if phase == PipelinePhase.SKEPTIC:
                        science_commit_ok, science_commit_reason = self._science_commit_gate(
                            science_cycle_trials["experiment"],
                            science_cycle_trials["skeptic"],
                        )
                        if not science_commit_ok:
                            self._log(
                                "warn",
                                f"  Auto-commit skipped: {science_commit_reason}",
                            )
                            cycle_science_commit_ok = False
                            if not cycle_science_commit_reason:
                                cycle_science_commit_reason = science_commit_reason

                    if (
                        config.auto_commit
                        and config.mode == "apply"
                        and commit_frequency == "per_phase"
                        and self._is_auto_commit_candidate_phase(phase)
                        and (phase != PipelinePhase.SKEPTIC or science_commit_ok)
                    ):
                        self._auto_commit_repo(
                            repo=repo,
                            cycle_num=cycle_num,
                            commit_scope=f"pipeline-{phase.value}",
                        )

                    if self._is_auto_commit_candidate_phase(phase):
                        cycle_has_auto_commit_candidate = True

                    if self._check_phase_boundary_token_budget(config):
                        cycle_aborted = True
                        break

                if (
                    not cycle_aborted
                    and config.auto_commit
                    and config.mode == "apply"
                    and commit_frequency == "per_cycle"
                ):
                    if not cycle_has_auto_commit_candidate:
                        self._log(
                            "info",
                            f"  Auto-commit skipped for cycle {cycle_num}: no eligible phases ran.",
                        )
                    elif not cycle_science_commit_ok:
                        reason = cycle_science_commit_reason or "science commit gate failed"
                        self._log(
                            "warn",
                            f"  Auto-commit skipped for cycle {cycle_num}: {reason}",
                        )
                    else:
                        self._auto_commit_repo(
                            repo=repo,
                            cycle_num=cycle_num,
                            commit_scope=f"pipeline-cycle-{cycle_num}",
                        )

                if cycle_aborted:
                    break

                self.state.total_cycles_completed = cycle_num

                # Brain progress assessment between cycles
                if brain.enabled and cycle_num >= 2:
                    progress = brain.assess_progress(
                        goal=self._brain_goal("progress-assessment"),
                        total_loops=cycle_num,
                        history_summary=self._build_history_summary(max_entries=30),
                    )
                    self._record_brain_note(
                        "progress_assessment",
                        f"Brain progress assessment action='{progress.action}'",
                        context={
                            "cycle": cycle_num,
                            "action": progress.action,
                            "reasoning": progress.reasoning[:500],
                        },
                    )
                    if progress.reasoning:
                        self._log("info", f"Brain assessment: {progress.reasoning[:200]}")
                    if progress.action == "stop":
                        self.state.stop_reason = "brain_converged"
                        self._log("info", "Brain recommends stopping - goals achieved")
                        break

                # Budget checks
                stop = self._check_stop_conditions(config, start_time)
                if stop:
                    self.state.stop_reason = stop
                    self._log("info", f"Stopping: {stop}")
                    break

            else:
                self.state.stop_reason = "max_cycles_reached"

        except Exception as exc:
            self._log("error", f"Unexpected error: {exc}")
            self.ledger.add(
                category="error",
                title=f"Pipeline error: {str(exc)[:60]}",
                detail=str(exc),
                severity="critical",
                source="pipeline:runtime",
                step_ref="",
            )
            self.state.stop_reason = f"error: {exc}"

        finally:
            self._finalize_run(start_time=start_time)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_phase_prompt(
        self,
        base_prompt: str,
        phase: PipelinePhase,
        context: str,
        cycle: int,
        iteration: int,
    ) -> str:
        """Assemble the full prompt for a phase execution."""
        parts: list[str] = []

        # Phase header
        parts.append(f"[Pipeline Cycle {cycle}, Phase: {phase.value}, Iteration {iteration}]")
        parts.append("")
        parts.append("--- REPOSITORY SCOPE (STRICT) ---")
        parts.append(f"You are operating inside repository `{self.repo_path}`.")
        parts.append("Use this repository as the single source of truth for this run.")
        parts.append("Do not search for, switch to, or request any other repository/project.")
        parts.append("Do not read or modify files outside this repository root.")
        parts.append("")

        # Base prompt
        parts.append(base_prompt)

        # Log file context (includes ledger when tracker was called with ledger)
        if context:
            parts.append("\n--- CURRENT LOG FILE CONTENTS ---\n")
            parts.append(context)

        # Previous results context
        if self.state.results:
            recent = self.state.results[-5:]
            parts.append("\n--- RECENT PIPELINE ACTIVITY ---\n")
            for r in recent:
                status = "OK" if r.success else "FAIL"
                parts.append(
                    f"  {r.phase} (iter {r.iteration}): {status}, "
                    f"tests={r.test_outcome}, "
                    f"files={r.files_changed}, "
                    f"lines={r.net_lines_changed:+d}"
                )

        # Log file paths (so the agent knows where they are)
        log_file = PHASE_LOG_FILES.get(phase)
        if log_file:
            log_path = self.tracker.path_for(log_file)
            parts.append(f"\nLog file location: {log_path}")

        parts.append("\n--- ITERATION CONTROL ---\n")
        parts.append(terminate_step_instruction("phase iteration"))

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Phase execution
    # ------------------------------------------------------------------

    def _execute_phase(
        self,
        runner: AgentRunner,
        evaluator: RepoEvaluator,
        repo: Path,
        config: PipelineConfig,
        phase: PipelinePhase,
        cycle: int,
        iteration: int,
        prompt: str,
    ) -> PhaseResult:
        """Execute a single phase iteration."""
        phase_start = time.monotonic()
        start_head_sha = ""
        try:
            start_head_sha = head_sha(repo)
        except Exception:
            start_head_sha = ""

        # Run the agent
        try:
            run_result = self._run_agent_with_keepalive(
                runner,
                repo,
                prompt,
                activity_label=f"Phase '{phase.value}' iteration {iteration}",
                timeout_seconds=config.timeout_per_phase,
                full_auto=True,
            )
        except Exception as exc:
            return PhaseResult(
                cycle=cycle,
                phase=phase.value,
                iteration=iteration,
                agent_success=False,
                validation_success=False,
                tests_passed=False,
                success=False,
                test_summary=str(exc),
                error_message=str(exc),
                agent_used=runner.name,
                prompt_used=prompt[:500],
                duration_seconds=round(time.monotonic() - phase_start, 1),
            )

        all_text_parts = [event.text for event in run_result.events if event.text]
        agent_output = run_result.final_message or "\n\n".join(all_text_parts)
        terminate_repeats = contains_terminate_step_signal(agent_output)
        if terminate_repeats:
            self._log(
                "info",
                f"  Agent emitted {TERMINATE_STEP_TAG}; remaining iterations can be skipped.",
            )

        # Evaluate
        eval_result = evaluator.evaluate(repo)
        outcome_level = (
            "info" if eval_result.test_outcome.value in ("passed", "skipped") else "warn"
        )
        self._log(
            outcome_level,
            f"  Tests: {eval_result.test_outcome.value} | "
            f"Files: {eval_result.files_changed} | "
            f"Net d: {eval_result.net_lines_changed:+d}",
        )
        repo_dirty = bool((eval_result.status_porcelain or "").strip())
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
                self._log(
                    "info",
                    (
                        f"  Detected agent-authored commit {end_head_sha} "
                        f"({files_changed} files, net {eval_result.net_lines_changed:+d})."
                    ),
                )

        # Handle commit phase specially - it commits, others go through
        # the normal apply/revert flow
        commit_sha = None
        if phase == PipelinePhase.COMMIT and config.mode == "apply" and repo_dirty:
            try:
                msg = generate_commit_message(
                    iteration,
                    "pipeline-commit-phase",
                    eval_result.test_outcome.value,
                )
                commit_sha = commit_all(repo, msg)
                self._log("info", f"  Committed: {commit_sha}")
            except Exception as exc:
                self._log("error", f"  Commit failed: {exc}")
                self.ledger.add(
                    category="error",
                    title=f"Commit failed: {str(exc)[:60]}",
                    detail=str(exc),
                    severity="major",
                    source="pipeline:commit",
                    step_ref=f"{phase.value}:iter{iteration}",
                )
        elif config.mode == "apply" and head_advanced and end_head_sha:
            commit_sha = end_head_sha
            self._log("info", f"  Using agent-authored commit: {commit_sha}")
        if config.mode == "dry-run":
            if head_advanced and start_head_sha:
                try:
                    reset_to_ref(repo, start_head_sha)
                    self._log(
                        "info",
                        (
                            "  Dry-run rollback restored repository to pre-phase HEAD "
                            f"({start_head_sha})."
                        ),
                    )
                except Exception as exc:
                    self._log("warn", f"  Could not reset dry-run commit(s): {exc}")
                    if not is_clean(repo):
                        revert_all(repo)
                        self._log("info", "  Changes reverted (dry-run)")
            elif repo_dirty:
                revert_all(repo)
                self._log("info", "  Changes reverted (dry-run)")

        tests_outcome = eval_result.test_outcome.value
        tests_passed = tests_outcome == "passed"
        tests_validation_success = tests_outcome in ("passed", "skipped")
        requires_repo_delta = self._phase_requires_repo_delta(phase)
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
                "  Validation marked phase as failed despite agent exit success: "
                + ", ".join(validation_failures),
            )

        error_message = "; ".join(run_result.errors) if run_result.errors else ""
        if validation_failures:
            validation_msg = "Validation failed: " + ", ".join(validation_failures)
            error_message = (
                f"{error_message}; {validation_msg}" if error_message else validation_msg
            )

        duration = time.monotonic() - phase_start
        return PhaseResult(
            cycle=cycle,
            phase=phase.value,
            iteration=iteration,
            agent_success=bool(run_result.success),
            validation_success=validation_success,
            tests_passed=tests_passed,
            success=final_success,
            test_outcome=tests_outcome,
            test_summary=eval_result.test_summary[:2000],
            test_exit_code=eval_result.test_exit_code,
            files_changed=eval_result.files_changed,
            net_lines_changed=eval_result.net_lines_changed,
            changed_files=eval_result.changed_files,
            commit_sha=commit_sha,
            error_message=error_message,
            duration_seconds=round(duration, 1),
            input_tokens=run_result.usage.input_tokens,
            output_tokens=run_result.usage.output_tokens,
            prompt_used=prompt[:500],
            agent_final_message=(agent_output or "")[:5000],
            terminate_repeats=terminate_repeats,
            agent_used=runner.name,
        )

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

    # ------------------------------------------------------------------
    # CUA visual test phase
    # ------------------------------------------------------------------

    def _run_cua_phase(self, config: PipelineConfig, cycle: int) -> bool:
        """Execute a CUA (Computer-Using Agent) visual test phase.

        Returns True if the session ran successfully.
        """
        try:
            from codex_manager.cua.actions import CUAProvider, CUASessionConfig
            from codex_manager.cua.session import run_cua_session_sync
        except ImportError as exc:
            self._log(
                "warn",
                f"CUA dependencies not installed, skipping visual test: {exc}. "
                "Install with: pip install codex-manager[cua] then python -m playwright install",
            )
            return False

        provider_str = getattr(config, "cua_provider", "openai")
        provider = CUAProvider.ANTHROPIC if provider_str == "anthropic" else CUAProvider.OPENAI

        target_url = getattr(config, "cua_target_url", "")
        task = getattr(config, "cua_task", "")
        if not task:
            task = (
                "Visually inspect the application UI. "
                "Navigate through the main views, test interactive elements, "
                "and report any visual bugs, broken layouts, or usability issues."
            )

        cua_config = CUASessionConfig(
            provider=provider,
            target_url=target_url,
            task=task,
            headless=getattr(config, "cua_headless", True),
            max_steps=30,
            timeout_seconds=config.timeout_per_phase,
            save_screenshots=True,
        )

        self._log("info", f"  CUA provider: {provider.value} | URL: {target_url or '(none)'}")
        self._log("info", f"  CUA task: {task[:120]}...")

        step_ref = f"cycle{cycle}:phase:visual_test"
        try:
            result = run_cua_session_sync(cua_config, ledger=self.ledger, step_ref=step_ref)

            self._log(
                "info" if result.success else "warn",
                f"  CUA finished: {result.total_steps} steps, "
                f"{result.duration_seconds}s, success={result.success}",
            )
            if result.summary:
                self._log("info", f"  CUA summary: {result.summary[:300]}")
            if result.error:
                self._log("error", f"  CUA error: {result.error}")
            if result.screenshots_saved:
                self._log(
                    "info",
                    f"  Screenshots saved: {len(result.screenshots_saved)} files",
                )

            # Log findings to TESTPLAN.md
            findings = f"\n## CUA Visual Test - Cycle {cycle}\n\n"
            findings += f"- **Provider**: {result.provider}\n"
            findings += f"- **Steps**: {result.total_steps}\n"
            findings += f"- **Duration**: {result.duration_seconds}s\n"
            findings += f"- **Success**: {result.success}\n"
            if result.observations:
                findings += f"\n### Observations ({len(result.observations)})\n\n"
                findings += result.observations_markdown()
                findings += "\n"
                self._log(
                    "info", f"  CUA captured {len(result.observations)} structured observations"
                )
            if result.summary:
                # Strip raw OBSERVATION lines from the summary for readability
                clean_summary = "\n".join(
                    line
                    for line in result.summary.splitlines()
                    if not line.strip().upper().startswith("OBSERVATION|")
                )
                if clean_summary.strip():
                    findings += f"\n### Summary\n\n{clean_summary.strip()}\n"
            if result.error:
                findings += f"\n### Errors\n\n{result.error}\n"

            self.tracker.append("TESTPLAN.md", findings)

            return result.success

        except Exception as exc:
            self._log("error", f"CUA visual test failed: {exc}")
            self.ledger.add(
                category="error",
                title=f"CUA visual test failed: {str(exc)[:60]}",
                detail=str(exc),
                severity="major",
                source="pipeline:visual_test",
                step_ref=f"cycle{cycle}:phase:visual_test",
            )
            return False

    # ------------------------------------------------------------------
    # History summary
    # ------------------------------------------------------------------

    def _build_history_summary(self, max_entries: int = 15) -> str:
        """Build a concise summary of recent results for the brain."""
        if not self.state.results:
            return "No previous results."
        recent = self.state.results[-max_entries:]
        lines = []
        for r in recent:
            status = "OK" if r.success else "FAIL"
            cycle_ref = r.cycle or self.state.current_cycle
            lines.append(
                f"  Cycle {cycle_ref}, {r.phase} (iter {r.iteration}): "
                f"{status}, tests={r.test_outcome}, "
                f"files={r.files_changed}, lines={r.net_lines_changed:+d}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stop conditions
    # ------------------------------------------------------------------

    def _check_stop_conditions(self, config: PipelineConfig, start_time: float) -> str | None:
        elapsed_min = (time.monotonic() - start_time) / 60

        if config.max_time_minutes > 0 and elapsed_min >= config.max_time_minutes:
            return "max_time_reached"

        if config.max_total_tokens > 0 and self.state.total_tokens >= config.max_total_tokens:
            return "budget_exhausted"

        # Improvement-threshold check (powers the "unlimited" mode)
        # Compare the latest full cycle to the previous one.
        # When unlimited is on, convergence is always checked.
        results = self.state.results
        prev_cycle: list[PhaseResult] = []
        curr_cycle: list[PhaseResult] = []

        if self.state.total_cycles_completed >= 2:
            prev_cycle_num = self.state.total_cycles_completed - 1
            curr_cycle_num = self.state.total_cycles_completed
            prev_cycle = [r for r in results if r.cycle == prev_cycle_num]
            curr_cycle = [r for r in results if r.cycle == curr_cycle_num]

            # Backward-compatible fallback for legacy states without cycle tags.
            if not prev_cycle or not curr_cycle:
                phase_count = len(config.get_phase_order())
                if len(results) >= phase_count * 2:
                    prev_cycle = results[-(phase_count * 2) : -phase_count]
                    curr_cycle = results[-phase_count:]

        if prev_cycle and curr_cycle:
            imp = self._compute_improvement(prev_cycle, curr_cycle)
            self.state.improvement_pct = round(imp, 2)

            threshold = config.improvement_threshold
            if (config.unlimited or config.stop_on_convergence) and imp < threshold:
                self._log(
                    "info",
                    f"Improvement dropped to {imp:.2f}% "
                    f"(threshold {threshold}%) - diminishing returns",
                )
                return "diminishing_returns"

        # Legacy simple convergence fallback (consistent with chain.py)
        if config.stop_on_convergence and len(results) >= 4:
            last_4 = results[-4:]
            all_pass = all(r.test_outcome in ("passed", "skipped") for r in last_4)
            all_low = all(abs(r.net_lines_changed) < 20 and r.files_changed <= 2 for r in last_4)
            if all_pass and all_low:
                return "convergence_detected"

        return None

    def _check_strict_token_budget(self, config: PipelineConfig) -> bool:
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
                f"Token budget reached ({self.state.total_tokens:,}/{config.max_total_tokens:,}) - strict budget mode stopping run now",
            )
        return True

    @staticmethod
    def _compute_improvement(prev: list[PhaseResult], curr: list[PhaseResult]) -> float:
        """Compute improvement percentage between two cycles."""

        def _activity(results: list[PhaseResult]) -> int:
            return sum(r.files_changed for r in results)

        def _magnitude(results: list[PhaseResult]) -> int:
            return sum(abs(r.net_lines_changed) for r in results)

        def _success_rate(results: list[PhaseResult]) -> float:
            if not results:
                return 0.0
            return sum(1 for r in results if r.success) / len(results) * 100

        prev_act = max(_activity(prev), 1)
        curr_act = _activity(curr)
        prev_mag = max(_magnitude(prev), 1)
        curr_mag = _magnitude(curr)
        prev_sr = _success_rate(prev)
        curr_sr = _success_rate(curr)

        activity_ratio = curr_act / prev_act * 100
        magnitude_ratio = curr_mag / prev_mag * 100
        success_delta = max(0, curr_sr - prev_sr + 50)

        score = activity_ratio * 0.45 + magnitude_ratio * 0.35 + success_delta * 0.20
        return min(score, 200.0)
