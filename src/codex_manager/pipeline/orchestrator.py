"""Pipeline orchestrator - drives the autonomous improvement pipeline.

The orchestrator runs multiple *cycles*, where each cycle executes all
enabled phases in order:

    Ideation -> Prioritization -> (Theorize -> Experiment -> Skeptic -> Analyze)
    -> Implementation -> Testing -> Debugging -> Commit -> Progress Review

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
import hashlib
import json
import logging
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from codex_manager.agent_runner import AgentRunner
from codex_manager.agent_signals import (
    TERMINATE_STEP_TAG,
    contains_terminate_step_signal,
    terminate_step_instruction,
)
from codex_manager.artifact_retention import RetentionPolicy, cleanup_runtime_artifacts
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
from codex_manager.managed_artifacts import (
    capture_artifact_snapshot,
    merge_eval_result_with_artifact_delta,
    summarize_artifact_delta,
)
from codex_manager.memory.vector_store import ProjectVectorMemory
from codex_manager.pipeline.phases import (
    PHASE_LOG_FILES,
    PhaseConfig,
    PhaseResult,
    PipelineConfig,
    PipelinePhase,
    PipelineState,
)
from codex_manager.pipeline.tracker import LogTracker
from codex_manager.preflight import (
    agent_preflight_issues as shared_agent_preflight_issues,
)
from codex_manager.preflight import (
    binary_exists as shared_binary_exists,
)
from codex_manager.preflight import (
    env_secret_issue as shared_env_secret_issue,
)
from codex_manager.preflight import (
    has_claude_auth as shared_has_claude_auth,
)
from codex_manager.preflight import (
    has_codex_auth as shared_has_codex_auth,
)
from codex_manager.preflight import (
    image_provider_auth_issue as shared_image_provider_auth_issue,
)
from codex_manager.preflight import (
    repo_worktree_counts as shared_repo_worktree_counts,
)
from codex_manager.prompt_logging import (
    format_prompt_log_line,
    format_prompt_preview,
    prompt_metadata,
)
from codex_manager.prompts.catalog import PromptCatalog, get_catalog
from codex_manager.research import DeepResearchSettings, run_native_deep_research
from codex_manager.schemas import EvalResult

logger = logging.getLogger(__name__)

_MUTATING_PHASES = {
    PipelinePhase.IMPLEMENTATION,
    PipelinePhase.DEBUGGING,
}
_GITHUB_API_TIMEOUT_SECONDS = 20
_GITHUB_PAT_SERVICE = "warpfoundry.github_auth"
_GITHUB_PAT_SERVICE_LEGACY = "codex_manager.github_auth"
_GITHUB_PAT_KEY = "pat"
_HISTORY_ERROR_CONTEXT_CHARS = 2000
_PIPELINE_MANAGED_ARTIFACT_GLOBS: tuple[str, ...] = (
    ".codex_manager/logs/WISHLIST.md",
    ".codex_manager/logs/TESTPLAN.md",
    ".codex_manager/logs/EXPERIMENTS.md",
    ".codex_manager/logs/RESEARCH.md",
    ".codex_manager/logs/SCIENTIST_REPORT.md",
)
_PIPELINE_LOG_QUEUE_MAX = 10_000
_PIPELINE_LOG_DROP_WARN_INTERVAL = 100


def _clip_text(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    text = str(value or "")
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


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
        *,
        resume_cycle: int = 1,
        resume_phase_index: int = 0,
        log_queue_maxsize: int = _PIPELINE_LOG_QUEUE_MAX,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.config = config or PipelineConfig()
        self.catalog = catalog or get_catalog()
        self.tracker = LogTracker(self.repo_path)
        self.ledger = KnowledgeLedger(self.repo_path)
        self.vector_memory = ProjectVectorMemory(
            self.repo_path,
            enabled=bool(getattr(self.config, "vector_memory_enabled", False)),
            backend=str(getattr(self.config, "vector_memory_backend", "chroma") or "chroma"),
            collection_name=str(getattr(self.config, "vector_memory_collection", "") or ""),
            default_top_k=int(getattr(self.config, "vector_memory_top_k", 8) or 8),
        )

        self.state = PipelineState()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused initially
        self._thread: threading.Thread | None = None
        self._log_queue_maxsize = max(100, int(log_queue_maxsize))
        self.log_queue: queue.Queue[dict] = queue.Queue(maxsize=self._log_queue_maxsize)
        self._next_log_event_id = 0
        self._log_queue_drops = 0
        self._log_callback = log_callback
        self._science_experiment_by_hypothesis: dict[str, str] = {}
        self._science_trials_payloads: list[dict[str, Any]] = []
        self._science_latest_analysis_text: str = ""
        self._science_action_items: list[str] = []
        self._brain_logbook: BrainLogbook | None = None
        self._history_logbook: HistoryLogbook | None = None
        self._pr_aware_state: dict[str, Any] = {}
        self._recovered_backlog_hashes: set[str] = set()
        self._missing_test_policy_warning_keys: set[str] = set()
        self._resume_cycle = max(1, int(resume_cycle))
        self._resume_phase_index = max(0, int(resume_phase_index))

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

        run_id = f"pipe_{uuid.uuid4().hex[:12]}"
        self.state = PipelineState(
            running=True,
            run_id=run_id,
            started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            resume_cycle=self._resume_cycle,
            resume_phase_index=self._resume_phase_index,
            pr_aware={},
        )
        self._pr_aware_state = {}
        self._recovered_backlog_hashes = set()
        self._missing_test_policy_warning_keys = set()
        self._next_log_event_id = 0
        self._log_queue_drops = 0
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
        run_id = f"pipe_{uuid.uuid4().hex[:12]}"
        self.state = PipelineState(
            running=True,
            run_id=run_id,
            started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            resume_cycle=self._resume_cycle,
            resume_phase_index=self._resume_phase_index,
            pr_aware={},
        )
        self._pr_aware_state = {}
        self._recovered_backlog_hashes = set()
        self._missing_test_policy_warning_keys = set()
        self._next_log_event_id = 0
        self._log_queue_drops = 0
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
            with suppress(queue.Empty):
                self.log_queue.get_nowait()
            self.log_queue.put_nowait(entry)
            self._log_queue_drops += 1
            if self._log_queue_drops == 1 or (
                self._log_queue_drops % _PIPELINE_LOG_DROP_WARN_INTERVAL == 0
            ):
                logger.warning(
                    "Pipeline log queue overflow: dropped %s oldest log event(s) (maxsize=%s).",
                    self._log_queue_drops,
                    self._log_queue_maxsize,
                )
        if self._log_callback:
            self._log_callback(level, message)
        getattr(logger, level if level != "warn" else "warning", logger.info)(
            "[pipeline] %s", message
        )
        self.state.last_log_epoch_ms = log_epoch_ms
        self.state.last_log_level = level
        self.state.last_log_message = message[:500]

    def get_log_events_since(
        self,
        after_id: int,
        *,
        limit: int = 500,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return non-destructive log replay entries newer than ``after_id``.

        Returns ``(events, replay_gap_detected)`` where replay_gap_detected is
        true when the in-memory queue already dropped entries newer than the
        caller's cursor.
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

    def _append_pipeline_debug_event(self, payload: dict[str, Any]) -> None:
        """Append one structured debug event for pipeline phase troubleshooting."""
        try:
            path = self.tracker.path_for("PIPELINE_DEBUG.jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Could not append pipeline debug event: %s", exc)

    def _derive_deep_research_topic(self, cycle: int) -> str:
        """Derive a compact research topic from WISHLIST context."""
        wishlist = self.tracker.read("WISHLIST.md")
        matches = re.findall(r"^### \[(WISH-[0-9]{3,})\]\s*(.+)$", wishlist, flags=re.MULTILINE)
        if matches:
            top = "; ".join(f"{wish_id}: {title.strip()}" for wish_id, title in matches[:3])
            return f"Cycle {cycle} priorities - {top}"

        title_matches = re.findall(r"^\s*[-*]\s+(.+)$", wishlist, flags=re.MULTILINE)
        if title_matches:
            top = "; ".join(item.strip() for item in title_matches[:3])
            return f"Cycle {cycle} priorities - {top}"

        return f"Cycle {cycle} repository improvement opportunities for {self.repo_path.name}"

    def _append_deep_research_log_entry(
        self,
        *,
        cycle: int,
        iteration: int,
        topic: str,
        providers: str,
        summary: str,
        reused_cache: bool,
    ) -> None:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        compact = re.sub(r"\s+", " ", (summary or "").strip())
        if len(compact) > 1800:
            compact = compact[:1797].rstrip() + "..."
        block = (
            "\n## Deep Research Entry\n\n"
            f"- Timestamp: {now}\n"
            f"- Cycle: {cycle}\n"
            f"- Iteration: {iteration}\n"
            f"- Providers: {providers}\n"
            f"- Cache Reuse: {'yes' if reused_cache else 'no'}\n"
            f"- Topic: {topic}\n\n"
            "### Summary\n\n"
            f"{compact}\n"
        )
        self.tracker.append("RESEARCH.md", block)

    def _build_native_deep_research_context(self, *, phase_context: str) -> str:
        """Build compact repository context passed to native research providers."""
        parts: list[str] = [
            "## Repository Scope",
            "",
            f"- Repository root: {self.repo_path}",
            "- Use this repository as the single source of truth.",
            "- Treat WISH-* references as entries in WISHLIST.md for this repository.",
            "",
        ]
        wishlist = self.tracker.read("WISHLIST.md").strip()
        if wishlist:
            excerpt = wishlist[:3000]
            if len(wishlist) > 3000:
                excerpt += "\n...[truncated]..."
            parts.append("## WISHLIST Snapshot\n")
            parts.append(excerpt)
        progress = self.tracker.read("PROGRESS.md").strip()
        if progress:
            excerpt = progress[:2000]
            if len(progress) > 2000:
                excerpt += "\n...[truncated]..."
            parts.append("\n## PROGRESS Snapshot\n")
            parts.append(excerpt)
        if phase_context:
            excerpt = phase_context[:2500]
            if len(phase_context) > 2500:
                excerpt += "\n...[truncated]..."
            parts.append("\n## Phase Context\n")
            parts.append(excerpt)
        return "\n".join(parts).strip()

    def _execute_native_deep_research(
        self,
        *,
        topic: str,
        cycle: int,
        iteration: int,
        phase_context: str,
    ) -> PhaseResult:
        """Execute provider-native deep research and map it into a PhaseResult."""
        started = time.monotonic()
        settings = DeepResearchSettings(
            providers=self.config.deep_research_providers,
            retry_attempts=self.config.deep_research_retry_attempts,
            daily_quota=self.config.deep_research_daily_quota,
            max_provider_tokens=self.config.deep_research_max_provider_tokens,
            daily_budget_usd=self.config.deep_research_budget_usd,
            openai_model=self.config.deep_research_openai_model,
            google_model=self.config.deep_research_google_model,
        )
        context = self._build_native_deep_research_context(phase_context=phase_context)
        native = run_native_deep_research(
            repo_path=self.repo_path,
            topic=topic,
            project_context=context,
            settings=settings,
        )
        provider_prompt_previews: dict[str, str] = {}
        provider_prompt_metadata: dict[str, dict[str, int | str]] = {}
        prompt_bundle_parts: list[str] = []
        for provider_name, provider_prompt in sorted(
            (native.provider_prompt_previews or {}).items()
        ):
            clean_provider = str(provider_name or "").strip().lower()
            if not clean_provider:
                continue
            prompt_text = str(provider_prompt or "")
            provider_prompt_previews[clean_provider] = format_prompt_preview(prompt_text)
            provider_prompt_metadata[clean_provider] = prompt_metadata(prompt_text)
            prompt_bundle_parts.append(f"## Provider: {clean_provider}\n{prompt_text}")

        prompt_bundle = "\n\n".join(prompt_bundle_parts).strip()
        if not prompt_bundle:
            prompt_bundle = (
                "Native deep research prompt previews unavailable from provider call.\n\n"
                f"Topic: {topic}\n\n"
                "Project context:\n"
                f"{context}"
            )
        native_prompt_meta = prompt_metadata(prompt_bundle)
        provider_results_payload: list[dict[str, Any]] = []
        for item in native.providers:
            provider_results_payload.append(
                {
                    "provider": item.provider,
                    "ok": bool(item.ok),
                    "input_tokens": int(item.input_tokens),
                    "output_tokens": int(item.output_tokens),
                    "estimated_cost_usd": float(item.estimated_cost_usd),
                    "sources_count": len(item.sources),
                    "sources_preview": list(item.sources[:20]),
                    "error": _clip_text(item.error, 4000),
                    "summary_excerpt": _clip_text(item.summary, 1200),
                }
            )
        self._append_pipeline_debug_event(
            {
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "cycle": cycle,
                "phase": PipelinePhase.DEEP_RESEARCH.value,
                "iteration": iteration,
                "mode": self.config.mode,
                "runner": f"deep_research:{self.config.deep_research_providers}",
                "native_deep_research": {
                    "topic": topic,
                    "topic_metadata": prompt_metadata(topic),
                    "project_context_preview": format_prompt_preview(context),
                    "project_context_metadata": prompt_metadata(context),
                    "provider_prompt_previews": provider_prompt_previews,
                    "provider_prompt_metadata": provider_prompt_metadata,
                    "settings": {
                        "providers": settings.providers,
                        "retry_attempts": settings.retry_attempts,
                        "daily_quota": settings.daily_quota,
                        "max_provider_tokens": settings.max_provider_tokens,
                        "daily_budget_usd": settings.daily_budget_usd,
                        "timeout_seconds": settings.timeout_seconds,
                        "openai_model": settings.openai_model,
                        "google_model": settings.google_model,
                    },
                    "result": {
                        "ok": bool(native.ok),
                        "quota_blocked": bool(native.quota_blocked),
                        "budget_blocked": bool(native.budget_blocked),
                        "governance_warnings": list(native.governance_warnings[:20]),
                        "filtered_source_count": int(native.filtered_source_count),
                        "merged_sources_count": len(native.merged_sources),
                        "merged_sources_preview": list(native.merged_sources[:30]),
                        "total_input_tokens": int(native.total_input_tokens),
                        "total_output_tokens": int(native.total_output_tokens),
                        "total_estimated_cost_usd": float(native.total_estimated_cost_usd),
                        "error": _clip_text(native.error, 4000),
                        "providers": provider_results_payload,
                    },
                },
                "prompt_length": native_prompt_meta["length_chars"],
                "prompt_sha256": native_prompt_meta["sha256"],
                "prompt_redaction_hits": native_prompt_meta["redaction_hits"],
                "prompt_preview": format_prompt_preview(prompt_bundle),
                "prompt_metadata": native_prompt_meta,
            }
        )
        if native.ok:
            provider_summaries = [
                f"{item.provider}: in={item.input_tokens:,}, out={item.output_tokens:,}, cost~${item.estimated_cost_usd:.4f}"
                for item in native.providers
                if item.ok
            ]
            if native.filtered_source_count:
                provider_summaries.append(f"sources_filtered={native.filtered_source_count}")
            summary = native.merged_summary.strip()
            if not summary:
                summary = "Native deep research completed with empty summary."
            test_summary = (
                "Native deep research complete. "
                + " | ".join(provider_summaries)
                + f" | total cost~${native.total_estimated_cost_usd:.4f}"
            )
            if native.governance_warnings:
                test_summary += " | governance warnings present"
            agent_message = summary
            if native.merged_sources:
                agent_message = (
                    f"{summary}\n\nSources:\n"
                    + "\n".join(f"- {source}" for source in native.merged_sources[:30])
                )
            if native.governance_warnings:
                agent_message += (
                    "\n\nSource Governance Warnings:\n"
                    + "\n".join(f"- {warning}" for warning in native.governance_warnings[:10])
                )
            return PhaseResult(
                cycle=cycle,
                phase=PipelinePhase.DEEP_RESEARCH.value,
                iteration=iteration,
                agent_success=True,
                validation_success=True,
                tests_passed=True,
                success=True,
                test_outcome="passed",
                test_summary=test_summary,
                test_exit_code=0,
                files_changed=0,
                net_lines_changed=0,
                changed_files=[],
                error_message="",
                duration_seconds=round(time.monotonic() - started, 1),
                input_tokens=native.total_input_tokens,
                output_tokens=native.total_output_tokens,
                prompt_used=f"Native deep research topic: {topic}",
                agent_final_message=agent_message,
                terminate_repeats=True,
                agent_used=f"deep_research:{self.config.deep_research_providers}",
            )

        provider_errors = [
            f"{item.provider}: {item.error}"
            for item in native.providers
            if (not item.ok) and item.error
        ]
        if native.error:
            provider_errors.insert(0, native.error)
        if native.quota_blocked:
            provider_errors.insert(0, "daily quota reached")
        if native.budget_blocked:
            provider_errors.insert(0, "daily budget reached")
        message = "; ".join(err for err in provider_errors if err) or "native deep research failed"
        return PhaseResult(
            cycle=cycle,
            phase=PipelinePhase.DEEP_RESEARCH.value,
            iteration=iteration,
            agent_success=False,
            validation_success=False,
            tests_passed=False,
            success=False,
            test_outcome="error",
            test_summary="Native deep research failed",
            test_exit_code=1,
            files_changed=0,
            net_lines_changed=0,
            changed_files=[],
            error_message=message,
            duration_seconds=round(time.monotonic() - started, 1),
            input_tokens=native.total_input_tokens,
            output_tokens=native.total_output_tokens,
            prompt_used=f"Native deep research topic: {topic}",
            agent_final_message="",
            terminate_repeats=False,
            agent_used=f"deep_research:{self.config.deep_research_providers}",
        )

    def _build_vector_memory_context(
        self,
        *,
        phase: PipelinePhase,
        base_prompt: str,
        log_context: str,
    ) -> str:
        if not self.vector_memory.enabled:
            return ""
        query = "\n".join(
            [
                f"phase: {phase.value}",
                base_prompt[:500],
                log_context[:900],
            ]
        ).strip()
        if not query:
            return ""
        categories: list[str] | None = None
        source_prefix = ""
        if phase in (
            PipelinePhase.THEORIZE,
            PipelinePhase.EXPERIMENT,
            PipelinePhase.SKEPTIC,
            PipelinePhase.ANALYZE,
        ):
            categories = [
                "scientist_trial",
                "scientist_report",
                "pipeline_phase",
                "deep_research",
                "recovered_backlog",
            ]
        elif phase == PipelinePhase.DEEP_RESEARCH:
            categories = [
                "deep_research",
                "pipeline_phase",
                "scientist_report",
                "recovered_backlog",
            ]
            source_prefix = "pipeline:"
        hits = self.vector_memory.search(
            query,
            top_k=int(getattr(self.config, "vector_memory_top_k", 8) or 8),
            categories=categories,
            source_prefix=source_prefix,
        )
        if not hits:
            return ""

        lines = ["## Similar Past Context (Vector Memory)", ""]
        for hit in hits:
            meta = hit.metadata or {}
            category = str(meta.get("category") or "note").strip() or "note"
            source = str(meta.get("source") or "").strip()
            snippet = re.sub(r"\s+", " ", hit.document).strip()
            if len(snippet) > 280:
                snippet = snippet[:277].rstrip() + "..."
            prefix = f"- [{category}] score={hit.score:.2f}"
            if source:
                prefix += f" source={source}"
            lines.append(f"{prefix}: {snippet}")
        return "\n".join(lines)

    def _record_vector_memory_note(
        self,
        *,
        phase: PipelinePhase,
        cycle: int,
        iteration: int,
        result: PhaseResult,
    ) -> None:
        if not self.vector_memory.enabled:
            return
        summary = (
            f"Phase {phase.value} cycle={cycle} iteration={iteration} "
            f"success={result.success} tests={result.test_outcome} "
            f"files={result.files_changed} lines={result.net_lines_changed:+d}. "
            f"Result: {(result.test_summary or result.error_message or result.agent_final_message or '').strip()}"
        )
        self.vector_memory.add_note(
            summary[:3500],
            category="pipeline_phase",
            source=f"pipeline:{phase.value}",
            metadata={
                "phase": phase.value,
                "cycle": cycle,
                "iteration": iteration,
                "success": bool(result.success),
                "test_outcome": result.test_outcome,
                "files_changed": int(result.files_changed),
                "net_lines_changed": int(result.net_lines_changed),
            },
        )

    def _record_science_trial_memory(self, payload: dict[str, Any]) -> None:
        """Index structured scientist trial artifacts into vector memory."""
        if not self.vector_memory.enabled:
            return
        hypothesis = payload.get("hypothesis") or {}
        post = payload.get("post") or {}
        verdict = str(payload.get("verdict") or "").strip()
        rationale = str(payload.get("verdict_rationale") or "").strip()
        summary = (
            "Scientist trial "
            f"id={payload.get('trial_id', '')} "
            f"phase={payload.get('phase', '')} "
            f"cycle={payload.get('cycle', '')} "
            f"verdict={verdict} "
            f"confidence={payload.get('confidence', '')}. "
            f"Hypothesis: {hypothesis.get('id', '')} {hypothesis.get('title', '')}. "
            f"Post-tests={post.get('test_outcome', '')}, files={post.get('files_changed', 0)}, "
            f"lines={post.get('net_lines_changed', 0)}. "
            f"Rationale: {rationale}"
        )
        self.vector_memory.add_note(
            summary[:3500],
            category="scientist_trial",
            source="pipeline:scientist",
            metadata={
                "phase": str(payload.get("phase", "")),
                "cycle": int(payload.get("cycle") or 0),
                "iteration": int(payload.get("iteration") or 0),
                "trial_id": str(payload.get("trial_id") or ""),
                "verdict": verdict,
                "confidence": str(payload.get("confidence") or ""),
            },
        )

    def _record_scientist_report_memory(self, report: str) -> None:
        """Index scientist report snapshots for cross-surface retrieval."""
        if not self.vector_memory.enabled:
            return
        content = str(report or "").strip()
        if not content:
            return
        compact = re.sub(r"\s+", " ", content)
        if len(compact) > 4000:
            compact = compact[:3997].rstrip() + "..."
        self.vector_memory.add_note(
            compact,
            category="scientist_report",
            source="pipeline:scientist_report",
            metadata={
                "cycle": int(self.state.current_cycle or 0),
                "phase": str(self.state.current_phase or ""),
            },
        )

    @staticmethod
    def _count_markdown_bullets(text: str) -> int:
        return sum(1 for line in str(text or "").splitlines() if line.startswith("- "))

    @staticmethod
    def _extract_recovered_backlog_section(context: str) -> str:
        marker = "## Recovered Pending Backlog (Cross-Run)"
        idx = str(context or "").find(marker)
        if idx < 0:
            return ""
        return str(context or "")[idx:].strip()

    def _record_recovered_backlog_memory(
        self,
        *,
        phase: PipelinePhase,
        cycle: int,
        context: str,
    ) -> None:
        if not self.vector_memory.enabled:
            return
        section = self._extract_recovered_backlog_section(context)
        if not section:
            return
        compact = re.sub(r"\s+", " ", section).strip()
        if not compact:
            return
        if len(compact) > 3900:
            compact = compact[:3897].rstrip() + "..."
        signature = hashlib.sha1(compact.encode("utf-8")).hexdigest()
        if signature in self._recovered_backlog_hashes:
            return
        self._recovered_backlog_hashes.add(signature)
        self.vector_memory.add_note(
            compact,
            category="recovered_backlog",
            source=f"pipeline:{phase.value}",
            metadata={
                "phase": phase.value,
                "cycle": int(cycle),
                "memory_kind": "recovered_backlog",
                "item_count": self._count_markdown_bullets(section),
            },
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
        if self.config.mode == "apply" and self.config.pr_aware_enabled:
            try:
                self._pr_aware_maybe_sync(
                    repo=self.repo_path,
                    reason="run_finalized",
                    force_description=True,
                )
            except Exception as exc:
                self._log("warn", f"PR-aware final sync failed: {exc}")
        history_context = {
            "run_id": self.state.run_id,
            "stop_reason": self.state.stop_reason,
            "total_cycles_completed": self.state.total_cycles_completed,
            "total_phases_completed": self.state.total_phases_completed,
            "total_tokens": self.state.total_tokens,
            "elapsed_seconds": round(self.state.elapsed_seconds, 1),
            "restart_required": bool(self.state.restart_required),
            "restart_checkpoint_path": self.state.restart_checkpoint_path,
            "resume_cycle": self.state.resume_cycle,
            "resume_phase_index": self.state.resume_phase_index,
            "pr_aware": self._pr_aware_snapshot(),
        }
        if extra_history_context:
            history_context.update(extra_history_context)
        self._record_history_note(
            "run_finished",
            f"Pipeline finished with stop_reason='{self.state.stop_reason}'.",
            level=history_level,
            context=history_context,
        )
        self._send_run_completion_webhooks()

    @staticmethod
    def _completion_webhook_kind(url: str) -> str:
        parsed = urlparse(url)
        host = str(parsed.hostname or "").strip().lower()
        path = str(parsed.path or "").strip().lower()
        if host.endswith("slack.com") and "/services/" in path:
            return "slack"
        if host.endswith("discord.com") and "/api/webhooks/" in path:
            return "discord"
        if host.endswith("discordapp.com") and "/api/webhooks/" in path:
            return "discord"
        return "generic"

    @staticmethod
    def _completion_webhook_url_valid(url: str) -> bool:
        parsed = urlparse(url)
        scheme = str(parsed.scheme or "").strip().lower()
        host = str(parsed.netloc or "").strip()
        return bool(host) and scheme in {"http", "https"}

    @staticmethod
    def _completion_outcome_counts(results: list[PhaseResult]) -> dict[str, int]:
        counts = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "error": 0,
            "unknown": 0,
        }
        for item in results:
            key = str(item.test_outcome or "").strip().lower()
            if key not in counts:
                key = "unknown"
            counts[key] += 1
        return counts

    @staticmethod
    def _completion_status(state: PipelineState) -> str:
        stop_reason = str(state.stop_reason or "").strip().lower()
        hard_failure_reasons = {
            "preflight_failed",
            "phase_failed_abort",
            "pr_aware_setup_failed",
            "branch_creation_failed",
            "budget_exhausted",
            "user_stopped",
            "max_time_reached",
        }
        if stop_reason.startswith("error:"):
            return "failure"
        if stop_reason in hard_failure_reasons:
            return "failure"
        if int(state.failures or 0) > 0:
            return "failure"
        if int(state.total_phases_completed or 0) <= 0:
            return "failure"
        return "success"

    def _completion_artifact_links(self) -> dict[str, str]:
        logs_dir = (self.repo_path / ".codex_manager" / "logs").resolve()
        outputs_dir = (self.repo_path / ".codex_manager" / "outputs").resolve()
        links: dict[str, str] = {
            "logs_dir": str(logs_dir),
            "outputs_dir": str(outputs_dir),
            "history_markdown": str((logs_dir / "HISTORY.md").resolve()),
            "history_jsonl": str((logs_dir / "HISTORY.jsonl").resolve()),
            "scientist_report": str((logs_dir / "SCIENTIST_REPORT.md").resolve()),
        }
        restart_checkpoint = str(self.state.restart_checkpoint_path or "").strip()
        if restart_checkpoint:
            links["restart_checkpoint"] = str(Path(restart_checkpoint).resolve())
        return links

    def _run_completion_payload(self) -> dict[str, Any]:
        tests = self._completion_outcome_counts(self.state.results)
        input_tokens = sum(max(0, int(item.input_tokens or 0)) for item in self.state.results)
        output_tokens = sum(max(0, int(item.output_tokens or 0)) for item in self.state.results)
        total_tokens = max(0, int(self.state.total_tokens or 0))
        if total_tokens <= 0:
            total_tokens = input_tokens + output_tokens
        run_id = str(self.state.run_id or "").strip()
        if not run_id:
            run_id = f"pipe_{uuid.uuid4().hex[:12]}"
            self.state.run_id = run_id
        status = self._completion_status(self.state)
        top_results = [
            {
                "cycle": int(item.cycle or 0),
                "phase": str(item.phase or ""),
                "iteration": int(item.iteration or 0),
                "success": bool(item.success),
                "test_outcome": str(item.test_outcome or ""),
                "files_changed": int(item.files_changed or 0),
                "net_lines_changed": int(item.net_lines_changed or 0),
                "commit_sha": str(item.commit_sha or ""),
            }
            for item in self.state.results[-20:]
        ]
        return {
            "event": "warpfoundry.pipeline.run.completed",
            "scope": "pipeline",
            "status": status,
            "success": status == "success",
            "repo_path": str(self.repo_path),
            "run_id": run_id,
            "mode": str(self.config.mode or ""),
            "stop_reason": str(self.state.stop_reason or ""),
            "started_at": str(self.state.started_at or ""),
            "finished_at": str(self.state.finished_at or ""),
            "elapsed_seconds": round(float(self.state.elapsed_seconds or 0.0), 1),
            "cycles_completed": int(self.state.total_cycles_completed or 0),
            "phases_completed": int(self.state.total_phases_completed or 0),
            "successes": int(self.state.successes or 0),
            "failures": int(self.state.failures or 0),
            "tests": tests,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": total_tokens,
            },
            "artifact_links": self._completion_artifact_links(),
            "phase_results": top_results,
        }

    @staticmethod
    def _slack_completion_payload(payload: dict[str, Any]) -> dict[str, Any]:
        tests = payload.get("tests") if isinstance(payload.get("tests"), dict) else {}
        tokens = payload.get("tokens") if isinstance(payload.get("tokens"), dict) else {}
        artifacts = (
            payload.get("artifact_links") if isinstance(payload.get("artifact_links"), dict) else {}
        )
        lines = [
            f"*WarpFoundry Pipeline {str(payload.get('status', 'unknown')).upper()}*",
            f"Repo: `{payload.get('repo_path', '')}`",
            f"Run ID: `{payload.get('run_id', '')}`",
            f"Stop reason: `{payload.get('stop_reason', '')}`",
            (
                "Tests: "
                f"passed={tests.get('passed', 0)}, "
                f"failed={tests.get('failed', 0)}, "
                f"error={tests.get('error', 0)}, "
                f"skipped={tests.get('skipped', 0)}, "
                f"unknown={tests.get('unknown', 0)}"
            ),
            f"Tokens: {int(tokens.get('total', 0) or 0):,}",
            (
                "Artifacts: "
                f"HISTORY={artifacts.get('history_jsonl', '')}, "
                f"outputs={artifacts.get('outputs_dir', '')}"
            ),
        ]
        return {"text": "\n".join(lines)}

    @staticmethod
    def _discord_completion_payload(payload: dict[str, Any]) -> dict[str, Any]:
        tests = payload.get("tests") if isinstance(payload.get("tests"), dict) else {}
        tokens = payload.get("tokens") if isinstance(payload.get("tokens"), dict) else {}
        artifacts = (
            payload.get("artifact_links") if isinstance(payload.get("artifact_links"), dict) else {}
        )
        lines = [
            f"**WarpFoundry Pipeline {str(payload.get('status', 'unknown')).upper()}**",
            f"Repo: `{payload.get('repo_path', '')}`",
            f"Run ID: `{payload.get('run_id', '')}`",
            f"Stop reason: `{payload.get('stop_reason', '')}`",
            (
                "Tests: "
                f"passed={tests.get('passed', 0)}, "
                f"failed={tests.get('failed', 0)}, "
                f"error={tests.get('error', 0)}, "
                f"skipped={tests.get('skipped', 0)}, "
                f"unknown={tests.get('unknown', 0)}"
            ),
            f"Tokens: {int(tokens.get('total', 0) or 0):,}",
            (
                "Artifacts: "
                f"HISTORY={artifacts.get('history_jsonl', '')}, "
                f"outputs={artifacts.get('outputs_dir', '')}"
            ),
        ]
        content = "\n".join(lines)
        if len(content) > 1900:
            content = content[:1897].rstrip() + "..."
        return {"content": content}

    @staticmethod
    def _post_completion_webhook_json(
        *,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: int,
    ) -> str:
        data = json.dumps(payload).encode("utf-8")
        request_obj = Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": "WarpFoundry/1.0",
            },
        )
        try:
            with urlopen(request_obj, timeout=timeout_seconds) as response:
                _ = response.read()
        except HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            detail = body.strip()
            if detail:
                detail = re.sub(r"\s+", " ", detail)[:280]
                return f"HTTP {exc.code}: {detail}"
            return f"HTTP {exc.code}"
        except URLError as exc:
            reason = str(getattr(exc, "reason", exc) or "").strip()
            return f"network error: {reason or exc}"
        except Exception as exc:  # pragma: no cover - defensive
            return f"request failed: {exc}"
        return ""

    def _send_run_completion_webhooks(self) -> None:
        raw_urls = list(getattr(self.config, "run_completion_webhooks", []) or [])
        urls = [str(item or "").strip() for item in raw_urls if str(item or "").strip()]
        if not urls:
            return

        timeout_seconds = max(
            2,
            min(60, int(getattr(self.config, "run_completion_webhook_timeout_seconds", 10) or 10)),
        )
        payload = self._run_completion_payload()
        delivered = 0

        for url in urls:
            if not self._completion_webhook_url_valid(url):
                self._log("warn", f"Skipping invalid run-completion webhook URL: {url}")
                continue
            kind = self._completion_webhook_kind(url)
            body: dict[str, Any]
            if kind == "slack":
                body = self._slack_completion_payload(payload)
            elif kind == "discord":
                body = self._discord_completion_payload(payload)
            else:
                body = payload
            error = self._post_completion_webhook_json(
                url=url,
                payload=body,
                timeout_seconds=timeout_seconds,
            )
            if error:
                self._log("warn", f"Run completion webhook delivery failed ({kind}): {error}")
                continue
            delivered += 1

        self._log(
            "info",
            f"Run completion webhooks delivered: {delivered}/{len(urls)}",
        )

    @staticmethod
    def _retention_policy(config: PipelineConfig) -> RetentionPolicy:
        """Build runtime-artifact retention policy from pipeline configuration."""
        return RetentionPolicy(
            enabled=bool(getattr(config, "artifact_retention_enabled", True)),
            max_age_days=max(1, int(getattr(config, "artifact_retention_max_age_days", 30))),
            max_files=max(1, int(getattr(config, "artifact_retention_max_files", 5000))),
            max_bytes=max(1, int(getattr(config, "artifact_retention_max_bytes", 2_000_000_000))),
            max_output_history_runs=max(
                1, int(getattr(config, "artifact_retention_max_output_runs", 30))
            ),
        )

    def _run_retention_cleanup(self, *, reason: str) -> None:
        """Apply retention cleanup for managed runtime artifacts."""
        try:
            cleanup = cleanup_runtime_artifacts(
                self.repo_path,
                policy=self._retention_policy(self.config),
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
        self._log(
            "info",
            (
                f"Retention cleanup ({reason}): removed "
                f"{cleanup['removed_files']} files, {cleanup['removed_dirs']} dirs, "
                f"{cleanup['removed_runs']} archived runs; "
                f"freed {int(cleanup.get('freed_bytes', 0))} bytes."
            ),
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
        self._science_trials_payloads.append(payload)
        if phase == PipelinePhase.ANALYZE:
            self._science_latest_analysis_text = output_text
            action_items = self._extract_science_action_items(output_text)
            if action_items:
                self._science_action_items = action_items
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
        self._record_science_trial_memory(payload)

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
    def _scientist_table_value(value: Any) -> str:
        text = str(value if value is not None else "").strip()
        if not text:
            return "-"
        text = re.sub(r"\s+", " ", text)
        return text.replace("|", r"\|")

    @staticmethod
    def _extract_science_action_items(text: str, *, limit: int = 10) -> list[str]:
        raw = (text or "").strip()
        if not raw:
            return []

        lines = raw.splitlines()
        items: list[str] = []
        seen: set[str] = set()
        in_action_section = False

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            heading_text = re.sub(r"^#{1,6}\s*", "", line).strip().lower()
            if re.search(
                r"\b(recommendations?|action plan|implementation handoff|checklist|next steps|todo|to-do)\b",
                heading_text,
            ):
                in_action_section = True
                continue

            if in_action_section and re.match(r"^#{1,6}\s+", line):
                in_action_section = False
                continue

            bullet_match = re.match(r"^\s*(?:[-*]|\d+[.)])\s+(.+)$", raw_line)
            if not bullet_match:
                continue
            if not in_action_section and items:
                continue

            candidate = re.sub(r"\s+", " ", bullet_match.group(1)).strip()
            if len(candidate) < 8:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(candidate)
            if len(items) >= limit:
                break

        return items

    def _build_scientist_report(self) -> str:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        trials = list(self._science_trials_payloads)
        hypotheses = {
            str((trial.get("hypothesis") or {}).get("id", "")).strip()
            for trial in trials
            if str((trial.get("hypothesis") or {}).get("id", "")).strip()
        }
        verdict_counts = {"supported": 0, "refuted": 0, "inconclusive": 0}
        rollback_count = 0
        trial_tokens = 0
        for trial in trials:
            verdict = str(trial.get("verdict", "")).strip().lower()
            if verdict in verdict_counts:
                verdict_counts[verdict] += 1
            if str(trial.get("rollback_action", "")).strip().lower() == "reverted":
                rollback_count += 1
            usage = trial.get("usage") or {}
            with suppress(Exception):
                trial_tokens += int(usage.get("total_tokens", 0) or 0)

        action_items = list(self._science_action_items)
        if not action_items and self._science_latest_analysis_text:
            action_items = self._extract_science_action_items(self._science_latest_analysis_text)

        implementation_results = [
            result
            for result in self.state.results
            if result.phase in {"implementation", "debugging", "commit"}
        ]
        changed_file_counts: dict[str, int] = {}
        for result in implementation_results:
            for entry in result.changed_files:
                path = str(entry.get("path", "")).strip()
                if path:
                    changed_file_counts[path] = changed_file_counts.get(path, 0) + 1

        lines: list[str] = [
            "# Scientist Mode Report",
            "",
            "> Auto-generated evidence and action dashboard for Scientist Mode.",
            "",
            "## Dashboard",
            f"- **Updated**: {now}",
            f"- **Current Cycle**: {self.state.current_cycle}",
            f"- **Science Trials Recorded**: {len(trials)}",
            f"- **Hypotheses Tracked**: {len(hypotheses)}",
            (
                "- **Verdicts**: "
                f"supported={verdict_counts['supported']}, "
                f"refuted={verdict_counts['refuted']}, "
                f"inconclusive={verdict_counts['inconclusive']}"
            ),
            f"- **Trial Tokens (total)**: {trial_tokens:,}",
            f"- **Science Rollbacks Applied**: {rollback_count}",
            f"- **Implementation Phases Executed**: {len(implementation_results)}",
            "",
            "## Action Plan (Implementation TODO)",
        ]

        if action_items:
            for item in action_items:
                lines.append(f"- [ ] {item}")
        elif trials:
            lines.append(
                "- [ ] No explicit checklist found in analyze output yet. Run analyze with a checklist section."
            )
        else:
            lines.append("- [ ] Run theorize/experiment/skeptic/analyze to generate evidence-backed TODOs.")

        lines.extend(
            [
                "",
                "## Science Trial Timeline",
            ]
        )
        if trials:
            lines.extend(
                [
                    "| Cycle | Phase | Hypothesis | Verdict | Confidence | Tests (baseline->post) | Repo Delta (files/lines) | Rollback |",
                    "|---:|---|---|---|---|---|---:|---|",
                ]
            )
            for trial in trials[-25:]:
                baseline = trial.get("baseline") or {}
                post = trial.get("post") or {}
                hypothesis = trial.get("hypothesis") or {}
                baseline_test = str(baseline.get("test_outcome", "n/a")).strip() or "n/a"
                post_test = str(post.get("test_outcome", "n/a")).strip() or "n/a"
                files_changed = post.get("files_changed", 0)
                net_lines = post.get("net_lines_changed", 0)
                try:
                    files_changed = int(files_changed)
                except Exception:
                    files_changed = 0
                try:
                    net_lines = int(net_lines)
                except Exception:
                    net_lines = 0
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            self._scientist_table_value(trial.get("cycle", 0)),
                            self._scientist_table_value(trial.get("phase", "")),
                            self._scientist_table_value(hypothesis.get("id", "") or "n/a"),
                            self._scientist_table_value(trial.get("verdict", "")),
                            self._scientist_table_value(trial.get("confidence", "")),
                            self._scientist_table_value(f"{baseline_test}->{post_test}"),
                            self._scientist_table_value(f"{files_changed}/{net_lines:+d}"),
                            self._scientist_table_value(trial.get("rollback_action", "")),
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("No scientist trials recorded yet.")

        lines.extend(
            [
                "",
                "## Implementation and Code Changes",
            ]
        )
        if implementation_results:
            lines.extend(
                [
                    "| Cycle | Phase | Iter | Status | Tests | Files | Net Delta | Commit |",
                    "|---:|---|---:|---|---|---:|---:|---|",
                ]
            )
            for result in implementation_results[-30:]:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            self._scientist_table_value(result.cycle or 0),
                            self._scientist_table_value(result.phase),
                            self._scientist_table_value(result.iteration),
                            self._scientist_table_value("ok" if result.success else "failed"),
                            self._scientist_table_value(result.test_outcome),
                            self._scientist_table_value(result.files_changed),
                            self._scientist_table_value(f"{result.net_lines_changed:+d}"),
                            self._scientist_table_value(result.commit_sha or "-"),
                        ]
                    )
                    + " |"
                )
            if changed_file_counts:
                lines.extend(
                    [
                        "",
                        "### Most-Touched Files",
                        "| File | Touches |",
                        "|---|---:|",
                    ]
                )
                for path, count in sorted(
                    changed_file_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:15]:
                    lines.append(f"| {self._scientist_table_value(path)} | {count} |")
        else:
            lines.append("No implementation/debugging/commit results recorded yet.")

        if self._science_latest_analysis_text:
            analysis = self._science_latest_analysis_text.strip()
            if len(analysis) > 2000:
                analysis = analysis[:2000].rstrip() + "\n...[truncated]..."
            lines.extend(
                [
                    "",
                    "## Latest Analyze Output (Excerpt)",
                    "```text",
                    analysis,
                    "```",
                ]
            )

        return "\n".join(lines).rstrip() + "\n"

    def _refresh_scientist_report(self) -> None:
        if not bool(self.config.science_enabled):
            return
        try:
            self.tracker.initialize_science()
            report = self._build_scientist_report()
            self.tracker.write("SCIENTIST_REPORT.md", report)
            self._record_scientist_report_memory(report)
        except Exception as exc:
            self._log("warn", f"Could not refresh Scientist report: {exc}")

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

    @staticmethod
    def _run_git_process(
        repo: Path,
        *args: str,
        timeout: int = 30,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    @staticmethod
    def _git_process_error(
        result: subprocess.CompletedProcess[str],
        fallback: str,
    ) -> str:
        stderr = str(result.stderr or "").strip()
        if stderr:
            return stderr
        stdout = str(result.stdout or "").strip()
        if stdout:
            return stdout
        return fallback

    def _git_current_branch_name(self, repo: Path) -> str:
        result = self._run_git_process(repo, "rev-parse", "--abbrev-ref", "HEAD")
        if result.returncode != 0:
            detail = self._git_process_error(result, "git rev-parse --abbrev-ref HEAD failed")
            raise RuntimeError(detail)
        return str(result.stdout or "").strip() or "HEAD"

    def _git_head_sha_full(self, repo: Path) -> str:
        result = self._run_git_process(repo, "rev-parse", "HEAD")
        if result.returncode != 0:
            detail = self._git_process_error(result, "git rev-parse HEAD failed")
            raise RuntimeError(detail)
        return str(result.stdout or "").strip()

    def _git_remote_names(self, repo: Path) -> list[str]:
        result = self._run_git_process(repo, "remote")
        if result.returncode != 0:
            detail = self._git_process_error(result, "git remote failed")
            raise RuntimeError(detail)
        names: list[str] = []
        seen: set[str] = set()
        for raw_line in str(result.stdout or "").splitlines():
            name = raw_line.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names

    def _git_remote_url(self, repo: Path, remote: str) -> str:
        result = self._run_git_process(repo, "remote", "get-url", remote)
        if result.returncode != 0:
            detail = self._git_process_error(result, f"git remote get-url {remote} failed")
            raise RuntimeError(detail)
        return str(result.stdout or "").strip()

    def _git_tracking_remote(self, repo: Path) -> str:
        result = self._run_git_process(
            repo,
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        )
        if result.returncode != 0:
            return ""
        tracking = str(result.stdout or "").strip()
        if "/" not in tracking:
            return ""
        return tracking.split("/", 1)[0].strip()

    def _git_remote_default_branch(self, repo: Path, remote: str) -> str:
        result = self._run_git_process(repo, "symbolic-ref", "--short", f"refs/remotes/{remote}/HEAD")
        if result.returncode != 0:
            return ""
        remote_head = str(result.stdout or "").strip()
        prefix = f"{remote}/"
        if remote_head.startswith(prefix):
            return remote_head[len(prefix) :].strip()
        return remote_head.rsplit("/", 1)[-1].strip()

    def _git_require_valid_branch_name(self, repo: Path, branch: str) -> None:
        probe = self._run_git_process(repo, "check-ref-format", "--branch", branch)
        if probe.returncode != 0:
            detail = self._git_process_error(probe, f"Invalid branch name: {branch}")
            raise RuntimeError(detail)

    def _git_local_branch_exists(self, repo: Path, branch: str) -> bool:
        probe = self._run_git_process(repo, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}")
        return probe.returncode == 0

    @staticmethod
    def _github_owner_repo_from_remote(remote_url: str) -> tuple[str, str]:
        raw = str(remote_url or "").strip()
        if not raw:
            return "", ""

        host = ""
        path = ""
        parsed = urlparse(raw)
        if parsed.scheme and parsed.netloc:
            host = str(parsed.hostname or parsed.netloc).strip().lower()
            path = str(parsed.path or "").strip()
        elif "://" not in raw and ":" in raw:
            left, right = raw.split(":", 1)
            if "@" in left and right.strip():
                host = left.split("@", 1)[1].strip().lower()
                path = "/" + right.strip()

        if host != "github.com":
            return "", ""
        normalized = path.strip().strip("/")
        if normalized.lower().endswith(".git"):
            normalized = normalized[:-4]
        parts = [part.strip() for part in normalized.split("/") if part.strip()]
        if len(parts) < 2:
            return "", ""
        return parts[0], parts[1]

    @staticmethod
    def _github_api_error_message(exc: HTTPError) -> str:
        detail = ""
        with suppress(Exception):
            body = exc.read().decode("utf-8", errors="replace")
            parsed = json.loads(body) if body else {}
            if isinstance(parsed, dict):
                detail = str(parsed.get("message") or "").strip()
        message = f"GitHub API returned HTTP {exc.code}."
        if detail:
            message += f" {detail[:220]}"
        return message

    @staticmethod
    def _github_pat_token() -> str:
        for key in ("CODEX_MANAGER_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
            token = str(os.getenv(key) or "").strip()
            if token:
                return token
        with suppress(Exception):
            import keyring

            for service in (_GITHUB_PAT_SERVICE, _GITHUB_PAT_SERVICE_LEGACY):
                token = str(keyring.get_password(service, _GITHUB_PAT_KEY) or "").strip()
                if token:
                    return token
        return ""

    def _github_api_request(
        self,
        *,
        method: str,
        path: str,
        token: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | list[Any] | None, str]:
        url = f"https://api.github.com{path}"
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "warpfoundry-pipeline-pr-aware",
        }
        token_value = str(token or "").strip()
        if token_value:
            headers["Authorization"] = f"Bearer {token_value}"

        data: bytes | None = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")

        request_obj = Request(url, headers=headers, data=data, method=method.upper())
        try:
            with urlopen(request_obj, timeout=_GITHUB_API_TIMEOUT_SECONDS) as response:
                body_text = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            return None, self._github_api_error_message(exc)
        except URLError as exc:
            reason = str(getattr(exc, "reason", exc) or "").strip()
            if reason:
                return None, f"Could not reach GitHub API: {reason}"
            return None, "Could not reach GitHub API."
        except Exception as exc:
            return None, f"GitHub API request failed: {exc}"

        if not body_text.strip():
            return {}, ""
        try:
            parsed = json.loads(body_text)
        except json.JSONDecodeError as exc:
            return None, f"GitHub API returned invalid JSON: {exc}"
        if isinstance(parsed, (dict, list)):
            return parsed, ""
        return None, "GitHub API returned an unexpected payload shape."

    def _pr_aware_snapshot(self) -> dict[str, Any]:
        state = self._pr_aware_state
        return {
            "enabled": bool(state.get("enabled")),
            "branch": str(state.get("branch") or ""),
            "remote": str(state.get("remote") or ""),
            "base_branch": str(state.get("base_branch") or ""),
            "auto_push": bool(state.get("auto_push")),
            "sync_description": bool(state.get("sync_description")),
            "pull_number": int(state.get("pull_number") or 0),
            "pull_request_url": str(state.get("pull_request_url") or ""),
            "last_pushed_head": str(state.get("last_pushed_head") or ""),
            "last_push_succeeded_at": str(state.get("last_push_succeeded_at") or ""),
            "last_sync_succeeded_at": str(state.get("last_sync_succeeded_at") or ""),
            "last_push_error": str(state.get("last_push_error") or ""),
            "last_sync_error": str(state.get("last_sync_error") or ""),
        }

    def _refresh_pr_aware_state(self) -> None:
        if not isinstance(self.state.pr_aware, dict):
            self.state.pr_aware = {}
        self.state.pr_aware = self._pr_aware_snapshot()

    def _resolve_pr_aware_remote(self, repo: Path, requested_remote: str) -> str:
        remotes = self._git_remote_names(repo)
        if not remotes:
            raise RuntimeError("PR-aware mode requires at least one configured git remote.")
        remote_set = set(remotes)

        candidate = str(requested_remote or "").strip()
        if candidate:
            if candidate not in remote_set:
                raise RuntimeError(
                    f"Configured PR remote '{candidate}' is not defined in this repository."
                )
            return candidate

        tracking_remote = self._git_tracking_remote(repo)
        if tracking_remote and tracking_remote in remote_set:
            return tracking_remote
        if "origin" in remote_set:
            return "origin"
        return remotes[0]

    def _build_pr_run_summary(self) -> str:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        stop_reason = str(self.state.stop_reason or "running")
        pull_url = str(self._pr_aware_state.get("pull_request_url") or "").strip()
        lines = [
            "## WarpFoundry Pipeline Summary",
            "",
            "> This description is managed automatically by PR-aware pipeline mode.",
            "",
            f"- Updated: {now}",
            f"- Repo: `{self.repo_path}`",
            f"- Mode: `{self.config.mode}`",
            f"- Stop reason: `{stop_reason}`",
            f"- Cycles completed: {int(self.state.total_cycles_completed or 0)}",
            f"- Phases completed: {int(self.state.total_phases_completed or 0)}",
            f"- Successes: {int(self.state.successes or 0)}",
            f"- Failures: {int(self.state.failures or 0)}",
            f"- Tokens used: {int(self.state.total_tokens or 0):,}",
            f"- Elapsed seconds: {round(float(self.state.elapsed_seconds or 0.0), 1)}",
        ]
        if pull_url:
            lines.append(f"- Pull request: {pull_url}")

        recent = self.state.results[-12:]
        lines.extend(["", "### Recent Phase Results", ""])
        if not recent:
            lines.append("No phase results recorded yet.")
            return "\n".join(lines)

        lines.append("| Cycle | Phase | Iter | Status | Tests | Files | Net | Commit |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for result in recent:
            status = "ok" if result.success else "failed"
            commit_short = str(result.commit_sha or "").strip()
            if len(commit_short) > 12:
                commit_short = commit_short[:12]
            lines.append(
                "| "
                f"{int(result.cycle or 0)} | "
                f"`{result.phase}` | "
                f"{int(result.iteration or 0)} | "
                f"{status} | "
                f"`{result.test_outcome}` | "
                f"{int(result.files_changed or 0)} | "
                f"{int(result.net_lines_changed or 0):+d} | "
                f"`{commit_short or '-'}` |"
            )
        return "\n".join(lines)

    def _pr_aware_find_pull_request(self) -> tuple[int, str]:
        state = self._pr_aware_state
        owner = str(state.get("owner") or "").strip()
        repo_name = str(state.get("repo_name") or "").strip()
        branch = str(state.get("branch") or "").strip()
        base_branch = str(state.get("base_branch") or "").strip()
        if not owner or not repo_name or not branch:
            return 0, ""

        query = f"state=open&head={quote(owner + ':' + branch, safe='')}"
        if base_branch:
            query += f"&base={quote(base_branch, safe='')}"
        path = (
            "/repos/"
            f"{quote(owner, safe='')}/{quote(repo_name, safe='')}/pulls?{query}"
        )
        payload, error = self._github_api_request(
            method="GET",
            path=path,
            token=str(state.get("token") or ""),
            payload=None,
        )
        if error:
            self._pr_aware_state["last_sync_error"] = error
            self._refresh_pr_aware_state()
            return 0, ""
        if not isinstance(payload, list) or not payload:
            return 0, ""
        for item in payload:
            if not isinstance(item, dict):
                continue
            number = int(item.get("number") or 0)
            url = str(item.get("html_url") or "").strip()
            if number > 0:
                return number, url
        return 0, ""

    def _pr_aware_create_pull_request(self) -> tuple[int, str, str]:
        state = self._pr_aware_state
        owner = str(state.get("owner") or "").strip()
        repo_name = str(state.get("repo_name") or "").strip()
        branch = str(state.get("branch") or "").strip()
        base_branch = str(state.get("base_branch") or "").strip()
        token = str(state.get("token") or "").strip()
        if not owner or not repo_name or not branch or not base_branch:
            return 0, "", "Missing owner/repo/head/base for pull request creation."
        if not token:
            return 0, "", "GitHub token is required to create pull requests."

        path = f"/repos/{quote(owner, safe='')}/{quote(repo_name, safe='')}/pulls"
        payload, error = self._github_api_request(
            method="POST",
            path=path,
            token=token,
            payload={
                "title": f"WarpFoundry pipeline updates ({branch})",
                "head": branch,
                "base": base_branch,
                "body": self._build_pr_run_summary(),
            },
        )
        if error:
            return 0, "", error
        if not isinstance(payload, dict):
            return 0, "", "Unexpected pull-request creation payload."
        number = int(payload.get("number") or 0)
        url = str(payload.get("html_url") or "").strip()
        if number <= 0:
            return 0, "", "GitHub pull-request creation returned an invalid PR number."
        return number, url, ""

    def _pr_aware_ensure_pull_request(self, *, create_if_missing: bool) -> None:
        state = self._pr_aware_state
        if not state.get("enabled"):
            return
        if state.get("pull_number"):
            return

        number, url = self._pr_aware_find_pull_request()
        if number > 0:
            state["pull_number"] = number
            state["pull_request_url"] = url
            state["last_sync_error"] = ""
            self._log("info", f"PR-aware mode linked existing PR #{number}.")
            self._refresh_pr_aware_state()
            return

        if not create_if_missing:
            self._refresh_pr_aware_state()
            return

        number, url, error = self._pr_aware_create_pull_request()
        if number <= 0:
            if error:
                state["last_sync_error"] = error
                self._log("warn", f"PR-aware mode could not create pull request: {error}")
            self._refresh_pr_aware_state()
            return

        state["pull_number"] = number
        state["pull_request_url"] = url
        state["last_sync_error"] = ""
        self._log("info", f"PR-aware mode created PR #{number}: {url}")
        self._refresh_pr_aware_state()

    def _pr_aware_push_updates(
        self,
        *,
        repo: Path,
        reason: str,
        set_upstream: bool,
        force: bool = False,
    ) -> bool:
        state = self._pr_aware_state
        if not state.get("enabled"):
            return False
        if not state.get("auto_push"):
            return False

        remote = str(state.get("remote") or "").strip()
        branch = str(state.get("branch") or "").strip()
        if not remote or not branch:
            state["last_push_error"] = "Missing remote/branch for PR-aware push."
            self._refresh_pr_aware_state()
            return False

        try:
            head = self._git_head_sha_full(repo)
        except Exception as exc:
            state["last_push_error"] = str(exc)
            self._refresh_pr_aware_state()
            return False

        if not force and head and head == str(state.get("last_pushed_head") or "").strip():
            return False

        push_args: list[str] = ["push"]
        needs_upstream = bool(set_upstream) or not bool(state.get("upstream_configured"))
        if needs_upstream:
            push_args.extend(["--set-upstream", remote, branch])
        else:
            push_args.extend([remote, branch])

        result = self._run_git_process(repo, *push_args)
        if result.returncode != 0:
            detail = self._git_process_error(result, f"git {' '.join(push_args)} failed")
            state["last_push_error"] = detail
            self._log("warn", f"PR-aware auto-push failed ({reason}): {detail}")
            self._refresh_pr_aware_state()
            return False

        state["last_pushed_head"] = head
        state["upstream_configured"] = True
        state["last_push_error"] = ""
        state["last_push_succeeded_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        self._log("info", f"PR-aware auto-push succeeded ({reason}) to {remote}/{branch}.")
        self._refresh_pr_aware_state()
        return True

    def _pr_aware_sync_pull_request_description(self, *, force: bool) -> bool:
        state = self._pr_aware_state
        if not state.get("enabled"):
            return False
        if not state.get("sync_description"):
            return False

        self._pr_aware_ensure_pull_request(create_if_missing=bool(state.get("auto_push")))
        pull_number = int(state.get("pull_number") or 0)
        if pull_number <= 0:
            return False

        token = str(state.get("token") or "").strip()
        if not token:
            state["last_sync_error"] = "GitHub token is required to update PR descriptions."
            self._refresh_pr_aware_state()
            return False

        owner = str(state.get("owner") or "").strip()
        repo_name = str(state.get("repo_name") or "").strip()
        if not owner or not repo_name:
            return False

        body = self._build_pr_run_summary()
        body_digest = hashlib.sha1(body.encode("utf-8")).hexdigest()
        if not force and body_digest == str(state.get("last_body_digest") or ""):
            return False

        path = f"/repos/{quote(owner, safe='')}/{quote(repo_name, safe='')}/pulls/{pull_number}"
        _payload, error = self._github_api_request(
            method="PATCH",
            path=path,
            token=token,
            payload={"body": body},
        )
        if error:
            state["last_sync_error"] = error
            self._log("warn", f"PR-aware description sync failed: {error}")
            self._refresh_pr_aware_state()
            return False

        state["last_sync_error"] = ""
        state["last_body_digest"] = body_digest
        state["last_sync_succeeded_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        self._log("info", f"PR-aware description synced (PR #{pull_number}).")
        self._refresh_pr_aware_state()
        return True

    def _pr_aware_maybe_sync(
        self,
        *,
        repo: Path,
        reason: str,
        force_description: bool = False,
    ) -> None:
        state = self._pr_aware_state
        if not state.get("enabled"):
            return
        pushed = self._pr_aware_push_updates(
            repo=repo,
            reason=reason,
            set_upstream=False,
            force=False,
        )
        self._pr_aware_sync_pull_request_description(force=force_description or pushed)

    def _setup_pr_aware_mode(
        self,
        *,
        repo: Path,
        config: PipelineConfig,
    ) -> None:
        now_tag = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        requested_branch = str(config.pr_feature_branch or "").strip()
        branch = requested_branch or f"warpfoundry/pr/{now_tag}"
        self._git_require_valid_branch_name(repo, branch)

        current = self._git_current_branch_name(repo)
        if self._git_local_branch_exists(repo, branch):
            if current != branch:
                checkout_result = self._run_git_process(repo, "checkout", branch)
                if checkout_result.returncode != 0:
                    detail = self._git_process_error(
                        checkout_result,
                        f"git checkout {branch} failed",
                    )
                    raise RuntimeError(detail)
                self._log("info", f"Switched to existing PR branch: {branch}")
            else:
                self._log("info", f"Using current PR branch: {branch}")
        else:
            create_branch(repo, branch_name=branch)
            self._log("info", f"Created PR branch: {branch}")

        remote = self._resolve_pr_aware_remote(repo, str(config.pr_remote or ""))
        remote_url = self._git_remote_url(repo, remote)
        base_branch = (
            str(config.pr_base_branch or "").strip()
            or self._git_remote_default_branch(repo, remote)
            or (current if current != "HEAD" else "main")
        )
        owner, repo_name = self._github_owner_repo_from_remote(remote_url)
        token = self._github_pat_token() if owner and repo_name else ""
        if config.pr_sync_description and owner and repo_name and not token:
            self._log(
                "warn",
                "PR-aware description sync is enabled but no GitHub token is configured.",
            )

        self._pr_aware_state = {
            "enabled": True,
            "branch": branch,
            "remote": remote,
            "base_branch": base_branch,
            "remote_url": remote_url,
            "owner": owner,
            "repo_name": repo_name,
            "token": token,
            "auto_push": bool(config.pr_auto_push),
            "sync_description": bool(config.pr_sync_description),
            "upstream_configured": False,
            "pull_number": 0,
            "pull_request_url": "",
            "last_pushed_head": "",
            "last_push_succeeded_at": "",
            "last_sync_succeeded_at": "",
            "last_push_error": "",
            "last_sync_error": "",
            "last_body_digest": "",
        }
        self._refresh_pr_aware_state()
        self._log(
            "info",
            "PR-aware mode enabled: "
            f"branch={branch}, remote={remote}, base={base_branch}.",
        )
        if not owner or not repo_name:
            self._log(
                "warn",
                "PR-aware GitHub automation is limited because the selected remote is not github.com.",
            )

        if config.pr_auto_push:
            self._pr_aware_push_updates(
                repo=repo,
                reason="startup",
                set_upstream=True,
                force=True,
            )
        self._pr_aware_ensure_pull_request(create_if_missing=bool(config.pr_auto_push))
        self._pr_aware_sync_pull_request_description(force=True)

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
        return shared_has_codex_auth()

    @staticmethod
    def _has_claude_auth() -> bool:
        return shared_has_claude_auth()

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
        default_agent = self._resolve_phase_agent_key(
            phase_agent="",
            default_agent=config.agent,
        )
        agents = {
            self._resolve_phase_agent_key(
                phase_agent=p.agent,
                default_agent=config.agent,
            )
            for p in phase_order
            if p.enabled
        }
        if not agents:
            agents = {default_agent}
        return agents

    @staticmethod
    def _normalize_agent_key(raw: str) -> str:
        key = (raw or "").strip().lower()
        if key in {"claude", "claude-code", "claude_code", "claudecode"}:
            return "claude_code"
        if key in {"", "auto"}:
            return "auto"
        return key

    def _resolve_phase_agent_key(self, *, phase_agent: str, default_agent: str) -> str:
        selected = self._normalize_agent_key(phase_agent)
        if selected == "auto":
            selected = self._normalize_agent_key(default_agent)
        if selected == "auto":
            selected = "codex"
        return selected

    @staticmethod
    def _image_provider_auth_issue(provider: str) -> str | None:
        return shared_image_provider_auth_issue(
            True,
            provider,
            codex_auth_detector=PipelineOrchestrator._has_codex_auth,
        )

    def _write_restart_checkpoint(
        self,
        *,
        config: PipelineConfig,
        next_cycle: int,
        next_phase_index: int,
    ) -> Path:
        checkpoint_dir = self.repo_path / ".codex_manager" / "state"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "pipeline_resume.json"
        payload = {
            "version": 1,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "repo_path": str(self.repo_path),
            "config": config.model_dump(),
            "resume_cycle": max(1, int(next_cycle)),
            "resume_phase_index": max(0, int(next_phase_index)),
        }
        checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return checkpoint_path

    def _preflight_issues(self, config: PipelineConfig, repo: Path) -> list[str]:
        issues: list[str] = []
        phase_order = config.get_phase_order()
        if not (repo / ".git").exists():
            issues.append(f"Not a git repository: {repo}")

        write_error = self._repo_write_error(repo)
        if write_error:
            issues.append(write_error)

        worktree_counts = shared_repo_worktree_counts(repo)
        if worktree_counts is not None:
            staged, unstaged, untracked = worktree_counts
            if staged or unstaged or untracked:
                issues.append(
                    "Repository worktree is dirty "
                    f"(staged {staged}, unstaged {unstaged}, untracked {untracked}). "
                    "Clean/stash local changes first."
                )

        if config.image_generation_enabled:
            image_issue = self._image_provider_auth_issue(config.image_provider)
            if image_issue:
                issues.append(image_issue)

        if config.vector_memory_enabled:
            backend = (config.vector_memory_backend or "chroma").strip().lower()
            if backend != "chroma":
                issues.append(
                    "Unsupported vector_memory_backend. Supported backend(s): chroma."
                )
            else:
                try:
                    import chromadb  # noqa: F401
                except Exception:
                    issues.append(
                        "Vector memory requires ChromaDB. Install with: pip install chromadb"
                    )

        if config.self_improvement_auto_restart and not config.self_improvement_enabled:
            issues.append(
                "self_improvement_auto_restart requires self_improvement_enabled to be true."
            )
        for raw_url in list(getattr(config, "run_completion_webhooks", []) or []):
            url = str(raw_url or "").strip()
            if not url:
                continue
            if not self._completion_webhook_url_valid(url):
                issues.append(
                    "Invalid run-completion webhook URL (must be http(s) with host): "
                    f"{url}"
                )

        if config.pr_aware_enabled:
            if config.mode != "apply":
                issues.append("pr_aware_enabled requires mode='apply'.")
            branch = str(config.pr_feature_branch or "").strip()
            if branch:
                probe = self._run_git_process(repo, "check-ref-format", "--branch", branch)
                if probe.returncode != 0:
                    detail = self._git_process_error(
                        probe,
                        f"Invalid pr_feature_branch value: {branch}",
                    )
                    issues.append(f"Invalid pr_feature_branch: {detail}")
            try:
                self._resolve_pr_aware_remote(repo, str(config.pr_remote or ""))
            except Exception as exc:
                issues.append(f"PR-aware remote setup failed: {exc}")

        if config.deep_research_enabled and config.deep_research_native_enabled:
            providers = (config.deep_research_providers or "both").strip().lower()
            if providers in {"both", "openai"}:
                openai_issue = shared_env_secret_issue(
                    ("OPENAI_API_KEY", "CODEX_API_KEY"),
                    "Native deep research (openai provider) requires OPENAI_API_KEY or CODEX_API_KEY.",
                )
                if openai_issue:
                    issues.append(openai_issue)
            if providers in {"both", "google"}:
                google_issue = shared_env_secret_issue(
                    ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
                    "Native deep research (google provider) requires GOOGLE_API_KEY or GEMINI_API_KEY.",
                )
                if google_issue:
                    issues.append(google_issue)

        issues.extend(
            shared_agent_preflight_issues(
                self._collect_required_agents(config),
                codex_binary=config.codex_binary,
                claude_binary=config.claude_binary,
                binary_exists_detector=self._binary_exists,
                codex_auth_detector=self._has_codex_auth,
                claude_auth_detector=self._has_claude_auth,
            )
        )

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
                openai_issue = shared_env_secret_issue(
                    ("OPENAI_API_KEY", "CODEX_API_KEY"),
                    "CUA visual test (openai provider) requires OPENAI_API_KEY or CODEX_API_KEY.",
                )
                if openai_issue:
                    issues.append(openai_issue)
            elif provider == "anthropic":
                try:
                    import anthropic  # noqa: F401
                except Exception:
                    issues.append(
                        "CUA visual test requires the Anthropic SDK. Install with: pip install anthropic"
                    )
                anthropic_issue = shared_env_secret_issue(
                    ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
                    "CUA visual test (anthropic provider) requires ANTHROPIC_API_KEY or CLAUDE_API_KEY.",
                )
                if anthropic_issue:
                    issues.append(anthropic_issue)

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
        try:
            config_snapshot = config.model_dump(mode="json")
        except Exception:
            try:
                config_snapshot = config.model_dump()
            except Exception:
                config_snapshot = {}

        self._record_history_note(
            "run_started",
            f"Pipeline run started in {config.mode} mode.",
            context={
                "run_id": self.state.run_id,
                "repo": str(repo),
                "mode": config.mode,
                "max_cycles": config.max_cycles,
                "unlimited": bool(config.unlimited),
                "phase_order": [p.phase.value for p in config.get_phase_order()],
                "science_enabled": bool(config.science_enabled),
                "brain_enabled": bool(config.brain_enabled),
                "resume_cycle": self._resume_cycle,
                "resume_phase_index": self._resume_phase_index,
                "vector_memory_enabled": bool(config.vector_memory_enabled),
                "vector_memory_backend": config.vector_memory_backend,
                "deep_research_enabled": bool(config.deep_research_enabled),
                "deep_research_providers": config.deep_research_providers,
                "deep_research_max_age_hours": int(config.deep_research_max_age_hours),
                "deep_research_dedupe": bool(config.deep_research_dedupe),
                "deep_research_native_enabled": bool(config.deep_research_native_enabled),
                "deep_research_retry_attempts": int(config.deep_research_retry_attempts),
                "deep_research_daily_quota": int(config.deep_research_daily_quota),
                "deep_research_max_provider_tokens": int(config.deep_research_max_provider_tokens),
                "deep_research_budget_usd": float(config.deep_research_budget_usd),
                "deep_research_openai_model": config.deep_research_openai_model,
                "deep_research_google_model": config.deep_research_google_model,
                "self_improvement_enabled": bool(config.self_improvement_enabled),
                "self_improvement_auto_restart": bool(config.self_improvement_auto_restart),
                "pr_aware_enabled": bool(config.pr_aware_enabled),
                "pr_feature_branch": str(config.pr_feature_branch or ""),
                "pr_remote": str(config.pr_remote or ""),
                "pr_base_branch": str(config.pr_base_branch or ""),
                "pr_auto_push": bool(config.pr_auto_push),
                "pr_sync_description": bool(config.pr_sync_description),
                "config_snapshot": config_snapshot,
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

        full_test_cmd = parse_test_command(config.test_cmd)
        smoke_test_cmd = parse_test_command(config.smoke_test_cmd)

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
                self._refresh_scientist_report()
            if config.vector_memory_enabled:
                if self.vector_memory.available:
                    self._log(
                        "info",
                        "Vector memory enabled: "
                        f"backend={self.vector_memory.backend}, "
                        f"collection={self.vector_memory.collection_name}",
                    )
                else:
                    self._log(
                        "warn",
                        f"Vector memory unavailable: {self.vector_memory.reason}",
                    )
            startup_recovered = self.tracker.get_recovered_backlog_context(max_items=20)
            startup_recovered_count = self._count_markdown_bullets(startup_recovered)
            if startup_recovered:
                self._record_recovered_backlog_memory(
                    phase=PipelinePhase.PROGRESS_REVIEW,
                    cycle=max(1, int(self._resume_cycle)),
                    context=startup_recovered,
                )
                self._record_history_note(
                    "recovered_backlog_loaded",
                    (
                        "Recovered pending cross-run backlog items from logs/history/"
                        "owner docs at startup."
                    ),
                    level="info",
                    context={
                        "items_recovered": startup_recovered_count,
                        "resume_cycle": self._resume_cycle,
                    },
                )
            self._run_retention_cleanup(reason="startup")
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
                if config.pr_aware_enabled:
                    self._setup_pr_aware_mode(repo=repo, config=config)
                else:
                    branch = create_branch(repo)
                    self._log("info", f"Created branch: {branch}")
            except Exception as exc:
                if config.pr_aware_enabled:
                    self._log("error", f"Failed to initialize PR-aware mode: {exc}")
                    self.ledger.add(
                        category="error",
                        title=f"PR-aware setup failed: {str(exc)[:60]}",
                        detail=str(exc),
                        severity="critical",
                        source="pipeline:startup",
                        step_ref="pr_aware_setup",
                    )
                    self.state.stop_reason = "pr_aware_setup_failed"
                else:
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
        start_cycle = max(1, int(self._resume_cycle))
        start_phase_index = max(0, int(self._resume_phase_index))
        if start_cycle > 1 or start_phase_index > 0:
            self._log(
                "info",
                "Resuming pipeline from checkpoint: "
                f"cycle={start_cycle}, phase_index={start_phase_index}",
            )
            self.state.total_cycles_completed = max(0, start_cycle - 1)

        try:
            for cycle_num in range(start_cycle, effective_max + 1):
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

                cycle_phase_start_index = start_phase_index if cycle_num == start_cycle else 0
                if cycle_phase_start_index > 0:
                    self._log(
                        "info",
                        f"Skipping to phase index {cycle_phase_start_index} for resumed cycle {cycle_num}.",
                    )

                for phase_idx, phase_cfg in enumerate(phase_order):
                    if phase_idx < cycle_phase_start_index:
                        continue
                    self._pause_event.wait()
                    if self._stop_event.is_set():
                        self.state.stop_reason = "user_stopped"
                        cycle_aborted = True
                        break

                    phase = phase_cfg.phase
                    self.state.current_phase = phase.value
                    deep_research_topic = ""

                    if phase == PipelinePhase.APPLY_UPGRADES_AND_RESTART:
                        self.state.current_iteration = 1
                        self.state.current_phase_started_at_epoch_ms = int(time.time() * 1000)

                        next_cycle = cycle_num
                        next_phase_index = phase_idx + 1
                        if next_phase_index >= len(phase_order):
                            next_cycle = cycle_num + 1
                            next_phase_index = 0

                        has_remaining_work = bool(config.unlimited) or next_cycle <= config.max_cycles
                        recovered_backlog = self.tracker.get_recovered_backlog_context(max_items=24)
                        recovered_backlog_count = self._count_markdown_bullets(recovered_backlog)
                        recovered_preview_lines: list[str] = []
                        if recovered_backlog:
                            recovered_preview_lines = [
                                line
                                for line in recovered_backlog.splitlines()
                                if line.startswith("- ")
                            ][:12]
                            self._record_recovered_backlog_memory(
                                phase=phase,
                                cycle=cycle_num,
                                context=recovered_backlog,
                            )
                            self._record_history_note(
                                "recovered_backlog_snapshot",
                                (
                                    "Captured cross-run backlog snapshot during self-improvement "
                                    "checkpoint."
                                ),
                                level="info",
                                context={
                                    "cycle": cycle_num,
                                    "phase": phase.value,
                                    "items_recovered": recovered_backlog_count,
                                },
                            )
                            progress_lines = [
                                "\n## Recovered Backlog Snapshot",
                                "",
                                f"- Captured: {dt.datetime.now(dt.timezone.utc).isoformat()}",
                                f"- Items recovered: {recovered_backlog_count}",
                                "- Sources: logs, owner docs, request history, run archives.",
                            ]
                            if recovered_preview_lines:
                                progress_lines.extend(
                                    [
                                        "",
                                        "### Top Carry-Forward Items",
                                        "",
                                        *recovered_preview_lines,
                                    ]
                                )
                            self.tracker.append("PROGRESS.md", "\n".join(progress_lines) + "\n")

                        summary = (
                            "Self-improvement checkpoint evaluated. "
                            f"next_cycle={next_cycle}, next_phase_index={next_phase_index}, "
                            f"recovered_backlog_items={recovered_backlog_count}."
                        )
                        self._log("info", summary)
                        restart_result = PhaseResult(
                            cycle=cycle_num,
                            phase=phase.value,
                            iteration=1,
                            agent_success=True,
                            validation_success=True,
                            tests_passed=True,
                            success=True,
                            test_outcome="skipped",
                            test_summary=summary,
                            test_exit_code=0,
                            files_changed=0,
                            net_lines_changed=0,
                            changed_files=[],
                            prompt_used="Self-improvement checkpoint",
                            agent_final_message=summary,
                            agent_used="system",
                        )
                        self.state.results.append(restart_result)
                        self.state.successes += 1
                        self.state.total_phases_completed += 1
                        self.state.elapsed_seconds = time.monotonic() - start_time

                        self._record_history_note(
                            "phase_result",
                            (
                                f"Cycle {cycle_num}, phase '{phase.value}' finished with status=ok."
                            ),
                            level="info",
                            context={
                                "cycle": cycle_num,
                                "phase": phase.value,
                                "iteration": 1,
                                "mode": config.mode,
                                "agent_success": True,
                                "validation_success": True,
                                "tests_passed": True,
                                "success": True,
                                "test_outcome": "skipped",
                                "files_changed": 0,
                                "net_lines_changed": 0,
                                "changed_files": [],
                                "duration_seconds": 0.0,
                                "commit_sha": None,
                                "terminate_repeats": False,
                                "error_message": "",
                                "recovered_backlog_items": recovered_backlog_count,
                            },
                        )

                        if has_remaining_work:
                            checkpoint_path = self._write_restart_checkpoint(
                                config=config,
                                next_cycle=next_cycle,
                                next_phase_index=next_phase_index,
                            )
                            self.state.restart_required = True
                            self.state.restart_checkpoint_path = str(checkpoint_path)
                            self.state.resume_cycle = max(1, int(next_cycle))
                            self.state.resume_phase_index = max(0, int(next_phase_index))
                            self.tracker.append(
                                "PROGRESS.md",
                                (
                                    "\n## Self-Improvement Checkpoint\n\n"
                                    f"- Created: {dt.datetime.now(dt.timezone.utc).isoformat()}\n"
                                    f"- Resume cycle: {self.state.resume_cycle}\n"
                                    f"- Resume phase index: {self.state.resume_phase_index}\n"
                                    f"- Checkpoint: {checkpoint_path}\n"
                                ),
                            )
                            self._record_history_note(
                                "self_restart_requested",
                                "Pipeline requested restart checkpoint for self-improvement.",
                                level="warn",
                                context={
                                    "checkpoint_path": str(checkpoint_path),
                                    "resume_cycle": self.state.resume_cycle,
                                    "resume_phase_index": self.state.resume_phase_index,
                                    "auto_restart": bool(config.self_improvement_auto_restart),
                                    "recovered_backlog_items": recovered_backlog_count,
                                },
                            )
                            self._log(
                                "warn",
                                "Self-improvement checkpoint created. Restart required to continue.",
                            )
                            self.state.stop_reason = "self_restart_requested"
                            cycle_aborted = True
                            break

                        self._log(
                            "info",
                            "Self-improvement checkpoint phase skipped restart because no work remains.",
                        )
                        continue

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

                    if phase == PipelinePhase.DEEP_RESEARCH:
                        deep_research_topic = self._derive_deep_research_topic(cycle_num)
                        if config.deep_research_dedupe and self.vector_memory.enabled:
                            cached = self.vector_memory.lookup_recent_deep_research(
                                deep_research_topic,
                                max_age_hours=config.deep_research_max_age_hours,
                            )
                            if cached:
                                self.state.current_iteration = 1
                                self.state.current_phase_started_at_epoch_ms = int(
                                    time.time() * 1000
                                )
                                cached_summary = str(cached.get("summary") or "").strip()
                                source = str(cached.get("source") or "memory-cache")
                                self._log(
                                    "info",
                                    "Deep research cache hit: "
                                    f"reusing recent findings from {source}.",
                                )
                                self._append_deep_research_log_entry(
                                    cycle=cycle_num,
                                    iteration=1,
                                    topic=deep_research_topic,
                                    providers=config.deep_research_providers,
                                    summary=cached_summary or "(cached summary missing)",
                                    reused_cache=True,
                                )
                                cached_result = PhaseResult(
                                    cycle=cycle_num,
                                    phase=phase.value,
                                    iteration=1,
                                    agent_success=True,
                                    validation_success=True,
                                    tests_passed=True,
                                    success=True,
                                    test_outcome="skipped",
                                    test_summary="Deep research cache reused.",
                                    test_exit_code=0,
                                    files_changed=0,
                                    net_lines_changed=0,
                                    changed_files=[],
                                    prompt_used="Deep research cache reuse",
                                    agent_final_message=(
                                        "Reused cached deep research findings for topic: "
                                        f"{deep_research_topic}"
                                    ),
                                    agent_used="system",
                                )
                                self.state.results.append(cached_result)
                                self.state.successes += 1
                                self.state.total_phases_completed += 1
                                self.state.elapsed_seconds = time.monotonic() - start_time
                                self.tracker.log_phase_result(
                                    phase.value,
                                    1,
                                    True,
                                    "Cache reuse: yes (recent deep research hit).",
                                )
                                self._record_history_note(
                                    "phase_result",
                                    (
                                        f"Cycle {cycle_num}, phase '{phase.value}' "
                                        "reused deep-research cache."
                                    ),
                                    context={
                                        "cycle": cycle_num,
                                        "phase": phase.value,
                                        "iteration": 1,
                                        "cache_reuse": True,
                                        "topic": deep_research_topic,
                                    },
                                )
                                self._record_vector_memory_note(
                                    phase=phase,
                                    cycle=cycle_num,
                                    iteration=1,
                                    result=cached_result,
                                )
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

                    if phase == PipelinePhase.DEEP_RESEARCH:
                        if not deep_research_topic:
                            deep_research_topic = self._derive_deep_research_topic(cycle_num)
                        base_prompt = (
                            f"{base_prompt}\n\n"
                            f"Research provider preference: {config.deep_research_providers}\n"
                            f"Research topic: {deep_research_topic}\n"
                            "If prior work already covers this topic, summarize reuse explicitly "
                            "instead of repeating expensive research."
                        )

                    phase_test_policy = self._normalized_test_policy(
                        getattr(phase_cfg, "test_policy", "skip")
                    )
                    phase_evaluator = self._build_phase_evaluator(
                        phase=phase,
                        phase_cfg=phase_cfg,
                        full_test_cmd=full_test_cmd,
                        smoke_test_cmd=smoke_test_cmd,
                        timeout_seconds=config.timeout_per_phase,
                    )
                    self._log_missing_test_policy_warning(
                        phase=phase,
                        phase_test_policy=phase_test_policy,
                        full_test_cmd=full_test_cmd,
                        smoke_test_cmd=smoke_test_cmd,
                    )

                    self._log(
                        "info",
                        f"Phase: {phase.value} "
                        f"({phase_cfg.iterations} iteration{'s' if phase_cfg.iterations > 1 else ''}) "
                        f"| test policy: {phase_test_policy}",
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
                            science_baseline_eval = phase_evaluator.evaluate(repo)
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

                        # Build phase context and execute either native research
                        # or regular agent-driven phase prompt.
                        context = self.tracker.get_context_for_phase(
                            phase.value, ledger=self.ledger
                        )
                        self._record_recovered_backlog_memory(
                            phase=phase,
                            cycle=cycle_num,
                            context=context,
                        )
                        full_prompt = ""
                        if phase == PipelinePhase.DEEP_RESEARCH and config.deep_research_native_enabled:
                            topic = deep_research_topic or self._derive_deep_research_topic(cycle_num)
                            self._log(
                                "info",
                                "  Executing native deep research "
                                f"({config.deep_research_providers}; topic='{topic[:100]}')",
                            )
                            result = self._execute_native_deep_research(
                                topic=topic,
                                cycle=cycle_num,
                                iteration=iteration,
                                phase_context=context,
                            )
                        else:
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

                            self._log(
                                "info",
                                f"  {format_prompt_log_line(full_prompt)}",
                            )

                            # Execute
                            agent_key = self._resolve_phase_agent_key(
                                phase_agent=phase_cfg.agent,
                                default_agent=config.agent,
                            )
                            runner = runners.get(agent_key, runners["codex"])

                            result = self._execute_phase(
                                runner,
                                phase_evaluator,
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
                        if (
                            config.mode == "apply"
                            and config.pr_aware_enabled
                            and bool(result.commit_sha)
                        ):
                            self._pr_aware_maybe_sync(
                                repo=repo,
                                reason=f"{phase.value}:iter{iteration}:phase_commit",
                            )
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
                        self._record_vector_memory_note(
                            phase=phase,
                            cycle=cycle_num,
                            iteration=iteration,
                            result=result,
                        )
                        if phase == PipelinePhase.DEEP_RESEARCH and result.success:
                            topic = deep_research_topic or self._derive_deep_research_topic(
                                cycle_num
                            )
                            summary = (
                                result.agent_final_message
                                or result.test_summary
                                or result.error_message
                            ).strip()
                            if summary:
                                self._append_deep_research_log_entry(
                                    cycle=cycle_num,
                                    iteration=iteration,
                                    topic=topic,
                                    providers=config.deep_research_providers,
                                    summary=summary,
                                    reused_cache=False,
                                )
                                if self.vector_memory.enabled:
                                    self.vector_memory.record_deep_research(
                                        topic=topic,
                                        summary=summary[:5000],
                                        providers=config.deep_research_providers,
                                        metadata={
                                            "phase": phase.value,
                                            "cycle": cycle_num,
                                            "iteration": iteration,
                                        },
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
                                "error_message": _clip_text(
                                    result.error_message, _HISTORY_ERROR_CONTEXT_CHARS
                                ),
                            },
                        )
                        if config.science_enabled:
                            self._refresh_scientist_report()

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
                                            "prompt_preview": format_prompt_preview(
                                                followup_prompt
                                            ),
                                            "prompt_metadata": prompt_metadata(followup_prompt),
                                        },
                                    )
                                    followup_result = self._execute_phase(
                                        runner,
                                        phase_evaluator,
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
                                    if (
                                        config.mode == "apply"
                                        and config.pr_aware_enabled
                                        and bool(followup_result.commit_sha)
                                    ):
                                        self._pr_aware_maybe_sync(
                                            repo=repo,
                                            reason=(
                                                f"{phase.value}:iter{iteration}:brain_followup_commit"
                                            ),
                                        )
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
                                            "error_message": _clip_text(
                                                followup_result.error_message,
                                                _HISTORY_ERROR_CONTEXT_CHARS,
                                            ),
                                            "brain_follow_up": True,
                                        },
                                    )
                                    if config.science_enabled:
                                        self._refresh_scientist_report()
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
                        committed = self._auto_commit_repo(
                            repo=repo,
                            cycle_num=cycle_num,
                            commit_scope=f"pipeline-{phase.value}",
                        )
                        if committed and config.pr_aware_enabled:
                            self._pr_aware_maybe_sync(
                                repo=repo,
                                reason=f"{phase.value}:per_phase_auto_commit",
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
                        committed = self._auto_commit_repo(
                            repo=repo,
                            cycle_num=cycle_num,
                            commit_scope=f"pipeline-cycle-{cycle_num}",
                        )
                        if committed and config.pr_aware_enabled:
                            self._pr_aware_maybe_sync(
                                repo=repo,
                                reason=f"cycle-{cycle_num}:per_cycle_auto_commit",
                            )

                if cycle_aborted:
                    break

                self.state.total_cycles_completed = cycle_num
                self.state.resume_cycle = cycle_num + 1
                self.state.resume_phase_index = 0
                self._run_retention_cleanup(reason=f"cycle-{cycle_num}")
                if config.mode == "apply" and config.pr_aware_enabled:
                    self._pr_aware_maybe_sync(
                        repo=repo,
                        reason=f"cycle-{cycle_num}:summary",
                        force_description=True,
                    )

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

        allow_path_creation = bool(getattr(self.config, "allow_path_creation", True))
        dep_policy = str(
            getattr(self.config, "dependency_install_policy", "project_only") or "project_only"
        ).strip().lower()
        if dep_policy not in {"disallow", "project_only", "allow_system"}:
            dep_policy = "project_only"
        image_enabled = bool(getattr(self.config, "image_generation_enabled", False))
        image_provider = str(getattr(self.config, "image_provider", "openai") or "openai").strip().lower()
        if image_provider not in {"openai", "google"}:
            image_provider = "openai"
        image_model = str(getattr(self.config, "image_model", "") or "").strip()
        if not image_model:
            image_model = "gpt-image-1" if image_provider == "openai" else "nano-banana"

        parts.append("--- CAPABILITY CONTRACT ---")
        parts.append(
            "File and directory creation inside repository: "
            + ("allowed." if allow_path_creation else "not allowed; edit existing paths only.")
        )
        if dep_policy == "disallow":
            parts.append(
                "Dependency installation: not allowed. Do not run pip/npm/brew/apt/choco install commands."
            )
        elif dep_policy == "project_only":
            parts.append(
                "Dependency installation: allowed only for project-scoped environments "
                "(for example venv, uv, poetry, npm/pnpm/yarn in this repo)."
            )
            parts.append(
                "Do not install global/system-wide dependencies. Prefer pinned versions and minimal additions."
            )
        else:
            parts.append(
                "Dependency installation: system-wide and project-scoped installs are allowed when required."
            )
            parts.append(
                "Keep changes minimal, document why installs are needed, and include rollback notes."
            )

        if image_enabled:
            parts.append(
                f"Image generation: enabled using provider `{image_provider}` and model `{image_model}`."
            )
            if image_provider == "openai":
                parts.append(
                    "Requires OPENAI_API_KEY or CODEX_API_KEY (or Codex CLI auth) in environment."
                )
            else:
                parts.append("Requires GOOGLE_API_KEY or GEMINI_API_KEY in environment.")
            parts.append(
                "Save generated assets under repository paths such as assets/icons or docs/images."
            )
        else:
            parts.append("Image generation: disabled for this run.")
        parts.append("")

        # Base prompt
        parts.append(base_prompt)

        # Log file context (includes ledger when tracker was called with ledger)
        if context:
            parts.append("\n--- CURRENT LOG FILE CONTENTS ---\n")
            parts.append(context)

        memory_context = self._build_vector_memory_context(
            phase=phase,
            base_prompt=base_prompt,
            log_context=context,
        )
        if memory_context:
            parts.append("\n--- LONG-TERM MEMORY CONTEXT ---\n")
            parts.append(memory_context)

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

    @staticmethod
    def _normalized_test_policy(raw_policy: object) -> str:
        policy = str(raw_policy or "skip").strip().lower()
        if policy in {"skip", "smoke", "full"}:
            return policy
        return "skip"

    def _build_phase_evaluator(
        self,
        *,
        phase: PipelinePhase,
        phase_cfg: PhaseConfig,
        full_test_cmd: list[str] | None,
        smoke_test_cmd: list[str] | None,
        timeout_seconds: int,
    ) -> RepoEvaluator:
        policy = self._normalized_test_policy(getattr(phase_cfg, "test_policy", "skip"))
        selected_cmd: list[str] | None = None
        if policy == "full":
            selected_cmd = full_test_cmd
        elif policy == "smoke":
            selected_cmd = smoke_test_cmd if smoke_test_cmd is not None else full_test_cmd
            if smoke_test_cmd is None and full_test_cmd is not None:
                self._log(
                    "info",
                    (
                        f"  Phase '{phase.value}' uses smoke policy; "
                        "falling back to full validation command."
                    ),
                )
        return RepoEvaluator(
            test_cmd=selected_cmd,
            timeout=timeout_seconds,
            skip_tests=(selected_cmd is None),
        )

    def _log_missing_test_policy_warning(
        self,
        *,
        phase: PipelinePhase,
        phase_test_policy: str,
        full_test_cmd: list[str] | None,
        smoke_test_cmd: list[str] | None,
    ) -> None:
        """Log actionable missing test-command warnings only once per run."""
        policy = self._normalized_test_policy(phase_test_policy)
        if policy not in {"smoke", "full"}:
            return

        has_full_cmd = full_test_cmd is not None
        has_smoke_cmd = smoke_test_cmd is not None

        if not has_full_cmd and not has_smoke_cmd:
            key = "global:no_validation_commands"
            if key in self._missing_test_policy_warning_keys:
                return
            self._missing_test_policy_warning_keys.add(key)
            self._log(
                "warn",
                (
                    "No validation command is configured. Phases requesting smoke/full tests "
                    "will skip validation. Configure 'Validation Command' (and optional "
                    "'Smoke Test Command') in Pipeline settings."
                ),
            )
            return

        if policy == "full" and not has_full_cmd:
            key = f"{phase.value}:full:missing_validation_command"
            if key in self._missing_test_policy_warning_keys:
                return
            self._missing_test_policy_warning_keys.add(key)
            self._log(
                "warn",
                (
                    f"  Phase '{phase.value}' requested full tests, but Validation Command "
                    "is empty; tests will be skipped for this phase."
                ),
            )

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
        artifact_extra_paths: list[str] = []
        phase_log_name = PHASE_LOG_FILES.get(phase)
        if phase_log_name:
            artifact_extra_paths.append(f".codex_manager/logs/{phase_log_name}")
        artifact_snapshot = capture_artifact_snapshot(
            repo,
            extra_globs=_PIPELINE_MANAGED_ARTIFACT_GLOBS,
            extra_rel_paths=artifact_extra_paths,
        )

        # Run the agent
        prompt_meta = prompt_metadata(prompt)
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
            phase_result = PhaseResult(
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
            self._append_pipeline_debug_event(
                {
                    "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "cycle": cycle,
                    "phase": phase.value,
                    "iteration": iteration,
                    "mode": config.mode,
                    "runner": runner.name,
                    "prompt_length": prompt_meta["length_chars"],
                    "prompt_sha256": prompt_meta["sha256"],
                    "prompt_redaction_hits": prompt_meta["redaction_hits"],
                    "prompt_preview": format_prompt_preview(prompt),
                    "prompt_metadata": prompt_meta,
                    "exception": _clip_text(str(exc), 4000),
                    "result": {
                        "agent_success": phase_result.agent_success,
                        "validation_success": phase_result.validation_success,
                        "success": phase_result.success,
                        "test_outcome": phase_result.test_outcome,
                        "files_changed": phase_result.files_changed,
                        "net_lines_changed": phase_result.net_lines_changed,
                        "error_message": _clip_text(phase_result.error_message, 4000),
                        "duration_seconds": phase_result.duration_seconds,
                    },
                }
            )
            return phase_result

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
        artifact_entries, artifact_insertions, artifact_deletions = summarize_artifact_delta(
            repo,
            artifact_snapshot,
            extra_globs=_PIPELINE_MANAGED_ARTIFACT_GLOBS,
            extra_rel_paths=artifact_extra_paths,
        )
        artifact_merge = merge_eval_result_with_artifact_delta(eval_result, artifact_entries)
        artifact_net = artifact_merge["insertions"] - artifact_merge["deletions"]
        if artifact_merge["files_added"] > 0:
            self._log(
                "info",
                (
                    "  Managed artifacts: "
                    f"{artifact_merge['files_added']} files, net {artifact_net:+d} "
                    "(counted outside git numstat)."
                ),
            )
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
        phase_result = PhaseResult(
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
        self._append_pipeline_debug_event(
            {
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "cycle": cycle,
                "phase": phase.value,
                "iteration": iteration,
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
                    "errors": [_clip_text(err, 4000) for err in run_result.errors],
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
                    "agent_success": phase_result.agent_success,
                    "tests_validation_success": tests_validation_success,
                    "requires_repo_delta": requires_repo_delta,
                    "has_repo_delta": has_repo_delta,
                    "terminate_repeats": terminate_repeats,
                    "validation_failures": validation_failures,
                    "validation_success": phase_result.validation_success,
                    "final_success": phase_result.success,
                },
                "messages": {
                    "test_summary": _clip_text(eval_result.test_summary, 2000),
                    "error_message": _clip_text(error_message, 4000),
                    "agent_output_excerpt": _clip_text(agent_output, 2000),
                },
            }
        )
        return phase_result

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
                "Install with: pip install warpfoundry[cua] then python -m playwright install",
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
