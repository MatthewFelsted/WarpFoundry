"""Structured markdown log file manager.

Manages the five log files that the pipeline reads from and writes to:

- **WISHLIST.md** — Feature ideas, prioritized and bundled
- **TESTPLAN.md** — Test cases and coverage tracking
- **ERRORS.md** — Error log with root cause analysis
- **EXPERIMENTS.md** — Hypotheses, experiments, and findings
- **PROGRESS.md** — Overall progress tracking

The tracker initializes these files with headers and provides methods to
read, append, and query their contents.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from codex_manager.file_io import append_text, atomic_write_text, read_text_utf8_resilient
from codex_manager.logbook_utils import rotate_if_needed

logger = logging.getLogger(__name__)

# ── File templates (initial content) ──────────────────────────────

_WISHLIST_TEMPLATE = """\
# WISHLIST — Feature Ideas & Improvements

> Auto-maintained by WarpFoundry Pipeline. Items are added during ideation,
> prioritized and bundled during prioritization, and implemented in order.

## Priority Queue

| Rank | Bundle | Items | Combined Score | Est. Effort |
|------|--------|-------|----------------|-------------|
| *(run prioritization phase to populate)* | | | | |

---

"""

_TESTPLAN_TEMPLATE = """\
# TEST PLAN — Test Cases & Coverage

> Auto-maintained by WarpFoundry Pipeline. Test cases are designed during
> the testing phase and tracked here.

## Coverage Summary

- **Unit Tests**: pending
- **Integration Tests**: pending
- **Edge Cases**: pending

---

"""

_ERRORS_TEMPLATE = """\
# ERROR LOG — Issues & Root Cause Analysis

> Auto-maintained by WarpFoundry Pipeline. Errors are logged during
> debugging with root cause analysis and fix descriptions.

## Summary

- **Total Errors**: 0
- **Fixed**: 0
- **Open**: 0

---

"""

_EXPERIMENTS_TEMPLATE = """\
# EXPERIMENTS — Hypotheses, Tests & Findings

> Auto-maintained by WarpFoundry Pipeline (Scientist Mode). Contains
> hypotheses, experimental protocols, results, and analysis.

## Summary

- **Hypotheses Proposed**: 0
- **Experiments Run**: 0
- **Conclusions Reached**: 0

---

"""

_PROGRESS_TEMPLATE = """\
# PROGRESS — Development Progress Tracker

> Auto-maintained by WarpFoundry Pipeline. Records what was accomplished
> each cycle and tracks overall project health.

## Project Status

- **Pipeline Cycles Completed**: 0
- **WISHLIST Items Completed**: 0
- **Tests Passing**: unknown
- **Open Errors**: 0

---

"""

_RESEARCH_TEMPLATE = """\
# RESEARCH — Deep Research Findings

> Auto-maintained by WarpFoundry Pipeline (Deep Research mode). Captures
> external research questions, summaries, and actionable implementation links.

## Summary

- **Research Runs**: 0
- **Cached Reuses**: 0
- **Last Topic**: none

---

"""

_AGENT_PROTOCOL_TEMPLATE = """\
# Agent Protocol

This file defines shared run-time coordination rules for all agents.

## Core Rules

1. Stay aligned to repository goals and current phase scope.
2. Reuse existing project context before proposing new work.
3. Avoid duplicate work when prior research or memory already covers a topic.
4. When uncertain, state assumptions explicitly in outputs/logs.
5. Keep edits incremental, testable, and reversible.
6. Avoid absolute marketing/legal guarantees unless verified by trusted sources.

## Coordination Contract

- Inputs:
  - `.codex_manager/logs/*.md` phase logs
  - `.codex_manager/ledger/*` open-item ledger
  - `.codex_manager/memory/*` long-term memory / research cache
- Outputs:
  - Update phase logs with concise, structured findings
  - Reference IDs for ledger/memory items when reusing prior context
  - Record decision rationale in `PROGRESS.md`

## Safety + Quality

- Prefer high-signal, directly actionable recommendations.
- Avoid speculative changes unrelated to project improvement.
- If blocked by missing credentials/tools, log the blocker and next step.
- For research-backed claims, include HTTPS source URLs from credible domains.
- Flag low-trust or policy-blocked sources for owner review.
"""

_TEMPLATES: dict[str, str] = {
    "WISHLIST.md": _WISHLIST_TEMPLATE,
    "TESTPLAN.md": _TESTPLAN_TEMPLATE,
    "ERRORS.md": _ERRORS_TEMPLATE,
    "EXPERIMENTS.md": _EXPERIMENTS_TEMPLATE,
    "PROGRESS.md": _PROGRESS_TEMPLATE,
    "RESEARCH.md": _RESEARCH_TEMPLATE,
}

_NORMALIZATION_MARKER = ".encoding_normalized_v1"
_DEFAULT_MARKDOWN_ROTATE_BYTES = 2_000_000
_DEFAULT_MARKDOWN_MAX_ARCHIVES = 12
_RECOVERY_CHECKBOX_RE = re.compile(r"^\s*[-*]\s+\[(?P<status>[ xX])\]\s+(?P<body>.+?)\s*$")
_RECOVERY_TODO_RE = re.compile(
    r"^\s*(?:[-*]\s+)?(?:TODO|FIXME|TBD)\s*[:\-]\s*(?P<body>.+)$",
    re.IGNORECASE,
)
_RECOVERY_DONE_STATUSES = {
    "done",
    "completed",
    "complete",
    "implemented",
    "resolved",
    "closed",
    "shipped",
}


class LogTracker:
    """Manages structured markdown log files for the pipeline.

    Parameters
    ----------
    repo_path:
        The root directory of the target repository.  Log files are
        created in a ``.codex_manager/logs/`` subdirectory.
    """

    def __init__(
        self,
        repo_path: str | Path,
        *,
        markdown_rotate_bytes: int | None = None,
        markdown_max_archives: int = _DEFAULT_MARKDOWN_MAX_ARCHIVES,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.logs_dir = self.repo_path / ".codex_manager" / "logs"
        rotate_env = os.getenv("CODEX_MANAGER_LOG_ROTATE_BYTES", "").strip()
        default_rotate = _DEFAULT_MARKDOWN_ROTATE_BYTES
        if rotate_env:
            try:
                default_rotate = max(0, int(rotate_env))
            except ValueError:
                logger.warning(
                    "Invalid CODEX_MANAGER_LOG_ROTATE_BYTES=%r; using %s",
                    rotate_env,
                    _DEFAULT_MARKDOWN_ROTATE_BYTES,
                )
        self.markdown_rotate_bytes = (
            max(0, int(markdown_rotate_bytes))
            if markdown_rotate_bytes is not None
            else default_rotate
        )
        self.markdown_max_archives = max(1, int(markdown_max_archives))
        self._tracker_archive_dir = self.logs_dir / "archive" / "tracker"
        self._encoding_events_seen: set[tuple[str, str]] = set()

    def initialize(self) -> None:
        """Create log directory and initialize any missing log files."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._tracker_archive_dir.mkdir(parents=True, exist_ok=True)
        for filename, template in _TEMPLATES.items():
            path = self.logs_dir / filename
            if not path.exists():
                atomic_write_text(path, template)
                logger.info("Created %s", path)
        protocol = self.repo_path / ".codex_manager" / "AGENT_PROTOCOL.md"
        if not protocol.exists():
            protocol.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(protocol, _AGENT_PROTOCOL_TEMPLATE)
            logger.info("Created %s", protocol)
        self._normalize_markdown_logs_once()
        self._migrate_legacy_error_log()

    # -- Scientist evidence artifacts ---------------------------------

    def science_dir(self) -> Path:
        """Return the directory used for Scientist-mode evidence files."""
        return self.logs_dir / "scientist"

    def initialize_science(self) -> None:
        """Create Scientist evidence files and folders if missing."""
        root = self.science_dir()
        (root / "prompts").mkdir(parents=True, exist_ok=True)
        (root / "outputs").mkdir(parents=True, exist_ok=True)
        (root / "snapshots").mkdir(parents=True, exist_ok=True)

        templates: dict[str, str] = {
            "README.md": (
                "# Scientist Evidence Artifacts\n\n"
                "This folder is auto-generated by Scientist Mode.\n\n"
                "- `../SCIENTIST_REPORT.md`: user-facing science dashboard and action plan\n"
                "- `TRIALS.jsonl`: structured machine-readable records (one trial per line)\n"
                "- `EVIDENCE.md`: human-readable per-trial summaries\n"
                "- `HYPOTHESES.md`: extracted hypothesis entries\n"
                "- `ANALYSIS.md`: analysis outputs from the analyze phase\n"
                "- `EXPERIMENTS_LATEST.md`: latest EXPERIMENTS.md snapshot\n"
                "- `prompts/`: exact prompts used for each science trial\n"
                "- `outputs/`: raw agent outputs for each science trial\n"
                "- `snapshots/`: snapshots of EXPERIMENTS.md per trial\n"
            ),
            "TRIALS.jsonl": "",
            "EVIDENCE.md": (
                "# Scientist Evidence Log\n\n"
                "> Evidence-first record of hypotheses, experiments, and outcomes.\n"
            ),
            "HYPOTHESES.md": (
                "# Scientist Hypotheses\n\n"
                "> Extracted from theorize phase outputs / EXPERIMENTS.md.\n"
            ),
            "ANALYSIS.md": (
                "# Scientist Analysis\n\n> Analyze-phase outputs, preserved verbatim for review.\n"
            ),
            "EXPERIMENTS_LATEST.md": (
                "# Latest EXPERIMENTS.md Snapshot\n\nNo scientist snapshot captured yet.\n"
            ),
        }

        for name, template in templates.items():
            path = root / name
            if not path.exists():
                atomic_write_text(path, template)
                logger.info("Created %s", path)

        report_path = self.logs_dir / "SCIENTIST_REPORT.md"
        if not report_path.exists():
            atomic_write_text(
                report_path,
                (
                    "# Scientist Mode Report\n\n"
                    "> Auto-generated science dashboard. Run Scientist Mode to populate.\n"
                ),
            )
            logger.info("Created %s", report_path)

    def science_path_for(self, relative_path: str) -> Path:
        """Return an absolute path inside the Scientist evidence folder."""
        return self.science_dir() / relative_path

    def write_science(self, relative_path: str, content: str) -> Path:
        """Write content to a Scientist evidence file."""
        path = self.science_path_for(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, content)
        return path

    def append_science(self, relative_path: str, content: str) -> Path:
        """Append content to a Scientist evidence file."""
        path = self.science_path_for(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        prefix = ""
        if path.exists() and path.stat().st_size > 0 and not content.startswith("\n"):
            prefix = "\n"
        append_text(path, prefix + content)
        return path

    def append_science_jsonl(self, relative_path: str, payload: dict[str, Any]) -> Path:
        """Append one JSON object line to a Scientist evidence JSONL file."""
        line = json.dumps(payload, ensure_ascii=False)
        path = self.science_path_for(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        append_text(path, line + "\n")
        return path

    def save_science_artifact(
        self,
        folder: str,
        stem: str,
        content: str,
        *,
        suffix: str = ".md",
    ) -> Path:
        """Save a named artifact in the Scientist evidence folder."""
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-") or "artifact"
        safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        relative = str(Path(folder) / f"{safe_stem}{safe_suffix}")
        return self.write_science(relative, content)

    def path_for(self, filename: str) -> Path:
        """Return the full path for a log file."""
        return self.logs_dir / filename

    def _read_text_utf8_resilient(self, path: Path) -> str:
        """Read *path* as text, recovering from legacy non-UTF8 bytes.

        If a fallback decoder succeeds, the file is rewritten as UTF-8 so
        future reads are stable.
        """
        result = read_text_utf8_resilient(path, normalize_to_utf8=True)
        if result.used_fallback:
            self._record_encoding_recovery(path=path, decoder=result.decoder)
        return result.text

    def _record_encoding_recovery(self, *, path: Path, decoder: str) -> None:
        key = (str(path), decoder)
        if key in self._encoding_events_seen:
            return
        self._encoding_events_seen.add(key)
        logger.warning(
            "Recovered non-UTF8 log file %s using %s; rewritten as UTF-8",
            path,
            decoder,
        )
        if path.name.upper() == "ERRORS.MD":
            return
        errors_path = self.path_for("ERRORS.md")
        note = (
            "\n### [encoding-recovery]\n"
            f"- **Time**: {dt.datetime.now(dt.timezone.utc).isoformat()}\n"
            f"- **File**: `{path}`\n"
            f"- **Decoder**: `{decoder}`\n"
            "- **Action**: Rewrote file as UTF-8 to avoid runtime decode failures.\n"
        )
        try:
            prefix = "\n" if errors_path.exists() and errors_path.stat().st_size > 0 else ""
            append_text(errors_path, prefix + note)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not record encoding recovery note: %s", exc)

    def _normalization_marker(self) -> Path:
        return self.logs_dir / _NORMALIZATION_MARKER

    def _normalize_markdown_logs_once(self) -> None:
        marker = self._normalization_marker()
        if marker.exists():
            return
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.logs_dir.glob("*.md")):
            result = read_text_utf8_resilient(path, normalize_to_utf8=True)
            if result.used_fallback:
                self._record_encoding_recovery(path=path, decoder=result.decoder)
        atomic_write_text(
            marker,
            f"normalized_at={dt.datetime.now(dt.timezone.utc).isoformat()}\n",
        )

    def _migrate_legacy_error_log(self) -> None:
        legacy_path = self.repo_path / ".codex_manager" / "ERRORS.md"
        if not legacy_path.exists() or not legacy_path.is_file():
            return
        result = read_text_utf8_resilient(legacy_path, normalize_to_utf8=True)
        migrated = result.text.strip()
        if migrated:
            block = (
                "\n## Legacy Runtime Error Log Migration\n\n"
                f"- **Migrated**: {dt.datetime.now(dt.timezone.utc).isoformat()}\n"
                f"- **Source**: `{legacy_path}`\n\n"
                "```text\n"
                f"{migrated[:12000]}\n"
                "```\n"
            )
            self.append("ERRORS.md", block)
        archive_dir = self.logs_dir / "archive" / "legacy"
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archived = archive_dir / f"ERRORS-legacy-{stamp}.md"
        idx = 1
        while archived.exists():
            idx += 1
            archived = archive_dir / f"ERRORS-legacy-{stamp}-{idx}.md"
        try:
            legacy_path.replace(archived)
            logger.info("Migrated legacy runtime error log to %s", archived)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not archive legacy runtime error log %s: %s", legacy_path, exc)

    def _rotate_markdown_if_needed(self, filename: str, path: Path) -> None:
        if self.markdown_rotate_bytes <= 0:
            return
        if path.suffix.lower() != ".md":
            return
        rotate_if_needed(
            path=path,
            max_bytes=self.markdown_rotate_bytes,
            archive_dir=self._tracker_archive_dir,
            markdown_header=_TEMPLATES.get(filename, ""),
            max_archives=self.markdown_max_archives,
        )

    def read(self, filename: str) -> str:
        """Read the contents of a log file."""
        path = self.logs_dir / filename
        return self._read_text_utf8_resilient(path)

    def write(self, filename: str, content: str) -> None:
        """Overwrite a log file with new content."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        path = self.logs_dir / filename
        atomic_write_text(path, content)

    def append(self, filename: str, content: str) -> None:
        """Append content to a log file."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        path = self.logs_dir / filename
        self._rotate_markdown_if_needed(filename, path)
        prefix = "\n" if path.exists() and path.stat().st_size > 0 and not content.startswith("\n") else ""
        append_text(path, prefix + content)

    # ── Structured queries ───────────────────────────────────────

    def count_items(self, filename: str, status: str = "pending") -> int:
        """Count items with a given status in a log file.

        Looks for lines matching ``**Status**: <status>`` pattern.
        """
        content = self.read(filename)
        pattern = re.compile(rf"\*\*Status\*\*:\s*{re.escape(status)}", re.IGNORECASE)
        return len(pattern.findall(content))

    def count_all_items(self, filename: str) -> dict[str, int]:
        """Count items by status in a log file."""
        content = self.read(filename)
        pattern = re.compile(r"\*\*Status\*\*:\s*(\w+)", re.IGNORECASE)
        counts: dict[str, int] = {}
        for match in pattern.finditer(content):
            status = match.group(1).lower()
            counts[status] = counts.get(status, 0) + 1
        return counts

    def next_item_id(self, filename: str, prefix: str) -> str:
        """Generate the next sequential ID for a log file.

        E.g., if WISHLIST.md has WISH-001 through WISH-012, returns "WISH-013".
        """
        content = self.read(filename)
        pattern = re.compile(rf"\[{re.escape(prefix)}-(\d+)\]")
        numbers = [int(m.group(1)) for m in pattern.finditer(content)]
        next_num = max(numbers, default=0) + 1
        return f"{prefix}-{next_num:03d}"

    @staticmethod
    def _clip_recovery_text(value: str, *, limit: int = 240) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            return ""
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    def _collect_markdown_recovery_items(
        self,
        path: Path,
        *,
        source: str,
        max_items: int,
    ) -> list[str]:
        if max_items <= 0 or not path.is_file():
            return []
        text = self._read_text_utf8_resilient(path)
        items: list[str] = []
        for line in text.splitlines():
            checkbox_match = _RECOVERY_CHECKBOX_RE.match(line)
            if checkbox_match:
                status = str(checkbox_match.group("status") or "").strip().lower()
                if status == "x":
                    continue
                body = self._clip_recovery_text(str(checkbox_match.group("body") or ""))
                if body:
                    items.append(f"[{source}] {body}")
                    if len(items) >= max_items:
                        break
                continue

            todo_match = _RECOVERY_TODO_RE.match(line)
            if todo_match:
                body = self._clip_recovery_text(str(todo_match.group("body") or ""))
                if body:
                    items.append(f"[{source}] {body}")
                    if len(items) >= max_items:
                        break
        return items

    def _collect_general_request_history_items(self, *, max_items: int) -> list[str]:
        if max_items <= 0:
            return []
        path = self.repo_path / ".codex_manager" / "owner" / "GENERAL_REQUEST_HISTORY.jsonl"
        if not path.is_file():
            return []
        text = self._read_text_utf8_resilient(path)
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return []
        items: list[str] = []
        for raw in reversed(lines[-600:]):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            status = str(payload.get("status") or "").strip().lower()
            if status in _RECOVERY_DONE_STATUSES:
                continue
            request = self._clip_recovery_text(str(payload.get("request") or ""))
            if not request:
                continue
            status_label = status or "pending"
            items.append(f"[owner/GENERAL_REQUEST_HISTORY.jsonl:{status_label}] {request}")
            if len(items) >= max_items:
                break
        return items

    def _collect_history_recovery_items(self, *, max_items: int) -> list[str]:
        if max_items <= 0:
            return []
        path = self.path_for("HISTORY.jsonl")
        if not path.is_file():
            return []
        text = self._read_text_utf8_resilient(path)
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return []
        items: list[str] = []
        for raw in reversed(lines[-800:]):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            event = str(payload.get("event") or "").strip().lower()
            level = str(payload.get("level") or "").strip().lower()
            summary = self._clip_recovery_text(str(payload.get("summary") or ""), limit=180)
            context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
            if event == "phase_result":
                if bool(context.get("success", True)):
                    continue
                phase = str(context.get("phase") or "").strip() or "unknown"
                cycle = str(context.get("cycle") or "?")
                detail = summary or "phase reported failure"
                items.append(
                    f"[logs/HISTORY.jsonl] Follow up failed phase '{phase}' (cycle {cycle}): {detail}"
                )
            elif event == "self_restart_requested":
                detail = summary or "restart checkpoint requested"
                items.append(f"[logs/HISTORY.jsonl] Restart handoff recorded: {detail}")
            elif level in {"warn", "error"}:
                if summary:
                    items.append(f"[logs/HISTORY.jsonl:{level}] {summary}")
            if len(items) >= max_items:
                break
        return items

    def _collect_output_history_recovery_items(self, *, max_items: int) -> list[str]:
        if max_items <= 0:
            return []
        output_history_root = self.repo_path / ".codex_manager" / "output_history"
        if not output_history_root.is_dir():
            return []

        run_dirs = sorted(
            (path for path in output_history_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not run_dirs:
            return []

        items: list[str] = []
        filename_keywords = (
            "todo",
            "wishlist",
            "plan",
            "progress",
            "research",
            "experiment",
            "scientist",
            "summary",
            "notes",
        )
        for run_dir in run_dirs[:3]:
            for file_path in sorted(run_dir.glob("*.md")):
                name = file_path.name.lower()
                if not any(keyword in name for keyword in filename_keywords):
                    continue
                source = f"output_history/{run_dir.name}/{file_path.name}"
                remaining = max_items - len(items)
                if remaining <= 0:
                    return items
                hits = self._collect_markdown_recovery_items(
                    file_path,
                    source=source,
                    max_items=remaining,
                )
                if hits:
                    items.extend(hits)
                if len(items) >= max_items:
                    return items
        return items

    def get_recovered_backlog_context(self, *, max_items: int = 24) -> str:
        """Return deduplicated pending items recovered from cross-run artifacts."""
        budget = max(1, int(max_items))
        recovered: list[str] = []
        seen: set[str] = set()

        def add_items(items: list[str]) -> bool:
            for item in items:
                normalized = re.sub(r"\s+", " ", item).strip().lower()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                recovered.append(item)
                if len(recovered) >= budget:
                    return False
            return True

        markdown_sources: list[tuple[Path, str]] = [
            (self.path_for("WISHLIST.md"), "logs/WISHLIST.md"),
            (self.path_for("PROGRESS.md"), "logs/PROGRESS.md"),
            (self.path_for("ERRORS.md"), "logs/ERRORS.md"),
            (self.path_for("RESEARCH.md"), "logs/RESEARCH.md"),
            (self.path_for("EXPERIMENTS.md"), "logs/EXPERIMENTS.md"),
            (self.path_for("SCIENTIST_REPORT.md"), "logs/SCIENTIST_REPORT.md"),
            (
                self.repo_path / ".codex_manager" / "owner" / "TODO_WISHLIST.md",
                "owner/TODO_WISHLIST.md",
            ),
            (
                self.repo_path / ".codex_manager" / "owner" / "FEATURE_DREAMS.md",
                "owner/FEATURE_DREAMS.md",
            ),
            (
                self.repo_path / "docs" / "REQUESTED_FEATURES_TODO.md",
                "docs/REQUESTED_FEATURES_TODO.md",
            ),
        ]

        for path, source in markdown_sources:
            remaining = budget - len(recovered)
            if remaining <= 0:
                break
            items = self._collect_markdown_recovery_items(
                path,
                source=source,
                max_items=remaining,
            )
            if not add_items(items):
                break

        if len(recovered) < budget:
            add_items(
                self._collect_general_request_history_items(max_items=budget - len(recovered))
            )
        if len(recovered) < budget:
            add_items(self._collect_history_recovery_items(max_items=budget - len(recovered)))
        if len(recovered) < budget:
            add_items(self._collect_output_history_recovery_items(max_items=budget - len(recovered)))

        if not recovered:
            return ""

        lines = [
            "## Recovered Pending Backlog (Cross-Run)",
            "",
            "Carry forward unresolved items from logs, owner docs, requests, and archives:",
        ]
        for item in recovered[:budget]:
            lines.append(f"- {item}")
        return "\n".join(lines)

    def get_context_for_phase(
        self,
        phase: str,
        ledger: Any = None,
    ) -> str:
        """Build context string for a pipeline phase.

        Returns relevant sections from log files (and optionally the
        project knowledge ledger) that the phase should be aware of.
        """
        from codex_manager.pipeline.phases import PHASE_LOG_FILES, PipelinePhase

        parts: list[str] = []

        try:
            phase_enum = PipelinePhase(phase)
        except ValueError:
            return ""

        # Knowledge ledger context (open items) when provided
        if ledger is not None:
            ledger_categories: list[str] | None = None
            if phase_enum == PipelinePhase.DEBUGGING:
                ledger_categories = ["error", "bug", "observation"]
            elif phase_enum == PipelinePhase.IDEATION:
                ledger_categories = ["suggestion", "wishlist", "feature"]
            elif phase_enum == PipelinePhase.DEEP_RESEARCH:
                ledger_categories = ["feature", "wishlist", "suggestion", "todo", "observation"]
            elif phase_enum in (PipelinePhase.IMPLEMENTATION, PipelinePhase.PRIORITIZATION):
                ledger_categories = ["todo", "feature", "suggestion", "wishlist"]
            elif phase_enum == PipelinePhase.TESTING:
                ledger_categories = ["observation", "bug", "error"]
            if ledger_categories:
                ledger_ctx = ledger.get_context_for_prompt(
                    categories=ledger_categories,
                    max_items=25,
                )
                if ledger_ctx:
                    parts.append(ledger_ctx)

        # Always include the primary log file
        primary = PHASE_LOG_FILES.get(phase_enum)
        if primary:
            content = self.read(primary)
            if content:
                parts.append(f"## Current {primary}\n\n{content[:5000]}")

        # Phase-specific extra context
        if phase_enum in (
            PipelinePhase.IMPLEMENTATION,
            PipelinePhase.DEBUGGING,
        ):
            # Implementation needs to know about errors too
            errors = self.read("ERRORS.md")
            if errors:
                parts.append(f"## Current ERRORS.md\n\n{errors[:2000]}")

        if phase_enum == PipelinePhase.IMPLEMENTATION:
            experiments = self.read("EXPERIMENTS.md")
            if experiments:
                parts.append(f"## Current EXPERIMENTS.md\n\n{experiments[:2500]}")
            research = self.read("RESEARCH.md")
            if research:
                parts.append(f"## Current RESEARCH.md\n\n{research[:2500]}")
            progress = self.read("PROGRESS.md")
            if progress:
                parts.append(f"## Recent PROGRESS.md\n\n{progress[-2200:]}")

        if phase_enum in (
            PipelinePhase.PRIORITIZATION,
            PipelinePhase.IMPLEMENTATION,
            PipelinePhase.DEBUGGING,
        ):
            science_report = self.read("SCIENTIST_REPORT.md")
            if science_report:
                parts.append(f"## Current SCIENTIST_REPORT.md\n\n{science_report[:4000]}")

        if phase_enum == PipelinePhase.PROGRESS_REVIEW:
            # Progress review needs everything
            for fname in ("WISHLIST.md", "TESTPLAN.md", "ERRORS.md", "EXPERIMENTS.md", "RESEARCH.md"):
                content = self.read(fname)
                if content:
                    parts.append(f"## Current {fname}\n\n{content[:2000]}")

        if phase_enum == PipelinePhase.APPLY_UPGRADES_AND_RESTART:
            for fname in ("WISHLIST.md", "EXPERIMENTS.md", "RESEARCH.md", "SCIENTIST_REPORT.md"):
                content = self.read(fname)
                if content:
                    parts.append(f"## Current {fname}\n\n{content[:2200]}")

        if phase_enum == PipelinePhase.TESTING:
            # Testing needs to know what was implemented
            progress = self.read("PROGRESS.md")
            if progress:
                parts.append(f"## Recent PROGRESS.md\n\n{progress[-2000:]}")

        protocol = self.repo_path / ".codex_manager" / "AGENT_PROTOCOL.md"
        if protocol.exists():
            text = self._read_text_utf8_resilient(protocol)
            if text:
                parts.append(f"## Agent Protocol\n\n{text[:2200]}")

        if phase_enum in (
            PipelinePhase.PRIORITIZATION,
            PipelinePhase.IMPLEMENTATION,
            PipelinePhase.DEBUGGING,
            PipelinePhase.DEEP_RESEARCH,
            PipelinePhase.PROGRESS_REVIEW,
            PipelinePhase.APPLY_UPGRADES_AND_RESTART,
        ):
            recovered = self.get_recovered_backlog_context(
                max_items=24 if phase_enum == PipelinePhase.APPLY_UPGRADES_AND_RESTART else 16
            )
            if recovered:
                parts.append(recovered)

        return "\n\n---\n\n".join(parts)

    def log_phase_result(
        self,
        phase: str,
        iteration: int,
        success: bool,
        summary: str,
    ) -> None:
        """Append a phase execution result to PROGRESS.md."""
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        status = "SUCCESS" if success else "FAILED"
        entry = f"\n### [{status}] {phase} (iteration {iteration}) — {timestamp}\n{summary}\n"
        self.append("PROGRESS.md", entry)
