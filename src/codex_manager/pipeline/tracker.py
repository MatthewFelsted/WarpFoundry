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
