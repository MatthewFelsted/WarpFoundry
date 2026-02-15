"""Persistent logbook for brain observations and control decisions.

The logbook keeps two artifacts under ``.codex_manager/logs``:
- ``BRAIN.md``: human-readable timeline
- ``BRAIN.jsonl``: structured machine-readable events

Both files are auto-rotated into ``.codex_manager/logs/archive/brain`` when
they exceed a configured size threshold.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Any

from codex_manager.logbook_utils import (
    SanitizeOptions,
    append_jsonl,
    ensure_log_paths,
    rotate_if_needed,
    sanitize_json_value,
    truncate_text,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_BYTES = 512_000
_DEFAULT_MAX_ARCHIVES = 10
_SANITIZE_OPTIONS = SanitizeOptions(
    max_depth=4,
    max_list_items=25,
    max_dict_items=40,
    max_key_len=80,
    max_str_len=1200,
    fallback_repr_len=300,
)


class BrainLogbook:
    """Append-only brain event log with archive rotation."""

    def __init__(
        self,
        repo_path: str | Path,
        *,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        max_archives: int = _DEFAULT_MAX_ARCHIVES,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.logs_dir = self.repo_path / ".codex_manager" / "logs"
        self.archive_dir = self.logs_dir / "archive" / "brain"
        self.markdown_path = self.logs_dir / "BRAIN.md"
        self.jsonl_path = self.logs_dir / "BRAIN.jsonl"
        self.max_bytes = max(64_000, int(max_bytes))
        self.max_archives = max(1, int(max_archives))
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """Ensure logbook files exist."""
        with self._lock:
            self._ensure_paths()

    def record(
        self,
        *,
        scope: str,
        event: str,
        summary: str,
        level: str = "info",
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record one brain event entry.

        Any write failure is swallowed to avoid impacting runtime execution.
        """
        payload = {
            "id": f"brain_{uuid.uuid4().hex[:12]}",
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "scope": truncate_text(scope, 40),
            "event": truncate_text(event, 80),
            "level": truncate_text(level, 16),
            "summary": truncate_text(summary, 600),
            "context": sanitize_json_value(context or {}, options=_SANITIZE_OPTIONS),
        }

        try:
            with self._lock:
                self._ensure_paths()
                self._rotate_if_needed(self.markdown_path)
                self._rotate_if_needed(self.jsonl_path)
                self._append_markdown(payload)
                self._append_jsonl(payload)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not append brain log entry: %s", exc)

    def _ensure_paths(self) -> None:
        ensure_log_paths(
            logs_dir=self.logs_dir,
            archive_dir=self.archive_dir,
            markdown_path=self.markdown_path,
            jsonl_path=self.jsonl_path,
            markdown_header=self._markdown_header(),
        )

    @staticmethod
    def _markdown_header() -> str:
        return "# BRAIN LOG\n\n> Auto-maintained brain observations and control decisions.\n\n"

    def _rotate_if_needed(self, path: Path) -> None:
        rotate_if_needed(
            path=path,
            max_bytes=self.max_bytes,
            archive_dir=self.archive_dir,
            markdown_header=self._markdown_header(),
            max_archives=self.max_archives,
        )

    def _append_markdown(self, payload: dict[str, Any]) -> None:
        context_dump = json.dumps(payload["context"], indent=2, ensure_ascii=True)
        entry = (
            f"## {payload['timestamp']} | {payload['scope']} | {payload['event']}\n\n"
            f"- **Level**: `{payload['level']}`\n"
            f"- **Summary**: {payload['summary']}\n"
            f"- **Event ID**: `{payload['id']}`\n\n"
            f"```json\n{context_dump}\n```\n\n"
        )
        with self.markdown_path.open("a", encoding="utf-8") as f:
            f.write(entry)

    def _append_jsonl(self, payload: dict[str, Any]) -> None:
        append_jsonl(self.jsonl_path, payload)
