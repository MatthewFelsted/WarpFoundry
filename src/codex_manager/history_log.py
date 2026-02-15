"""Persistent execution history log with archive rotation.

Writes human-readable and machine-readable history entries for chain/pipeline
runs under ``.codex_manager/logs``.
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
    rotate_file,
    rotate_if_needed,
    sanitize_json_value,
    truncate_text,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_BYTES = 1_200_000
_DEFAULT_MAX_ARCHIVES = 24
_SANITIZE_OPTIONS = SanitizeOptions(
    max_depth=5,
    max_list_items=80,
    max_dict_items=80,
    max_key_len=120,
    max_str_len=2000,
    fallback_repr_len=400,
)


class HistoryLogbook:
    """Append-only run history log with archive rotation."""

    def __init__(
        self,
        repo_path: str | Path,
        *,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        max_archives: int = _DEFAULT_MAX_ARCHIVES,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.logs_dir = self.repo_path / ".codex_manager" / "logs"
        self.archive_dir = self.logs_dir / "archive" / "history"
        self.markdown_path = self.logs_dir / "HISTORY.md"
        self.jsonl_path = self.logs_dir / "HISTORY.jsonl"
        self.meta_path = self.logs_dir / "HISTORY.meta.json"
        self.max_bytes = max(128_000, int(max_bytes))
        self.max_archives = max(1, int(max_archives))
        self._lock = threading.Lock()

    def initialize(self) -> None:
        with self._lock:
            self._ensure_paths()
            self._rotate_if_month_changed()
            self._rotate_if_needed(self.markdown_path)
            self._rotate_if_needed(self.jsonl_path)

    def record(
        self,
        *,
        scope: str,
        event: str,
        summary: str,
        level: str = "info",
        context: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "id": f"hist_{uuid.uuid4().hex[:12]}",
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "scope": truncate_text(scope, 40),
            "event": truncate_text(event, 80),
            "level": truncate_text(level, 16),
            "summary": truncate_text(summary, 800),
            "context": sanitize_json_value(context or {}, options=_SANITIZE_OPTIONS),
        }

        try:
            with self._lock:
                self._ensure_paths()
                self._rotate_if_month_changed()
                self._rotate_if_needed(self.markdown_path)
                self._rotate_if_needed(self.jsonl_path)
                self._append_markdown(payload)
                self._append_jsonl(payload)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not append history log entry: %s", exc)

    def _ensure_paths(self) -> None:
        ensure_log_paths(
            logs_dir=self.logs_dir,
            archive_dir=self.archive_dir,
            markdown_path=self.markdown_path,
            jsonl_path=self.jsonl_path,
            markdown_header=self._markdown_header(),
        )
        if not self.meta_path.exists():
            month = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m")
            self.meta_path.write_text(json.dumps({"active_month": month}), encoding="utf-8")

    @staticmethod
    def _markdown_header() -> str:
        return (
            "# RUN HISTORY\n\n> Auto-maintained execution history for chain and pipeline runs.\n\n"
        )

    def _rotate_if_month_changed(self) -> None:
        now_month = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m")
        try:
            raw = self.meta_path.read_text(encoding="utf-8")
            meta = json.loads(raw) if raw.strip() else {}
            prev_month = str(meta.get("active_month") or now_month)
        except Exception:
            prev_month = now_month

        if prev_month == now_month:
            return

        if (
            self.markdown_path.exists()
            and self.markdown_path.stat().st_size > len(self._markdown_header()) + 8
        ):
            self._rotate_file(self.markdown_path)
        if self.jsonl_path.exists() and self.jsonl_path.stat().st_size > 0:
            self._rotate_file(self.jsonl_path)

        self.meta_path.write_text(json.dumps({"active_month": now_month}), encoding="utf-8")

    def _rotate_if_needed(self, path: Path) -> None:
        rotate_if_needed(
            path=path,
            max_bytes=self.max_bytes,
            archive_dir=self.archive_dir,
            markdown_header=self._markdown_header(),
            max_archives=self.max_archives,
        )

    def _rotate_file(self, path: Path) -> None:
        rotate_file(
            path=path,
            archive_dir=self.archive_dir,
            markdown_header=self._markdown_header(),
            max_archives=self.max_archives,
        )

    def _append_markdown(self, payload: dict[str, Any]) -> None:
        details = self._format_markdown_context(payload["context"])
        entry = (
            f"## {payload['timestamp']} | {payload['scope']} | {payload['event']}\n\n"
            f"- **Level**: `{payload['level']}`\n"
            f"- **Summary**: {payload['summary']}\n"
            f"- **Event ID**: `{payload['id']}`\n\n"
            f"{details}\n\n"
        )
        with self.markdown_path.open("a", encoding="utf-8") as f:
            f.write(entry)

    @staticmethod
    def _format_markdown_context(context: dict[str, Any]) -> str:
        changed = context.get("changed_files")
        if isinstance(changed, list) and changed:
            lines = ["### Changed Files", ""]
            for item in changed[:80]:
                path = str(item.get("path", "(unknown)"))
                ins = item.get("insertions")
                dels = item.get("deletions")
                if isinstance(ins, int) and isinstance(dels, int):
                    lines.append(f"- `{path}` | +{ins} -{dels}")
                else:
                    lines.append(f"- `{path}` | binary/non-text")
            json_copy = dict(context)
            json_copy.pop("changed_files", None)
            lines.append("")
            lines.append("### Context")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(json_copy, indent=2, ensure_ascii=True))
            lines.append("```")
            return "\n".join(lines)

        return "```json\n" + json.dumps(context, indent=2, ensure_ascii=True) + "\n```"

    def _append_jsonl(self, payload: dict[str, Any]) -> None:
        append_jsonl(self.jsonl_path, payload)
