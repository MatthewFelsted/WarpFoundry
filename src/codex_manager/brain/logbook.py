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

logger = logging.getLogger(__name__)

_DEFAULT_MAX_BYTES = 512_000
_DEFAULT_MAX_ARCHIVES = 10


def _truncate(text: str, max_len: int) -> str:
    clean = (text or "").strip()
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


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
            "scope": _truncate(scope, 40),
            "event": _truncate(event, 80),
            "level": _truncate(level, 16),
            "summary": _truncate(summary, 600),
            "context": self._sanitize_context(context or {}),
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
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        if not self.markdown_path.exists():
            self.markdown_path.write_text(self._markdown_header(), encoding="utf-8")
        if not self.jsonl_path.exists():
            self.jsonl_path.touch()

    @staticmethod
    def _markdown_header() -> str:
        return "# BRAIN LOG\n\n> Auto-maintained brain observations and control decisions.\n\n"

    def _rotate_if_needed(self, path: Path) -> None:
        if not path.exists():
            return
        if path.stat().st_size < self.max_bytes:
            return

        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = self.archive_dir / f"{path.stem}-{stamp}{path.suffix}"
        idx = 1
        while target.exists():
            idx += 1
            target = self.archive_dir / f"{path.stem}-{stamp}-{idx}{path.suffix}"

        path.replace(target)
        if path.suffix.lower() == ".md":
            path.write_text(self._markdown_header(), encoding="utf-8")
        else:
            path.touch()

        self._prune_archives(path.stem, path.suffix)

    def _prune_archives(self, stem: str, suffix: str) -> None:
        files = sorted(
            self.archive_dir.glob(f"{stem}-*{suffix}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in files[self.max_archives :]:
            try:
                old.unlink(missing_ok=True)
            except Exception:
                continue

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
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _sanitize_context(self, value: Any, depth: int = 0) -> Any:
        if depth > 4:
            return "[truncated-depth]"
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            if isinstance(value, str):
                return _truncate(value, 1200)
            return value
        if isinstance(value, list):
            return [self._sanitize_context(v, depth + 1) for v in value[:25]]
        if isinstance(value, tuple):
            return [self._sanitize_context(v, depth + 1) for v in value[:25]]
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for i, (k, v) in enumerate(value.items()):
                if i >= 40:
                    out["__truncated__"] = f"{len(value) - 40} more key(s)"
                    break
                out[_truncate(str(k), 80)] = self._sanitize_context(v, depth + 1)
            return out
        return _truncate(repr(value), 300)
