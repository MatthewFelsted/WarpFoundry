"""Project Knowledge Ledger — per-repo structured store for observations, errors, suggestions.

All parts of the system (CUA, pipeline, chain, brain) read from and write to the ledger
so that findings from one step are visible to later steps (e.g. debugging knows what
CUA found). Entries are append-only; status (open/resolved) is tracked in the index.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LEDGER_DIR = "ledger"
ENTRIES_FILE = "entries.jsonl"
INDEX_FILE = "index.json"

OPEN_STATUSES = frozenset({"open", "in_progress"})
RESOLVED_STATUSES = frozenset({"resolved", "wontfix", "deferred"})


@dataclass
class LedgerEntry:
    """A single ledger entry (error, observation, suggestion, etc.)."""

    id: str
    category: str  # error | bug | observation | suggestion | todo | wishlist | feature
    severity: str  # critical | major | minor | cosmetic | positive | info
    status: str  # open | in_progress | resolved | wontfix | deferred
    title: str
    detail: str
    source: str  # e.g. "cua:anthropic", "pipeline:debugging"
    resolution: str = ""
    file_path: str = ""
    step_ref: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LedgerEntry:
        return cls(
            id=d.get("id", ""),
            category=d.get("category", ""),
            severity=d.get("severity", "info"),
            status=d.get("status", "open"),
            title=d.get("title", ""),
            detail=d.get("detail", ""),
            source=d.get("source", ""),
            resolution=d.get("resolution", ""),
            file_path=d.get("file_path", ""),
            step_ref=d.get("step_ref", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


@dataclass
class LedgerIndex:
    """Derived state: counts and open entries."""

    total_entries: int = 0
    open_count: int = 0
    resolved_count: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    by_severity: dict[str, int] = field(default_factory=dict)
    open_entries: list[LedgerEntry] = field(default_factory=list)


class KnowledgeLedger:
    """Per-repo knowledge ledger: append-only entries + index for queries."""

    def __init__(self, repo_path: str | Path) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.ledger_dir = self.repo_path / ".codex_manager" / LEDGER_DIR
        self.entries_path = self.ledger_dir / ENTRIES_FILE
        self.index_path = self.ledger_dir / INDEX_FILE
        self._entries: list[LedgerEntry] = []
        self._index: LedgerIndex | None = None
        self._lock = threading.RLock()
        self._ensure_dir()
        self._load()

    def _ensure_dir(self) -> None:
        self.ledger_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        """Load entries from JSONL (last occurrence per id wins) and rebuild index."""
        by_id: dict[str, LedgerEntry] = {}
        order: list[str] = []
        if self.entries_path.exists():
            with open(self.entries_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = LedgerEntry.from_dict(json.loads(line))
                        if e.id not in by_id:
                            order.append(e.id)
                        by_id[e.id] = e
                    except (json.JSONDecodeError, KeyError) as ex:
                        logger.warning("Skip invalid ledger line: %s", ex)
        with self._lock:
            self._entries = [by_id[oid] for oid in order]
            self._rebuild_index()

    def _next_id(self) -> str:
        """Generate next LED-XXX id."""
        max_n = 0
        for e in self._entries:
            if e.id.startswith("LED-"):
                try:
                    n = int(e.id.split("-", 1)[1])
                    max_n = max(max_n, n)
                except ValueError:
                    pass
        return f"LED-{max_n + 1:03d}"

    def _rebuild_index(self) -> None:
        """Rebuild index from current entries."""
        by_cat: dict[str, int] = {}
        by_sev: dict[str, int] = {}
        open_entries: list[LedgerEntry] = []
        resolved = 0
        for e in self._entries:
            by_cat[e.category] = by_cat.get(e.category, 0) + 1
            by_sev[e.severity] = by_sev.get(e.severity, 0) + 1
            if e.status in OPEN_STATUSES:
                open_entries.append(e)
            elif e.status in RESOLVED_STATUSES:
                resolved += 1
        self._index = LedgerIndex(
            total_entries=len(self._entries),
            open_count=len(open_entries),
            resolved_count=resolved,
            by_category=by_cat,
            by_severity=by_sev,
            open_entries=open_entries,
        )

    def _append_entry(self, entry: LedgerEntry) -> None:
        """Append one entry and refresh in-memory state.

        Caller must hold ``self._lock``.
        """
        with open(self.entries_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        self._entries.append(entry)
        self._rebuild_index()

    def _write_entry_update(self, entry: LedgerEntry) -> None:
        """Append a status-update line (append-only). In-memory state updated by id."""
        # Caller must hold ``self._lock``.
        with open(self.entries_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        for i, e in enumerate(self._entries):
            if e.id == entry.id:
                self._entries[i] = entry
                break
        else:
            self._entries.append(entry)
        self._rebuild_index()

    def add(
        self,
        category: str,
        title: str,
        detail: str,
        severity: str = "info",
        source: str = "",
        file_path: str = "",
        step_ref: str = "",
    ) -> LedgerEntry:
        """Append a new entry. Returns the created entry."""
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            entry = LedgerEntry(
                id=self._next_id(),
                category=category,
                severity=severity,
                status="open",
                title=title,
                detail=detail,
                source=source,
                resolution="",
                file_path=file_path,
                step_ref=step_ref,
                created_at=now,
                updated_at=now,
            )
            self._append_entry(entry)
        logger.debug("Ledger add: %s [%s] %s", entry.id, category, title[:50])
        return entry

    def resolve(self, entry_id: str, resolution: str, source: str = "") -> bool:
        """Mark entry as resolved with resolution text."""
        with self._lock:
            for e in self._entries:
                if e.id == entry_id:
                    e.status = "resolved"
                    e.resolution = resolution
                    e.updated_at = datetime.now(timezone.utc).isoformat()
                    if source:
                        e.source = e.source + "; resolved by " + source
                    self._write_entry_update(e)
                    return True
        return False

    def update_status(self, entry_id: str, status: str, note: str = "") -> bool:
        """Change status (in_progress, deferred, wontfix)."""
        valid = {"open", "in_progress", "resolved", "wontfix", "deferred"}
        if status not in valid:
            return False
        with self._lock:
            for e in self._entries:
                if e.id == entry_id:
                    e.status = status
                    if note:
                        e.resolution = note
                    e.updated_at = datetime.now(timezone.utc).isoformat()
                    self._write_entry_update(e)
                    return True
        return False

    def get_entry(self, entry_id: str) -> LedgerEntry | None:
        """Return the latest state of an entry (by id)."""
        with self._lock:
            for e in reversed(self._entries):
                if e.id == entry_id:
                    return e
        return None

    def query(
        self,
        category: str | None = None,
        status: str | None = None,
        severity: str | None = None,
        limit: int = 100,
    ) -> list[LedgerEntry]:
        """Filter entries. Returns most recent first (by order in file)."""
        with self._lock:
            result: list[LedgerEntry] = []
            seen_ids: set[str] = set()
            for e in reversed(self._entries):
                if e.id in seen_ids:
                    continue
                seen_ids.add(e.id)
                if category is not None and e.category != category:
                    continue
                if status is not None and e.status != status:
                    continue
                if severity is not None and e.severity != severity:
                    continue
                result.append(e)
                if len(result) >= limit:
                    break
            return result

    def get_open_errors(self) -> list[LedgerEntry]:
        """Open entries that are errors or bugs."""
        with self._lock:
            return [
                e
                for e in (self._index.open_entries if self._index else [])
                if e.category in ("error", "bug")
            ]

    def get_open_suggestions(self) -> list[LedgerEntry]:
        """Open entries that are suggestions, wishlist, or feature."""
        with self._lock:
            return [
                e
                for e in (self._index.open_entries if self._index else [])
                if e.category in ("suggestion", "wishlist", "feature")
            ]

    def get_context_for_prompt(
        self,
        categories: list[str] | None = None,
        statuses: list[str] | None = None,
        max_items: int = 30,
    ) -> str:
        """Formatted markdown for injection into prompts."""
        if statuses is None:
            statuses = ["open", "in_progress"]
        with self._lock:
            entries: list[LedgerEntry] = []
            for e in reversed(self._entries):
                if e.status not in statuses:
                    continue
                if categories is not None and e.category not in categories:
                    continue
                entries.append(e)
                if len(entries) >= max_items:
                    break
        if not entries:
            return ""
        lines = ["## Open Items (Knowledge Ledger)", ""]
        for e in entries:
            lines.append(f"- **{e.id}** [{e.severity.upper()}] [{e.category}] {e.title}")
            lines.append(f"  Detail: {e.detail[:300]}{'...' if len(e.detail) > 300 else ''}")
            if e.source:
                lines.append(f"  Source: {e.source}")
            if e.file_path:
                lines.append(f"  File: {e.file_path}")
            lines.append("")
        lines.append(
            "When you fix an issue, note the LED-ID in your commit or response so it can be marked resolved."
        )
        return "\n".join(lines)

    def stats(self) -> LedgerIndex:
        """Summary counts."""
        with self._lock:
            if self._index is None:
                self._rebuild_index()
            return self._index or LedgerIndex()
