"""Tests for KnowledgeLedger persistence, querying, and context rendering."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from codex_manager.ledger import KnowledgeLedger


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    return repo


def test_add_query_and_stats(tmp_path: Path) -> None:
    ledger = KnowledgeLedger(_make_repo(tmp_path))
    first = ledger.add(
        category="error",
        title="Crash on startup",
        detail="traceback",
        severity="major",
        source="pipeline:debugging",
        file_path="app.py",
        step_ref="step-1",
    )
    second = ledger.add(
        category="suggestion",
        title="Improve docs",
        detail="Add usage examples",
        severity="minor",
    )

    assert first.id == "LED-001"
    assert second.id == "LED-002"
    assert first.created_at
    assert second.updated_at

    recent = ledger.query(limit=1)
    assert [entry.id for entry in recent] == ["LED-002"]

    stats = ledger.stats()
    assert stats.total_entries == 2
    assert stats.open_count == 2
    assert stats.resolved_count == 0
    assert stats.by_category == {"error": 1, "suggestion": 1}
    assert stats.by_severity == {"major": 1, "minor": 1}

    by_status = ledger.query(status="open")
    assert {entry.id for entry in by_status} == {"LED-001", "LED-002"}
    assert ledger.get_entry("LED-001") is not None
    assert ledger.get_entry("LED-999") is None


def test_load_deduplicates_latest_entry_and_skips_invalid_lines(caplog, tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    entries_path = repo / ".codex_manager" / "ledger" / "entries.jsonl"
    entries_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        {
            "id": "LED-001",
            "category": "bug",
            "severity": "major",
            "status": "open",
            "title": "Old state",
            "detail": "first",
            "source": "phase:a",
        },
        {
            "id": "LED-001",
            "category": "bug",
            "severity": "major",
            "status": "resolved",
            "title": "New state",
            "detail": "second",
            "source": "phase:b",
            "resolution": "fixed",
        },
    ]
    entries_path.write_text(
        json.dumps(lines[0]) + "\n" + "not-json\n" + json.dumps(lines[1]) + "\n",
        encoding="utf-8",
    )

    with caplog.at_level("WARNING"):
        ledger = KnowledgeLedger(repo)

    assert "Skip invalid ledger line" in caplog.text
    loaded = ledger.get_entry("LED-001")
    assert loaded is not None
    assert loaded.status == "resolved"
    assert loaded.title == "New state"
    assert ledger.stats().total_entries == 1
    assert ledger.stats().resolved_count == 1


def test_resolve_and_update_status(tmp_path: Path) -> None:
    ledger = KnowledgeLedger(_make_repo(tmp_path))
    entry = ledger.add(
        category="todo",
        title="Implement feature",
        detail="needs design",
        source="pipeline:implementation",
    )

    assert ledger.resolve(entry.id, "implemented", source="agent:codex") is True
    resolved = ledger.get_entry(entry.id)
    assert resolved is not None
    assert resolved.status == "resolved"
    assert resolved.resolution == "implemented"
    assert "resolved by agent:codex" in resolved.source

    assert ledger.update_status(entry.id, "deferred", note="postpone") is True
    deferred = ledger.get_entry(entry.id)
    assert deferred is not None
    assert deferred.status == "deferred"
    assert deferred.resolution == "postpone"

    assert ledger.update_status(entry.id, "not-a-status") is False
    assert ledger.resolve("LED-999", "n/a") is False


def test_open_helpers_and_prompt_context_rendering(tmp_path: Path) -> None:
    ledger = KnowledgeLedger(_make_repo(tmp_path))
    long_detail = "x" * 350
    err = ledger.add(
        category="error",
        title="Failure",
        detail=long_detail,
        severity="critical",
        source="cua:openai",
        file_path="ui/page.tsx",
    )
    bug = ledger.add(
        category="bug",
        title="Edge case bug",
        detail="repro steps",
        severity="major",
    )
    suggestion = ledger.add(
        category="suggestion",
        title="Nice improvement",
        detail="add linting",
    )

    assert ledger.update_status(bug.id, "in_progress")
    assert ledger.update_status(suggestion.id, "resolved")

    open_errors = ledger.get_open_errors()
    assert {entry.id for entry in open_errors} == {err.id, bug.id}
    assert all(entry.category in {"error", "bug"} for entry in open_errors)

    open_suggestions = ledger.get_open_suggestions()
    assert open_suggestions == []

    context = ledger.get_context_for_prompt(categories=["error", "bug"], max_items=5)
    assert "## Open Items (Knowledge Ledger)" in context
    assert err.id in context
    assert "Source: cua:openai" in context
    assert "File: ui/page.tsx" in context
    assert "..." in context
    assert "When you fix an issue" in context

    resolved_only = ledger.get_context_for_prompt(categories=["suggestion"], statuses=["resolved"])
    assert suggestion.id in resolved_only


def test_context_empty_and_stats_rebuild_when_index_missing(tmp_path: Path) -> None:
    ledger = KnowledgeLedger(_make_repo(tmp_path))
    assert ledger.get_context_for_prompt() == ""

    ledger.add(category="feature", title="New feature", detail="todo")
    ledger._index = None
    stats = ledger.stats()
    assert stats.total_entries == 1
    assert stats.open_count == 1


def test_open_suggestions_includes_wishlist_and_feature_categories(tmp_path: Path) -> None:
    ledger = KnowledgeLedger(_make_repo(tmp_path))
    wishlist = ledger.add(
        category="wishlist",
        title="Wishlist item",
        detail="candidate",
    )
    suggestion = ledger.add(
        category="suggestion",
        title="Suggestion item",
        detail="candidate",
    )
    feature = ledger.add(
        category="feature",
        title="Feature item",
        detail="candidate",
    )
    ledger.update_status(feature.id, "resolved")

    open_suggestions = ledger.get_open_suggestions()
    assert {entry.id for entry in open_suggestions} == {wishlist.id, suggestion.id}
    assert all(
        entry.category in {"wishlist", "suggestion", "feature"} for entry in open_suggestions
    )


def test_add_is_thread_safe_under_concurrent_writes(monkeypatch, tmp_path: Path) -> None:
    ledger = KnowledgeLedger(_make_repo(tmp_path))
    original_append = ledger._append_entry

    def slow_append(entry) -> None:
        # Widen the race window so this test reliably catches missing locks.
        time.sleep(0.01)
        original_append(entry)

    monkeypatch.setattr(ledger, "_append_entry", slow_append)

    workers = 12
    start_barrier = threading.Barrier(workers)

    def add_entry(i: int):
        start_barrier.wait(timeout=5)
        return ledger.add(
            category="error",
            title=f"Concurrent issue {i}",
            detail="race check",
            severity="minor",
            source="test",
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        created = list(pool.map(add_entry, range(workers)))

    ids = [entry.id for entry in created]
    assert len(ids) == workers
    assert len(set(ids)) == workers
    assert ledger.stats().total_entries == workers

    reloaded = KnowledgeLedger(ledger.repo_path)
    assert reloaded.stats().total_entries == workers
