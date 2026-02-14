"""Tests for persistent run-history logging."""

from __future__ import annotations

import json
from pathlib import Path

from codex_manager.history_log import HistoryLogbook


def test_history_logbook_writes_markdown_and_jsonl(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    logbook = HistoryLogbook(repo)
    logbook.initialize()
    logbook.record(
        scope="pipeline",
        event="phase_result",
        summary="Ideation phase completed.",
        context={
            "phase": "ideation",
            "files_changed": 2,
            "changed_files": [
                {"path": "README.md", "insertions": 8, "deletions": 1},
                {"path": "docs/plan.md", "insertions": 12, "deletions": 0},
            ],
        },
    )

    md_path = repo / ".codex_manager" / "logs" / "HISTORY.md"
    jsonl_path = repo / ".codex_manager" / "logs" / "HISTORY.jsonl"

    assert md_path.exists()
    assert jsonl_path.exists()
    markdown = md_path.read_text(encoding="utf-8")
    assert "Ideation phase completed." in markdown
    assert "Changed Files" in markdown
    assert "`README.md` | +8 -1" in markdown

    lines = [ln for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines
    payload = json.loads(lines[0])
    assert payload["scope"] == "pipeline"
    assert payload["event"] == "phase_result"
    assert payload["context"]["phase"] == "ideation"


def test_history_logbook_rotates_when_size_limit_exceeded(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    logbook = HistoryLogbook(repo, max_bytes=64_000, max_archives=3)
    logbook.initialize()

    for i in range(50):
        logbook.record(
            scope="chain",
            event=f"bulk_entry_{i}",
            summary="x" * 1_100,
            context={"blob": "y" * 2_000, "idx": i},
        )
    logbook.record(
        scope="chain",
        event="post_rotation",
        summary="rotation trigger",
        context={},
    )

    archive_dir = repo / ".codex_manager" / "logs" / "archive" / "history"
    archived = list(archive_dir.glob("HISTORY-*"))
    assert archived

    active_md = repo / ".codex_manager" / "logs" / "HISTORY.md"
    active_jsonl = repo / ".codex_manager" / "logs" / "HISTORY.jsonl"
    assert active_md.exists()
    assert active_jsonl.exists()
    assert "rotation trigger" in active_md.read_text(encoding="utf-8")
