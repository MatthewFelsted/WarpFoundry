"""Tests for persistent brain observation logging."""

from __future__ import annotations

import json
from pathlib import Path

from codex_manager.brain.logbook import BrainLogbook


def test_brain_logbook_writes_markdown_and_jsonl(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    logbook = BrainLogbook(repo)
    logbook.initialize()
    logbook.record(
        scope="chain",
        event="evaluate_step",
        summary="Brain selected follow_up",
        context={"loop": 1, "step": "implementation", "action": "follow_up"},
    )

    md_path = repo / ".codex_manager" / "logs" / "BRAIN.md"
    jsonl_path = repo / ".codex_manager" / "logs" / "BRAIN.jsonl"

    assert md_path.exists()
    assert jsonl_path.exists()
    assert "Brain selected follow_up" in md_path.read_text(encoding="utf-8")

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert lines
    payload = json.loads(lines[0])
    assert payload["scope"] == "chain"
    assert payload["event"] == "evaluate_step"
    assert payload["context"]["action"] == "follow_up"


def test_brain_logbook_rotates_when_size_limit_exceeded(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    logbook = BrainLogbook(repo, max_bytes=64_000, max_archives=3)
    logbook.initialize()

    # Inflate the active log beyond threshold.
    for i in range(45):
        logbook.record(
            scope="pipeline",
            event=f"bulk_entry_{i}",
            summary="x" * 1_200,
            context={"blob": "y" * 2_000, "idx": i},
        )
    # Next record triggers pre-append rotation.
    logbook.record(
        scope="pipeline",
        event="post_rotation",
        summary="rotation trigger",
        context={},
    )

    archive_dir = repo / ".codex_manager" / "logs" / "archive" / "brain"
    archived = list(archive_dir.glob("BRAIN-*"))
    assert archived

    active_md = repo / ".codex_manager" / "logs" / "BRAIN.md"
    active_jsonl = repo / ".codex_manager" / "logs" / "BRAIN.jsonl"
    assert active_md.exists()
    assert active_jsonl.exists()
    assert "rotation trigger" in active_md.read_text(encoding="utf-8")
