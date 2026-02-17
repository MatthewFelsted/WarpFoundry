"""Additional month-rotation edge tests for the history logbook."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import codex_manager.history_log as history_log_module
from codex_manager.history_log import HistoryLogbook

pytestmark = pytest.mark.integration


def test_rotate_if_month_changed_rotates_active_logs_and_updates_meta(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    logbook = HistoryLogbook(repo)
    logbook.initialize()

    logbook.markdown_path.write_text(
        logbook._markdown_header() + "entry\nentry\n",
        encoding="utf-8",
    )
    logbook.jsonl_path.write_text('{"id":"one"}\n', encoding="utf-8")
    logbook.meta_path.write_text('{"active_month":"2000-01"}', encoding="utf-8")

    rotated: list[Path] = []
    monkeypatch.setattr(logbook, "_rotate_file", lambda path: rotated.append(path))

    logbook._rotate_if_month_changed()

    assert logbook.markdown_path in rotated
    assert logbook.jsonl_path in rotated
    meta = json.loads(logbook.meta_path.read_text(encoding="utf-8"))
    expected_month = history_log_module.dt.datetime.now(history_log_module.dt.timezone.utc).strftime(
        "%Y-%m"
    )
    assert meta["active_month"] == expected_month


def test_rotate_if_month_changed_ignores_malformed_meta(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    logbook = HistoryLogbook(repo)
    logbook.initialize()
    logbook.markdown_path.write_text(
        logbook._markdown_header() + "entry\nentry\n",
        encoding="utf-8",
    )
    logbook.jsonl_path.write_text('{"id":"one"}\n', encoding="utf-8")
    logbook.meta_path.write_text("{not-json", encoding="utf-8")

    rotated: list[Path] = []
    monkeypatch.setattr(logbook, "_rotate_file", lambda path: rotated.append(path))

    logbook._rotate_if_month_changed()

    assert rotated == []
