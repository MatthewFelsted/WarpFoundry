"""Tests for runtime artifact retention cleanup policy."""

from __future__ import annotations

import os
import time
from pathlib import Path

from codex_manager.artifact_retention import RetentionPolicy, cleanup_runtime_artifacts


def _write_bytes(path: Path, *, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def _set_age_seconds(path: Path, *, seconds_ago: int) -> None:
    ts = time.time() - seconds_ago
    os.utime(path, (ts, ts))


def test_cleanup_runtime_artifacts_prunes_old_files_and_respects_caps(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    archive_root = repo / ".codex_manager" / "logs" / "archive"
    old_file = archive_root / "old.log"
    recent_a = archive_root / "recent-a.log"
    recent_b = archive_root / "recent-b.log"
    recent_c = archive_root / "recent-c.log"

    _write_bytes(old_file, size=4)
    _write_bytes(recent_a, size=6)
    _write_bytes(recent_b, size=6)
    _write_bytes(recent_c, size=6)

    _set_age_seconds(old_file, seconds_ago=10 * 86_400)
    _set_age_seconds(recent_a, seconds_ago=90)
    _set_age_seconds(recent_b, seconds_ago=60)
    _set_age_seconds(recent_c, seconds_ago=30)

    policy = RetentionPolicy(
        enabled=True,
        max_age_days=1,
        max_files=2,
        max_bytes=12,
        max_output_history_runs=30,
    )

    result = cleanup_runtime_artifacts(repo, policy=policy)

    assert result["removed_files"] >= 2
    assert not old_file.exists()
    assert not recent_a.exists()
    assert recent_b.exists()
    assert recent_c.exists()


def test_cleanup_runtime_artifacts_keeps_active_output_history_runs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    history_root = repo / ".codex_manager" / "output_history"
    run_old_active = history_root / "2026-02-15T000000-active"
    run_old_remove = history_root / "2026-02-15T010000-remove"
    run_new_keep = history_root / "2026-02-15T020000-keep"

    _write_bytes(run_old_active / "report.md", size=2)
    _write_bytes(run_old_remove / "report.md", size=2)
    _write_bytes(run_new_keep / "report.md", size=2)

    _set_age_seconds(run_old_active, seconds_ago=300)
    _set_age_seconds(run_old_remove, seconds_ago=200)
    _set_age_seconds(run_new_keep, seconds_ago=100)

    policy = RetentionPolicy(
        enabled=True,
        max_age_days=90,
        max_files=500,
        max_bytes=10_000,
        max_output_history_runs=1,
    )
    result = cleanup_runtime_artifacts(
        repo,
        policy=policy,
        active_paths=[run_old_active],
    )

    assert result["removed_runs"] == 1
    assert run_old_active.exists()
    assert not run_old_remove.exists()
    assert run_new_keep.exists()


def test_cleanup_runtime_artifacts_disabled_is_noop(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    result = cleanup_runtime_artifacts(repo, policy=RetentionPolicy(enabled=False))

    assert result == {
        "removed_files": 0,
        "removed_dirs": 0,
        "removed_runs": 0,
        "freed_bytes": 0,
    }
