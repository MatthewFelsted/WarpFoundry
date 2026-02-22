"""Additional retention-policy edge cases for runtime artifact cleanup."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from codex_manager.artifact_retention import RetentionPolicy, cleanup_runtime_artifacts

pytestmark = pytest.mark.unit


def _write_bytes(path: Path, *, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def _set_age_seconds(path: Path, *, seconds_ago: int) -> None:
    ts = time.time() - seconds_ago
    os.utime(path, (ts, ts))


def test_cleanup_runtime_artifacts_noop_when_manager_root_missing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    result = cleanup_runtime_artifacts(repo, policy=RetentionPolicy(enabled=True))

    assert result == {
        "removed_files": 0,
        "removed_dirs": 0,
        "removed_runs": 0,
        "freed_bytes": 0,
    }


def test_cleanup_runtime_artifacts_prunes_oldest_files_to_satisfy_byte_budget(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    archive = repo / ".codex_manager" / "logs" / "archive"
    oldest = archive / "oldest.log"
    middle = archive / "middle.log"
    newest = archive / "newest.log"

    _write_bytes(oldest, size=5)
    _write_bytes(middle, size=5)
    _write_bytes(newest, size=5)
    _set_age_seconds(oldest, seconds_ago=120)
    _set_age_seconds(middle, seconds_ago=60)
    _set_age_seconds(newest, seconds_ago=30)

    result = cleanup_runtime_artifacts(
        repo,
        policy=RetentionPolicy(
            enabled=True,
            max_age_days=365,
            max_files=100,
            max_bytes=10,
            max_output_history_runs=10,
        ),
    )

    assert result["removed_files"] == 1
    assert result["freed_bytes"] == 5
    assert not oldest.exists()
    assert middle.exists()
    assert newest.exists()


def test_cleanup_runtime_artifacts_keeps_active_empty_directories(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    manager_cache = repo / ".codex_manager" / "cache"
    active_run = manager_cache / "active-run"
    stale_empty = manager_cache / "stale-empty"
    active_run.mkdir(parents=True, exist_ok=True)
    stale_empty.mkdir(parents=True, exist_ok=True)

    result = cleanup_runtime_artifacts(
        repo,
        policy=RetentionPolicy(
            enabled=True,
            max_age_days=365,
            max_files=100,
            max_bytes=100_000,
            max_output_history_runs=10,
        ),
        active_paths=[active_run],
    )

    assert active_run.exists()
    assert not stale_empty.exists()
    assert result["removed_dirs"] >= 1


def test_cleanup_runtime_artifacts_continues_when_file_unlink_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    archive = repo / ".codex_manager" / "logs" / "archive"
    protected = archive / "protected.log"
    removable = archive / "removable.log"

    _write_bytes(protected, size=8)
    _write_bytes(removable, size=8)
    _set_age_seconds(protected, seconds_ago=120)
    _set_age_seconds(removable, seconds_ago=60)

    original_unlink = Path.unlink

    def guarded_unlink(self: Path, missing_ok: bool = False) -> None:
        if self.resolve() == protected.resolve():
            raise PermissionError("cannot delete protected file")
        original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", guarded_unlink)

    result = cleanup_runtime_artifacts(
        repo,
        policy=RetentionPolicy(
            enabled=True,
            max_age_days=365,
            max_files=100,
            max_bytes=8,
            max_output_history_runs=10,
        ),
    )

    assert protected.exists()
    assert not removable.exists()
    assert result["removed_files"] == 1


def test_cleanup_runtime_artifacts_skips_unstatable_files_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    archive = repo / ".codex_manager" / "logs" / "archive"
    broken = archive / "broken.log"
    removable = archive / "removable.log"

    _write_bytes(broken, size=4)
    _write_bytes(removable, size=4)
    _set_age_seconds(broken, seconds_ago=30)
    _set_age_seconds(removable, seconds_ago=30)

    broken_str = str(broken)
    original_stat = Path.stat

    def flaky_stat(self: Path, *args: object, **kwargs: object):  # type: ignore[override]
        if str(self) == broken_str:
            raise OSError("stat failure")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)

    result = cleanup_runtime_artifacts(
        repo,
        policy=RetentionPolicy(
            enabled=True,
            max_age_days=365,
            max_files=100,
            max_bytes=1,
            max_output_history_runs=10,
        ),
    )
    monkeypatch.setattr(Path, "stat", original_stat)

    assert broken.exists()
    assert not removable.exists()
    assert result["removed_files"] == 1
