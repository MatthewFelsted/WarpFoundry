"""Retention helpers for managed runtime artifacts under ``.codex_manager``."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """Configurable limits for runtime artifact cleanup."""

    enabled: bool = True
    max_age_days: int = 30
    max_files: int = 5000
    max_bytes: int = 2_000_000_000
    max_output_history_runs: int = 30


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _is_active_path(path: Path, active_roots: list[Path]) -> bool:
    return any(_is_within(path, root) for root in active_roots)


def _candidate_roots(manager_root: Path) -> list[Path]:
    return [
        manager_root / "output_history",
        manager_root / "logs" / "archive",
        manager_root / "logs" / "cua",
        manager_root / "cua_screenshots",
        manager_root / "cache",
    ]


def _prune_empty_dirs(root: Path, active_roots: list[Path]) -> int:
    removed = 0
    if not root.exists():
        return removed
    for directory in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        if _is_active_path(directory, active_roots):
            continue
        try:
            next(directory.iterdir())
        except StopIteration:
            directory.rmdir()
            removed += 1
        except Exception:
            continue
    return removed


def _prune_output_history_runs(
    output_history_root: Path,
    *,
    max_runs: int,
    active_roots: list[Path],
) -> tuple[int, int]:
    removed_runs = 0
    removed_bytes = 0
    if not output_history_root.exists() or max_runs < 1:
        return removed_runs, removed_bytes
    runs = sorted(
        (path for path in output_history_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in runs[max_runs:]:
        if _is_active_path(run_dir, active_roots):
            continue
        size = sum(
            file_path.stat().st_size
            for file_path in run_dir.rglob("*")
            if file_path.is_file()
        )
        shutil.rmtree(run_dir, ignore_errors=True)
        removed_runs += 1
        removed_bytes += int(size)
    return removed_runs, removed_bytes


def cleanup_runtime_artifacts(
    repo_path: str | Path,
    *,
    policy: RetentionPolicy,
    active_paths: list[Path] | None = None,
) -> dict[str, int]:
    """Apply retention policy and return cleanup counters."""
    if not policy.enabled:
        return {
            "removed_files": 0,
            "removed_dirs": 0,
            "removed_runs": 0,
            "freed_bytes": 0,
        }

    repo = Path(repo_path).resolve()
    manager_root = repo / ".codex_manager"
    if not manager_root.exists():
        return {
            "removed_files": 0,
            "removed_dirs": 0,
            "removed_runs": 0,
            "freed_bytes": 0,
        }

    active_roots = [Path(path).resolve() for path in (active_paths or [])]
    removed_files = 0
    removed_dirs = 0
    removed_runs = 0
    freed_bytes = 0

    output_history_root = manager_root / "output_history"
    runs_removed, bytes_removed = _prune_output_history_runs(
        output_history_root,
        max_runs=max(1, int(policy.max_output_history_runs)),
        active_roots=active_roots,
    )
    removed_runs += runs_removed
    freed_bytes += bytes_removed

    now = time.time()
    cutoff = now - max(1, int(policy.max_age_days)) * 86_400

    files: list[Path] = []
    for root in _candidate_roots(manager_root):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if _is_active_path(path, active_roots):
                continue
            files.append(path)

    # Age-based pruning.
    for path in list(files):
        try:
            if path.stat().st_mtime >= cutoff:
                continue
            size = int(path.stat().st_size)
            path.unlink(missing_ok=True)
            freed_bytes += size
            removed_files += 1
            files.remove(path)
        except Exception:
            continue

    # Count/size caps prune oldest first.
    files.sort(key=lambda path: path.stat().st_mtime)
    max_files = max(1, int(policy.max_files))
    max_bytes = max(1, int(policy.max_bytes))
    total_bytes = sum(path.stat().st_size for path in files if path.exists())

    while len(files) > max_files:
        path = files.pop(0)
        try:
            size = int(path.stat().st_size)
            path.unlink(missing_ok=True)
            removed_files += 1
            freed_bytes += size
            total_bytes -= size
        except Exception:
            continue

    while files and total_bytes > max_bytes:
        path = files.pop(0)
        try:
            size = int(path.stat().st_size)
            path.unlink(missing_ok=True)
            removed_files += 1
            freed_bytes += size
            total_bytes -= size
        except Exception:
            continue

    for root in _candidate_roots(manager_root):
        removed_dirs += _prune_empty_dirs(root, active_roots)

    return {
        "removed_files": removed_files,
        "removed_dirs": removed_dirs,
        "removed_runs": removed_runs,
        "freed_bytes": max(0, int(freed_bytes)),
    }
