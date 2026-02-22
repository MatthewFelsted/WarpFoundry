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


@dataclass(frozen=True, slots=True)
class _FileCandidate:
    path: Path
    mtime: float
    size: int


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
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
    for directory in sorted((p for p in root.rglob("*") if _safe_is_dir(p)), reverse=True):
        if _is_active_path(directory, active_roots):
            continue
        try:
            next(directory.iterdir())
        except StopIteration:
            directory.rmdir()
            removed += 1
        except OSError:
            continue
    return removed


def _safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _safe_file_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0


def _safe_file_size(path: Path) -> int:
    try:
        return max(0, int(path.stat().st_size))
    except OSError:
        return 0


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
        key=_safe_file_mtime,
        reverse=True,
    )
    for run_dir in runs[max_runs:]:
        if _is_active_path(run_dir, active_roots):
            continue
        size = sum(_safe_file_size(file_path) for file_path in run_dir.rglob("*") if file_path.is_file())
        shutil.rmtree(run_dir, ignore_errors=True)
        removed_runs += 1
        removed_bytes += int(size)
    return removed_runs, removed_bytes


def _file_candidate(path: Path) -> _FileCandidate | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return _FileCandidate(
        path=path,
        mtime=float(stat.st_mtime),
        size=max(0, int(stat.st_size)),
    )


def _collect_file_candidates(manager_root: Path, active_roots: list[Path]) -> list[_FileCandidate]:
    candidates: list[_FileCandidate] = []
    for root in _candidate_roots(manager_root):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            try:
                is_file = path.is_file()
            except OSError:
                continue
            if not is_file or _is_active_path(path, active_roots):
                continue
            candidate = _file_candidate(path)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _unlink_file(path: Path) -> bool:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def _prune_candidates_by_age(
    candidates: list[_FileCandidate],
    *,
    cutoff: float,
) -> tuple[list[_FileCandidate], int, int]:
    kept: list[_FileCandidate] = []
    removed_files = 0
    removed_bytes = 0
    for candidate in candidates:
        if candidate.mtime >= cutoff:
            kept.append(candidate)
            continue
        if _unlink_file(candidate.path):
            removed_files += 1
            removed_bytes += candidate.size
        else:
            kept.append(candidate)
    return kept, removed_files, removed_bytes


def _prune_candidates_by_limits(
    candidates: list[_FileCandidate],
    *,
    max_files: int,
    max_bytes: int,
) -> tuple[list[_FileCandidate], int, int]:
    ordered = sorted(candidates, key=lambda candidate: candidate.mtime)
    removed_paths: set[Path] = set()
    removed_files = 0
    removed_bytes = 0
    remaining_files = len(ordered)
    remaining_bytes = sum(candidate.size for candidate in ordered)

    for candidate in ordered:
        if remaining_files <= max_files and remaining_bytes <= max_bytes:
            break
        if not _unlink_file(candidate.path):
            continue
        removed_paths.add(candidate.path)
        removed_files += 1
        removed_bytes += candidate.size
        remaining_files -= 1
        remaining_bytes = max(0, remaining_bytes - candidate.size)

    if not removed_paths:
        return ordered, 0, 0
    kept = [candidate for candidate in ordered if candidate.path not in removed_paths]
    return kept, removed_files, removed_bytes


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
    files = _collect_file_candidates(manager_root, active_roots)
    files, removed_by_age, bytes_by_age = _prune_candidates_by_age(files, cutoff=cutoff)
    removed_files += removed_by_age
    freed_bytes += bytes_by_age

    # Count/size caps prune oldest first.
    max_files = max(1, int(policy.max_files))
    max_bytes = max(1, int(policy.max_bytes))
    files, removed_by_limit, bytes_by_limit = _prune_candidates_by_limits(
        files,
        max_files=max_files,
        max_bytes=max_bytes,
    )
    removed_files += removed_by_limit
    freed_bytes += bytes_by_limit

    for root in _candidate_roots(manager_root):
        removed_dirs += _prune_empty_dirs(root, active_roots)

    return {
        "removed_files": removed_files,
        "removed_dirs": removed_dirs,
        "removed_runs": removed_runs,
        "freed_bytes": max(0, int(freed_bytes)),
    }
