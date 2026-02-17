"""Helpers for counting managed documentation artifacts outside git numstat."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from codex_manager.schemas import EvalResult

_DEFAULT_GLOBS: tuple[str, ...] = (".codex_manager/owner/**/*",)


def capture_artifact_snapshot(
    repo: str | Path,
    *,
    extra_globs: Sequence[str] | None = None,
    extra_rel_paths: Sequence[str] | None = None,
) -> dict[str, str | None]:
    """Capture a text snapshot of managed artifacts under ``repo``."""
    root = Path(repo).resolve()
    candidates = _collect_candidates(
        root,
        extra_globs=extra_globs,
        extra_rel_paths=extra_rel_paths,
    )
    snapshot: dict[str, str | None] = {}
    for rel in sorted(candidates):
        snapshot[rel] = _read_text_or_none(root / rel)
    return snapshot


def summarize_artifact_delta(
    repo: str | Path,
    before_snapshot: Mapping[str, str | None],
    *,
    extra_globs: Sequence[str] | None = None,
    extra_rel_paths: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Summarize line deltas for managed artifacts since ``before_snapshot``."""
    root = Path(repo).resolve()
    candidates = _collect_candidates(
        root,
        extra_globs=extra_globs,
        extra_rel_paths=extra_rel_paths,
    )
    candidates.update(before_snapshot.keys())

    entries: list[dict[str, Any]] = []
    total_insertions = 0
    total_deletions = 0
    for rel in sorted(candidates):
        before_text = before_snapshot.get(rel)
        after_text = _read_text_or_none(root / rel)
        if before_text == after_text:
            continue
        insertions, deletions = _line_delta(before_text, after_text)
        entries.append(
            {
                "path": rel,
                "insertions": insertions,
                "deletions": deletions,
                "source": "managed_artifact",
            }
        )
        total_insertions += insertions
        total_deletions += deletions

    return entries, total_insertions, total_deletions


def merge_eval_result_with_artifact_delta(
    eval_result: EvalResult,
    artifact_entries: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Merge managed artifact entries into an :class:`EvalResult` in-place."""
    existing_paths = {
        str(entry.get("path") or "").strip()
        for entry in eval_result.changed_files
        if str(entry.get("path") or "").strip()
    }

    merged_entries: list[dict[str, Any]] = []
    insertions = 0
    deletions = 0
    for entry in artifact_entries:
        path = str(entry.get("path") or "").strip()
        if not path or path in existing_paths:
            continue
        ins_raw = entry.get("insertions")
        del_raw = entry.get("deletions")
        ins = int(ins_raw) if isinstance(ins_raw, int) else 0
        dels = int(del_raw) if isinstance(del_raw, int) else 0
        merged_entries.append(
            {
                "path": path,
                "insertions": ins,
                "deletions": dels,
                "source": str(entry.get("source") or "managed_artifact"),
            }
        )
        insertions += ins
        deletions += dels
        existing_paths.add(path)

    if merged_entries:
        eval_result.changed_files.extend(merged_entries)
        eval_result.files_changed += len(merged_entries)
        eval_result.net_lines_changed += insertions - deletions
        if not eval_result.diff_stat:
            total_insertions = 0
            total_deletions = 0
            for item in eval_result.changed_files:
                ins_val = item.get("insertions")
                del_val = item.get("deletions")
                if isinstance(ins_val, int):
                    total_insertions += ins_val
                if isinstance(del_val, int):
                    total_deletions += del_val
            eval_result.diff_stat = (
                f"{eval_result.files_changed} files changed, "
                f"{total_insertions} insertions(+), {total_deletions} deletions(-)"
            )

    return {
        "files_added": len(merged_entries),
        "insertions": insertions,
        "deletions": deletions,
    }


def _collect_candidates(
    repo: Path,
    *,
    extra_globs: Sequence[str] | None,
    extra_rel_paths: Sequence[str] | None,
) -> set[str]:
    candidates: set[str] = set()
    patterns = [*_DEFAULT_GLOBS, *(extra_globs or ())]
    for pattern in patterns:
        pattern = str(pattern or "").strip()
        if not pattern:
            continue
        for path in repo.glob(pattern):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(repo).as_posix()
            except ValueError:
                continue
            candidates.add(rel)

    for raw in extra_rel_paths or ():
        rel = _normalize_relpath(raw)
        if rel:
            candidates.add(rel)
    return candidates


def _normalize_relpath(raw: str) -> str:
    value = str(raw or "").strip().replace("\\", "/")
    if not value:
        return ""
    candidate = Path(value)
    if candidate.is_absolute():
        return ""
    parts = [part for part in candidate.parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        return ""
    return Path(*parts).as_posix()


def _read_text_or_none(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _line_delta(before_text: str | None, after_text: str | None) -> tuple[int, int]:
    before_lines = [] if before_text is None else before_text.splitlines()
    after_lines = [] if after_text is None else after_text.splitlines()
    matcher = SequenceMatcher(a=before_lines, b=after_lines)
    insertions = 0
    deletions = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            insertions += j2 - j1
        elif tag == "delete":
            deletions += i2 - i1
        elif tag == "replace":
            deletions += i2 - i1
            insertions += j2 - j1
    return insertions, deletions
