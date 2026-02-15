"""Shared helpers for rotating logbooks and JSON-safe payload handling."""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SanitizeOptions:
    """Limits controlling recursive payload sanitization."""

    max_depth: int
    max_list_items: int
    max_dict_items: int
    max_key_len: int
    max_str_len: int
    fallback_repr_len: int


def truncate_text(text: str, max_len: int) -> str:
    """Trim whitespace and clamp to ``max_len`` characters."""
    clean = (text or "").strip()
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def sanitize_json_value(value: Any, *, options: SanitizeOptions, depth: int = 0) -> Any:
    """Recursively sanitize values so they can be serialized as strict JSON."""
    if depth > options.max_depth:
        return "[truncated-depth]"
    if value is None:
        return None
    if isinstance(value, str):
        return truncate_text(value, options.max_str_len)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [
            sanitize_json_value(item, options=options, depth=depth + 1)
            for item in value[: options.max_list_items]
        ]
    if isinstance(value, tuple):
        return [
            sanitize_json_value(item, options=options, depth=depth + 1)
            for item in value[: options.max_list_items]
        ]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= options.max_dict_items:
                out["__truncated__"] = f"{len(value) - options.max_dict_items} more key(s)"
                break
            out[truncate_text(str(key), options.max_key_len)] = sanitize_json_value(
                item, options=options, depth=depth + 1
            )
        return out
    return truncate_text(repr(value), options.fallback_repr_len)


def ensure_log_paths(
    *,
    logs_dir: Path,
    archive_dir: Path,
    markdown_path: Path,
    jsonl_path: Path,
    markdown_header: str,
) -> None:
    """Create log directories/files when they do not already exist."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    if not markdown_path.exists():
        markdown_path.write_text(markdown_header, encoding="utf-8")
    if not jsonl_path.exists():
        jsonl_path.touch()


def rotate_if_needed(
    *,
    path: Path,
    max_bytes: int,
    archive_dir: Path,
    markdown_header: str,
    max_archives: int,
) -> None:
    """Rotate ``path`` into archive storage if it exceeds the max size."""
    if not path.exists() or path.stat().st_size < max_bytes:
        return
    rotate_file(
        path=path,
        archive_dir=archive_dir,
        markdown_header=markdown_header,
        max_archives=max_archives,
    )


def rotate_file(
    *,
    path: Path,
    archive_dir: Path,
    markdown_header: str,
    max_archives: int,
) -> None:
    """Rotate ``path`` and prune old archives for the same file stem/suffix."""
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = archive_dir / f"{path.stem}-{stamp}{path.suffix}"
    idx = 1
    while target.exists():
        idx += 1
        target = archive_dir / f"{path.stem}-{stamp}-{idx}{path.suffix}"

    path.replace(target)
    if path.suffix.lower() == ".md":
        path.write_text(markdown_header, encoding="utf-8")
    else:
        path.touch()

    prune_archives(
        archive_dir=archive_dir,
        stem=path.stem,
        suffix=path.suffix,
        max_archives=max_archives,
    )


def prune_archives(*, archive_dir: Path, stem: str, suffix: str, max_archives: int) -> None:
    """Delete older archive files and keep the newest ``max_archives`` entries."""
    files = sorted(
        archive_dir.glob(f"{stem}-*{suffix}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in files[max_archives:]:
        old.unlink(missing_ok=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append a JSON payload to a JSONL file using strict JSON encoding."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
