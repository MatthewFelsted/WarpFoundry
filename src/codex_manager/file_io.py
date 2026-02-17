"""Cross-platform text I/O helpers with resilient decoding and atomic writes."""

from __future__ import annotations

import os
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

_ATOMIC_REPLACE_MAX_RETRIES = 8
_ATOMIC_REPLACE_RETRY_SECONDS = 0.01

_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


def _path_lock_key(path: Path) -> str:
    return str(path.resolve())


def _path_lock(path: Path) -> threading.RLock:
    key = _path_lock_key(path)
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
    return lock


@contextmanager
def locked_path(path: Path) -> Iterator[None]:
    """Serialize file access for a single process using a per-path lock."""
    lock = _path_lock(path)
    with lock:
        yield


def _replace_file_with_retry(src: Path, dst: Path) -> None:
    last_error: OSError | None = None
    for attempt in range(_ATOMIC_REPLACE_MAX_RETRIES):
        try:
            src.replace(dst)
            return
        except PermissionError as exc:
            last_error = exc
        except OSError as exc:
            if exc.errno != 13:
                raise
            last_error = exc
        if attempt < _ATOMIC_REPLACE_MAX_RETRIES - 1:
            time.sleep(_ATOMIC_REPLACE_RETRY_SECONDS * (attempt + 1))
    if last_error is not None:
        raise last_error


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write text to disk atomically to avoid partial/corrupt files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(content)
        with locked_path(path):
            _replace_file_with_retry(tmp_path, path)
    finally:
        with suppress(OSError):
            tmp_path.unlink(missing_ok=True)


def append_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Append text under a per-path lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with locked_path(path), path.open("a", encoding=encoding) as handle:
        handle.write(content)


@dataclass(frozen=True, slots=True)
class ResilientTextRead:
    """Result payload for resilient text decoding."""

    text: str
    used_fallback: bool = False
    decoder: str = "utf-8"
    used_replacement: bool = False
    normalized_to_utf8: bool = False


def read_text_utf8_resilient(
    path: Path,
    *,
    normalize_to_utf8: bool = True,
) -> ResilientTextRead:
    """Read text as UTF-8, recovering from common legacy encodings."""
    if not path.exists():
        return ResilientTextRead(text="")

    try:
        return ResilientTextRead(text=path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        raw = path.read_bytes()

    for decoder in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(decoder)
        except UnicodeDecodeError:
            continue
        normalized = False
        if normalize_to_utf8:
            atomic_write_text(path, text)
            normalized = True
        return ResilientTextRead(
            text=text,
            used_fallback=True,
            decoder=decoder,
            normalized_to_utf8=normalized,
        )

    text = raw.decode("utf-8", errors="replace")
    normalized = False
    if normalize_to_utf8:
        atomic_write_text(path, text)
        normalized = True
    return ResilientTextRead(
        text=text,
        used_fallback=True,
        decoder="utf-8-replace",
        used_replacement=True,
        normalized_to_utf8=normalized,
    )
