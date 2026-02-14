"""Shared helpers for CLI runner implementations."""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codex_manager.schemas import CodexEvent

logger = logging.getLogger(__name__)


def resolve_binary(name: str) -> str:
    """Resolve a binary name to a full executable path when possible."""
    resolved = shutil.which(name)
    if resolved:
        return resolved
    return name


def coerce_int(value: Any) -> int:
    """Best-effort integer coercion for loosely typed CLI payloads."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:  # NaN
            return 0
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned:
            return 0
        try:
            return int(cleaned)
        except ValueError:
            try:
                return int(float(cleaned))
            except (TypeError, ValueError, OverflowError):
                return 0
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


@dataclass(slots=True)
class StreamExecutionResult:
    """Captured output and metadata from a runner subprocess."""

    events: list[CodexEvent]
    raw_lines: list[str]
    stderr_lines: list[str]
    exit_code: int
    timed_out: bool

    @property
    def stderr_text(self) -> str:
        return "\n".join(self.stderr_lines).strip()


def execute_streaming_json_command(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: int,
    parse_stdout_line: Callable[[str], CodexEvent | None],
    process_name: str,
) -> StreamExecutionResult:
    """Run a subprocess and parse stdout JSONL with inactivity timeout support."""
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if proc.stdout is None or proc.stderr is None:
        raise RuntimeError(f"{process_name} subprocess pipes are unexpectedly unavailable")

    events: list[CodexEvent] = []
    raw_lines: list[str] = []
    stderr_lines: list[str] = []
    stream_queue: queue.Queue[tuple[str, str | object]] = queue.Queue()
    done_sentinel = object()

    def _pump_stream(stream_name: str, stream: Any) -> None:
        try:
            for line in stream:
                stream_queue.put((stream_name, line.rstrip("\n\r")))
        finally:
            stream_queue.put((stream_name, done_sentinel))

    def _collect_stdout_line(line: str) -> None:
        raw_lines.append(line)
        try:
            event = parse_stdout_line(line)
        except Exception:  # pragma: no cover - defensive parser isolation
            logger.warning(
                "Failed to parse %s JSONL stdout line; preserving raw output", process_name
            )
            return
        if event is not None:
            events.append(event)

    stdout_thread = threading.Thread(
        target=_pump_stream, args=("stdout", proc.stdout), daemon=True
    )
    stderr_thread = threading.Thread(
        target=_pump_stream, args=("stderr", proc.stderr), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    inactivity_timeout = timeout_seconds if timeout_seconds > 0 else None
    last_activity = time.monotonic()
    closed_streams: set[str] = set()
    timed_out = False

    try:
        while len(closed_streams) < 2:
            if inactivity_timeout is not None and (time.monotonic() - last_activity) >= inactivity_timeout:
                timed_out = True
                break

            wait_seconds = 0.25
            if inactivity_timeout is not None:
                remaining = inactivity_timeout - (time.monotonic() - last_activity)
                wait_seconds = max(0.05, min(0.5, remaining))

            try:
                stream_name, payload = stream_queue.get(timeout=wait_seconds)
            except queue.Empty:
                if (
                    proc.poll() is not None
                    and not stdout_thread.is_alive()
                    and not stderr_thread.is_alive()
                ):
                    break
                continue

            if payload is done_sentinel:
                closed_streams.add(stream_name)
                continue

            last_activity = time.monotonic()
            if not payload:
                continue
            line = str(payload)

            if stream_name == "stdout":
                _collect_stdout_line(line)
            else:
                stderr_lines.append(line)

        if timed_out:
            proc.kill()

        _wait_for_process(proc)

        # Drain buffered lines produced just before process exit.
        while True:
            try:
                stream_name, payload = stream_queue.get_nowait()
            except queue.Empty:
                break
            if payload is done_sentinel or not payload:
                continue
            line = str(payload)
            if stream_name == "stdout":
                _collect_stdout_line(line)
            else:
                stderr_lines.append(line)

        return StreamExecutionResult(
            events=events,
            raw_lines=raw_lines,
            stderr_lines=stderr_lines,
            exit_code=proc.returncode if proc.returncode is not None else -1,
            timed_out=timed_out,
        )
    finally:
        stdout_thread.join(timeout=1.0)
        stderr_thread.join(timeout=1.0)
        if proc.stdout is not None and not proc.stdout.closed:
            proc.stdout.close()
        if proc.stderr is not None and not proc.stderr.closed:
            proc.stderr.close()


def _wait_for_process(proc: subprocess.Popen[str]) -> None:
    """Wait for child process exit and force-kill if it refuses to terminate."""
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive
        proc.kill()
        proc.wait(timeout=5.0)
