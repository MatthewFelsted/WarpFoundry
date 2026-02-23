"""Shared helpers for CLI runner implementations."""

from __future__ import annotations

import logging
import math
import os
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codex_manager.schemas import CodexEvent, RunResult

logger = logging.getLogger(__name__)
if os.name != "nt":  # pragma: no cover - platform-specific import
    import signal

_DEFAULT_MAX_CAPTURED_EVENTS = 20_000
_DEFAULT_MAX_CAPTURED_STDOUT_LINES = 20_000
_DEFAULT_MAX_CAPTURED_STDERR_LINES = 10_000
_COMMAND_LINE_LENGTH_ERROR_CODES = {7, 87, 206}
_COMMAND_LINE_LENGTH_ERROR_SUBSTRINGS = (
    "command line is too long",
    "filename or extension is too long",
    "argument list too long",
)


def _streaming_process_isolation_kwargs() -> dict[str, object]:
    """Return subprocess kwargs that isolate child signal/control handling.

    On Windows we combine CREATE_NEW_PROCESS_GROUP (prevents the child from
    receiving Ctrl+C aimed at the parent's group) **and** CREATE_NO_WINDOW
    (detaches the child from the parent's console entirely).  Without the
    latter, a terminating child can send CTRL_C_EVENT back through the
    shared console, killing the GUI server with an unexpected SIGINT.

    On POSIX, using a new session provides equivalent isolation.
    """
    if os.name == "nt":
        new_pg = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        no_win = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        flags = new_pg | no_win
        return {"creationflags": flags} if flags else {}
    return {"start_new_session": True}


def resolve_binary(name: str) -> str:
    """Resolve a binary name to a full executable path when possible."""
    expanded = os.path.expandvars(os.path.expanduser(str(name or "").strip()))
    if len(expanded) >= 2 and expanded[0] == expanded[-1] and expanded[0] in {"'", '"'}:
        # Accept copy/paste paths wrapped in shell quotes.
        expanded = expanded[1:-1].strip()
    if not expanded:
        return ""
    resolved = shutil.which(expanded)
    if resolved:
        return resolved
    return expanded


def coerce_int(value: Any) -> int:
    """Best-effort integer coercion for loosely typed CLI payloads."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
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


def is_command_line_too_long_error(exc: BaseException) -> bool:
    """Return True when an exception indicates argv length exceeded OS limits."""
    message = str(exc or "").strip().lower()
    if any(token in message for token in _COMMAND_LINE_LENGTH_ERROR_SUBSTRINGS):
        return True

    for attr_name in ("winerror", "errno"):
        raw_code = getattr(exc, attr_name, None)
        try:
            code = int(raw_code)
        except (TypeError, ValueError, OverflowError):
            continue
        if code in _COMMAND_LINE_LENGTH_ERROR_CODES:
            return True
    return False


@dataclass(slots=True)
class StreamExecutionResult:
    """Captured output and metadata from a runner subprocess."""

    events: list[CodexEvent]
    raw_lines: list[str]
    stderr_lines: list[str]
    exit_code: int
    timed_out: bool
    cancelled: bool = False

    @property
    def stderr_text(self) -> str:
        return "\n".join(self.stderr_lines).strip()


@dataclass(slots=True)
class PromptExecutionOutcome:
    """Outcome of one runner attempt including stdin fallback state."""

    result: RunResult | None
    used_stdin_prompt: bool
    error: Exception | None


def execute_with_prompt_transport_fallback(
    *,
    cwd: Path,
    prompt: str,
    use_stdin_prompt: bool,
    process_name: str,
    build_command: Callable[[str], list[str]],
    execute: Callable[[list[str], Path, str | None], RunResult],
) -> PromptExecutionOutcome:
    """Execute a command and retry once with stdin when argv is too long."""
    prompt_arg = "-" if use_stdin_prompt else prompt
    stdin_text = prompt if use_stdin_prompt else None
    cmd = build_command(prompt_arg)

    try:
        result = execute(cmd, cwd, stdin_text)
        return PromptExecutionOutcome(
            result=result,
            used_stdin_prompt=use_stdin_prompt,
            error=None,
        )
    except OSError as exc:
        if not use_stdin_prompt and is_command_line_too_long_error(exc):
            logger.warning(
                "%s argv exceeded command-line limits; retrying with stdin prompt transport.",
                process_name,
            )
            try:
                result = execute(build_command("-"), cwd, prompt)
                return PromptExecutionOutcome(
                    result=result,
                    used_stdin_prompt=True,
                    error=None,
                )
            except Exception as retry_exc:  # pragma: no cover - caller handles surfaced failures
                return PromptExecutionOutcome(
                    result=None,
                    used_stdin_prompt=True,
                    error=retry_exc,
                )
        return PromptExecutionOutcome(
            result=None,
            used_stdin_prompt=use_stdin_prompt,
            error=exc,
        )
    except Exception as exc:  # pragma: no cover - caller handles surfaced failures
        return PromptExecutionOutcome(
            result=None,
            used_stdin_prompt=use_stdin_prompt,
            error=exc,
        )


def execute_streaming_json_command(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: int,
    parse_stdout_line: Callable[[str], CodexEvent | None],
    process_name: str,
    stdin_text: str | None = None,
    cancel_event: threading.Event | None = None,
    max_events: int | None = None,
    max_stdout_lines: int | None = None,
    max_stderr_lines: int | None = None,
) -> StreamExecutionResult:
    """Run a subprocess and parse stdout JSONL with inactivity timeout support."""
    max_events = _normalize_capture_limit(max_events, _DEFAULT_MAX_CAPTURED_EVENTS)
    max_stdout_lines = _normalize_capture_limit(
        max_stdout_lines, _DEFAULT_MAX_CAPTURED_STDOUT_LINES
    )
    max_stderr_lines = _normalize_capture_limit(
        max_stderr_lines, _DEFAULT_MAX_CAPTURED_STDERR_LINES
    )

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE if stdin_text is not None else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        **_streaming_process_isolation_kwargs(),
    )
    if proc.stdout is None or proc.stderr is None:
        raise RuntimeError(f"{process_name} subprocess pipes are unexpectedly unavailable")

    events: deque[CodexEvent] = deque(maxlen=max_events)
    raw_lines: deque[str] = deque(maxlen=max_stdout_lines)
    stderr_lines: deque[str] = deque(maxlen=max_stderr_lines)
    events_dropped = 0
    raw_lines_dropped = 0
    stderr_lines_dropped = 0
    stream_queue: queue.Queue[tuple[str, str | object]] = queue.Queue()
    done_sentinel = object()

    def _append_event(event: CodexEvent) -> None:
        nonlocal events_dropped
        if len(events) == events.maxlen:
            events_dropped += 1
        events.append(event)

    def _append_stdout_line(line: str) -> None:
        nonlocal raw_lines_dropped
        if len(raw_lines) == raw_lines.maxlen:
            raw_lines_dropped += 1
        raw_lines.append(line)

    def _append_stderr_line(line: str) -> None:
        nonlocal stderr_lines_dropped
        if len(stderr_lines) == stderr_lines.maxlen:
            stderr_lines_dropped += 1
        stderr_lines.append(line)

    def _pump_stream(stream_name: str, stream: Any) -> None:
        try:
            for line in stream:
                stream_queue.put((stream_name, line.rstrip("\n\r")))
        finally:
            stream_queue.put((stream_name, done_sentinel))

    def _collect_stdout_line(line: str) -> None:
        _append_stdout_line(line)
        try:
            event = parse_stdout_line(line)
        except Exception:  # pragma: no cover - defensive parser isolation
            logger.warning(
                "Failed to parse %s JSONL stdout line; preserving raw output", process_name
            )
            return
        if event is not None:
            _append_event(event)

    def _pump_stdin(stream: Any, text: str) -> None:
        try:
            stream.write(text)
            if text and not text.endswith("\n"):
                stream.write("\n")
            stream.flush()
        except Exception:  # pragma: no cover - stdin write failures are non-fatal
            logger.debug("%s stdin write failed", process_name)
        finally:
            with suppress(Exception):
                stream.close()

    stdin_thread: threading.Thread | None = None
    if stdin_text is not None and proc.stdin is not None:
        stdin_thread = threading.Thread(
            target=_pump_stdin,
            args=(proc.stdin, stdin_text),
            daemon=True,
        )
        stdin_thread.start()

    stdout_thread = threading.Thread(target=_pump_stream, args=("stdout", proc.stdout), daemon=True)
    stderr_thread = threading.Thread(target=_pump_stream, args=("stderr", proc.stderr), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    inactivity_timeout = timeout_seconds if timeout_seconds > 0 else None
    last_activity = time.monotonic()
    closed_streams: set[str] = set()
    timed_out = False
    cancelled = False

    try:
        while len(closed_streams) < 2:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                _terminate_process_with_fallback(
                    proc,
                    process_name=process_name,
                    reason="stop request",
                )
                break

            if (
                inactivity_timeout is not None
                and (time.monotonic() - last_activity) >= inactivity_timeout
            ):
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
                _append_stderr_line(line)

        if timed_out:
            _terminate_process_with_fallback(
                proc,
                process_name=process_name,
                reason="inactivity timeout",
            )

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
                _append_stderr_line(line)

        if events_dropped:
            logger.warning(
                "%s emitted more than %s parsed events; dropped %s oldest event(s)",
                process_name,
                max_events,
                events_dropped,
            )
        if raw_lines_dropped:
            logger.warning(
                "%s emitted more than %s stdout lines; dropped %s oldest line(s)",
                process_name,
                max_stdout_lines,
                raw_lines_dropped,
            )
        if stderr_lines_dropped:
            logger.warning(
                "%s emitted more than %s stderr lines; dropped %s oldest line(s)",
                process_name,
                max_stderr_lines,
                stderr_lines_dropped,
            )

        return StreamExecutionResult(
            events=list(events),
            raw_lines=list(raw_lines),
            stderr_lines=list(stderr_lines),
            exit_code=proc.returncode if proc.returncode is not None else -1,
            timed_out=timed_out,
            cancelled=cancelled,
        )
    finally:
        if stdin_thread is not None:
            stdin_thread.join(timeout=1.0)
        stdout_thread.join(timeout=1.0)
        stderr_thread.join(timeout=1.0)
        if proc.stdin is not None and not proc.stdin.closed:
            with suppress(Exception):
                proc.stdin.close()
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


def _terminate_process_with_fallback(
    proc: subprocess.Popen[str],
    *,
    process_name: str,
    reason: str,
    terminate_timeout_seconds: float = 1.5,
) -> None:
    """Request graceful terminate first, then force-kill if still alive."""
    if proc.poll() is not None:
        return

    _terminate_process(proc)
    timeout = max(0.1, float(terminate_timeout_seconds))
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        logger.warning(
            "%s did not exit after terminate during %s; forcing kill.",
            process_name,
            reason,
        )

    _kill_process(proc)
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:  # pragma: no cover - extreme edge case
        logger.warning("%s ignored kill during %s.", process_name, reason)


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    """Best-effort graceful termination for a child process (and its group)."""
    if os.name != "nt":
        _signal_process_group(proc, signal.SIGTERM)
    with suppress(Exception):
        proc.terminate()


def _kill_process(proc: subprocess.Popen[str]) -> None:
    """Best-effort force kill for a child process (and its group)."""
    if os.name != "nt":
        _signal_process_group(proc, signal.SIGKILL)
    with suppress(Exception):
        proc.kill()


def _signal_process_group(proc: subprocess.Popen[str], sig: int) -> None:
    """Best-effort signal delivery to the subprocess process-group on POSIX."""
    if os.name == "nt":  # pragma: no cover - Windows-only runtime branch
        return
    pid = int(getattr(proc, "pid", 0) or 0)
    if pid <= 0:
        return
    with suppress(Exception):
        os.killpg(os.getpgid(pid), sig)


def _normalize_capture_limit(value: int | None, default: int) -> int:
    """Normalize output capture limits and enforce a minimum of one entry."""
    if value is None:
        return default
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(1, normalized)
