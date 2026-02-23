"""Interface to Anthropic Claude Code CLI (``claude``)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from codex_manager.agent_runner import AgentRunner, register_agent
from codex_manager.prompt_logging import prompt_metadata
from codex_manager.runner_common import (
    coerce_int,
    execute_streaming_json_command,
    execute_with_prompt_transport_fallback,
    resolve_binary,
)
from codex_manager.schemas import (
    CodexEvent,
    EventKind,
    RunResult,
    UsageInfo,
)

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT = 600  # 10 minutes of inactivity
_WINDOWS_COMMAND_LINE_LIMIT = 32767
_WINDOWS_CMD_EXE_LIMIT = 8191
_COMMAND_LINE_SAFETY_MARGIN = 2048
_POSIX_PROMPT_ARG_LIMIT = 60000


class ClaudeCodeRunner(AgentRunner):
    """Spawn ``claude -p`` and parse its stream-json output.

    Claude Code's non-interactive mode works as follows::

        claude -p "prompt"                         # plain text
        claude -p "prompt" --output-format json    # single JSON blob
        claude -p "prompt" --output-format stream-json  # streaming JSONL

    We use ``stream-json`` to get real-time JSONL events, similar to
    ``codex exec --json``.

    Parameters
    ----------
    claude_binary:
        Path or name of the Claude Code CLI binary.
    timeout:
        Maximum seconds without stdout/stderr activity before the child
        process is killed. ``0`` disables the timeout.
    env_overrides:
        Extra environment variables forwarded to the child process.
    max_turns:
        Maximum agent turns (``--max-turns``).  ``0`` means unlimited.
    model:
        Override the model Claude Code uses (``--model``).  Leave blank
        for the default.
    """

    name = "Claude Code"

    def __init__(
        self,
        claude_binary: str = "claude",
        timeout: int = DEFAULT_TIMEOUT,
        env_overrides: dict[str, str] | None = None,
        max_turns: int = 0,
        model: str = "",
    ) -> None:
        self.claude_binary = claude_binary
        self.timeout = max(0, coerce_int(timeout))
        self.env_overrides = env_overrides or {}
        self.max_turns = max(0, coerce_int(max_turns))
        self.model = (model or "").strip()
        self._cancel_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        repo_path: str | Path,
        prompt: str,
        *,
        full_auto: bool = False,
        extra_args: list[str] | None = None,
    ) -> RunResult:
        """Execute a single Claude Code invocation and return results."""
        repo_path = Path(repo_path).resolve()
        if not repo_path.is_dir():
            return RunResult(
                success=False,
                exit_code=-1,
                errors=[f"repo_path does not exist: {repo_path}"],
            )
        self._cancel_event.clear()

        use_stdin_prompt = self._should_pipe_prompt_via_stdin(
            prompt,
            full_auto=full_auto,
            extra_args=extra_args,
        )
        prompt_meta = prompt_metadata(prompt)
        logger.info(
            "Running Claude Code CLI (cwd=%s, prompt_transport=%s, prompt_len=%s, prompt_sha256=%s)",
            repo_path,
            "stdin" if use_stdin_prompt else "argv",
            prompt_meta["length_chars"],
            prompt_meta["sha256"],
        )

        start = time.monotonic()
        def _build_command(prompt_arg: str) -> list[str]:
            return self._build_command(prompt_arg, full_auto=full_auto, extra_args=extra_args)

        def _execute_with_prompt(
            command: list[str],
            cwd: Path,
            stdin_text: str | None,
        ) -> RunResult:
            return self._execute(command, cwd=cwd, stdin_text=stdin_text)

        attempt_outcome = execute_with_prompt_transport_fallback(
            cwd=repo_path,
            prompt=prompt,
            use_stdin_prompt=use_stdin_prompt,
            process_name="Claude Code",
            build_command=_build_command,
            execute=_execute_with_prompt,
        )

        if attempt_outcome.error is not None:
            return RunResult(
                success=False,
                exit_code=-1,
                errors=[f"Failed to execute claude: {attempt_outcome.error}"],
                duration_seconds=time.monotonic() - start,
            )
        result = attempt_outcome.result
        if result is None:
            return RunResult(
                success=False,
                exit_code=-1,
                errors=["Failed to execute claude: missing result"],
                duration_seconds=time.monotonic() - start,
            )
        if self._cancel_event.is_set():
            return RunResult(
                success=False,
                exit_code=-1,
                errors=["Execution cancelled by stop request"],
                duration_seconds=time.monotonic() - start,
            )
        result.duration_seconds = time.monotonic() - start
        return result

    def stop(self) -> None:
        """Request cancellation of the active Claude subprocess, if any."""
        self._cancel_event.set()

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def _build_command(
        self,
        prompt: str,
        *,
        full_auto: bool,
        extra_args: list[str] | None,
    ) -> list[str]:
        cmd = [resolve_binary(self.claude_binary), "-p", prompt, "--output-format", "stream-json"]

        if full_auto:
            # Skip interactive permission prompts — equivalent to Codex's --full-auto
            cmd.append("--dangerously-skip-permissions")

        if self.max_turns > 0:
            cmd.extend(["--max-turns", str(self.max_turns)])

        has_model_override = False
        if extra_args:
            for arg in extra_args:
                normalized = (arg or "").strip().lower()
                if normalized in {"--model", "-m"} or normalized.startswith("--model="):
                    has_model_override = True
                    break

        if self.model and not has_model_override:
            cmd.extend(["--model", self.model])

        if extra_args:
            cmd.extend(extra_args)

        return cmd

    def _should_pipe_prompt_via_stdin(
        self,
        prompt: str,
        *,
        full_auto: bool,
        extra_args: list[str] | None,
    ) -> bool:
        """Return True when prompt should be supplied via stdin instead of argv."""
        if os.name != "nt":
            return len(prompt) >= _POSIX_PROMPT_ARG_LIMIT

        if not prompt:
            return False

        base_cmd = self._build_command("", full_auto=full_auto, extra_args=extra_args)
        try:
            prompt_index = base_cmd.index("-p") + 1
        except ValueError:
            prompt_index = len(base_cmd)

        probe_cmd = list(base_cmd)
        if prompt_index < len(probe_cmd):
            probe_cmd[prompt_index] = prompt
        else:
            probe_cmd.append(prompt)

        try:
            estimated = len(subprocess.list2cmdline(probe_cmd))
        except Exception:
            estimated = sum(len(part) for part in probe_cmd) + len(probe_cmd) + 1

        command_limit = self._effective_windows_command_limit(base_cmd)
        return estimated >= (command_limit - _COMMAND_LINE_SAFETY_MARGIN)

    @staticmethod
    def _effective_windows_command_limit(base_cmd: list[str]) -> int:
        """Return the best-effort command length ceiling for the resolved launcher."""
        if not base_cmd:
            return _WINDOWS_COMMAND_LINE_LIMIT
        launcher = (base_cmd[0] or "").strip().lower()
        if launcher.endswith(".cmd") or launcher.endswith(".bat"):
            return _WINDOWS_CMD_EXE_LIMIT
        return _WINDOWS_COMMAND_LINE_LIMIT

    # ------------------------------------------------------------------
    # Execution + JSONL parsing
    # ------------------------------------------------------------------

    def _execute(self, cmd: list[str], cwd: Path, *, stdin_text: str | None = None) -> RunResult:
        """Spawn the subprocess and consume its streaming-JSON stdout."""
        env = {**os.environ, **self.env_overrides}
        inactivity_timeout = self.timeout if self.timeout > 0 else None
        execution = execute_streaming_json_command(
            cmd=cmd,
            cwd=cwd,
            env=env,
            timeout_seconds=self.timeout,
            parse_stdout_line=self._parse_line,
            process_name="Claude Code",
            stdin_text=stdin_text,
            cancel_event=self._cancel_event,
        )
        stderr_text = execution.stderr_text

        if execution.cancelled:
            return RunResult(
                success=False,
                exit_code=-1,
                events=execution.events,
                errors=["Execution cancelled by stop request"]
                + ([stderr_text] if stderr_text else []),
            )

        if execution.timed_out:
            timeout_msg = (
                f"Claude Code process timed out after {inactivity_timeout}s with no output activity"
            )
            return RunResult(
                success=False,
                exit_code=-1,
                events=execution.events,
                errors=[timeout_msg] + ([stderr_text] if stderr_text else []),
            )

        return self._aggregate(
            execution.events,
            execution.exit_code,
            stderr_text,
            execution.raw_lines,
        )

    # ------------------------------------------------------------------
    # JSONL parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_line(line: str) -> CodexEvent | None:
        """Parse one line of Claude Code stream-json output.

        Claude Code's ``stream-json`` format emits objects like::

            {"type": "system", ...}
            {"type": "assistant", "message": {...}, "session_id": "..."}
            {"type": "result", "result": "...", "session_id": "..."}

        We normalise these into the same CodexEvent model used by Codex.
        """
        try:
            data: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("Non-JSON line from claude: %s", line[:200])
            return None

        kind = _classify_claude_event(data)
        text = _extract_claude_text(data, kind)
        return CodexEvent(kind=kind, raw=data, text=text)

    @staticmethod
    def _aggregate(
        events: list[CodexEvent],
        exit_code: int,
        stderr: str,
        raw_lines: list[str],
    ) -> RunResult:
        """Combine parsed events into a single RunResult."""
        errors: list[str] = []
        if stderr:
            errors.append(stderr)

        file_changes: list[dict[str, Any]] = []
        command_execs: list[dict[str, Any]] = []
        usage = UsageInfo()
        final_message = ""

        for ev in events:
            if ev.kind == EventKind.ERROR:
                errors.append(ev.text or json.dumps(ev.raw))
            elif ev.kind == EventKind.FILE_CHANGE:
                file_changes.append(ev.raw)
            elif ev.kind == EventKind.COMMAND_EXEC:
                command_execs.append(ev.raw)
            elif ev.kind == EventKind.TURN_COMPLETED:
                usage = _extract_claude_usage(ev.raw)
            if ev.kind == EventKind.AGENT_MESSAGE and ev.text:
                final_message = ev.text

        # Fallback: extract from result event
        if not final_message:
            for ev in reversed(events):
                if ev.raw.get("type") == "result":
                    result = ev.raw.get("result", "")
                    if isinstance(result, str):
                        final_message = result
                    elif isinstance(result, dict):
                        final_message = result.get("text") or result.get("content") or ""
                    break

        # Second fallback: last non-JSON line
        if not final_message and raw_lines:
            for rl in reversed(raw_lines):
                try:
                    json.loads(rl)
                except json.JSONDecodeError:
                    final_message = rl
                    break

        # Keep non-zero exits from showing up as opaque "unknown error".
        if exit_code != 0 and not errors:
            inferred = _infer_claude_error(events)
            if inferred:
                errors.append(inferred)
            elif final_message:
                errors.append(final_message[:500])
            else:
                errors.append(
                    f"Claude Code exited with status {exit_code} but produced no explicit error output"
                )

        return RunResult(
            success=exit_code == 0,
            exit_code=exit_code,
            final_message=final_message,
            events=events,
            file_changes=file_changes,
            command_executions=command_execs,
            usage=usage,
            errors=errors,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _classify_claude_event(data: dict[str, Any]) -> EventKind:
    """Map a raw Claude Code JSON event to a :class:`EventKind`.

    Claude Code stream-json events use a ``type`` field with values like:

    - ``"system"`` — session metadata
    - ``"assistant"`` — agent messages and tool use
    - ``"result"`` — final aggregated result
    """
    etype = (data.get("type") or "").lower().strip()

    # Direct type mapping
    if etype == "result":
        return EventKind.TURN_COMPLETED

    if etype == "system":
        return EventKind.UNKNOWN  # metadata, not actionable

    if etype == "assistant":
        # Check if this is a tool use (file edit or command)
        message = data.get("message", {})
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type", "")
                tool_name = block.get("name", "").lower()

                if block_type == "tool_use":
                    if any(k in tool_name for k in ("write", "edit", "file", "create")):
                        return EventKind.FILE_CHANGE
                    if any(k in tool_name for k in ("bash", "command", "exec", "terminal")):
                        return EventKind.COMMAND_EXEC
                    # Other tool uses are still agent actions
                    return EventKind.AGENT_MESSAGE
                if block_type == "text":
                    return EventKind.AGENT_MESSAGE
        return EventKind.AGENT_MESSAGE

    if etype == "error":
        return EventKind.ERROR

    # Fallback: look for common keys
    if "error" in data:
        return EventKind.ERROR

    return EventKind.UNKNOWN


def _extract_claude_text(data: dict[str, Any], kind: EventKind) -> str | None:
    """Pull human-readable text from a Claude Code event."""
    # Result event
    if data.get("type") == "result":
        result = data.get("result")
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return result.get("text") or result.get("content", "")

    # Assistant message
    if data.get("type") == "assistant":
        message = data.get("message", {})
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            texts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            tool_name = block.get("name", "unknown")
                            tool_input = block.get("input", {})
                            if isinstance(tool_input, dict):
                                # For file writes, show the path
                                path = tool_input.get("file_path") or tool_input.get("path", "")
                                cmd = tool_input.get("command", "")
                                if path:
                                    texts.append(f"[{tool_name}: {path}]")
                                elif cmd:
                                    texts.append(f"[{tool_name}: {cmd[:100]}]")
                return "\n".join(texts).strip() or None

    # Error
    if kind == EventKind.ERROR:
        for key in ("error", "message", "text"):
            val = data.get(key)
            if isinstance(val, str):
                return val
            if isinstance(val, dict):
                return val.get("message") or val.get("text", "")

    return None


def _extract_claude_usage(data: dict[str, Any]) -> UsageInfo:
    """Extract token usage from a Claude Code result event."""
    # Usage can be at top level or nested in the result
    usage_raw: dict[str, Any] = {}
    top_level_usage = data.get("usage")
    if isinstance(top_level_usage, dict):
        usage_raw = top_level_usage
    if not usage_raw:
        result = data.get("result")
        if isinstance(result, dict):
            nested_usage = result.get("usage")
            if isinstance(nested_usage, dict):
                usage_raw = nested_usage

    # Claude Code reports input/output tokens
    input_tokens = max(0, coerce_int(usage_raw.get("input_tokens", 0)))
    output_tokens = max(0, coerce_int(usage_raw.get("output_tokens", 0)))

    # Also check for cache tokens (Claude Code reports these)
    cached_input = max(0, coerce_int(usage_raw.get("cached_input_tokens", 0)))
    cache_read = max(0, coerce_int(usage_raw.get("cache_read_input_tokens", 0)))
    cache_creation = max(0, coerce_int(usage_raw.get("cache_creation_input_tokens", 0)))

    total_tokens = max(0, coerce_int(usage_raw.get("total_tokens", 0)))
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens + cached_input + cache_read + cache_creation

    return UsageInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        model=data.get("model") or usage_raw.get("model"),
    )


def _infer_claude_error(events: list[CodexEvent]) -> str | None:
    """Extract a useful error string from Claude stream-json events."""
    for ev in reversed(events):
        raw = ev.raw
        for key in ("error", "message", "reason", "detail"):
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            if isinstance(val, dict):
                nested = val.get("message") or val.get("text") or val.get("detail")
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()

        result = raw.get("result")
        if isinstance(result, dict):
            err = result.get("error")
            if isinstance(err, str) and err.strip():
                return err.strip()
            if isinstance(err, dict):
                nested = err.get("message") or err.get("text") or err.get("detail")
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()

    return None


# ── Register with the agent registry ─────────────────────────────
register_agent("claude_code", ClaudeCodeRunner)
