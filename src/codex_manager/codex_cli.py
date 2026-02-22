"""Interface to the OpenAI Codex CLI (``codex exec``)."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from codex_manager.agent_runner import AgentRunner, register_agent
from codex_manager.runner_common import (
    coerce_int,
    execute_streaming_json_command,
    execute_with_prompt_transport_fallback,
    resolve_binary,
)
from codex_manager.prompt_logging import prompt_metadata
from codex_manager.schemas import (
    CodexEvent,
    EventKind,
    RunResult,
    UsageInfo,
)

logger = logging.getLogger(__name__)


# Default inactivity timeout for a single Codex run (seconds).
DEFAULT_TIMEOUT = 600  # 10 minutes
DEFAULT_CODEX_MODEL = "gpt-5.3-codex"
_WINDOWS_COMMAND_LINE_LIMIT = 32767
_WINDOWS_CMD_EXE_LIMIT = 8191
_COMMAND_LINE_SAFETY_MARGIN = 2048
_POSIX_PROMPT_ARG_LIMIT = 60000
_DEFAULT_TRANSIENT_NETWORK_RETRIES = 2
_DEFAULT_TRANSIENT_RETRY_BACKOFF_SECONDS = 1.0
_MAX_TRANSIENT_RETRY_BACKOFF_SECONDS = 8.0
_TRANSIENT_NETWORK_ERROR_SUBSTRINGS = (
    "stream disconnected before completion",
    "error sending request for url",
    "network error",
    "connection reset",
    "connection aborted",
    "connection refused",
    "failed to refresh available models",
    "service unavailable",
    "temporarily unavailable",
    "timed out",
    "timeout",
)


class CodexRunner(AgentRunner):
    """Spawn ``codex exec --json`` and parse its JSONL event stream.

    Parameters
    ----------
    codex_binary:
        Path or name of the Codex CLI binary.  Defaults to ``"codex"`` (must
        be on ``$PATH``).
    timeout:
        Maximum seconds without stdout/stderr activity before the child
        process is killed. ``0`` disables the timeout.
    env_overrides:
        Extra environment variables forwarded to the child process.  Use this
        to inject ``CODEX_API_KEY`` without touching the system environment.
    sandbox_mode:
        Sandbox mode used in ``full_auto`` runs (``read-only``,
        ``workspace-write``, or ``danger-full-access``).
    approval_policy:
        Approval policy injected via ``-c approval_policy=...`` during
        ``full_auto`` runs to avoid interactive prompts in background chains.
    reasoning_effort:
        Codex reasoning effort override sent as
        ``-c model_reasoning_effort=...``. Use ``"inherit"`` to keep the
        user's Codex CLI default from ``~/.codex/config.toml``.
    model:
        Codex model passed via ``--model`` when no explicit model is present
        in ``extra_args``.
    bypass_approvals_and_sandbox:
        If True, uses ``--dangerously-bypass-approvals-and-sandbox``.
    transient_network_retries:
        Number of automatic retries for transient network transport failures.
    transient_retry_backoff_seconds:
        Base backoff delay used between transient retries.
    """

    name = "Codex"

    def __init__(
        self,
        codex_binary: str = "codex",
        timeout: int = DEFAULT_TIMEOUT,
        env_overrides: dict[str, str] | None = None,
        sandbox_mode: str = "workspace-write",
        approval_policy: str = "never",
        reasoning_effort: str = "xhigh",
        model: str = DEFAULT_CODEX_MODEL,
        bypass_approvals_and_sandbox: bool = False,
        transient_network_retries: int = _DEFAULT_TRANSIENT_NETWORK_RETRIES,
        transient_retry_backoff_seconds: float = _DEFAULT_TRANSIENT_RETRY_BACKOFF_SECONDS,
    ) -> None:
        self.codex_binary = codex_binary
        self.timeout = max(0, coerce_int(timeout))
        self.env_overrides = env_overrides or {}
        self.sandbox_mode = str(sandbox_mode or "workspace-write").strip() or "workspace-write"
        self.approval_policy = str(approval_policy or "never").strip() or "never"
        normalized_effort = (reasoning_effort or "xhigh").strip().lower()
        valid_efforts = {"inherit", "low", "medium", "high", "xhigh"}
        if normalized_effort not in valid_efforts:
            logger.warning(
                "Invalid codex reasoning effort '%s'; falling back to 'xhigh'",
                reasoning_effort,
            )
            normalized_effort = "xhigh"
        self.reasoning_effort = normalized_effort
        self.model = (model or DEFAULT_CODEX_MODEL).strip() or DEFAULT_CODEX_MODEL
        self.bypass_approvals_and_sandbox = bypass_approvals_and_sandbox
        self.transient_network_retries = max(0, coerce_int(transient_network_retries))
        try:
            backoff_seconds = float(transient_retry_backoff_seconds)
        except (TypeError, ValueError, OverflowError):
            backoff_seconds = _DEFAULT_TRANSIENT_RETRY_BACKOFF_SECONDS
        if backoff_seconds < 0:
            backoff_seconds = _DEFAULT_TRANSIENT_RETRY_BACKOFF_SECONDS
        self.transient_retry_backoff_seconds = backoff_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        repo_path: str | Path,
        prompt: str,
        *,
        use_json: bool = True,
        full_auto: bool = False,
        extra_args: list[str] | None = None,
    ) -> RunResult:
        """Execute a single Codex CLI invocation and return structured results.

        Parameters
        ----------
        repo_path:
            Working directory (the target git repository).
        prompt:
            Natural-language task prompt sent to Codex.
        use_json:
            If *True* (default), pass ``--json`` so Codex emits JSONL events.
        full_auto:
            If *True*, pass ``--full-auto`` to allow Codex to write files and
            run commands autonomously.  **Only enable when the user has
            explicitly chosen "apply" mode.**
        extra_args:
            Additional CLI flags forwarded verbatim.
        """
        repo_path = Path(repo_path).resolve()
        if not repo_path.is_dir():
            return RunResult(
                success=False,
                exit_code=-1,
                errors=[f"repo_path does not exist: {repo_path}"],
            )

        use_stdin_prompt = self._should_pipe_prompt_via_stdin(
            prompt,
            repo_path=repo_path,
            use_json=use_json,
            full_auto=full_auto,
            extra_args=extra_args,
        )
        prompt_meta = prompt_metadata(prompt)
        logger.info(
            "Running Codex CLI (cwd=%s, prompt_transport=%s, prompt_len=%s, prompt_sha256=%s)",
            repo_path,
            "stdin" if use_stdin_prompt else "argv",
            prompt_meta["length_chars"],
            prompt_meta["sha256"],
        )

        start = time.monotonic()
        max_attempts = self.transient_network_retries + 1
        result: RunResult | None = None

        def _build_command(prompt_arg: str) -> list[str]:
            return self._build_command(
                prompt_arg,
                repo_path,
                use_json=use_json,
                full_auto=full_auto,
                extra_args=extra_args,
            )

        def _execute_with_prompt(
            command: list[str],
            cwd: Path,
            stdin_text: str | None,
        ) -> RunResult:
            return self._execute(command, cwd=cwd, stdin_text=stdin_text)

        for attempt in range(1, max_attempts + 1):
            attempt_outcome = execute_with_prompt_transport_fallback(
                cwd=repo_path,
                prompt=prompt,
                use_stdin_prompt=use_stdin_prompt,
                process_name="Codex",
                build_command=_build_command,
                execute=_execute_with_prompt,
            )
            use_stdin_prompt = attempt_outcome.used_stdin_prompt

            if attempt_outcome.error is not None:
                return RunResult(
                    success=False,
                    exit_code=-1,
                    errors=[f"Failed to execute codex: {attempt_outcome.error}"],
                    duration_seconds=time.monotonic() - start,
                )
            result = attempt_outcome.result
            if result is None:
                return RunResult(
                    success=False,
                    exit_code=-1,
                    errors=["Failed to execute codex: missing result"],
                    duration_seconds=time.monotonic() - start,
                )

            if result.success:
                break
            if attempt >= max_attempts or not self._is_transient_network_failure(result):
                break

            delay_seconds = self._retry_delay_seconds(attempt)
            logger.warning(
                "Transient Codex transport failure (attempt %d/%d); retrying in %.1fs",
                attempt,
                max_attempts,
                delay_seconds,
            )
            if delay_seconds > 0:
                time.sleep(delay_seconds)

        if result is None:  # pragma: no cover - defensive, loop always assigns.
            result = RunResult(
                success=False,
                exit_code=-1,
                errors=["Codex execution produced no result"],
            )
        result.duration_seconds = time.monotonic() - start
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_command(
        self,
        prompt: str,
        repo_path: Path,
        *,
        use_json: bool,
        full_auto: bool,
        extra_args: list[str] | None,
    ) -> list[str]:
        cmd = [resolve_binary(self.codex_binary), "exec"]
        # Explicit workspace so sandbox allows writes in this directory
        cmd.extend(["--cd", str(repo_path)])
        if use_json:
            cmd.append("--json")
        if self.reasoning_effort != "inherit":
            cmd.extend(["-c", f"model_reasoning_effort={json.dumps(self.reasoning_effort)}"])
        if full_auto:
            if self.bypass_approvals_and_sandbox:
                cmd.append("--dangerously-bypass-approvals-and-sandbox")
            else:
                cmd.append("--full-auto")
                # Keep writes scoped to the selected repo and force non-interactive approval mode.
                cmd.extend(["--sandbox", self.sandbox_mode])
                cmd.extend(["-c", f"approval_policy={json.dumps(self.approval_policy)}"])
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
        cmd.append(prompt)
        return cmd

    def _should_pipe_prompt_via_stdin(
        self,
        prompt: str,
        *,
        repo_path: Path,
        use_json: bool,
        full_auto: bool,
        extra_args: list[str] | None,
    ) -> bool:
        """Return True when prompt should be supplied via stdin instead of argv.

        On Windows, large prompts can exceed either:
        - CreateProcess command-line limits (~32k), or
        - cmd.exe wrapper limits (~8k) when the resolved CLI is a .cmd/.bat shim.
        """
        if os.name != "nt":
            return len(prompt) >= _POSIX_PROMPT_ARG_LIMIT

        if not prompt:
            return False

        base_cmd = self._build_command(
            "",
            repo_path,
            use_json=use_json,
            full_auto=full_auto,
            extra_args=extra_args,
        )[:-1]
        try:
            estimated = len(subprocess.list2cmdline([*base_cmd, prompt]))
        except Exception:
            estimated = sum(len(part) for part in base_cmd) + len(prompt) + len(base_cmd) + 1

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

    def _execute(self, cmd: list[str], cwd: Path, *, stdin_text: str | None = None) -> RunResult:
        """Spawn the subprocess and consume its JSONL stdout."""
        env = {**os.environ, **self.env_overrides}
        inactivity_timeout = self.timeout if self.timeout > 0 else None
        execution = execute_streaming_json_command(
            cmd=cmd,
            cwd=cwd,
            env=env,
            timeout_seconds=self.timeout,
            parse_stdout_line=self._parse_line,
            process_name="Codex",
            stdin_text=stdin_text,
        )
        stderr_text = execution.stderr_text

        if execution.timed_out:
            timeout_msg = (
                f"Codex process timed out after {inactivity_timeout}s with no output activity"
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

    def _is_transient_network_failure(self, result: RunResult) -> bool:
        """Return True when result looks like a retriable transient transport failure."""
        if result.success:
            return False

        parts = [str(item or "") for item in result.errors]
        if result.final_message:
            parts.append(result.final_message)
        if not parts:
            return False

        haystack = "\n".join(parts).lower()
        return any(marker in haystack for marker in _TRANSIENT_NETWORK_ERROR_SUBSTRINGS)

    def _retry_delay_seconds(self, attempt: int) -> float:
        """Return exponential backoff delay for transient retry attempts."""
        base = max(0.0, float(self.transient_retry_backoff_seconds or 0.0))
        if base <= 0:
            return 0.0
        exponent = max(0, coerce_int(attempt) - 1)
        return min(_MAX_TRANSIENT_RETRY_BACKOFF_SECONDS, base * (2**exponent))

    # ------------------------------------------------------------------
    # JSONL parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_line(line: str) -> CodexEvent | None:
        """Attempt to parse one line of JSONL output into a CodexEvent."""
        try:
            data: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("Non-JSON line from codex: %s", line[:200])
            return None

        kind = _classify_event(data)
        text = _extract_text(data, kind)
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

        # Collect structural data
        file_changes: list[dict[str, Any]] = []
        command_execs: list[dict[str, Any]] = []
        usage = UsageInfo()
        final_message = ""
        agent_text_candidates: list[str] = []

        # Log event type distribution for debugging
        type_counts: dict[str, int] = {}
        for ev in events:
            type_counts[ev.kind.value] = type_counts.get(ev.kind.value, 0) + 1

            if ev.kind == EventKind.ERROR:
                errors.append(ev.text or json.dumps(ev.raw))
            elif ev.kind == EventKind.FILE_CHANGE:
                file_changes.append(ev.raw)
            elif ev.kind == EventKind.COMMAND_EXEC:
                command_execs.append(ev.raw)
            elif ev.kind == EventKind.TURN_COMPLETED:
                usage = _extract_usage(ev.raw)
            if ev.kind == EventKind.AGENT_MESSAGE and ev.text:
                cleaned = ev.text.strip()
                if cleaned:
                    agent_text_candidates.append(cleaned)
                    final_message = cleaned

        if type_counts:
            logger.info("Codex event types: %s", type_counts)

        if final_message and _looks_like_status_only_text(final_message):
            final_message = ""

        # Fallback 1: use latest meaningful agent-message text.
        if not final_message:
            for text in reversed(agent_text_candidates):
                if not _looks_like_status_only_text(text):
                    final_message = text
                    break

        # Fallback 2: non-JSON lines from raw output
        if not final_message and raw_lines:
            non_json = []
            for rl in raw_lines:
                try:
                    json.loads(rl)
                except json.JSONDecodeError:
                    if rl.strip():
                        non_json.append(rl)
            if non_json:
                candidate = "\n".join(non_json).strip()
                if candidate and not _looks_like_status_only_text(candidate):
                    final_message = candidate

        # If codex exits non-zero without stderr/error events, surface a
        # best-effort diagnostic so callers do not show "unknown error".
        if exit_code != 0 and not errors:
            inferred = _infer_error_from_events(events)
            if inferred:
                errors.append(inferred)
            elif final_message:
                errors.append(final_message[:500])
            else:
                errors.append(
                    f"Codex exited with status {exit_code} but produced no explicit error output"
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


def _classify_event(data: dict[str, Any]) -> EventKind:
    """Map a raw JSON object to one of the known event kinds.

    Codex CLI 0.98+ uses a nested structure::

        {"type": "item.completed", "item": {"type": "agent_message", ...}}
        {"type": "turn.completed", "usage": {...}}

    Earlier versions used flat ``{"type": "agent_message", ...}``.
    We handle both formats.
    """
    etype = (data.get("type") or data.get("event") or "").lower()

    # ── Codex 0.98+ nested format ────────────────────────────────
    if etype in ("item.completed", "item.started"):
        item = data.get("item") or {}
        item_type = (item.get("type") or "").lower().replace(".", "_").replace("-", "_")
        nested_map: dict[str, EventKind] = {
            "agent_message": EventKind.AGENT_MESSAGE,
            "message": EventKind.AGENT_MESSAGE,
            "command_execution": EventKind.COMMAND_EXEC,
            "file_change": EventKind.FILE_CHANGE,
            "file_edit": EventKind.FILE_CHANGE,
            "reasoning": EventKind.UNKNOWN,  # internal thinking, not actionable
        }
        return nested_map.get(item_type, EventKind.UNKNOWN)

    if etype == "turn.completed":
        return EventKind.TURN_COMPLETED

    if etype in ("thread.started", "turn.started"):
        return EventKind.UNKNOWN

    # ── Legacy flat format ────────────────────────────────────────
    etype_norm = etype.replace(".", "_").replace("-", "_")
    flat_map: dict[str, EventKind] = {
        "agent_message": EventKind.AGENT_MESSAGE,
        "message": EventKind.AGENT_MESSAGE,
        "output_text": EventKind.AGENT_MESSAGE,
        "response": EventKind.AGENT_MESSAGE,
        "assistant": EventKind.AGENT_MESSAGE,
        "result": EventKind.AGENT_MESSAGE,
        "file_change": EventKind.FILE_CHANGE,
        "file_edit": EventKind.FILE_CHANGE,
        "command_exec": EventKind.COMMAND_EXEC,
        "command_execution": EventKind.COMMAND_EXEC,
        "exec_command": EventKind.COMMAND_EXEC,
        "turn_completed": EventKind.TURN_COMPLETED,
        "turn_complete": EventKind.TURN_COMPLETED,
        "error": EventKind.ERROR,
    }
    return flat_map.get(etype_norm, EventKind.UNKNOWN)


def _extract_text(data: dict[str, Any], kind: EventKind) -> str | None:
    """Pull human-readable text from an event payload.

    Handles both Codex 0.98+ nested format (``item.content[].text``)
    and legacy flat format.
    """
    # Only message-like events should contribute text payloads.
    if kind not in {
        EventKind.AGENT_MESSAGE,
        EventKind.COMMAND_EXEC,
        EventKind.ERROR,
    }:
        return None

    # Codex 0.98+ nested: text is in item.text or item.content[].text.
    item = data.get("item")
    if isinstance(item, dict):
        if kind == EventKind.COMMAND_EXEC and item.get("type") == "command_execution":
            cmd = item.get("command", "")
            exit_code = item.get("exit_code", "?")
            if cmd:
                return f"[exec: {cmd[:200]}] (exit {exit_code})"

        if isinstance(item.get("text"), str) and item["text"].strip():
            return item["text"]

        content = item.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    t = block.get("text") or block.get("content") or ""
                    if t:
                        parts.append(t)
            joined = "\n".join(parts).strip()
            if joined:
                return joined

    # Legacy flat format.
    if kind == EventKind.COMMAND_EXEC and isinstance(data.get("command"), str):
        cmd = data["command"]
        exit_code = data.get("exit_code", "?")
        return f"[exec: {cmd[:200]}] (exit {exit_code})"

    for key in ("text", "message", "content", "output", "result"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val
        if isinstance(val, list):
            parts = [p.get("text", "") for p in val if isinstance(p, dict) and p.get("text")]
            joined = "\n".join(parts).strip()
            if joined:
                return joined
    return None


def _looks_like_status_only_text(text: str) -> bool:
    """Return True for generic CLI status text that is not useful task output."""
    normalized = re.sub(r"\s+", " ", (text or "")).strip().lower().replace("\u2019", "'")
    if not normalized:
        return True

    if (
        normalized.startswith("working in `")
        and "share the task you want implemented" in normalized
    ):
        return True

    placeholders = (
        "share the task you want implemented",
        "share the task you'd like implemented",
        "tell me what you want implemented",
        "provide the task you want implemented",
    )
    return any(p in normalized for p in placeholders)


def _extract_usage(data: dict[str, Any]) -> UsageInfo:
    """Extract token usage from a turn.completed event."""
    usage_raw: dict[str, Any] = {}
    top_level_usage = data.get("usage")
    if isinstance(top_level_usage, dict):
        usage_raw = top_level_usage
    else:
        nested_data = data.get("data")
        if isinstance(nested_data, dict):
            nested_usage = nested_data.get("usage")
            if isinstance(nested_usage, dict):
                usage_raw = nested_usage

    input_tokens = max(0, coerce_int(usage_raw.get("input_tokens", 0)))
    output_tokens = max(0, coerce_int(usage_raw.get("output_tokens", 0)))
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
        model=usage_raw.get("model") or data.get("model"),
    )


def _infer_error_from_events(events: list[CodexEvent]) -> str | None:
    """Extract a useful error string from non-success event payloads."""
    for ev in reversed(events):
        raw = ev.raw

        # Check top-level error-like keys first.
        for key in ("error", "message", "reason", "detail"):
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            if isinstance(val, dict):
                nested = val.get("message") or val.get("text") or val.get("detail")
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()

        # Check nested `item` payloads emitted by Codex 0.98+.
        item = raw.get("item")
        if isinstance(item, dict):
            item_type = str(item.get("type", "")).lower()
            if "error" in item_type:
                for key in ("text", "message", "error", "reason", "detail"):
                    val = item.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()

        # Some terminal events may encode status/exit text fields.
        status = raw.get("status")
        if isinstance(status, str) and status.lower() in {"error", "failed"}:
            text = raw.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()

    return None


register_agent("codex", CodexRunner)
