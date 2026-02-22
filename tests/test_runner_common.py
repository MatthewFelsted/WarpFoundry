"""Tests for shared runner helpers."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import codex_manager.runner_common as runner_common_module
from codex_manager.runner_common import (
    coerce_int,
    execute_streaming_json_command,
    execute_with_prompt_transport_fallback,
    resolve_binary,
)
from codex_manager.schemas import CodexEvent, EventKind, RunResult


def _make_executable(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    if os.name == "nt":
        path.write_text("@echo off\r\nexit /b 0\r\n", encoding="utf-8")
    else:
        path.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_coerce_int_handles_non_finite_floats() -> None:
    assert coerce_int(float("nan")) == 0
    assert coerce_int(float("inf")) == 0
    assert coerce_int(float("-inf")) == 0


def test_resolve_binary_expands_environment_variables(monkeypatch, tmp_path: Path) -> None:
    var_name = "CODEX_MANAGER_TEST_BIN_DIR"
    monkeypatch.setenv(var_name, str(tmp_path))

    if os.name == "nt":
        tool = _make_executable(tmp_path, "codex-tool.cmd")
        configured = f"%{var_name}%\\{tool.name}"
    else:
        tool = _make_executable(tmp_path, "codex-tool")
        configured = f"${var_name}/{tool.name}"

    resolved = resolve_binary(configured)
    assert Path(resolved).resolve() == tool.resolve()


def test_resolve_binary_accepts_wrapped_quotes(tmp_path: Path) -> None:
    if os.name == "nt":
        tool = _make_executable(tmp_path, "codex tool.cmd")
    else:
        tool = _make_executable(tmp_path, "codex tool")

    resolved = resolve_binary(f'"{tool}"')
    assert Path(resolved).resolve() == tool.resolve()


def test_streaming_process_isolation_kwargs_matches_platform() -> None:
    kwargs = runner_common_module._streaming_process_isolation_kwargs()
    if os.name == "nt":
        expected_flag = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        if expected_flag:
            assert int(kwargs.get("creationflags") or 0) & expected_flag
        else:
            assert kwargs == {}
    else:
        assert kwargs.get("start_new_session") is True


def test_execute_streaming_json_command_limits_captured_output(tmp_path: Path) -> None:
    script = (
        "import json\n"
        "for i in range(30):\n"
        "    print(json.dumps({'i': i, 'text': f'line-{i}'}), flush=True)\n"
    )
    cmd = [sys.executable, "-c", script]

    def _parse_line(line: str) -> CodexEvent:
        payload = json.loads(line)
        return CodexEvent(kind=EventKind.AGENT_MESSAGE, raw=payload, text=str(payload["i"]))

    result = execute_streaming_json_command(
        cmd=cmd,
        cwd=tmp_path,
        env=dict(os.environ),
        timeout_seconds=10,
        parse_stdout_line=_parse_line,
        process_name="test-runner",
        max_events=5,
        max_stdout_lines=7,
        max_stderr_lines=3,
    )

    assert result.exit_code == 0
    assert result.timed_out is False
    assert [int(ev.text or "-1") for ev in result.events] == [25, 26, 27, 28, 29]
    assert len(result.raw_lines) == 7
    assert '"i": 23' in result.raw_lines[0]
    assert '"i": 29' in result.raw_lines[-1]
    assert result.stderr_lines == []


def test_execute_streaming_json_command_writes_stdin_text(tmp_path: Path) -> None:
    script = (
        "import json\n"
        "import sys\n"
        "payload = sys.stdin.read().strip()\n"
        "print(json.dumps({'stdin': payload}), flush=True)\n"
    )
    cmd = [sys.executable, "-c", script]

    def _parse_line(line: str) -> CodexEvent:
        payload = json.loads(line)
        return CodexEvent(
            kind=EventKind.AGENT_MESSAGE,
            raw=payload,
            text=str(payload.get("stdin", "")),
        )

    result = execute_streaming_json_command(
        cmd=cmd,
        cwd=tmp_path,
        env=dict(os.environ),
        timeout_seconds=10,
        parse_stdout_line=_parse_line,
        process_name="test-runner",
        stdin_text="hello from stdin",
    )

    assert result.exit_code == 0
    assert result.timed_out is False
    assert len(result.events) == 1
    assert result.events[0].text == "hello from stdin"


def test_execute_with_prompt_transport_fallback_retries_with_stdin(tmp_path: Path) -> None:
    prompt = "very long prompt payload"
    captured: list[tuple[list[str], Path, str | None]] = []

    def _build_command(prompt_arg: str) -> list[str]:
        return ["runner", prompt_arg]

    def _execute(command: list[str], cwd: Path, stdin_text: str | None) -> RunResult:
        captured.append((list(command), cwd, stdin_text))
        if len(captured) == 1:
            raise OSError("The command line is too long.")
        return RunResult(success=True, exit_code=0, final_message="ok")

    outcome = execute_with_prompt_transport_fallback(
        cwd=tmp_path,
        prompt=prompt,
        use_stdin_prompt=False,
        process_name="Test Runner",
        build_command=_build_command,
        execute=_execute,
    )

    assert outcome.error is None
    assert outcome.used_stdin_prompt is True
    assert outcome.result is not None
    assert outcome.result.success is True
    assert len(captured) == 2
    assert captured[0][0][-1] == prompt
    assert captured[0][2] is None
    assert captured[1][0][-1] == "-"
    assert captured[1][2] == prompt


def test_execute_with_prompt_transport_fallback_returns_non_overflow_errors(tmp_path: Path) -> None:
    def _build_command(prompt_arg: str) -> list[str]:
        return ["runner", prompt_arg]

    def _execute(command: list[str], cwd: Path, stdin_text: str | None) -> RunResult:
        raise OSError("permission denied")

    outcome = execute_with_prompt_transport_fallback(
        cwd=tmp_path,
        prompt="hello",
        use_stdin_prompt=False,
        process_name="Test Runner",
        build_command=_build_command,
        execute=_execute,
    )

    assert outcome.result is None
    assert isinstance(outcome.error, OSError)
    assert outcome.used_stdin_prompt is False
