"""Integration tests for runner subprocess behavior using stub CLIs."""

from __future__ import annotations

import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from codex_manager.claude_code import ClaudeCodeRunner
from codex_manager.codex_cli import CodexRunner

pytestmark = pytest.mark.integration


def _make_stub_cli(tmp_path: Path, name: str, script_body: str) -> str:
    """Create a cross-platform executable wrapper for a Python stub script."""
    impl = tmp_path / f"{name}_impl.py"
    impl.write_text(textwrap.dedent(script_body), encoding="utf-8")

    if os.name == "nt":
        wrapper = tmp_path / f"{name}.cmd"
        wrapper.write_text(
            f'@echo off\r\n"{sys.executable}" "{impl}" %*\r\n',
            encoding="utf-8",
        )
        return str(wrapper)

    wrapper = tmp_path / name
    wrapper.write_text(
        f'#!/usr/bin/env sh\n"{sys.executable}" "{impl}" "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    return str(wrapper)


CODEX_STUB_SCRIPT = """
import json
import os
import sys
import time
from pathlib import Path


def emit(obj):
    print(json.dumps(obj), flush=True)


def parse_cd(argv):
    for i, token in enumerate(argv):
        if token == "--cd" and i + 1 < len(argv):
            return argv[i + 1]
    return "."


args = sys.argv[1:]
if args and args[0] == "exec":
    args = args[1:]

mode = os.environ.get("STUB_MODE", "success")
repo = Path(parse_cd(args))

if mode == "success":
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "stub-created.txt").write_text("ok\\n", encoding="utf-8")
    emit({"type": "thread.started", "thread_id": "stub"})
    emit({"type": "turn.started"})
    emit({"type": "item.completed", "item": {"type": "file_change", "path": "stub-created.txt"}})
    emit({"type": "item.completed", "item": {"type": "agent_message", "text": "created"}})
    emit({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}})
    sys.exit(0)

if mode == "blocked":
    emit({"type": "turn.failed", "error": "Permission denied by sandbox"})
    sys.exit(1)

if mode == "nonzero_message":
    emit({"type": "item.completed", "item": {"type": "agent_message", "text": "Tool crashed with no stderr"}})
    sys.exit(1)

if mode == "sleep_then_success":
    time.sleep(2.2)
    emit({"type": "item.completed", "item": {"type": "agent_message", "text": "late success"}})
    emit({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}})
    sys.exit(0)

if mode == "heartbeat":
    for i in range(4):
        emit({"type": "item.completed", "item": {"type": "agent_message", "text": f"tick {i}"}})
        time.sleep(0.6)
    emit({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}})
    sys.exit(0)

emit({"type": "error", "message": "unknown mode"})
sys.exit(1)
"""


CLAUDE_STUB_SCRIPT = """
import json
import os
import sys
import time


def emit(obj):
    print(json.dumps(obj), flush=True)


mode = os.environ.get("STUB_MODE", "nonzero_message")

if mode == "nonzero_message":
    emit({
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": "Tool crashed with no stderr"}]},
    })
    sys.exit(1)

if mode == "sleep_then_success":
    time.sleep(2.2)
    emit({"type": "result", "result": "OK", "usage": {"input_tokens": 1, "output_tokens": 1}})
    sys.exit(0)

if mode == "heartbeat":
    for i in range(4):
        emit({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": f"tick {i}"}]},
        })
        time.sleep(0.6)
    emit({"type": "result", "result": "OK", "usage": {"input_tokens": 1, "output_tokens": 1}})
    sys.exit(0)

emit({"type": "result", "result": "OK", "usage": {"input_tokens": 1, "output_tokens": 1}})
sys.exit(0)
"""


class TestCodexRunnerIntegration:
    def test_successful_write_in_repo(self, tmp_path: Path):
        stub = _make_stub_cli(tmp_path, "codex_stub", CODEX_STUB_SCRIPT)
        repo = tmp_path / "repo"
        repo.mkdir()

        runner = CodexRunner(codex_binary=stub, timeout=30, env_overrides={"STUB_MODE": "success"})
        result = runner.run(repo, "write file", full_auto=True)

        assert result.success is True
        assert (repo / "stub-created.txt").exists()

    def test_parser_failures_do_not_abort_subprocess_collection(self, tmp_path: Path):
        stub = _make_stub_cli(tmp_path, "codex_stub", CODEX_STUB_SCRIPT)
        repo = tmp_path / "repo"
        repo.mkdir()

        runner = CodexRunner(codex_binary=stub, timeout=30, env_overrides={"STUB_MODE": "success"})

        def _raise_parser_error(_line: str):
            raise RuntimeError("bad parser")

        runner._parse_line = _raise_parser_error  # type: ignore[method-assign]
        result = runner.run(repo, "write file", full_auto=True)

        assert result.success is True
        assert result.events == []

    def test_blocked_permission_path_surfaces_error(self, tmp_path: Path):
        stub = _make_stub_cli(tmp_path, "codex_stub", CODEX_STUB_SCRIPT)
        repo = tmp_path / "repo"
        repo.mkdir()

        runner = CodexRunner(codex_binary=stub, timeout=30, env_overrides={"STUB_MODE": "blocked"})
        result = runner.run(repo, "blocked", full_auto=True)

        assert result.success is False
        assert result.errors
        assert any(
            "permission denied" in err.lower() or "sandbox" in err.lower() for err in result.errors
        )

    def test_nonzero_exit_without_stderr_still_has_error_text(self, tmp_path: Path):
        stub = _make_stub_cli(tmp_path, "codex_stub", CODEX_STUB_SCRIPT)
        repo = tmp_path / "repo"
        repo.mkdir()

        runner = CodexRunner(
            codex_binary=stub,
            timeout=30,
            env_overrides={"STUB_MODE": "nonzero_message"},
        )
        result = runner.run(repo, "trigger message", full_auto=True)

        assert result.success is False
        assert result.errors
        assert "crashed" in result.errors[0].lower()

    def test_inactivity_timeout_triggers_when_no_output(self, tmp_path: Path):
        stub = _make_stub_cli(tmp_path, "codex_stub", CODEX_STUB_SCRIPT)
        repo = tmp_path / "repo"
        repo.mkdir()

        runner = CodexRunner(
            codex_binary=stub,
            timeout=1,
            env_overrides={"STUB_MODE": "sleep_then_success"},
        )
        result = runner.run(repo, "wait", full_auto=True)

        assert result.success is False
        assert any("timed out" in err.lower() for err in result.errors)

    def test_zero_timeout_disables_inactivity_cutoff(self, tmp_path: Path):
        stub = _make_stub_cli(tmp_path, "codex_stub", CODEX_STUB_SCRIPT)
        repo = tmp_path / "repo"
        repo.mkdir()

        runner = CodexRunner(
            codex_binary=stub,
            timeout=0,
            env_overrides={"STUB_MODE": "sleep_then_success"},
        )
        result = runner.run(repo, "wait", full_auto=True)

        assert result.success is True
        assert "late success" in (result.final_message or "")

    def test_periodic_output_prevents_timeout(self, tmp_path: Path):
        stub = _make_stub_cli(tmp_path, "codex_stub", CODEX_STUB_SCRIPT)
        repo = tmp_path / "repo"
        repo.mkdir()

        runner = CodexRunner(
            codex_binary=stub,
            timeout=1,
            env_overrides={"STUB_MODE": "heartbeat"},
        )
        result = runner.run(repo, "heartbeat", full_auto=True)

        assert result.success is True


class TestClaudeCodeRunnerIntegration:
    def test_nonzero_exit_without_stderr_still_has_error_text(self, tmp_path: Path):
        stub = _make_stub_cli(tmp_path, "claude_stub", CLAUDE_STUB_SCRIPT)
        repo = tmp_path / "repo"
        repo.mkdir()

        runner = ClaudeCodeRunner(
            claude_binary=stub,
            timeout=30,
            env_overrides={"STUB_MODE": "nonzero_message"},
        )
        result = runner.run(repo, "trigger message", full_auto=True)

        assert result.success is False
        assert result.errors
        assert "crashed" in result.errors[0].lower()

    def test_inactivity_timeout_triggers_when_no_output(self, tmp_path: Path):
        stub = _make_stub_cli(tmp_path, "claude_stub", CLAUDE_STUB_SCRIPT)
        repo = tmp_path / "repo"
        repo.mkdir()

        runner = ClaudeCodeRunner(
            claude_binary=stub,
            timeout=1,
            env_overrides={"STUB_MODE": "sleep_then_success"},
        )
        result = runner.run(repo, "wait", full_auto=True)

        assert result.success is False
        assert any("timed out" in err.lower() for err in result.errors)

    def test_zero_timeout_disables_inactivity_cutoff(self, tmp_path: Path):
        stub = _make_stub_cli(tmp_path, "claude_stub", CLAUDE_STUB_SCRIPT)
        repo = tmp_path / "repo"
        repo.mkdir()

        runner = ClaudeCodeRunner(
            claude_binary=stub,
            timeout=0,
            env_overrides={"STUB_MODE": "sleep_then_success"},
        )
        result = runner.run(repo, "wait", full_auto=True)

        assert result.success is True

    def test_periodic_output_prevents_timeout(self, tmp_path: Path):
        stub = _make_stub_cli(tmp_path, "claude_stub", CLAUDE_STUB_SCRIPT)
        repo = tmp_path / "repo"
        repo.mkdir()

        runner = ClaudeCodeRunner(
            claude_binary=stub,
            timeout=1,
            env_overrides={"STUB_MODE": "heartbeat"},
        )
        result = runner.run(repo, "heartbeat", full_auto=True)

        assert result.success is True
