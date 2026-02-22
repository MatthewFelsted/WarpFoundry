"""Unit tests for codex_cli module."""

from __future__ import annotations

from pathlib import Path

import pytest

from codex_manager.codex_cli import (
    CodexRunner,
    _classify_event,
    _extract_text,
    _extract_usage,
)
from codex_manager.schemas import CodexEvent, EventKind, RunResult


class TestClassifyEvent:
    @pytest.mark.parametrize(
        "data,expected",
        [
            ({"type": "agent_message"}, EventKind.AGENT_MESSAGE),
            ({"type": "message"}, EventKind.AGENT_MESSAGE),
            ({"type": "file_change"}, EventKind.FILE_CHANGE),
            ({"type": "file_edit"}, EventKind.FILE_CHANGE),
            ({"type": "command_exec"}, EventKind.COMMAND_EXEC),
            ({"type": "turn.completed"}, EventKind.TURN_COMPLETED),
            ({"type": "error"}, EventKind.ERROR),
            ({"event": "agent_message"}, EventKind.AGENT_MESSAGE),
            ({"type": "something_else"}, EventKind.UNKNOWN),
            ({}, EventKind.UNKNOWN),
        ],
    )
    def test_classification(self, data, expected):
        assert _classify_event(data) == expected


class TestExtractText:
    def test_simple_text(self):
        assert _extract_text({"text": "hello"}, EventKind.AGENT_MESSAGE) == "hello"

    def test_content_list(self):
        data = {"content": [{"text": "part1"}, {"text": "part2"}]}
        result = _extract_text(data, EventKind.AGENT_MESSAGE)
        assert "part1" in result
        assert "part2" in result

    def test_non_message_returns_none(self):
        assert _extract_text({"text": "hello"}, EventKind.FILE_CHANGE) is None


class TestExtractUsage:
    def test_direct_usage(self):
        data = {"usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}}
        u = _extract_usage(data)
        assert u.input_tokens == 10
        assert u.output_tokens == 20

    def test_nested_usage(self):
        data = {"data": {"usage": {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15}}}
        u = _extract_usage(data)
        assert u.total_tokens == 15

    def test_missing_usage(self):
        u = _extract_usage({})
        assert u.total_tokens == 0

    def test_cached_usage_fallback_total(self):
        data = {"usage": {"input_tokens": 10, "output_tokens": 5, "cached_input_tokens": 7}}
        u = _extract_usage(data)
        assert u.total_tokens == 22

    def test_string_token_values_are_coerced(self):
        data = {"usage": {"input_tokens": "10", "output_tokens": "5", "total_tokens": "15"}}
        u = _extract_usage(data)
        assert u.input_tokens == 10
        assert u.output_tokens == 5
        assert u.total_tokens == 15

    def test_malformed_usage_payload_is_safe(self):
        data = {
            "usage": "unexpected",
            "data": "bad-shape",
        }
        u = _extract_usage(data)
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.total_tokens == 0

    def test_non_numeric_usage_values_fall_back_to_zero(self):
        data = {
            "usage": {
                "input_tokens": "n/a",
                "output_tokens": None,
                "cached_input_tokens": "3",
                "cache_read_input_tokens": "invalid",
                "cache_creation_input_tokens": "",
                "total_tokens": "bad",
            }
        }
        u = _extract_usage(data)
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.total_tokens == 3


class TestCodexRunnerBuildCommand:
    def test_basic(self):
        runner = CodexRunner()
        cmd = runner._build_command(
            "do stuff", Path("/repo"), use_json=True, full_auto=False, extra_args=None
        )
        assert Path(cmd[0]).name.lower().startswith("codex")
        assert cmd[1] == "exec"
        assert "--cd" in cmd
        assert str(Path("/repo")) in cmd
        assert "--json" in cmd
        assert any(arg == 'model_reasoning_effort="xhigh"' for arg in cmd)
        assert "--model" in cmd
        assert "gpt-5.3-codex" in cmd
        assert "do stuff" in cmd

    def test_full_auto(self):
        runner = CodexRunner()
        cmd = runner._build_command(
            "fix", Path("/repo"), use_json=True, full_auto=True, extra_args=None
        )
        assert "--full-auto" in cmd
        assert "--sandbox" in cmd
        assert "workspace-write" in cmd
        assert "-c" in cmd
        assert any(arg == 'approval_policy="never"' for arg in cmd)

    def test_custom_sandbox_and_approval_policy(self):
        runner = CodexRunner(sandbox_mode="danger-full-access", approval_policy="on-failure")
        cmd = runner._build_command(
            "fix", Path("/repo"), use_json=True, full_auto=True, extra_args=None
        )
        assert "danger-full-access" in cmd
        assert any(arg == 'approval_policy="on-failure"' for arg in cmd)

    def test_bypass_approvals_and_sandbox(self):
        runner = CodexRunner(bypass_approvals_and_sandbox=True)
        cmd = runner._build_command(
            "fix", Path("/repo"), use_json=True, full_auto=True, extra_args=None
        )
        assert "--dangerously-bypass-approvals-and-sandbox" in cmd
        assert "--sandbox" not in cmd

    def test_extra_args(self):
        runner = CodexRunner()
        cmd = runner._build_command(
            "fix", Path("/repo"), use_json=False, full_auto=False, extra_args=["--model", "o3"]
        )
        assert "--model" in cmd
        assert "o3" in cmd
        assert "gpt-5.3-codex" not in cmd
        assert "--json" not in cmd

    def test_custom_default_model(self):
        runner = CodexRunner(model="gpt-5.2")
        cmd = runner._build_command(
            "fix", Path("/repo"), use_json=True, full_auto=False, extra_args=None
        )
        assert "--model" in cmd
        assert "gpt-5.2" in cmd

    def test_custom_reasoning_effort(self):
        runner = CodexRunner(reasoning_effort="medium")
        cmd = runner._build_command(
            "fix", Path("/repo"), use_json=True, full_auto=False, extra_args=None
        )
        assert any(arg == 'model_reasoning_effort="medium"' for arg in cmd)

    def test_inherit_reasoning_effort_omits_override(self):
        runner = CodexRunner(reasoning_effort="inherit")
        cmd = runner._build_command(
            "fix", Path("/repo"), use_json=True, full_auto=False, extra_args=None
        )
        assert not any(arg.startswith("model_reasoning_effort=") for arg in cmd)

    def test_invalid_reasoning_effort_falls_back_to_xhigh(self):
        runner = CodexRunner(reasoning_effort="extreme")
        cmd = runner._build_command(
            "fix", Path("/repo"), use_json=True, full_auto=False, extra_args=None
        )
        assert any(arg == 'model_reasoning_effort="xhigh"' for arg in cmd)

    def test_invalid_timeout_is_coerced_to_zero(self):
        runner = CodexRunner(timeout="not-a-number")  # type: ignore[arg-type]
        assert runner.timeout == 0

    def test_blank_sandbox_and_approval_policy_use_defaults(self):
        runner = CodexRunner(sandbox_mode="   ", approval_policy="")
        assert runner.sandbox_mode == "workspace-write"
        assert runner.approval_policy == "never"

    def test_bad_repo_path(self):
        runner = CodexRunner()
        result = runner.run("/nonexistent/path/12345", "test")
        assert result.success is False
        assert "does not exist" in result.errors[0]

    def test_run_uses_stdin_when_prompt_piping_enabled(self, monkeypatch, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        runner = CodexRunner(codex_binary="codex")

        monkeypatch.setattr(
            runner,
            "_should_pipe_prompt_via_stdin",
            lambda *_args, **_kwargs: True,
        )

        captured: dict[str, object] = {}

        def _fake_execute(cmd: list[str], cwd: Path, *, stdin_text: str | None = None) -> RunResult:
            captured["cmd"] = cmd
            captured["cwd"] = cwd
            captured["stdin_text"] = stdin_text
            return RunResult(success=True, exit_code=0)

        monkeypatch.setattr(runner, "_execute", _fake_execute)

        result = runner.run(repo, "very long prompt payload", full_auto=True)
        assert result.success is True
        assert captured["cwd"] == repo.resolve()
        assert isinstance(captured["cmd"], list)
        assert captured["cmd"][-1] == "-"
        assert captured["stdin_text"] == "very long prompt payload"

    def test_run_retries_with_stdin_when_command_line_is_too_long(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        runner = CodexRunner(codex_binary="codex")

        monkeypatch.setattr(
            runner,
            "_should_pipe_prompt_via_stdin",
            lambda *_args, **_kwargs: False,
        )

        prompt = "very long prompt payload"
        captured: list[tuple[list[str], str | None]] = []

        def _fake_execute(cmd: list[str], cwd: Path, *, stdin_text: str | None = None) -> RunResult:
            assert cwd == repo.resolve()
            captured.append((list(cmd), stdin_text))
            if len(captured) == 1:
                raise OSError("The command line is too long.")
            return RunResult(success=True, exit_code=0, final_message="ok")

        monkeypatch.setattr(runner, "_execute", _fake_execute)

        result = runner.run(repo, prompt, full_auto=True)

        assert result.success is True
        assert len(captured) == 2
        assert captured[0][0][-1] == prompt
        assert captured[0][1] is None
        assert captured[1][0][-1] == "-"
        assert captured[1][1] == prompt

    def test_run_logging_omits_raw_prompt_text(self, monkeypatch, tmp_path: Path, caplog) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        runner = CodexRunner(codex_binary="codex")
        prompt = "api_key=sk-proj-super-secret-value"

        monkeypatch.setattr(
            runner,
            "_should_pipe_prompt_via_stdin",
            lambda *_args, **_kwargs: False,
        )
        monkeypatch.setattr(
            runner,
            "_execute",
            lambda *_args, **_kwargs: RunResult(success=True, exit_code=0),
        )

        with caplog.at_level("INFO", logger="codex_manager.codex_cli"):
            result = runner.run(repo, prompt)

        assert result.success is True
        joined = "\n".join(record.getMessage() for record in caplog.records)
        assert prompt not in joined
        assert "prompt_len=" in joined
        assert "prompt_sha256=" in joined

    def test_should_pipe_prompt_early_for_windows_batch_shims(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        runner = CodexRunner(codex_binary=str(tmp_path / "codex.cmd"))
        monkeypatch.setattr("codex_manager.codex_cli.os.name", "nt")

        # Above cmd.exe practical limit once quoted.
        prompt = "x" * 7000
        assert (
            runner._should_pipe_prompt_via_stdin(
                prompt,
                repo_path=repo,
                use_json=True,
                full_auto=True,
                extra_args=None,
            )
            is True
        )

    def test_transient_network_error_retries_then_recovers(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        runner = CodexRunner(
            codex_binary="codex",
            transient_network_retries=2,
            transient_retry_backoff_seconds=0,
        )

        calls = {"count": 0}

        def _fake_execute(_cmd: list[str], cwd: Path, *, stdin_text: str | None = None) -> RunResult:
            calls["count"] += 1
            if calls["count"] == 1:
                return RunResult(
                    success=False,
                    exit_code=1,
                    errors=[
                        "stream disconnected before completion: "
                        "error sending request for url (https://chatgpt.com/backend-api/codex/models)"
                    ],
                )
            return RunResult(success=True, exit_code=0, final_message="ok")

        monkeypatch.setattr(runner, "_execute", _fake_execute)
        result = runner.run(repo, "retry me")

        assert result.success is True
        assert calls["count"] == 2

    def test_non_transient_failure_does_not_retry(self, monkeypatch, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        runner = CodexRunner(
            codex_binary="codex",
            transient_network_retries=2,
            transient_retry_backoff_seconds=0,
        )

        calls = {"count": 0}

        def _fake_execute(_cmd: list[str], cwd: Path, *, stdin_text: str | None = None) -> RunResult:
            calls["count"] += 1
            return RunResult(success=False, exit_code=1, errors=["permission denied"])

        monkeypatch.setattr(runner, "_execute", _fake_execute)
        result = runner.run(repo, "fail fast")

        assert result.success is False
        assert calls["count"] == 1

    def test_transient_network_retry_respects_retry_cap(self, monkeypatch, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        runner = CodexRunner(
            codex_binary="codex",
            transient_network_retries=1,
            transient_retry_backoff_seconds=0,
        )

        calls = {"count": 0}

        def _fake_execute(_cmd: list[str], cwd: Path, *, stdin_text: str | None = None) -> RunResult:
            calls["count"] += 1
            return RunResult(
                success=False,
                exit_code=1,
                errors=["network error: error sending request for url (https://chatgpt.com)"],
            )

        monkeypatch.setattr(runner, "_execute", _fake_execute)
        result = runner.run(repo, "still failing")

        assert result.success is False
        assert calls["count"] == 2


class TestCodexRunnerAggregate:
    def test_infers_error_from_unknown_event_on_nonzero_exit(self):
        events = [
            CodexEvent(
                kind=EventKind.UNKNOWN,
                raw={"type": "turn.failed", "error": "Command blocked by sandbox"},
            )
        ]
        result = CodexRunner._aggregate(events, exit_code=1, stderr="", raw_lines=[])
        assert result.success is False
        assert "blocked by sandbox" in result.errors[0].lower()

    def test_ignores_placeholder_status_text_as_final_message(self):
        events = [
            CodexEvent(
                kind=EventKind.UNKNOWN,
                raw={"type": "thread.started"},
                text="Working in `C:\\repo` now. Share the task you want implemented and I'll proceed.",
            )
        ]
        result = CodexRunner._aggregate(events, exit_code=0, stderr="", raw_lines=[])
        assert result.success is True
        assert result.final_message == ""
