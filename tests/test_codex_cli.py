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
from codex_manager.schemas import CodexEvent, EventKind


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

    def test_bad_repo_path(self):
        runner = CodexRunner()
        result = runner.run("/nonexistent/path/12345", "test")
        assert result.success is False
        assert "does not exist" in result.errors[0]


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
