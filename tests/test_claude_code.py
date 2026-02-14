"""Unit tests for claude_code module."""

from __future__ import annotations

from pathlib import Path

from codex_manager.claude_code import ClaudeCodeRunner, _extract_claude_usage
from codex_manager.schemas import CodexEvent, EventKind


class TestClaudeCodeRunnerBuildCommand:
    def test_basic(self):
        runner = ClaudeCodeRunner()
        cmd = runner._build_command("hello", full_auto=False, extra_args=None)
        assert Path(cmd[0]).name.lower().startswith("claude")
        assert "-p" in cmd
        assert "hello" in cmd
        assert "--output-format" in cmd
        assert "stream-json" in cmd

    def test_full_auto(self):
        runner = ClaudeCodeRunner()
        cmd = runner._build_command("hello", full_auto=True, extra_args=None)
        assert "--dangerously-skip-permissions" in cmd

    def test_extra_args_model_override_skips_default_model(self):
        runner = ClaudeCodeRunner(model="claude-3-7-sonnet")
        cmd = runner._build_command(
            "hello",
            full_auto=False,
            extra_args=["--model", "claude-opus-4-1"],
        )
        assert cmd.count("--model") == 1
        assert "claude-opus-4-1" in cmd
        assert "claude-3-7-sonnet" not in cmd

    def test_model_is_trimmed(self):
        runner = ClaudeCodeRunner(model="  claude-3-7-sonnet  ")
        cmd = runner._build_command("hello", full_auto=False, extra_args=None)
        model_idx = cmd.index("--model")
        assert cmd[model_idx + 1] == "claude-3-7-sonnet"


class TestClaudeCodeRunnerAggregate:
    def test_infers_error_from_result_payload(self):
        events = [
            CodexEvent(
                kind=EventKind.TURN_COMPLETED,
                raw={"type": "result", "result": {"error": "permission denied"}},
            )
        ]
        result = ClaudeCodeRunner._aggregate(events, exit_code=1, stderr="", raw_lines=[])
        assert result.success is False
        assert "permission denied" in result.errors[0].lower()


class TestExtractClaudeUsage:
    def test_string_usage_values_are_coerced(self):
        usage = _extract_claude_usage(
            {
                "usage": {
                    "input_tokens": "8",
                    "output_tokens": "4",
                    "cached_input_tokens": "3",
                    "cache_read_input_tokens": "2",
                    "cache_creation_input_tokens": "1",
                }
            }
        )
        assert usage.input_tokens == 8
        assert usage.output_tokens == 4
        assert usage.total_tokens == 18

    def test_total_tokens_preferred_when_present(self):
        usage = _extract_claude_usage(
            {
                "usage": {
                    "input_tokens": 8,
                    "output_tokens": 4,
                    "total_tokens": 30,
                }
            }
        )
        assert usage.total_tokens == 30

    def test_malformed_usage_payload_is_safe(self):
        usage = _extract_claude_usage({"usage": "invalid", "result": "not-a-dict"})
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
