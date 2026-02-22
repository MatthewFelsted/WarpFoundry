"""Unit tests for claude_code module."""

from __future__ import annotations

from pathlib import Path

from codex_manager.claude_code import ClaudeCodeRunner, _extract_claude_usage
from codex_manager.schemas import CodexEvent, EventKind, RunResult


class TestClaudeCodeRunnerBuildCommand:
    def test_invalid_timeout_and_max_turns_are_coerced(self):
        runner = ClaudeCodeRunner(timeout="bad", max_turns="7")  # type: ignore[arg-type]
        assert runner.timeout == 0
        assert runner.max_turns == 7

    def test_negative_max_turns_is_clamped(self):
        runner = ClaudeCodeRunner(max_turns=-4)
        assert runner.max_turns == 0

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

    def test_run_uses_stdin_when_prompt_piping_enabled(self, monkeypatch, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        runner = ClaudeCodeRunner(claude_binary="claude")

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
        cmd = captured["cmd"]
        assert isinstance(cmd, list)
        prompt_index = cmd.index("-p") + 1
        assert cmd[prompt_index] == "-"
        assert captured["stdin_text"] == "very long prompt payload"

    def test_run_retries_with_stdin_when_command_line_is_too_long(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        runner = ClaudeCodeRunner(claude_binary="claude")

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
        first_prompt_index = captured[0][0].index("-p") + 1
        second_prompt_index = captured[1][0].index("-p") + 1
        assert captured[0][0][first_prompt_index] == prompt
        assert captured[0][1] is None
        assert captured[1][0][second_prompt_index] == "-"
        assert captured[1][1] == prompt

    def test_should_pipe_prompt_early_for_windows_batch_shims(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        runner = ClaudeCodeRunner(claude_binary=str(tmp_path / "claude.cmd"))
        monkeypatch.setattr("codex_manager.claude_code.os.name", "nt")

        prompt = "x" * 7000
        assert (
            runner._should_pipe_prompt_via_stdin(
                prompt,
                full_auto=True,
                extra_args=None,
            )
            is True
        )


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
