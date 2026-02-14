"""Unit tests for eval_tools module."""

from __future__ import annotations

from codex_manager.eval_tools import RepoEvaluator, _summarise_output, parse_test_command
from codex_manager.schemas import TestOutcome


class TestSummariseOutput:
    def test_short_text(self):
        text = "All 5 tests passed."
        assert _summarise_output(text) == text

    def test_long_text_truncated(self):
        lines = [f"line {i}" for i in range(100)]
        result = _summarise_output("\n".join(lines), max_lines=30)
        assert "lines omitted" in result


class TestRepoEvaluator:
    def test_custom_test_cmd(self):
        evaluator = RepoEvaluator(test_cmd=["echo", "ok"])
        assert evaluator.test_cmd == ["echo", "ok"]

    def test_custom_test_cmd_is_copied(self):
        command = ["echo", "ok"]
        evaluator = RepoEvaluator(test_cmd=command)
        command.append("mutated")
        assert evaluator.test_cmd == ["echo", "ok"]

    def test_default_test_cmd(self):
        evaluator = RepoEvaluator()
        assert evaluator.test_cmd == ["python", "-m", "pytest", "-q"]

    def test_run_tests_skips_when_test_cmd_empty(self, tmp_path):
        evaluator = RepoEvaluator(test_cmd=["echo", "ok"])
        evaluator.test_cmd = []

        outcome, summary, exit_code = evaluator._run_tests(tmp_path)
        assert outcome == TestOutcome.SKIPPED
        assert "empty test command" in summary
        assert exit_code == 0

    def test_run_tests_returns_error_on_invalid_configuration(self, monkeypatch, tmp_path):
        evaluator = RepoEvaluator(test_cmd=["echo", "ok"])
        evaluator.timeout = -1

        def _raise_value_error(*args, **kwargs):
            raise ValueError("timeout out of range")

        monkeypatch.setattr("subprocess.run", _raise_value_error)

        outcome, summary, exit_code = evaluator._run_tests(tmp_path)
        assert outcome == TestOutcome.ERROR
        assert "Invalid test command configuration" in summary
        assert exit_code == -1


class TestParseTestCommand:
    def test_empty_returns_none(self):
        assert parse_test_command("") is None
        assert parse_test_command("   ") is None
        assert parse_test_command(None) is None

    def test_parses_quoted_args(self):
        assert parse_test_command('pytest -k "slow suite" -q') == [
            "pytest",
            "-k",
            "slow suite",
            "-q",
        ]

    def test_parses_windows_style_quoted_executable(self):
        assert parse_test_command('"C:\\Program Files\\Python\\python.exe" -m pytest -q') == [
            r"C:\Program Files\Python\python.exe",
            "-m",
            "pytest",
            "-q",
        ]

    def test_normalizes_sequence_input(self):
        assert parse_test_command(["pytest", " ", "-q"]) == ["pytest", "-q"]

    def test_normalizes_sequence_input_ignores_none(self):
        assert parse_test_command(["pytest", None, " ", "-q"]) == ["pytest", "-q"]

    def test_parses_unquoted_windows_backslash_path(self, monkeypatch):
        monkeypatch.setattr("codex_manager.eval_tools._is_windows_platform", lambda: True)
        assert parse_test_command(r"python -m pytest tests\test_eval_tools.py -q") == [
            "python",
            "-m",
            "pytest",
            r"tests\test_eval_tools.py",
            "-q",
        ]

    def test_parses_quoted_args_on_windows(self, monkeypatch):
        monkeypatch.setattr("codex_manager.eval_tools._is_windows_platform", lambda: True)
        assert parse_test_command('pytest -k "slow suite" -q') == [
            "pytest",
            "-k",
            "slow suite",
            "-q",
        ]


def test_evaluate_includes_changed_files(monkeypatch, tmp_path):
    evaluator = RepoEvaluator(test_cmd=["echo", "ok"])

    monkeypatch.setattr(
        RepoEvaluator,
        "_run_tests",
        lambda self, cwd: (TestOutcome.PASSED, "ok", 0),
    )
    monkeypatch.setattr("codex_manager.eval_tools.diff_stat", lambda repo: "1 file changed")
    monkeypatch.setattr("codex_manager.eval_tools.status_porcelain", lambda repo: " M src/app.py")
    monkeypatch.setattr("codex_manager.eval_tools.diff_numstat", lambda repo: (2, 15, 4))
    monkeypatch.setattr(
        "codex_manager.eval_tools.diff_numstat_entries",
        lambda repo: [
            {"path": "src/app.py", "insertions": 10, "deletions": 2},
            {"path": "README.md", "insertions": 5, "deletions": 2},
        ],
    )

    result = evaluator.evaluate(tmp_path)
    assert result.test_outcome == TestOutcome.PASSED
    assert result.files_changed == 2
    assert result.net_lines_changed == 11
    assert result.changed_files == [
        {"path": "src/app.py", "insertions": 10, "deletions": 2},
        {"path": "README.md", "insertions": 5, "deletions": 2},
    ]
