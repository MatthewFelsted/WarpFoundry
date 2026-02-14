"""Tests for shared preflight diagnostics."""

from __future__ import annotations

import os
from pathlib import Path

import codex_manager.preflight as preflight


def _check(report: preflight.PreflightReport, category: str, key: str) -> preflight.PreflightCheck:
    for check in report.checks:
        if check.category == category and check.key == key:
            return check
    raise AssertionError(f"Missing check {category}/{key}")


def test_parse_agents_normalizes_and_deduplicates() -> None:
    parsed = preflight.parse_agents("codex, auto claude_code codex")
    assert parsed == ["codex", "claude_code"]


def test_parse_agents_supports_claude_aliases() -> None:
    parsed = preflight.parse_agents("claude, claude-code claudecode claude_code")
    assert parsed == ["claude_code"]


def test_parse_agents_supports_semicolons_and_quoted_tokens() -> None:
    parsed = preflight.parse_agents('"codex"; "claude" ; auto')
    assert parsed == ["codex", "claude_code"]


def test_binary_exists_rejects_directories(tmp_path: Path) -> None:
    assert preflight.binary_exists(str(tmp_path)) is False


def test_binary_exists_requires_explicit_file_to_be_executable(tmp_path: Path) -> None:
    if os.name == "nt":
        runnable = tmp_path / "tool.cmd"
        runnable.write_text("@echo off\r\nexit /b 0\r\n", encoding="utf-8")
        assert preflight.binary_exists(str(runnable)) is True

        not_runnable = tmp_path / "tool.txt"
        not_runnable.write_text("not executable\r\n", encoding="utf-8")
        assert preflight.binary_exists(str(not_runnable)) is False
        return

    candidate = tmp_path / "tool"
    candidate.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
    assert preflight.binary_exists(str(candidate)) is False

    candidate.chmod(candidate.stat().st_mode | 0o111)
    assert preflight.binary_exists(str(candidate)) is True


def test_binary_exists_expands_environment_variables(monkeypatch, tmp_path: Path) -> None:
    var_name = "CODEX_MANAGER_TEST_BIN_DIR"
    monkeypatch.setenv(var_name, str(tmp_path))

    if os.name == "nt":
        runnable = tmp_path / "tool.cmd"
        runnable.write_text("@echo off\r\nexit /b 0\r\n", encoding="utf-8")
        configured = f"%{var_name}%\\tool.cmd"
    else:
        runnable = tmp_path / "tool"
        runnable.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
        runnable.chmod(runnable.stat().st_mode | 0o111)
        configured = f"${var_name}/tool"

    assert preflight.binary_exists(configured) is True


def test_build_preflight_report_warns_when_repo_not_provided(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "binary_exists", lambda _binary: True)
    monkeypatch.setattr(preflight, "has_codex_auth", lambda: True)

    report = preflight.build_preflight_report(
        repo_path="",
        agents=["auto", "unknown-agent"],
    )

    assert report.ready is True
    assert report.requested_agents == ["codex", "unknown-agent"]
    assert report.summary["warn"] >= 1
    assert any(c.category == "agents" and c.status == "warn" for c in report.checks)


def test_build_preflight_report_reports_codex_failures(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    monkeypatch.setattr(preflight, "binary_exists", lambda _binary: False)
    monkeypatch.setattr(preflight, "has_codex_auth", lambda: False)
    monkeypatch.setattr(preflight, "repo_write_error", lambda _repo: None)

    report = preflight.build_preflight_report(
        repo_path=repo,
        agents=["codex"],
        codex_binary="codex-test",
    )

    failures = report.failure_messages()
    assert report.ready is False
    assert any("Codex CLI binary available" in msg for msg in failures)
    assert any("Codex authentication detected" in msg for msg in failures)
    actions = report.next_actions
    assert any(a.key == "install_codex_cli" for a in actions)
    assert any(a.key == "codex_login" for a in actions)
    assert any(a.key == "rerun_doctor" for a in actions)


def test_build_preflight_report_to_dict_includes_summary(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    monkeypatch.setattr(preflight, "binary_exists", lambda _binary: True)
    monkeypatch.setattr(preflight, "has_codex_auth", lambda: True)
    monkeypatch.setattr(preflight, "repo_write_error", lambda _repo: None)

    report = preflight.build_preflight_report(
        repo_path=repo,
        agents=["codex"],
    )
    payload = report.to_dict()

    assert payload["ready"] is True
    assert payload["codex_binary"] == "codex"
    assert payload["claude_binary"] == "claude"
    assert isinstance(payload["checks"], list)
    assert payload["summary"]["fail"] == 0
    assert isinstance(payload["next_actions"], list)
    assert any(a["key"] == "first_dry_run" for a in payload["next_actions"])


def test_has_codex_auth_ignores_placeholder_env_values(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(preflight.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-your-key-here")
    monkeypatch.delenv("CODEX_API_KEY", raising=False)

    assert preflight.has_codex_auth() is False

    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-real-secret-value")
    assert preflight.has_codex_auth() is True


def test_has_claude_auth_ignores_placeholder_env_values(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(preflight.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-your-key-here")
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)

    assert preflight.has_claude_auth() is False

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real-secret-value")
    assert preflight.has_claude_auth() is True


def test_next_actions_rerun_doctor_preserves_custom_binary_flags(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    monkeypatch.setattr(preflight, "binary_exists", lambda _binary: False)
    monkeypatch.setattr(preflight, "has_codex_auth", lambda: False)
    monkeypatch.setattr(preflight, "repo_write_error", lambda _repo: None)

    report = preflight.build_preflight_report(
        repo_path=repo,
        agents=["codex"],
        codex_binary="codex-custom",
        claude_binary="claude-custom",
    )

    rerun = next(a for a in report.next_actions if a.key == "rerun_doctor")
    assert '--codex-bin "codex-custom"' in rerun.command
    assert '--claude-bin "claude-custom"' in rerun.command


def test_build_preflight_report_flags_placeholder_env_values(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(preflight.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(preflight, "binary_exists", lambda _binary: True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-your-key-here")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-your-key-here")
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)

    report = preflight.build_preflight_report(
        repo_path="",
        agents=["codex", "claude_code"],
    )

    codex_auth = _check(report, "codex", "auth")
    claude_auth = _check(report, "claude_code", "auth")
    assert codex_auth.status == "fail"
    assert "placeholder" in codex_auth.detail.lower()
    assert "OPENAI_API_KEY" in codex_auth.detail
    assert claude_auth.status == "fail"
    assert "placeholder" in claude_auth.detail.lower()
    assert "ANTHROPIC_API_KEY" in claude_auth.detail
