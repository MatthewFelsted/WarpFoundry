"""Tests for shared preflight diagnostics."""

from __future__ import annotations

from pathlib import Path

import codex_manager.preflight as preflight


def test_parse_agents_normalizes_and_deduplicates() -> None:
    parsed = preflight.parse_agents("codex, auto claude_code codex")
    assert parsed == ["codex", "claude_code"]


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
    assert isinstance(payload["checks"], list)
    assert payload["summary"]["fail"] == 0
    assert isinstance(payload["next_actions"], list)
    assert any(a["key"] == "first_dry_run" for a in payload["next_actions"])
