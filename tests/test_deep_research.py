"""Unit tests for deep-research governance helpers."""

from __future__ import annotations

from codex_manager.research import deep_research as deep_research_module


def test_filter_sources_by_policy_filters_insecure_and_blocked_domains(monkeypatch):
    monkeypatch.delenv("DEEP_RESEARCH_ALLOWED_SOURCE_DOMAINS", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_BLOCKED_SOURCE_DOMAINS", raising=False)

    accepted, warnings = deep_research_module._filter_sources_by_policy(
        [
            "http://trusted.example/a",
            "https://example.com/x",
            "https://docs.python.org/3/",
        ]
    )

    assert accepted == ["https://docs.python.org/3/"]
    assert any("non-https" in warning.lower() for warning in warnings)
    assert any("low-trust source domains" in warning.lower() for warning in warnings)


def test_filter_sources_by_policy_respects_allowlist(monkeypatch):
    monkeypatch.setenv("DEEP_RESEARCH_ALLOWED_SOURCE_DOMAINS", "docs.python.org")
    monkeypatch.delenv("DEEP_RESEARCH_BLOCKED_SOURCE_DOMAINS", raising=False)

    accepted, warnings = deep_research_module._filter_sources_by_policy(
        [
            "https://docs.python.org/3/library/pathlib.html",
            "https://pypi.org/project/requests/",
        ]
    )

    assert accepted == ["https://docs.python.org/3/library/pathlib.html"]
    assert any("allowed_source_domains" in warning.lower() for warning in warnings)
