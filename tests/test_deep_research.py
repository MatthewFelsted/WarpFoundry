"""Unit tests for deep-research governance helpers."""

from __future__ import annotations

import threading
import time
from pathlib import Path

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


def test_run_native_deep_research_serializes_quota_checks(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    first_call_started = threading.Event()
    allow_provider_return = threading.Event()
    call_count = 0
    call_count_lock = threading.Lock()

    def _fake_openai_native(*, topic, guidance, model, max_output_tokens, timeout_seconds):
        nonlocal call_count
        with call_count_lock:
            call_count += 1
        first_call_started.set()
        assert allow_provider_return.wait(timeout=2.0)
        return {
            "summary": "See https://docs.python.org/3/library/threading.html",
            "input_tokens": 10,
            "output_tokens": 20,
        }

    monkeypatch.setattr(deep_research_module, "_call_openai_native", _fake_openai_native)

    settings = deep_research_module.DeepResearchSettings(
        providers="openai",
        retry_attempts=1,
        daily_quota=1,
        max_provider_tokens=800,
        timeout_seconds=30,
    )

    results: list[deep_research_module.DeepResearchRunResult] = []
    results_lock = threading.Lock()

    def _worker() -> None:
        result = deep_research_module.run_native_deep_research(
            repo_path=repo,
            topic="thread safety",
            project_context="quota race check",
            settings=settings,
        )
        with results_lock:
            results.append(result)

    t1 = threading.Thread(target=_worker)
    t2 = threading.Thread(target=_worker)
    t1.start()
    assert first_call_started.wait(timeout=1.0)
    t2.start()
    time.sleep(0.15)

    # Second concurrent run should not enter provider execution while
    # the first call is still in-flight for this repository.
    assert call_count == 1

    allow_provider_return.set()
    t1.join(timeout=3.0)
    t2.join(timeout=3.0)

    assert len(results) == 2
    assert len([r for r in results if r.ok]) == 1
    blocked = [r for r in results if not r.ok and r.quota_blocked]
    assert len(blocked) == 1


def test_run_native_deep_research_returns_provider_prompt_previews(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    def _fake_openai_native(*, topic, guidance, model, max_output_tokens, timeout_seconds):
        return {
            "summary": "OpenAI findings with source https://docs.python.org/3/library/json.html",
            "input_tokens": 12,
            "output_tokens": 21,
        }

    def _fake_google_native(*, topic, guidance, model, max_output_tokens, timeout_seconds):
        return {
            "summary": "Google findings with source https://developer.mozilla.org/en-US/docs/Web/HTTP",
            "input_tokens": 8,
            "output_tokens": 13,
        }

    monkeypatch.setattr(deep_research_module, "_call_openai_native", _fake_openai_native)
    monkeypatch.setattr(deep_research_module, "_call_google_native", _fake_google_native)

    settings = deep_research_module.DeepResearchSettings(
        providers="both",
        retry_attempts=1,
        daily_quota=5,
        max_provider_tokens=1000,
        timeout_seconds=30,
    )

    result = deep_research_module.run_native_deep_research(
        repo_path=repo,
        topic="repo hardening",
        project_context="WISH-001 and WISH-008 are complete.",
        settings=settings,
    )

    assert result.ok is True
    assert set(result.provider_prompt_previews.keys()) == {"openai", "google"}
    assert "Topic: repo hardening" in result.provider_prompt_previews["openai"]
    assert "Project context:" in result.provider_prompt_previews["openai"]
    assert "Topic: repo hardening" in result.provider_prompt_previews["google"]
    assert "Project context:" in result.provider_prompt_previews["google"]


def test_call_openai_native_uses_non_placeholder_fallback_key(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def _fake_http_json(url, *, method, headers=None, payload=None, timeout_seconds=0):
        captured["auth"] = str((headers or {}).get("Authorization") or "")
        return {
            "output_text": "OpenAI summary",
            "usage": {"input_tokens": 11, "output_tokens": 7},
        }

    monkeypatch.setattr(deep_research_module, "_http_json", _fake_http_json)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-your-key-here")
    monkeypatch.setenv("CODEX_API_KEY", "sk-proj-real-secret")

    result = deep_research_module._call_openai_native(
        topic="Auth fallback",
        guidance="check fallback key selection",
        model="gpt-5.2",
        max_output_tokens=300,
        timeout_seconds=15,
    )

    assert captured["auth"] == "Bearer sk-proj-real-secret"
    assert result["summary"] == "OpenAI summary"


def test_call_google_native_uses_non_placeholder_fallback_key(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def _fake_http_json(url, *, method, headers=None, payload=None, timeout_seconds=0):
        captured["url"] = str(url)
        return {
            "candidates": [{"content": {"parts": [{"text": "Google summary"}]}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 9},
        }

    monkeypatch.setattr(deep_research_module, "_http_json", _fake_http_json)
    monkeypatch.setenv("GOOGLE_API_KEY", "your-key-here")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-real-secret")

    result = deep_research_module._call_google_native(
        topic="Auth fallback",
        guidance="check fallback key selection",
        model="gemini-3.1-pro-preview",
        max_output_tokens=300,
        timeout_seconds=15,
    )

    assert "key=gemini-real-secret" in captured["url"]
    assert result["summary"] == "Google summary"


def test_call_google_native_sends_thinking_defaults(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_http_json(url, *, method, headers=None, payload=None, timeout_seconds=0):
        captured["payload"] = payload
        return {
            "candidates": [{"content": {"parts": [{"text": "Google summary"}]}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 9},
        }

    monkeypatch.setattr(deep_research_module, "_http_json", _fake_http_json)
    monkeypatch.setenv("GOOGLE_API_KEY", "gemini-real-secret")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    deep_research_module._call_google_native(
        topic="Thinking config",
        guidance="verify thinking defaults",
        model="gemini-3.1-pro-preview",
        max_output_tokens=65536,
        timeout_seconds=15,
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    generation = payload.get("generationConfig")
    assert isinstance(generation, dict)
    assert generation["maxOutputTokens"] == 65536
    thinking = generation.get("thinkingConfig")
    assert isinstance(thinking, dict)
    assert thinking["thinkingBudget"] == 32000
    assert thinking["includeThoughts"] is True


def test_call_google_native_retries_without_thinking_config_on_rejection(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_http_json(url, *, method, headers=None, payload=None, timeout_seconds=0):
        calls.append({"payload": payload})
        if len(calls) == 1:
            raise RuntimeError("HTTP 400 ... Unknown name 'thinkingConfig'")
        return {
            "candidates": [{"content": {"parts": [{"text": "Google summary"}]}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 9},
        }

    monkeypatch.setattr(deep_research_module, "_http_json", _fake_http_json)
    monkeypatch.setenv("GOOGLE_API_KEY", "gemini-real-secret")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = deep_research_module._call_google_native(
        topic="Thinking fallback",
        guidance="verify compatibility fallback",
        model="gemini-3.1-pro-preview",
        max_output_tokens=2048,
        timeout_seconds=15,
    )

    assert result["summary"] == "Google summary"
    assert len(calls) == 2
    first_generation = calls[0]["payload"]["generationConfig"]
    second_generation = calls[1]["payload"]["generationConfig"]
    assert "thinkingConfig" in first_generation
    assert "thinkingConfig" not in second_generation

