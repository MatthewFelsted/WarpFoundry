"""Tests for connector environment parsing hardening."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

_CONNECTOR_ENV_KEYS = (
    "AI_TEXT_ONLY",
    "AI_PER_REQUEST_TIMEOUT_S",
    "AI_RECONNECT_WAIT_S",
    "AI_CLIENT_MAX_AGE_S",
    "AI_RESULT_CACHE_PATH",
    "AI_LEAD_MODEL",
    "AI_CHEAP_MODEL",
    "AI_MEDIUM_MODEL",
    "AI_FREE_MODEL",
    "AI_OPENAI_MAX_ATTEMPTS",
    "AI_GEMINI_MAX_ATTEMPTS",
    "AI_ANTHROPIC_MAX_ATTEMPTS",
    "AI_XAI_MAX_ATTEMPTS",
    "AI_OLLAMA_MAX_ATTEMPTS",
    "AI_RETRY_BACKOFF_BASE",
    "AI_RETRY_BACKOFF_MAX_S",
    "OLLAMA_AUTO_START",
    "OLLAMA_BASE_URL",
    "ANTHROPIC_MAX_TOKENS",
    "CONNECTOR_CACHE_BASE",
)


def _reload_connector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    env: dict[str, str],
):
    for key in _CONNECTOR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    # Keep cache writes scoped to the pytest temp directory.
    monkeypatch.setenv("AI_RESULT_CACHE_PATH", str(tmp_path / "cache" / "ai_cache.db"))
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    import codex_manager.brain.connector as connector

    return importlib.reload(connector)


def test_invalid_connector_env_values_fall_back_to_safe_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    connector = _reload_connector(
        monkeypatch,
        tmp_path,
        env={
            "AI_TEXT_ONLY": "maybe",
            "AI_PER_REQUEST_TIMEOUT_S": "not-a-number",
            "AI_RECONNECT_WAIT_S": "broken",
            "AI_CLIENT_MAX_AGE_S": "0",
            "AI_OPENAI_MAX_ATTEMPTS": "-2",
            "AI_GEMINI_MAX_ATTEMPTS": "0",
            "AI_ANTHROPIC_MAX_ATTEMPTS": "invalid",
            "AI_XAI_MAX_ATTEMPTS": "-9",
            "AI_OLLAMA_MAX_ATTEMPTS": "",
            "AI_RETRY_BACKOFF_BASE": "0.01",
            "AI_RETRY_BACKOFF_MAX_S": "0",
            "OLLAMA_AUTO_START": "maybe",
            "OLLAMA_BASE_URL": "   ",
            "ANTHROPIC_MAX_TOKENS": "invalid",
        },
    )

    assert connector.DEFAULT_TEXT_ONLY is True
    assert pytest.approx(600.0) == connector.DEFAULT_PER_REQUEST_TIMEOUT_S
    assert connector.DEFAULT_RECONNECT_WAIT_S == 30
    assert connector.CLIENT_MAX_AGE_S == 1
    assert connector.OPENAI_MAX_ATTEMPTS == 1
    assert connector.GEMINI_MAX_ATTEMPTS == 1
    assert connector.ANTHROPIC_MAX_ATTEMPTS == 3
    assert connector.XAI_MAX_ATTEMPTS == 1
    assert connector.OLLAMA_MAX_ATTEMPTS == 3
    assert pytest.approx(1.0) == connector.RETRY_BACKOFF_BASE
    assert pytest.approx(0.1) == connector.RETRY_BACKOFF_MAX_S
    assert connector.OLLAMA_AUTO_START is False
    assert connector.OLLAMA_BASE_URL == "http://localhost:11434"


def test_connector_env_boolean_aliases_and_path_expansion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    connector = _reload_connector(
        monkeypatch,
        tmp_path,
        env={
            "CONNECTOR_CACHE_BASE": str(tmp_path),
            "AI_TEXT_ONLY": "0",
            "OLLAMA_AUTO_START": "YES",
            "AI_RECONNECT_WAIT_S": "12.7",
            "AI_RESULT_CACHE_PATH": "$CONNECTOR_CACHE_BASE/custom/ai_cache.db",
            "OLLAMA_BASE_URL": "http://localhost:11434/",
        },
    )

    expected_cache_path = os.path.expandvars(
        os.path.expanduser("$CONNECTOR_CACHE_BASE/custom/ai_cache.db")
    )

    assert connector.DEFAULT_TEXT_ONLY is False
    assert connector.OLLAMA_AUTO_START is True
    assert connector.DEFAULT_RECONNECT_WAIT_S == 12
    assert expected_cache_path == connector._CACHE_PATH
    assert connector.OLLAMA_BASE_URL == "http://localhost:11434"


def test_connect_dispatches_using_provider_from_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    connector = _reload_connector(monkeypatch, tmp_path, env={})
    observed: list[dict[str, object]] = []

    def fake_openai(
        model: str,
        prompt: str,
        text_only: bool,
        timeout_s: float,
        *,
        disable_cache: bool = False,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        observed.append(
            {
                "model": model,
                "prompt": prompt,
                "text_only": text_only,
                "timeout_s": timeout_s,
                "disable_cache": disable_cache,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
            }
        )
        return "stub-response"

    monkeypatch.setattr(connector, "provider_from_model", lambda _model: "openai")
    monkeypatch.setattr(connector, "_connect_openai", fake_openai)

    result = connector.connect(
        "custom-model-name",
        "audit this",
        text_only=False,
        per_request_timeout=12.5,
        disable_cache=True,
        max_output_tokens=512,
        temperature=0.2,
    )

    assert result == "stub-response"
    assert observed == [
        {
            "model": "custom-model-name",
            "prompt": "audit this",
            "text_only": False,
            "timeout_s": 12.5,
            "disable_cache": True,
            "max_output_tokens": 512,
            "temperature": 0.2,
        }
    ]


def test_prompt_all_respects_explicit_empty_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    connector = _reload_connector(monkeypatch, tmp_path, env={})
    called: list[str] = []

    def fake_connect(*_args, **_kwargs):
        called.append("called")
        return "unused"

    monkeypatch.setattr(connector, "connect", fake_connect)
    assert connector.prompt_all("ping", models=[]) == []
    assert called == []
