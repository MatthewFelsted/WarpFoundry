"""Tests for CUA provider auth-key fallback behavior."""

from __future__ import annotations

import sys
import types

import pytest

from codex_manager.cua.anthropic_cua import AnthropicCUA
from codex_manager.cua.openai_cua import OpenAICUA


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("openai")

    class _FakeOpenAI:
        def __init__(self, *, api_key: str):
            self.api_key = api_key

    module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)


def _install_fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("anthropic")

    class _FakeAnthropic:
        def __init__(self, *, api_key: str):
            self.api_key = api_key

    module.Anthropic = _FakeAnthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)


def test_openai_cua_uses_codex_key_when_openai_is_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-your-key-here")
    monkeypatch.setenv("CODEX_API_KEY", "sk-proj-real-secret")

    provider = OpenAICUA()
    client = provider._get_client()

    assert getattr(client, "api_key") == "sk-proj-real-secret"


def test_openai_cua_rejects_placeholder_only_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-your-key-here")
    monkeypatch.delenv("CODEX_API_KEY", raising=False)

    provider = OpenAICUA()

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY \\(or CODEX_API_KEY\\) is not set"):
        provider._get_client()


def test_anthropic_cua_uses_claude_key_when_anthropic_is_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_anthropic(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-your-key-here")
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-ant-real-secret")

    provider = AnthropicCUA()
    client = provider._get_client()

    assert getattr(client, "api_key") == "sk-ant-real-secret"


def test_anthropic_cua_rejects_placeholder_only_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_anthropic(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-your-key-here")
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)

    provider = AnthropicCUA()

    with pytest.raises(
        RuntimeError,
        match="ANTHROPIC_API_KEY \\(or CLAUDE_API_KEY\\) is not set",
    ):
        provider._get_client()
