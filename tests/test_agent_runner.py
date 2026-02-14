"""Tests for agent runner registry helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

import codex_manager.agent_runner as agent_runner_module
from codex_manager.agent_runner import AgentRunner, get_agent_class, list_agents, register_agent
from codex_manager.schemas import RunResult


class _DummyRunner(AgentRunner):
    name = "dummy"

    def run(
        self,
        repo_path: str | Path,
        prompt: str,
        *,
        full_auto: bool = False,
        extra_args: list[str] | None = None,
    ) -> RunResult:
        return RunResult(success=True, exit_code=0)


def test_register_get_and_list_agents(monkeypatch) -> None:
    monkeypatch.setattr(agent_runner_module, "_REGISTRY", {})

    register_agent("b", _DummyRunner)
    register_agent("a", _DummyRunner)

    assert get_agent_class("a") is _DummyRunner
    assert list_agents() == ["a", "b"]


def test_get_agent_class_raises_helpful_error(monkeypatch) -> None:
    monkeypatch.setattr(agent_runner_module, "_REGISTRY", {})
    with pytest.raises(KeyError, match=r"Unknown agent 'missing'. Available: \(none\)"):
        get_agent_class("missing")

    register_agent("codex", _DummyRunner)
    with pytest.raises(KeyError, match=r"Available: codex"):
        get_agent_class("missing")
