"""Abstract base class for coding-agent runners.

All agent runners (Codex CLI, Claude Code, future agents) implement
the same interface so the chain executor can dispatch to any of them
interchangeably.
"""

from __future__ import annotations

import abc
from pathlib import Path

from codex_manager.schemas import RunResult


class AgentRunner(abc.ABC):
    """Common interface for coding-agent CLI wrappers.

    Subclasses must implement :meth:`run` which accepts a repo path and
    a prompt and returns a :class:`RunResult`.
    """

    #: Human-readable name shown in the GUI (e.g. "Codex", "Claude Code").
    name: str = "base"

    @abc.abstractmethod
    def run(
        self,
        repo_path: str | Path,
        prompt: str,
        *,
        full_auto: bool = False,
        extra_args: list[str] | None = None,
    ) -> RunResult:
        """Execute a single agent invocation and return structured results.

        Parameters
        ----------
        repo_path:
            Working directory (the target git repository).
        prompt:
            Natural-language task prompt.
        full_auto:
            If True, allow the agent to write files and run commands
            autonomously.
        extra_args:
            Additional CLI flags forwarded verbatim.
        """


# ── Registry ──────────────────────────────────────────────────────

_REGISTRY: dict[str, type[AgentRunner]] = {}


def register_agent(key: str, cls: type[AgentRunner]) -> None:
    """Register an agent runner class under a lookup key."""
    normalized_key = (key or "").strip()
    if not normalized_key:
        raise ValueError("Agent key must be a non-empty string")
    if not isinstance(cls, type) or not issubclass(cls, AgentRunner):
        raise TypeError("Registered agent must be an AgentRunner subclass")

    existing = _REGISTRY.get(normalized_key)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"Agent '{normalized_key}' is already registered with {existing.__name__}"
        )

    _REGISTRY[normalized_key] = cls


def get_agent_class(key: str) -> type[AgentRunner]:
    """Look up a registered agent runner class by key."""
    normalized_key = (key or "").strip()
    if normalized_key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown agent '{normalized_key}'. Available: {available}")
    return _REGISTRY[normalized_key]


def list_agents() -> list[str]:
    """Return all registered agent keys."""
    return sorted(_REGISTRY)
