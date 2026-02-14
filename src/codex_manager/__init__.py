"""Codex Manager - orchestrate AI coding agents for iterative repo improvement."""

from importlib.metadata import PackageNotFoundError, version

from codex_manager.schemas import EvalResult, LoopState, RunResult

__all__ = ["EvalResult", "LoopState", "RunResult"]

try:
    __version__ = version("codex-manager")
except PackageNotFoundError:
    __version__ = "0.0.0"
