"""Computer-Using Agent (CUA) integration.

Supports OpenAI's ``computer-use-preview`` model and Anthropic's Claude
computer use tool. Uses Playwright for browser automation.
"""

from codex_manager.cua.actions import (
    CUAProvider,
    CUASessionConfig,
    CUASessionResult,
)
from codex_manager.cua.session import run_cua_session, run_cua_session_sync

__all__ = [
    "CUAProvider",
    "CUASessionConfig",
    "CUASessionResult",
    "run_cua_session",
    "run_cua_session_sync",
]
