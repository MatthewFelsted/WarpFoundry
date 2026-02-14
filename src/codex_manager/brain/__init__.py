"""Brain package — multi-AI connector + intelligent decision layer.

The brain provides the "thinking" capability for the AI Manager:
- Connect to multiple AI providers (OpenAI, Anthropic, Gemini, xAI, Ollama)
- Plan and refine prompts before sending them to Codex
- Evaluate results and decide follow-up actions
- Handle errors and decide when to escalate to a human
"""

from __future__ import annotations

from codex_manager.brain.manager import BrainManager

__all__ = ["BrainManager"]
