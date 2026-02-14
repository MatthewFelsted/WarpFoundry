"""Shared agent control tags used by orchestrators."""

from __future__ import annotations

import re

TERMINATE_STEP_TAG = "[TERMINATE_STEP]"
"""Canonical tag that instructs orchestrators to skip remaining repeats."""

_TERMINATE_STEP_RE = re.compile(r"\[\s*terminate[\s_-]*step\s*\]", re.IGNORECASE)


def contains_terminate_step_signal(text: str) -> bool:
    """Return True when *text* contains a terminate-step control tag."""
    if not text:
        return False
    return bool(_TERMINATE_STEP_RE.search(text))


def terminate_step_instruction(scope: str) -> str:
    """Return prompt guidance for signaling no-op repeats."""
    return (
        f"If there is no additional meaningful work for this {scope}, "
        f"output `{TERMINATE_STEP_TAG}` on its own line. The runner will "
        "skip the remaining repeats/iterations."
    )
