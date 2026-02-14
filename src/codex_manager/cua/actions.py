"""Data models for computer-use agent actions and results.

These models represent the actions a CUA model can request (click, type,
scroll, etc.) and the results of executing those actions.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CUAProvider(str, Enum):
    """Supported CUA providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ActionType(str, Enum):
    """Types of actions the CUA model can request."""

    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    SCROLL = "scroll"
    TYPE = "type"
    KEYPRESS = "keypress"
    KEY = "key"
    MOUSE_MOVE = "mouse_move"
    DRAG = "drag"
    WAIT = "wait"
    SCREENSHOT = "screenshot"


@dataclass
class CUAAction:
    """A single action requested by the CUA model."""

    action_type: ActionType
    x: int = 0
    y: int = 0
    button: str = "left"
    text: str = ""
    keys: list[str] = field(default_factory=list)
    scroll_x: int = 0
    scroll_y: int = 0
    # For drag
    start_x: int = 0
    start_y: int = 0
    end_x: int = 0
    end_y: int = 0
    # Raw provider data
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class CUAStepResult:
    """Result of executing a single CUA action."""

    action: CUAAction
    success: bool = True
    error: str = ""
    screenshot_b64: str = ""  # base64-encoded screenshot after the action
    reasoning: str = ""  # model's reasoning for this action
    timestamp: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())


@dataclass
class CUASessionConfig:
    """Configuration for a CUA session."""

    provider: CUAProvider = CUAProvider.OPENAI
    # Target URL to test — if empty, the CUA just takes a screenshot of the desktop/browser
    target_url: str = ""
    # The task/goal to accomplish
    task: str = ""
    # Browser viewport
    viewport_width: int = 1280
    viewport_height: int = 800
    # Limits
    max_steps: int = 50
    timeout_seconds: int = 300
    # OpenAI-specific (override with CUA_OPENAI_MODEL env var; may require Tier 3+ access)
    openai_model: str = field(
        default_factory=lambda: os.getenv("CUA_OPENAI_MODEL", "computer-use-preview")
    )
    # Anthropic-specific
    anthropic_model: str = "claude-opus-4-6"
    anthropic_tool_version: str = "computer_20251124"
    anthropic_beta: str = "computer-use-2025-11-24"
    # Whether to run browser headless (invisible) or visible
    headless: bool = True
    # Save screenshots to disk
    save_screenshots: bool = True
    screenshots_dir: str = ""


@dataclass
class CUAObservation:
    """A single structured observation from the CUA visual test.

    Each observation documents a finding (bug, layout issue, positive note)
    captured by the CUA during its session.
    """

    element: str = ""          # what element or view is affected
    expected: str = ""         # expected behavior / appearance
    actual: str = ""           # what actually happened
    severity: str = "minor"    # "critical" | "major" | "minor" | "cosmetic" | "positive"
    step_number: int = 0       # which step discovered this
    screenshot: str = ""       # path to the relevant screenshot (if any)
    category: str = ""         # "layout" | "interaction" | "navigation" | "content" | "performance"
    recommendation: str = ""   # suggested fix or improvement


@dataclass
class CUASessionResult:
    """Complete result of a CUA session."""

    task: str = ""
    provider: str = ""
    success: bool = False
    steps: list[CUAStepResult] = field(default_factory=list)
    total_steps: int = 0
    summary: str = ""
    observations: list[CUAObservation] = field(default_factory=list)
    error: str = ""
    duration_seconds: float = 0.0
    screenshots_saved: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    def observations_markdown(self) -> str:
        """Render observations as a Markdown report suitable for TESTPLAN.md or chain context."""
        if not self.observations:
            return "_No structured observations captured._"
        lines: list[str] = []
        # Group by severity
        for sev in ("critical", "major", "minor", "cosmetic", "positive"):
            obs_group = [o for o in self.observations if o.severity == sev]
            if not obs_group:
                continue
            icon = {"critical": "🔴", "major": "🟠", "minor": "🟡", "cosmetic": "⚪", "positive": "🟢"}.get(sev, "•")
            lines.append(f"\n#### {icon} {sev.title()} ({len(obs_group)})\n")
            for o in obs_group:
                lines.append(f"- **{o.element or 'Unknown element'}**")
                if o.expected:
                    lines.append(f"  - Expected: {o.expected}")
                if o.actual:
                    lines.append(f"  - Actual: {o.actual}")
                if o.category:
                    lines.append(f"  - Category: {o.category}")
                if o.recommendation:
                    lines.append(f"  - Recommendation: {o.recommendation}")
                if o.screenshot:
                    lines.append(f"  - Screenshot: `{o.screenshot}`")
        return "\n".join(lines)


# ── Enhanced task prompt suffix ──────────────────────────────────
# Appended to any CUA task so the model produces structured output
# at the end of its session.

CUA_OBSERVATION_SUFFIX = """

IMPORTANT: After you finish testing, provide a structured report as your FINAL message.
Use this EXACT format for each finding (one per line, use | as separator):

OBSERVATION|<severity>|<category>|<element>|<expected>|<actual>|<recommendation>

Where:
- severity: critical, major, minor, cosmetic, or positive
- category: layout, interaction, navigation, content, performance
- element: what element/view/page is affected
- expected: what should happen
- actual: what actually happened (or what you observed)
- recommendation: suggested fix or improvement

Example:
OBSERVATION|minor|layout|sidebar menu|Menu items should be evenly spaced|Items overlap on narrow viewport|Add flex-wrap or reduce padding
OBSERVATION|positive|interaction|Chain Builder|Drag-and-drop should reorder steps|Works smoothly with visual feedback|N/A

End your report with a brief overall summary paragraph.
"""
