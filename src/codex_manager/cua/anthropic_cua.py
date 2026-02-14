"""Anthropic Claude computer use provider.

Uses Claude's ``computer_20251124`` (or ``computer_20250124``) tool via
the Messages API with the appropriate beta header.

Requires: ``pip install anthropic``
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    _pkg_root = Path(__file__).resolve().parent.parent.parent.parent
    for d in (_pkg_root, _pkg_root.parent, Path.cwd()):
        env_file = d / ".env"
        if env_file.is_file():
            load_dotenv(env_file)
            break
    else:
        load_dotenv()
except ImportError:
    pass

from codex_manager.cua.actions import (
    ActionType,
    CUAAction,
)

logger = logging.getLogger(__name__)

# Map Anthropic action type strings to our ActionType enum
_ANTHROPIC_ACTION_MAP: dict[str, ActionType] = {
    "screenshot": ActionType.SCREENSHOT,
    "left_click": ActionType.CLICK,
    "right_click": ActionType.RIGHT_CLICK,
    "double_click": ActionType.DOUBLE_CLICK,
    "middle_click": ActionType.CLICK,
    "triple_click": ActionType.CLICK,
    "type": ActionType.TYPE,
    "key": ActionType.KEY,
    "mouse_move": ActionType.MOUSE_MOVE,
    "left_click_drag": ActionType.DRAG,
    "scroll": ActionType.SCROLL,
    "wait": ActionType.WAIT,
}


def parse_anthropic_action(tool_input: dict[str, Any]) -> CUAAction:
    """Parse an Anthropic computer use tool_input into our CUAAction model."""
    action_type_str = tool_input.get("action", "screenshot")
    action_type = _ANTHROPIC_ACTION_MAP.get(action_type_str, ActionType.SCREENSHOT)

    coordinate = tool_input.get("coordinate", [0, 0])
    x = int(coordinate[0]) if len(coordinate) > 0 else 0
    y = int(coordinate[1]) if len(coordinate) > 1 else 0

    return CUAAction(
        action_type=action_type,
        x=x,
        y=y,
        text=tool_input.get("text", ""),
        keys=[tool_input.get("text", "")] if action_type == ActionType.KEY else [],
        scroll_x=int(tool_input.get("scroll_x", 0)),
        scroll_y=int(tool_input.get("scroll_y", 0)),
        start_x=int(tool_input.get("start_coordinate", [0, 0])[0]) if "start_coordinate" in tool_input else x,
        start_y=int(tool_input.get("start_coordinate", [0, 0])[1]) if "start_coordinate" in tool_input else y,
        end_x=x,
        end_y=y,
        raw=dict(tool_input),
    )


class AnthropicCUA:
    """Anthropic Claude computer use provider.

    Parameters
    ----------
    model:
        Claude model to use (default: ``claude-opus-4-6``).
    tool_version:
        Computer tool version (``computer_20251124`` or ``computer_20250124``).
    beta_flag:
        Beta header value.
    display_width:
        Browser viewport width.
    display_height:
        Browser viewport height.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        tool_version: str = "computer_20251124",
        beta_flag: str = "computer-use-2025-11-24",
        display_width: int = 1280,
        display_height: int = 800,
    ) -> None:
        self.model = model
        self.tool_version = tool_version
        self.beta_flag = beta_flag
        self.display_width = display_width
        self.display_height = display_height
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-initialize the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "Anthropic SDK is required for Claude CUA. "
                    "Install with: pip install anthropic"
                ) from exc
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Set it in your environment or .env file."
                )
            self._client = Anthropic(api_key=api_key)
        return self._client

    def _tools(self) -> list[dict[str, Any]]:
        """Return the tool definitions for computer use."""
        return [
            {
                "type": self.tool_version,
                "name": "computer",
                "display_width_px": self.display_width,
                "display_height_px": self.display_height,
            },
        ]

    def create_initial_request(
        self,
        task: str,
        screenshot_b64: str | None = None,
    ) -> Any:
        """Send the initial request to Claude with computer use.

        Parameters
        ----------
        task:
            Natural-language description of what to do.
        screenshot_b64:
            Optional base64-encoded screenshot of the initial state.

        Returns
        -------
        The Anthropic Message response.
        """
        client = self._get_client()

        content: list[dict[str, Any]] = [
            {"type": "text", "text": task},
        ]
        if screenshot_b64:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": screenshot_b64,
                },
            })

        response = client.beta.messages.create(
            model=self.model,
            max_tokens=4096,
            tools=self._tools(),
            messages=[{"role": "user", "content": content}],
            betas=[self.beta_flag],
        )
        return response

    def send_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_use_id: str,
        screenshot_b64: str,
    ) -> Any:
        """Send a tool result (screenshot) back to Claude.

        Parameters
        ----------
        messages:
            Full conversation history so far.
        tool_use_id:
            The ``tool_use`` block id we're responding to.
        screenshot_b64:
            Base64-encoded screenshot after executing the action.

        Returns
        -------
        The next Anthropic Message response.
        """
        client = self._get_client()

        # Add the tool result
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": screenshot_b64,
                            },
                        }
                    ],
                }
            ],
        })

        response = client.beta.messages.create(
            model=self.model,
            max_tokens=4096,
            tools=self._tools(),
            messages=messages,
            betas=[self.beta_flag],
        )
        return response

    @staticmethod
    def extract_computer_call(response: Any) -> tuple[str | None, dict[str, Any] | None, str]:
        """Extract a computer tool_use from the response.

        Returns
        -------
        (tool_use_id, tool_input, reasoning):
            - tool_use_id: The block id, or None if no computer use requested.
            - tool_input: The input dict with action details, or None.
            - reasoning: Any text reasoning from the response.
        """
        tool_use_id = None
        tool_input = None
        reasoning = ""

        for block in response.content:
            block_type = getattr(block, "type", "")
            if block_type == "tool_use" and getattr(block, "name", "") == "computer":
                tool_use_id = block.id
                tool_input = block.input
            elif block_type == "text":
                text = getattr(block, "text", "")
                if text:
                    reasoning += text + " "
            elif block_type == "thinking":
                thinking_text = getattr(block, "thinking", "")
                if thinking_text:
                    reasoning += thinking_text + " "

        return tool_use_id, tool_input, reasoning.strip()

    @staticmethod
    def extract_text_output(response: Any) -> str:
        """Extract final text output from the response."""
        parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", "") == "text":
                text = getattr(block, "text", "")
                if text:
                    parts.append(text)
        return "\n".join(parts)
