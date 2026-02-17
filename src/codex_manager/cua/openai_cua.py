"""OpenAI Computer-Using Agent (CUA) provider.

Uses the ``computer-use-preview`` model via the OpenAI Responses API.
The model views screenshots and returns click/type/scroll actions.

Requires: ``pip install openai``
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    # Load .env from package root or cwd so OPENAI_API_KEY is set when CUA runs
    _pkg_root = (
        Path(__file__).resolve().parent.parent.parent.parent
    )  # cua/openai_cua.py -> codex_manager/
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
from codex_manager.preflight import first_env_secret

logger = logging.getLogger(__name__)

# Map OpenAI action type strings to our ActionType enum
_OPENAI_ACTION_MAP: dict[str, ActionType] = {
    "click": ActionType.CLICK,
    "double_click": ActionType.DOUBLE_CLICK,
    "right_click": ActionType.RIGHT_CLICK,
    "scroll": ActionType.SCROLL,
    "type": ActionType.TYPE,
    "keypress": ActionType.KEYPRESS,
    "key": ActionType.KEY,
    "mouse_move": ActionType.MOUSE_MOVE,
    "drag": ActionType.DRAG,
    "wait": ActionType.WAIT,
    "screenshot": ActionType.SCREENSHOT,
}


def parse_openai_action(action_data: Any) -> CUAAction:
    """Parse an OpenAI CUA action object into our CUAAction model.

    The action_data comes from ``response.output[i].action`` where
    the item type is ``computer_call``.
    """
    raw = {}
    if hasattr(action_data, "__dict__"):
        raw = {k: v for k, v in action_data.__dict__.items() if not k.startswith("_")}
    elif isinstance(action_data, dict):
        raw = dict(action_data)

    action_type_str = getattr(action_data, "type", raw.get("type", "click"))
    action_type = _OPENAI_ACTION_MAP.get(action_type_str, ActionType.CLICK)

    return CUAAction(
        action_type=action_type,
        x=int(getattr(action_data, "x", raw.get("x", 0)) or 0),
        y=int(getattr(action_data, "y", raw.get("y", 0)) or 0),
        button=str(getattr(action_data, "button", raw.get("button", "left")) or "left"),
        text=str(getattr(action_data, "text", raw.get("text", "")) or ""),
        keys=list(getattr(action_data, "keys", raw.get("keys", [])) or []),
        scroll_x=int(
            getattr(action_data, "scrollX", raw.get("scrollX", raw.get("scroll_x", 0))) or 0
        ),
        scroll_y=int(
            getattr(action_data, "scrollY", raw.get("scrollY", raw.get("scroll_y", 0))) or 0
        ),
        raw=raw,
    )


class OpenAICUA:
    """OpenAI CUA provider using the Responses API.

    Parameters
    ----------
    model:
        The CUA model to use (default: ``computer-use-preview``).
    display_width:
        Browser viewport width sent to the model.
    display_height:
        Browser viewport height sent to the model.
    environment:
        Environment type (``browser``, ``windows``, ``mac``, ``ubuntu``).
    """

    def __init__(
        self,
        model: str = "computer-use-preview",
        display_width: int = 1280,
        display_height: int = 800,
        environment: str = "browser",
    ) -> None:
        self.model = model
        self.display_width = display_width
        self.display_height = display_height
        self.environment = environment
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "OpenAI SDK is required for CUA. Install with: pip install openai"
                ) from exc
            api_key = first_env_secret(("OPENAI_API_KEY", "CODEX_API_KEY"))
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY (or CODEX_API_KEY) is not set. "
                    "Set it in your environment or in a .env file "
                    "(same key as for OpenAI chatbots)."
                )
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _tools(self) -> list[dict[str, Any]]:
        """Return the tool definition for computer use."""
        return [
            {
                "type": "computer_use_preview",
                "display_width": self.display_width,
                "display_height": self.display_height,
                "environment": self.environment,
            }
        ]

    def create_initial_request(
        self,
        task: str,
        screenshot_b64: str | None = None,
    ) -> Any:
        """Send the initial request to the CUA model.

        Parameters
        ----------
        task:
            Natural-language description of what to do.
        screenshot_b64:
            Optional base64-encoded screenshot of the initial state.

        Returns
        -------
        The OpenAI Response object.
        """
        client = self._get_client()

        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": task},
        ]
        if screenshot_b64:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{screenshot_b64}",
                }
            )

        response = client.responses.create(
            model=self.model,
            tools=self._tools(),
            input=[{"role": "user", "content": content}],
            reasoning={"summary": "concise"},
            truncation="auto",
        )
        return response

    def send_screenshot(
        self,
        previous_response_id: str,
        call_id: str,
        screenshot_b64: str,
    ) -> Any:
        """Send a screenshot back to the model (continuing the CUA loop).

        Parameters
        ----------
        previous_response_id:
            The ``response.id`` from the previous turn.
        call_id:
            The ``computer_call.call_id`` we're responding to.
        screenshot_b64:
            Base64-encoded screenshot after executing the action.

        Returns
        -------
        The next OpenAI Response object.
        """
        client = self._get_client()

        response = client.responses.create(
            model=self.model,
            previous_response_id=previous_response_id,
            tools=self._tools(),
            input=[
                {
                    "call_id": call_id,
                    "type": "computer_call_output",
                    "output": {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{screenshot_b64}",
                    },
                }
            ],
            truncation="auto",
        )
        return response

    @staticmethod
    def extract_computer_call(response: Any) -> tuple[str | None, Any | None, str]:
        """Extract the computer_call from a response, if any.

        Returns
        -------
        (call_id, action, reasoning):
            - call_id: The call_id to respond to, or None if no computer call.
            - action: The raw action object, or None.
            - reasoning: Any reasoning summary text.
        """
        call_id = None
        action = None
        reasoning = ""

        for item in response.output:
            item_type = getattr(item, "type", "")
            if item_type == "computer_call":
                call_id = getattr(item, "call_id", None)
                action = getattr(item, "action", None)
            elif item_type == "reasoning":
                summaries = getattr(item, "summary", [])
                if summaries:
                    for s in summaries:
                        text = getattr(s, "text", "")
                        if text:
                            reasoning += text + " "
            elif item_type == "message" or item_type == "text":
                text = getattr(item, "text", "")
                if text:
                    reasoning += text + " "

        return call_id, action, reasoning.strip()

    @staticmethod
    def extract_text_output(response: Any) -> str:
        """Extract any text output from the response (final answer)."""
        parts: list[str] = []
        for item in response.output:
            item_type = getattr(item, "type", "")
            if item_type == "message":
                content = getattr(item, "content", [])
                for c in content:
                    text = getattr(c, "text", "")
                    if text:
                        parts.append(text)
            elif item_type == "text":
                text = getattr(item, "text", "")
                if text:
                    parts.append(text)
        return "\n".join(parts)
