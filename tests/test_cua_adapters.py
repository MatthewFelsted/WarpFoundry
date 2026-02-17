"""Unit tests for CUA provider adapters (OpenAI + Anthropic)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from codex_manager.cua.actions import ActionType
from codex_manager.cua.anthropic_cua import AnthropicCUA, parse_anthropic_action
from codex_manager.cua.openai_cua import OpenAICUA, parse_openai_action


def test_parse_openai_action_supports_dict_aliases_and_unknown_type() -> None:
    action = parse_openai_action(
        {
            "type": "not_a_real_action",
            "x": "12",
            "y": "34",
            "button": "",
            "text": None,
            "keys": ["Enter"],
            "scroll_x": 5,
            "scrollY": "9",
        }
    )

    assert action.action_type == ActionType.CLICK
    assert action.x == 12
    assert action.y == 34
    assert action.button == "left"
    assert action.text == ""
    assert action.keys == ["Enter"]
    assert action.scroll_x == 5
    assert action.scroll_y == 9
    assert action.raw["type"] == "not_a_real_action"


def test_parse_openai_action_reads_public_object_fields() -> None:
    class _ActionData:
        def __init__(self) -> None:
            self.type = "scroll"
            self.x = "7"
            self.y = None
            self.scrollX = 11
            self._private = "ignored"

    parsed = parse_openai_action(_ActionData())

    assert parsed.action_type == ActionType.SCROLL
    assert parsed.x == 7
    assert parsed.y == 0
    assert parsed.scroll_x == 11
    assert "_private" not in parsed.raw


def test_openai_extract_computer_call_collects_reasoning_and_message_text() -> None:
    action_obj = SimpleNamespace(type="click", x=4, y=5)
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="reasoning",
                summary=[SimpleNamespace(text="think-a"), SimpleNamespace(text="think-b")],
            ),
            SimpleNamespace(type="message", text="message-note"),
            SimpleNamespace(type="text", text="text-note"),
            SimpleNamespace(type="computer_call", call_id="call-1", action=action_obj),
        ]
    )

    call_id, action, reasoning = OpenAICUA.extract_computer_call(response)

    assert call_id == "call-1"
    assert action is action_obj
    assert reasoning == "think-a think-b message-note text-note"


def test_openai_extract_text_output_reads_message_content_and_text_items() -> None:
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(text="line-1"), SimpleNamespace(text="line-2")],
            ),
            SimpleNamespace(type="text", text="line-3"),
        ]
    )

    assert OpenAICUA.extract_text_output(response) == "line-1\nline-2\nline-3"


def test_openai_create_initial_request_includes_image_when_present() -> None:
    captured: dict[str, Any] = {}

    class _Responses:
        def create(self, **kwargs: Any) -> dict[str, str]:
            captured.update(kwargs)
            return {"id": "r-openai"}

    provider = OpenAICUA(model="computer-use-test", display_width=900, display_height=700)
    provider._client = SimpleNamespace(responses=_Responses())

    response = provider.create_initial_request("Do work", screenshot_b64="abc123")

    assert response == {"id": "r-openai"}
    assert captured["model"] == "computer-use-test"
    assert captured["tools"][0]["display_width"] == 900
    assert captured["tools"][0]["display_height"] == 700
    assert captured["input"][0]["role"] == "user"
    content = captured["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "Do work"}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"] == "data:image/png;base64,abc123"


def test_openai_send_screenshot_builds_computer_call_output_payload() -> None:
    captured: dict[str, Any] = {}

    class _Responses:
        def create(self, **kwargs: Any) -> dict[str, str]:
            captured.update(kwargs)
            return {"id": "r-next"}

    provider = OpenAICUA(model="computer-use-test")
    provider._client = SimpleNamespace(responses=_Responses())

    response = provider.send_screenshot("r-prev", "call-42", "img-b64")

    assert response == {"id": "r-next"}
    assert captured["previous_response_id"] == "r-prev"
    assert captured["input"][0]["call_id"] == "call-42"
    assert captured["input"][0]["type"] == "computer_call_output"
    output_payload = captured["input"][0]["output"]
    assert output_payload["type"] == "input_image"
    assert output_payload["image_url"] == "data:image/png;base64,img-b64"


def test_openai_get_client_raises_runtime_error_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "openai", types.ModuleType("openai"))

    provider = OpenAICUA()

    with pytest.raises(RuntimeError, match="OpenAI SDK is required for CUA"):
        provider._get_client()


def test_openai_get_client_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    module = types.ModuleType("openai")

    class _FakeOpenAI:
        def __init__(self, *, api_key: str) -> None:
            calls.append(api_key)

    module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-real")
    monkeypatch.delenv("CODEX_API_KEY", raising=False)

    provider = OpenAICUA()
    first = provider._get_client()
    second = provider._get_client()

    assert first is second
    assert calls == ["sk-openai-real"]


def test_parse_anthropic_action_uses_coordinate_boundaries_and_key_mapping() -> None:
    action = parse_anthropic_action(
        {
            "action": "key",
            "text": "Enter",
            "coordinate": [17],
            "scroll_x": "3",
            "scroll_y": "4",
            "start_coordinate": [1, 2],
        }
    )

    assert action.action_type == ActionType.KEY
    assert action.x == 17
    assert action.y == 0
    assert action.keys == ["Enter"]
    assert action.scroll_x == 3
    assert action.scroll_y == 4
    assert action.start_x == 1
    assert action.start_y == 2
    assert action.end_x == 17
    assert action.end_y == 0


def test_anthropic_extract_computer_call_collects_text_and_thinking() -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="line-a"),
            SimpleNamespace(type="thinking", thinking="line-b"),
            SimpleNamespace(type="tool_use", name="not-computer", id="skip", input={"x": 1}),
            SimpleNamespace(
                type="tool_use",
                name="computer",
                id="tool-1",
                input={"action": "left_click", "coordinate": [1, 2]},
            ),
        ]
    )

    tool_use_id, tool_input, reasoning = AnthropicCUA.extract_computer_call(response)

    assert tool_use_id == "tool-1"
    assert tool_input == {"action": "left_click", "coordinate": [1, 2]}
    assert reasoning == "line-a line-b"


def test_anthropic_extract_text_output_collects_text_blocks() -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="A"),
            SimpleNamespace(type="tool_use", name="computer", id="t1", input={}),
            SimpleNamespace(type="text", text="B"),
        ]
    )

    assert AnthropicCUA.extract_text_output(response) == "A\nB"


def test_anthropic_create_initial_request_includes_image_when_present() -> None:
    captured: dict[str, Any] = {}

    class _Messages:
        def create(self, **kwargs: Any) -> dict[str, str]:
            captured.update(kwargs)
            return {"id": "anthropic-1"}

    provider = AnthropicCUA(model="claude-test", tool_version="computer_test")
    provider._client = SimpleNamespace(beta=SimpleNamespace(messages=_Messages()))

    response = provider.create_initial_request("Do work", screenshot_b64="img64")

    assert response == {"id": "anthropic-1"}
    assert captured["model"] == "claude-test"
    assert captured["tools"][0]["type"] == "computer_test"
    payload = captured["messages"][0]["content"]
    assert payload[0] == {"type": "text", "text": "Do work"}
    assert payload[1]["type"] == "image"
    assert payload[1]["source"]["data"] == "img64"


def test_anthropic_send_tool_result_appends_user_tool_result_and_calls_api() -> None:
    captured: dict[str, Any] = {}

    class _Messages:
        def create(self, **kwargs: Any) -> dict[str, str]:
            captured.update(kwargs)
            return {"id": "anthropic-2"}

    provider = AnthropicCUA(model="claude-test")
    provider._client = SimpleNamespace(beta=SimpleNamespace(messages=_Messages()))

    messages: list[dict[str, Any]] = [{"role": "assistant", "content": [{"type": "text"}]}]
    response = provider.send_tool_result(messages, "tool-9", "ss-b64")

    assert response == {"id": "anthropic-2"}
    assert len(messages) == 2
    appended = messages[-1]
    assert appended["role"] == "user"
    tool_result = appended["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "tool-9"
    assert tool_result["content"][0]["source"]["data"] == "ss-b64"
    assert captured["messages"] is messages


def test_anthropic_get_client_raises_runtime_error_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", types.ModuleType("anthropic"))

    provider = AnthropicCUA()

    with pytest.raises(RuntimeError, match="Anthropic SDK is required for Claude CUA"):
        provider._get_client()


def test_anthropic_get_client_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    module = types.ModuleType("anthropic")

    class _FakeAnthropic:
        def __init__(self, *, api_key: str) -> None:
            calls.append(api_key)

    module.Anthropic = _FakeAnthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)

    provider = AnthropicCUA()
    first = provider._get_client()
    second = provider._get_client()

    assert first is second
    assert calls == ["sk-ant-real"]
