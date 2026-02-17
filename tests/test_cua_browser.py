"""Unit tests for Playwright browser environment helpers used by CUA."""

from __future__ import annotations

import asyncio
import builtins
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from codex_manager.cua.actions import ActionType, CUAAction
from codex_manager.cua.browser import BrowserEnvironment, _map_key


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class _RecordingMouse:
    def __init__(self, calls: list[tuple[Any, ...]], *, click_error: Exception | None = None) -> None:
        self._calls = calls
        self._click_error = click_error

    async def click(self, x: int, y: int, button: str = "left") -> None:
        if self._click_error is not None:
            raise self._click_error
        self._calls.append(("mouse.click", x, y, button))

    async def dblclick(self, x: int, y: int) -> None:
        self._calls.append(("mouse.dblclick", x, y))

    async def move(self, x: int, y: int) -> None:
        self._calls.append(("mouse.move", x, y))

    async def down(self) -> None:
        self._calls.append(("mouse.down",))

    async def up(self) -> None:
        self._calls.append(("mouse.up",))


class _RecordingKeyboard:
    def __init__(self, calls: list[tuple[Any, ...]]) -> None:
        self._calls = calls

    async def type(self, text: str) -> None:
        self._calls.append(("keyboard.type", text))

    async def press(self, key: str) -> None:
        self._calls.append(("keyboard.press", key))


class _RecordingPage:
    def __init__(
        self,
        *,
        screenshot_bytes: bytes = b"png-bytes",
        click_error: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.mouse = _RecordingMouse(self.calls, click_error=click_error)
        self.keyboard = _RecordingKeyboard(self.calls)
        self._screenshot_bytes = screenshot_bytes

    async def goto(self, url: str, timeout: int) -> None:
        self.calls.append(("goto", url, timeout))

    async def wait_for_load_state(self, state: str) -> None:
        self.calls.append(("wait_for_load_state", state))

    async def screenshot(self, *, type: str) -> bytes:
        self.calls.append(("screenshot", type))
        return self._screenshot_bytes

    async def evaluate(self, script: str) -> None:
        self.calls.append(("evaluate", script))

    async def wait_for_timeout(self, ms: int) -> None:
        self.calls.append(("wait_for_timeout", ms))


def test_browser_page_property_requires_start() -> None:
    env = BrowserEnvironment()

    with pytest.raises(RuntimeError, match="Browser not started"):
        _ = env.page


def test_map_key_supports_known_and_unknown_keys() -> None:
    assert _map_key("enter") == "Enter"
    assert _map_key("Ctrl+Shift+I") == "Control+Shift+i"
    assert _map_key("custom-key") == "custom-key"


def test_start_raises_runtime_error_when_playwright_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _fake_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "playwright.async_api":
            raise ImportError("playwright unavailable")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    env = BrowserEnvironment()
    with pytest.raises(RuntimeError, match="Playwright is required for CUA"):
        _run(env.start())


def test_start_and_stop_initialize_and_cleanup_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    fake_page = _RecordingPage()

    class _FakeContext:
        async def new_page(self) -> _RecordingPage:
            captured["new_page"] = True
            return fake_page

    class _FakeBrowser:
        async def new_context(self, *, viewport: dict[str, int]) -> _FakeContext:
            captured["viewport"] = viewport
            return _FakeContext()

        async def close(self) -> None:
            captured["browser_closed"] = True

    class _FakeChromium:
        async def launch(self, *, headless: bool, args: list[str]) -> _FakeBrowser:
            captured["headless"] = headless
            captured["launch_args"] = args
            return _FakeBrowser()

    class _FakePlaywright:
        chromium = _FakeChromium()

        async def stop(self) -> None:
            captured["playwright_stopped"] = True

    class _Starter:
        async def start(self) -> _FakePlaywright:
            return _FakePlaywright()

    async_api_module = types.ModuleType("playwright.async_api")
    async_api_module.async_playwright = lambda: _Starter()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "playwright.async_api", async_api_module)

    env = BrowserEnvironment(width=1010, height=770, headless=False)
    _run(env.start())
    assert env.page is fake_page
    assert captured["headless"] is False
    assert captured["viewport"] == {"width": 1010, "height": 770}
    assert "--no-sandbox" in captured["launch_args"]

    _run(env.stop())
    assert captured["browser_closed"] is True
    assert captured["playwright_stopped"] is True
    assert env._browser is None
    assert env._playwright is None


def test_navigate_and_screenshot_helpers(tmp_path: Path) -> None:
    env = BrowserEnvironment()
    page = _RecordingPage(screenshot_bytes=b"\x89PNG")
    env._page = page

    _run(env.navigate("https://example.test", timeout_ms=1234))
    assert ("goto", "https://example.test", 1234) in page.calls
    assert ("wait_for_load_state", "domcontentloaded") in page.calls

    b64 = _run(env.screenshot_b64())
    assert b64 == "iVBORw=="

    out_path = tmp_path / "shot.png"
    _run(env.save_screenshot(out_path))
    assert out_path.read_bytes() == b"\x89PNG"


def test_execute_action_click_falls_back_to_left_button_for_invalid_value() -> None:
    env = BrowserEnvironment()
    page = _RecordingPage()
    env._page = page

    action = CUAAction(action_type=ActionType.CLICK, x=10, y=20, button="not-a-button")
    _run(env.execute_action(action))

    assert ("mouse.click", 10, 20, "left") in page.calls


def test_execute_action_scroll_uses_raw_scroll_offsets_when_explicit_values_missing() -> None:
    env = BrowserEnvironment()
    page = _RecordingPage()
    env._page = page

    action = CUAAction(
        action_type=ActionType.SCROLL,
        x=5,
        y=6,
        scroll_x=0,
        scroll_y=0,
        raw={"scrollX": 12, "scroll_y": -4},
    )
    _run(env.execute_action(action))

    assert ("mouse.move", 5, 6) in page.calls
    assert ("evaluate", "window.scrollBy(12, -4)") in page.calls


def test_execute_action_keypress_maps_known_keys_and_preserves_unknown() -> None:
    env = BrowserEnvironment()
    page = _RecordingPage()
    env._page = page

    action = CUAAction(
        action_type=ActionType.KEYPRESS,
        keys=["enter", "ctrl+shift+i", "my-custom-key"],
    )
    _run(env.execute_action(action))

    assert ("keyboard.press", "Enter") in page.calls
    assert ("keyboard.press", "Control+Shift+i") in page.calls
    assert ("keyboard.press", "my-custom-key") in page.calls


def test_execute_action_key_uses_text_as_fallback_key() -> None:
    env = BrowserEnvironment()
    page = _RecordingPage()
    env._page = page

    action = CUAAction(action_type=ActionType.KEY, text="tab")
    _run(env.execute_action(action))

    assert ("keyboard.press", "Tab") in page.calls


def test_execute_action_drag_wait_and_screenshot_branches() -> None:
    env = BrowserEnvironment()
    page = _RecordingPage()
    env._page = page

    drag_action = CUAAction(action_type=ActionType.DRAG, x=11, y=12, end_x=20, end_y=22)
    _run(env.execute_action(drag_action))
    assert ("mouse.move", 11, 12) in page.calls
    assert ("mouse.down",) in page.calls
    assert ("mouse.move", 20, 22) in page.calls
    assert ("mouse.up",) in page.calls

    wait_default = CUAAction(action_type=ActionType.WAIT, raw={})
    _run(env.execute_action(wait_default))
    assert ("wait_for_timeout", 2000) in page.calls

    wait_custom = CUAAction(action_type=ActionType.WAIT, raw={"ms": "150"})
    _run(env.execute_action(wait_custom))
    assert ("wait_for_timeout", 150) in page.calls

    before = list(page.calls)
    _run(env.execute_action(CUAAction(action_type=ActionType.SCREENSHOT)))
    assert page.calls == before


def test_execute_action_reraises_underlying_page_errors() -> None:
    env = BrowserEnvironment()
    env._page = _RecordingPage(click_error=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        _run(env.execute_action(CUAAction(action_type=ActionType.CLICK, x=1, y=2)))
