"""Playwright browser environment for CUA sessions.

Provides screenshot capture and action execution on a real browser instance.
Supports both headless and headed (visible) modes.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from codex_manager.cua.actions import ActionType, CUAAction

logger = logging.getLogger(__name__)


class BrowserEnvironment:
    """Manages a Playwright browser instance for CUA interactions.

    Usage::

        async with BrowserEnvironment(width=1280, height=800) as env:
            await env.navigate("http://localhost:5088")
            screenshot = await env.screenshot_b64()
            await env.execute_action(some_action)
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 800,
        headless: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.headless = headless
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None

    async def __aenter__(self) -> BrowserEnvironment:
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()

    async def start(self) -> None:
        """Launch the browser."""
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is required for CUA. Install with:\n"
                "  pip install playwright\n"
                "  python -m playwright install"
            ) from exc

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-extensions",
                "--disable-file-system",
                "--no-sandbox",
            ],
        )
        context = await self._browser.new_context(
            viewport={"width": self.width, "height": self.height},
            # Don't expose host env
        )
        self._page = await context.new_page()
        logger.info(
            "Browser started: %dx%d headless=%s",
            self.width, self.height, self.headless,
        )

    async def stop(self) -> None:
        """Close the browser."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        logger.info("Browser stopped")

    @property
    def page(self) -> Any:
        """The active Playwright page."""
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")
        return self._page

    async def navigate(self, url: str, timeout_ms: int = 30_000) -> None:
        """Navigate to a URL."""
        logger.info("Navigating to %s", url)
        await self.page.goto(url, timeout=timeout_ms)
        await self.page.wait_for_load_state("domcontentloaded")

    async def screenshot_bytes(self) -> bytes:
        """Capture a full-page screenshot as PNG bytes."""
        return await self.page.screenshot(type="png")

    async def screenshot_b64(self) -> str:
        """Capture a screenshot and return as base64-encoded string."""
        raw = await self.screenshot_bytes()
        return base64.b64encode(raw).decode("ascii")

    async def save_screenshot(self, path: str | Path) -> None:
        """Save a screenshot to disk."""
        raw = await self.screenshot_bytes()
        Path(path).write_bytes(raw)

    async def execute_action(self, action: CUAAction) -> None:
        """Execute a CUA action on the browser page."""
        page = self.page
        at = action.action_type

        try:
            if at == ActionType.CLICK:
                btn = action.button if action.button in ("left", "right", "middle") else "left"
                await page.mouse.click(action.x, action.y, button=btn)
                logger.debug("click (%d, %d) button=%s", action.x, action.y, btn)

            elif at == ActionType.DOUBLE_CLICK:
                await page.mouse.dblclick(action.x, action.y)
                logger.debug("dblclick (%d, %d)", action.x, action.y)

            elif at == ActionType.RIGHT_CLICK:
                await page.mouse.click(action.x, action.y, button="right")
                logger.debug("right_click (%d, %d)", action.x, action.y)

            elif at == ActionType.SCROLL:
                await page.mouse.move(action.x, action.y)
                sx = action.scroll_x or action.raw.get("scrollX", 0) or action.raw.get("scroll_x", 0)
                sy = action.scroll_y or action.raw.get("scrollY", 0) or action.raw.get("scroll_y", 0)
                await page.evaluate(f"window.scrollBy({sx}, {sy})")
                logger.debug("scroll (%d, %d) dx=%d dy=%d", action.x, action.y, sx, sy)

            elif at == ActionType.TYPE:
                await page.keyboard.type(action.text)
                logger.debug("type '%s'", action.text[:50])

            elif at in (ActionType.KEYPRESS, ActionType.KEY):
                keys = action.keys or ([action.text] if action.text else [])
                for k in keys:
                    mapped = _map_key(k)
                    await page.keyboard.press(mapped)
                    logger.debug("keypress '%s'", mapped)

            elif at == ActionType.MOUSE_MOVE:
                await page.mouse.move(action.x, action.y)
                logger.debug("mouse_move (%d, %d)", action.x, action.y)

            elif at == ActionType.DRAG:
                sx, sy = action.start_x or action.x, action.start_y or action.y
                ex, ey = action.end_x, action.end_y
                await page.mouse.move(sx, sy)
                await page.mouse.down()
                await page.mouse.move(ex, ey)
                await page.mouse.up()
                logger.debug("drag (%d,%d)->(%d,%d)", sx, sy, ex, ey)

            elif at == ActionType.WAIT:
                wait_ms = int(action.raw.get("ms", 2000))
                await page.wait_for_timeout(wait_ms)
                logger.debug("wait %dms", wait_ms)

            elif at == ActionType.SCREENSHOT:
                # No action needed — screenshot is always taken after every step
                logger.debug("screenshot (no-op)")

            else:
                logger.warning("Unknown action type: %s", at)

        except Exception as exc:
            logger.error("Action %s failed: %s", at, exc)
            raise


# ── Key mapping helpers ──────────────────────────────────────────

_KEY_MAP: dict[str, str] = {
    "enter": "Enter",
    "return": "Enter",
    "tab": "Tab",
    "space": " ",
    "backspace": "Backspace",
    "delete": "Delete",
    "escape": "Escape",
    "esc": "Escape",
    "arrowup": "ArrowUp",
    "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft",
    "arrowright": "ArrowRight",
    "home": "Home",
    "end": "End",
    "pageup": "PageUp",
    "pagedown": "PageDown",
    "ctrl+a": "Control+a",
    "ctrl+c": "Control+c",
    "ctrl+v": "Control+v",
    "ctrl+s": "Control+s",
    "ctrl+z": "Control+z",
    "ctrl+shift+i": "Control+Shift+i",
}


def _map_key(key: str) -> str:
    """Map a key name from CUA model output to Playwright key name."""
    lower = key.lower().strip()
    return _KEY_MAP.get(lower, key)
