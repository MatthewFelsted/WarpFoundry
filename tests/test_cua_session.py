"""Unit tests for CUA session orchestration loops and wrappers."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

import codex_manager.cua.anthropic_cua as anthropic_cua_module
import codex_manager.cua.openai_cua as openai_cua_module
import codex_manager.cua.session as session_module
from codex_manager.cua.actions import ActionType, CUAAction, CUAProvider, CUASessionConfig, CUASessionResult


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


async def _immediate_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
    return func(*args, **kwargs)


async def _no_sleep(_seconds: float) -> None:
    return None


class _LoopBrowser:
    def __init__(self, *, screenshot_b64: str = "s" * 140, fail_execute: bool = False) -> None:
        self.executed: list[CUAAction] = []
        self.saved: list[str] = []
        self.screenshot_value = screenshot_b64
        self.fail_execute = fail_execute

    async def execute_action(self, action: CUAAction) -> None:
        self.executed.append(action)
        if self.fail_execute:
            raise RuntimeError("action-failed")

    async def screenshot_b64(self) -> str:
        return self.screenshot_value

    async def save_screenshot(self, path: str | Path) -> None:
        self.saved.append(str(path))
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"png")


def test_parse_observations_ignores_malformed_lines_and_attaches_latest_screenshot() -> None:
    text = "\n".join(
        [
            "OBSERVATION|major|layout|sidebar|Aligned items|Items overlap|Use flex-wrap",
            "not-an-observation-line",
            "OBSERVATION|minor|content|title|Readable title|Truncated title|Increase width",
            "OBSERVATION|too|short",
        ]
    )

    observations = session_module._parse_observations(text, screenshots=["one.png", "two.png"])

    assert len(observations) == 2
    assert observations[0].severity == "major"
    assert observations[0].category == "layout"
    assert observations[0].element == "sidebar"
    assert observations[1].severity == "minor"
    assert observations[1].category == "content"
    assert all(obs.screenshot == "two.png" for obs in observations)


def test_parse_observations_returns_empty_for_blank_text() -> None:
    assert session_module._parse_observations("", screenshots=["x.png"]) == []
    assert session_module._parse_observations(" \n\t", screenshots=None) == []


def test_openai_loop_completes_when_provider_returns_no_computer_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module.asyncio, "to_thread", _immediate_to_thread)
    monkeypatch.setattr(session_module.asyncio, "sleep", _no_sleep)

    class _Response:
        id = "resp-1"

    class _Provider:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def create_initial_request(self, task: str, screenshot_b64: str | None = None) -> _Response:
            assert task == "enhanced-task"
            assert screenshot_b64 == "initial"
            return _Response()

        @staticmethod
        def extract_computer_call(_response: _Response) -> tuple[None, None, str]:
            return None, None, "provider-reasoning"

        @staticmethod
        def extract_text_output(_response: _Response) -> str:
            return "final-output"

    monkeypatch.setattr(openai_cua_module, "OpenAICUA", _Provider)
    monkeypatch.setattr(openai_cua_module, "parse_openai_action", lambda _data: None)

    config = CUASessionConfig(provider=CUAProvider.OPENAI, task="task", max_steps=2, timeout_seconds=0)
    result = CUASessionResult(task="task", provider="openai")
    browser = _LoopBrowser()

    _run(
        session_module._openai_loop(
            config=config,
            browser=browser,
            initial_screenshot="initial",
            result=result,
            screenshots_dir=None,
            start_time=time.monotonic(),
            task_text="enhanced-task",
        )
    )

    assert result.summary == "final-output"
    assert result.steps == []


def test_openai_loop_records_action_failures_and_hits_max_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(session_module.asyncio, "to_thread", _immediate_to_thread)
    monkeypatch.setattr(session_module.asyncio, "sleep", _no_sleep)

    class _Response:
        def __init__(self, response_id: str) -> None:
            self.id = response_id

    class _Provider:
        send_calls: list[tuple[str, str, str]] = []

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def create_initial_request(self, _task: str, _screenshot_b64: str | None = None) -> _Response:
            return _Response("resp-1")

        @staticmethod
        def extract_computer_call(_response: _Response) -> tuple[str, dict[str, Any], str]:
            return "call-1", {"type": "click", "x": 3, "y": 4}, "click it"

        def send_screenshot(self, response_id: str, call_id: str, screenshot_b64: str) -> _Response:
            self.send_calls.append((response_id, call_id, screenshot_b64))
            return _Response("resp-2")

        @staticmethod
        def extract_text_output(_response: _Response) -> str:
            return ""

    monkeypatch.setattr(openai_cua_module, "OpenAICUA", _Provider)
    monkeypatch.setattr(
        openai_cua_module,
        "parse_openai_action",
        lambda _data: CUAAction(action_type=ActionType.CLICK, x=3, y=4),
    )

    config = CUASessionConfig(provider=CUAProvider.OPENAI, task="task", max_steps=1, timeout_seconds=0)
    result = CUASessionResult(task="task", provider="openai")
    browser = _LoopBrowser(fail_execute=True)

    _run(
        session_module._openai_loop(
            config=config,
            browser=browser,
            initial_screenshot="initial",
            result=result,
            screenshots_dir=tmp_path,
            start_time=time.monotonic(),
            task_text="task",
        )
    )

    assert len(result.steps) == 1
    assert result.steps[0].success is False
    assert "action-failed" in result.steps[0].error
    assert result.steps[0].screenshot_b64.endswith("...")
    assert result.summary == "Reached max steps (1)"
    assert browser.saved and browser.saved[0].endswith("step_001_click.png")
    assert _Provider.send_calls == [("resp-1", "call-1", browser.screenshot_value)]


def test_openai_loop_stops_on_timeout_before_extracting_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module.asyncio, "to_thread", _immediate_to_thread)

    class _Response:
        id = "resp-timeout"

    class _Provider:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        @staticmethod
        def create_initial_request(_task: str, _screenshot_b64: str | None = None) -> _Response:
            return _Response()

        @staticmethod
        def extract_computer_call(_response: _Response) -> tuple[None, None, str]:
            raise AssertionError("extract_computer_call should not be reached after timeout")

    monkeypatch.setattr(openai_cua_module, "OpenAICUA", _Provider)
    monkeypatch.setattr(openai_cua_module, "parse_openai_action", lambda _data: None)

    config = CUASessionConfig(provider=CUAProvider.OPENAI, task="task", max_steps=2, timeout_seconds=1)
    result = CUASessionResult(task="task", provider="openai")

    _run(
        session_module._openai_loop(
            config=config,
            browser=_LoopBrowser(),
            initial_screenshot="initial",
            result=result,
            screenshots_dir=None,
            start_time=time.monotonic() - 5,
            task_text="task",
        )
    )

    assert result.error == "Timeout after 1s"
    assert result.steps == []


def test_anthropic_loop_stops_on_end_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module.asyncio, "to_thread", _immediate_to_thread)
    monkeypatch.setattr(session_module.asyncio, "sleep", _no_sleep)

    class _Response:
        stop_reason = "end_turn"
        content: list[Any] = []

    class _Provider:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        @staticmethod
        def create_initial_request(_task: str, _screenshot_b64: str | None = None) -> _Response:
            return _Response()

        @staticmethod
        def extract_text_output(_response: _Response) -> str:
            return "done"

        @staticmethod
        def extract_computer_call(_response: _Response) -> tuple[str | None, dict[str, Any] | None, str]:
            raise AssertionError("extract_computer_call should not be called on end_turn")

    monkeypatch.setattr(anthropic_cua_module, "AnthropicCUA", _Provider)
    monkeypatch.setattr(
        anthropic_cua_module,
        "parse_anthropic_action",
        lambda _tool_input: CUAAction(action_type=ActionType.SCREENSHOT),
    )

    config = CUASessionConfig(provider=CUAProvider.ANTHROPIC, task="task", max_steps=2, timeout_seconds=0)
    result = CUASessionResult(task="task", provider="anthropic")

    _run(
        session_module._anthropic_loop(
            config=config,
            browser=_LoopBrowser(),
            initial_screenshot="initial",
            result=result,
            screenshots_dir=None,
            start_time=time.monotonic(),
            task_text="task",
        )
    )

    assert result.summary == "done"
    assert result.steps == []


def test_anthropic_loop_handles_screenshot_action_without_browser_execute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(session_module.asyncio, "to_thread", _immediate_to_thread)
    monkeypatch.setattr(session_module.asyncio, "sleep", _no_sleep)

    class _Response:
        def __init__(self, stop_reason: str, content: list[Any] | None = None) -> None:
            self.stop_reason = stop_reason
            self.content = content or []

    class _Provider:
        send_messages: list[dict[str, Any]] | None = None

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self._responses = [_Response("tool_use"), _Response("end_turn")]

        def create_initial_request(self, _task: str, _screenshot_b64: str | None = None) -> _Response:
            return self._responses[0]

        @staticmethod
        def extract_computer_call(_response: _Response) -> tuple[str, dict[str, Any], str]:
            return "tool-1", {"action": "screenshot"}, "inspect"

        def send_tool_result(
            self,
            messages: list[dict[str, Any]],
            _tool_use_id: str,
            _screenshot_b64: str,
        ) -> _Response:
            type(self).send_messages = messages
            return self._responses[1]

        @staticmethod
        def extract_text_output(_response: _Response) -> str:
            return "all done"

    monkeypatch.setattr(anthropic_cua_module, "AnthropicCUA", _Provider)
    monkeypatch.setattr(
        anthropic_cua_module,
        "parse_anthropic_action",
        lambda _tool_input: CUAAction(action_type=ActionType.SCREENSHOT),
    )

    config = CUASessionConfig(provider=CUAProvider.ANTHROPIC, task="task", max_steps=2, timeout_seconds=0)
    result = CUASessionResult(task="task", provider="anthropic")
    browser = _LoopBrowser()

    _run(
        session_module._anthropic_loop(
            config=config,
            browser=browser,
            initial_screenshot="initial",
            result=result,
            screenshots_dir=tmp_path,
            start_time=time.monotonic(),
            task_text="task",
        )
    )

    assert len(result.steps) == 1
    assert result.steps[0].success is True
    assert browser.executed == []
    assert result.summary == "all done"
    assert browser.saved and browser.saved[0].endswith("step_001_screenshot.png")
    assert _Provider.send_messages is not None
    assert _Provider.send_messages[0]["role"] == "user"


def test_anthropic_loop_uses_reasoning_fallback_when_no_tool_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module.asyncio, "to_thread", _immediate_to_thread)

    class _Response:
        stop_reason = "tool_use"
        content: list[Any] = []

    class _Provider:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        @staticmethod
        def create_initial_request(_task: str, _screenshot_b64: str | None = None) -> _Response:
            return _Response()

        @staticmethod
        def extract_computer_call(_response: _Response) -> tuple[None, None, str]:
            return None, None, "reasoning-only"

        @staticmethod
        def extract_text_output(_response: _Response) -> str:
            return ""

    monkeypatch.setattr(anthropic_cua_module, "AnthropicCUA", _Provider)
    monkeypatch.setattr(
        anthropic_cua_module,
        "parse_anthropic_action",
        lambda _tool_input: CUAAction(action_type=ActionType.CLICK),
    )

    config = CUASessionConfig(provider=CUAProvider.ANTHROPIC, task="task", max_steps=2, timeout_seconds=0)
    result = CUASessionResult(task="task", provider="anthropic")

    _run(
        session_module._anthropic_loop(
            config=config,
            browser=_LoopBrowser(),
            initial_screenshot="initial",
            result=result,
            screenshots_dir=None,
            start_time=time.monotonic(),
            task_text="task",
        )
    )

    assert result.summary == "reasoning-only"
    assert result.steps == []


def test_run_cua_session_parses_observations_and_writes_ledger_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(session_module.asyncio, "sleep", _no_sleep)

    class _FakeBrowserEnvironment:
        instances: list[_FakeBrowserEnvironment] = []

        def __init__(self, width: int, height: int, headless: bool) -> None:
            self.width = width
            self.height = height
            self.headless = headless
            self.navigated: list[str] = []
            self.saved: list[str] = []

        async def __aenter__(self) -> _FakeBrowserEnvironment:
            type(self).instances.append(self)
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def navigate(self, url: str) -> None:
            self.navigated.append(url)

        async def screenshot_b64(self) -> str:
            return "initial-b64"

        async def save_screenshot(self, path: str | Path) -> None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"png")
            self.saved.append(str(path))

    class _Ledger:
        def __init__(self) -> None:
            self.entries: list[dict[str, Any]] = []

        def add(self, **kwargs: Any) -> None:
            self.entries.append(kwargs)

    async def _fake_openai_loop(
        config: CUASessionConfig,
        browser: Any,
        initial_screenshot: str,
        result: CUASessionResult,
        screenshots_dir: Path | None,
        start_time: float = 0.0,
        task_text: str = "",
    ) -> None:
        assert config.provider == CUAProvider.OPENAI
        assert initial_screenshot == "initial-b64"
        assert isinstance(start_time, float)
        assert screenshots_dir is not None
        assert "IMPORTANT: After you finish testing" in task_text
        result.summary = (
            "OBSERVATION|major|layout|sidebar|Aligned rows|Rows overlap|Use spacing\n"
            "OBSERVATION|positive|interaction|toolbar|Quick response|Works instantly|N/A"
        )
        result.screenshots_saved.append("post-step.png")

    monkeypatch.setattr(session_module, "BrowserEnvironment", _FakeBrowserEnvironment)
    monkeypatch.setattr(session_module, "_openai_loop", _fake_openai_loop)

    config = CUASessionConfig(
        provider=CUAProvider.OPENAI,
        task="Visual QA",
        target_url="https://example.test",
        save_screenshots=True,
        screenshots_dir=str(tmp_path),
    )
    ledger = _Ledger()

    result = _run(session_module.run_cua_session(config, ledger=ledger, step_ref="Loop 1, Step 4"))

    assert result.success is True
    assert len(result.observations) == 2
    assert result.observations[0].severity == "major"
    assert result.observations[1].severity == "positive"
    assert ledger.entries and len(ledger.entries) == 2
    assert ledger.entries[0]["source"] == "cua:openai"
    assert ledger.entries[0]["step_ref"] == "Loop 1, Step 4"
    assert _FakeBrowserEnvironment.instances[0].navigated == ["https://example.test"]
    assert result.screenshots_saved[0].endswith("step_000_initial.png")


def test_run_cua_session_adds_openai_access_hint_for_404_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingBrowserEnvironment:
        def __init__(self, width: int, height: int, headless: bool) -> None:
            self.width = width
            self.height = height
            self.headless = headless

        async def __aenter__(self) -> _FailingBrowserEnvironment:
            raise RuntimeError("404 model_not_found: model does not exist or you do not have access")

        async def __aexit__(self, *args: Any) -> None:
            return None

    monkeypatch.setattr(session_module, "BrowserEnvironment", _FailingBrowserEnvironment)

    config = CUASessionConfig(provider=CUAProvider.OPENAI, task="Visual QA", save_screenshots=False)
    result = _run(session_module.run_cua_session(config))

    assert result.success is False
    assert "Tier 3+" in result.error
    assert "platform.openai.com/docs/models/computer-use-preview" in result.error


def test_run_cua_session_marks_unsupported_provider_as_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module.asyncio, "sleep", _no_sleep)

    class _NoopBrowserEnvironment:
        def __init__(self, width: int, height: int, headless: bool) -> None:
            self.width = width
            self.height = height
            self.headless = headless

        async def __aenter__(self) -> _NoopBrowserEnvironment:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def navigate(self, _url: str) -> None:
            return None

        async def screenshot_b64(self) -> str:
            return "initial"

        async def save_screenshot(self, _path: str | Path) -> None:
            return None

    class _UnsupportedProvider:
        value = "custom"

        def __eq__(self, _other: object) -> bool:
            return False

        def __str__(self) -> str:
            return "custom"

    monkeypatch.setattr(session_module, "BrowserEnvironment", _NoopBrowserEnvironment)

    config = CUASessionConfig(task="task", save_screenshots=False)
    config.provider = _UnsupportedProvider()  # type: ignore[assignment]
    result = _run(session_module.run_cua_session(config))

    assert result.provider == "custom"
    assert "Unsupported CUA provider" in result.error
    assert result.success is False


def test_run_cua_session_sync_delegates_to_async_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    sentinel = CUASessionResult(task="sync", provider="openai", success=True)

    async def _fake_run_cua_session(
        config: CUASessionConfig,
        ledger: object | None = None,
        step_ref: str = "",
    ) -> CUASessionResult:
        captured["config"] = config
        captured["ledger"] = ledger
        captured["step_ref"] = step_ref
        return sentinel

    monkeypatch.setattr(session_module, "run_cua_session", _fake_run_cua_session)

    config = CUASessionConfig(task="sync-task")
    ledger = object()
    result = session_module.run_cua_session_sync(config, ledger=ledger, step_ref="S4")

    assert result is sentinel
    assert captured["config"] is config
    assert captured["ledger"] is ledger
    assert captured["step_ref"] == "S4"
