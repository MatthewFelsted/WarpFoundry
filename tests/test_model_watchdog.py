"""Tests for provider/dependency model watchdog snapshots."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from codex_manager.monitoring.model_watchdog import (
    AVAILABLE_MODEL_PROVIDERS,
    DEFAULT_HTTP_USER_AGENT,
    ModelCatalogWatchdog,
)


def _default_fetchers(state: dict[str, list[str]]):
    return {
        provider: (lambda _timeout, provider=provider: list(state.get(provider, [])))
        for provider in AVAILABLE_MODEL_PROVIDERS
    }


def test_watchdog_run_once_writes_snapshot_and_diff(tmp_path: Path):
    state = {
        "openai": ["gpt-alpha", "gpt-beta"],
        "ollama": ["gemma3:27b"],
    }
    watchdog = ModelCatalogWatchdog(
        root_dir=tmp_path / "watchdog",
        provider_fetchers=_default_fetchers(state),
        dependency_packages=(),
        default_interval_hours=24,
    )
    watchdog.update_config({"providers": ["openai", "ollama"], "history_limit": 50})

    first = watchdog.run_once(force=True, reason="manual")
    assert first["status"] == "ok"
    assert Path(first["snapshot_path"]).is_file()
    assert Path(first["history_path"]).is_file()

    state["openai"] = ["gpt-beta", "gpt-gamma"]
    second = watchdog.run_once(force=True, reason="manual")
    assert second["status"] == "ok"
    assert second["catalog_changed"] is True

    openai_diff = second["diff"]["providers"]["openai"]
    assert openai_diff["added"] == ["gpt-gamma"]
    assert openai_diff["removed"] == ["gpt-alpha"]

    alerts = watchdog.latest_alerts()
    assert alerts["has_alerts"] is True
    warn_alerts = [row for row in alerts["alerts"] if row.get("severity") == "warn"]
    assert warn_alerts
    playbook = warn_alerts[0].get("playbook") or {}
    assert playbook.get("id") == "openai_model_remediation"
    migrations = playbook.get("migrations") or []
    assert migrations
    assert migrations[0]["from_model"] == "gpt-alpha"
    assert migrations[0]["to_model"] == "gpt-gamma"
    assert alerts.get("playbooks")

    latest = json.loads(watchdog.latest_snapshot_path.read_text(encoding="utf-8"))
    assert latest["providers"]["openai"]["models"] == ["gpt-beta", "gpt-gamma"]

    history_lines = watchdog.history_path.read_text(encoding="utf-8").splitlines()
    assert len(history_lines) == 2


def test_watchdog_run_once_skips_when_not_due(tmp_path: Path):
    state = {"openai": ["gpt-alpha"]}
    watchdog = ModelCatalogWatchdog(
        root_dir=tmp_path / "watchdog",
        provider_fetchers=_default_fetchers(state),
        dependency_packages=(),
        default_interval_hours=24,
    )
    watchdog.update_config({"providers": ["openai"]})

    first = watchdog.run_once(force=True, reason="manual")
    assert first["status"] == "ok"

    skipped = watchdog.run_once(force=False, reason="scheduled")
    assert skipped["status"] == "skipped_not_due"
    assert skipped["reason"] == "scheduled"


def test_watchdog_update_config_normalizes_values(tmp_path: Path):
    state = {}
    watchdog = ModelCatalogWatchdog(
        root_dir=tmp_path / "watchdog",
        provider_fetchers=_default_fetchers(state),
        dependency_packages=(),
    )

    config = watchdog.update_config(
        {
            "enabled": "yes",
            "interval_hours": 0,
            "providers": ["openai", "not-real"],
            "request_timeout_seconds": 999,
            "history_limit": 1,
            "auto_run_on_start": 0,
        }
    )

    assert config["enabled"] is True
    assert config["interval_hours"] == 1
    assert config["providers"] == ["openai"]
    assert config["request_timeout_seconds"] == 60
    assert config["history_limit"] == 10
    assert config["auto_run_on_start"] is False


def test_watchdog_stop_keeps_thread_reference_until_joined(tmp_path: Path) -> None:
    watchdog = ModelCatalogWatchdog(
        root_dir=tmp_path / "watchdog",
        provider_fetchers={},
        dependency_packages=(),
    )

    class _StickyThread(threading.Thread):
        def __init__(self) -> None:
            super().__init__(daemon=True)
            self.alive = True
            self.join_calls = 0

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float | None = None) -> None:
            self.join_calls += 1

    sticky = _StickyThread()
    watchdog._thread = sticky

    watchdog.stop()

    assert sticky.join_calls == 1
    assert watchdog._thread is sticky

    sticky.alive = False
    watchdog.stop()
    assert watchdog._thread is None


def test_fetch_openai_models_uses_non_placeholder_fallback_key(
    monkeypatch, tmp_path: Path
) -> None:
    watchdog = ModelCatalogWatchdog(
        root_dir=tmp_path / "watchdog",
        provider_fetchers={},
        dependency_packages=(),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-your-key-here")
    monkeypatch.setenv("CODEX_API_KEY", "sk-proj-real-secret")

    captured: dict[str, str] = {}

    def _fake_http_json(url: str, *, headers: dict[str, str], timeout_s: int):
        captured["auth"] = str(headers.get("Authorization") or "")
        return {"data": [{"id": "gpt-test-model"}]}

    monkeypatch.setattr(watchdog, "_http_json", _fake_http_json)

    models = watchdog._fetch_openai_models(timeout_s=10)

    assert models == ["gpt-test-model"]
    assert captured["auth"] == "Bearer sk-proj-real-secret"


def test_fetch_xai_models_supports_grok_api_key_alias(monkeypatch, tmp_path: Path) -> None:
    watchdog = ModelCatalogWatchdog(
        root_dir=tmp_path / "watchdog",
        provider_fetchers={},
        dependency_packages=(),
    )
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("GROK_API_KEY", "xai-real-secret")

    captured: dict[str, str] = {}

    def _fake_http_json(url: str, *, headers: dict[str, str], timeout_s: int):
        captured["auth"] = str(headers.get("Authorization") or "")
        return {"data": [{"id": "grok-test-model"}]}

    monkeypatch.setattr(watchdog, "_http_json", _fake_http_json)

    models = watchdog._fetch_xai_models(timeout_s=10)

    assert models == ["grok-test-model"]
    assert captured["auth"] == "Bearer xai-real-secret"


def test_http_json_sets_default_user_agent_and_accept(monkeypatch, tmp_path: Path) -> None:
    watchdog = ModelCatalogWatchdog(
        root_dir=tmp_path / "watchdog",
        provider_fetchers={},
        dependency_packages=(),
    )
    captured: dict[str, str] = {}

    class _FakeHeaders:
        @staticmethod
        def get_content_charset() -> str:
            return "utf-8"

    class _FakeResponse:
        headers = _FakeHeaders()

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        @staticmethod
        def read() -> bytes:
            return b'{"data":[]}'

    def _fake_urlopen(request, timeout):
        header_map = {str(k).lower(): str(v) for k, v in request.header_items()}
        captured["user-agent"] = header_map.get("user-agent", "")
        captured["accept"] = header_map.get("accept", "")
        return _FakeResponse()

    monkeypatch.setattr("codex_manager.monitoring.model_watchdog.urlopen", _fake_urlopen)

    payload = watchdog._http_json("https://example.com/v1/models", headers={}, timeout_s=5)

    assert payload == {"data": []}
    assert captured["user-agent"] == DEFAULT_HTTP_USER_AGENT
    assert captured["accept"] == "application/json"


def test_watchdog_fetch_failure_does_not_report_false_model_removals(tmp_path: Path) -> None:
    state: dict[str, object] = {"openai": ["gpt-alpha", "gpt-beta"]}

    def _fetch_openai(_timeout: int) -> list[str]:
        value = state["openai"]
        if isinstance(value, Exception):
            raise value
        if not isinstance(value, list):
            raise TypeError("openai state must be a list")
        return [str(item) for item in value]

    watchdog = ModelCatalogWatchdog(
        root_dir=tmp_path / "watchdog",
        provider_fetchers={"openai": _fetch_openai},
        dependency_packages=(),
        default_interval_hours=24,
    )
    watchdog.update_config({"providers": ["openai"], "history_limit": 50})

    first = watchdog.run_once(force=True, reason="manual")
    assert first["status"] == "ok"

    state["openai"] = RuntimeError("network timeout while refreshing catalog")
    second = watchdog.run_once(force=True, reason="manual")

    assert second["status"] == "error"
    assert second["catalog_changed"] is False
    diff = second["diff"]["providers"]["openai"]
    assert diff["removed_count"] == 0
    assert diff["added_count"] == 0
    assert diff["compared_models"] is False
    assert second["failed_providers"] == ["openai"]

    alerts = watchdog.latest_alerts()
    assert alerts["has_alerts"] is True
    assert any(
        "catalog refresh failed" in str(alert.get("title") or "").lower()
        for alert in alerts["alerts"]
    )


def test_watchdog_partial_provider_failure_reports_degraded_status(tmp_path: Path) -> None:
    def _fetch_openai(_timeout: int) -> list[str]:
        return ["gpt-alpha"]

    def _fetch_xai(_timeout: int) -> list[str]:
        raise RuntimeError("xai endpoint temporarily unavailable")

    watchdog = ModelCatalogWatchdog(
        root_dir=tmp_path / "watchdog",
        provider_fetchers={"openai": _fetch_openai, "xai": _fetch_xai},
        dependency_packages=(),
        default_interval_hours=24,
    )
    watchdog.update_config({"providers": ["openai", "xai"], "history_limit": 50})

    result = watchdog.run_once(force=True, reason="manual")

    assert result["status"] == "degraded"
    assert result["successful_providers"] == ["openai"]
    assert result["failed_providers"] == ["xai"]
    assert "xai" in result["provider_errors"]

    status = watchdog.status()
    assert status["state"]["last_status"] == "degraded"
    assert "xai" in str(status["state"]["last_error"] or "")
