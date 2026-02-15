"""Tests for provider/dependency model watchdog snapshots."""

from __future__ import annotations

import json
from pathlib import Path

from codex_manager.monitoring.model_watchdog import AVAILABLE_MODEL_PROVIDERS, ModelCatalogWatchdog


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
