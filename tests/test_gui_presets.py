"""Tests for GUI preset loading, cache behavior, and prompt accessors."""

from __future__ import annotations

from typing import ClassVar

import codex_manager.gui.presets as presets_module


class _CatalogStub:
    raw: ClassVar[dict[str, dict[str, dict[str, object]]]] = {
        "presets": {"custom": {}, "secondary": {}}
    }

    def preset_detail(self, key: str):
        if key == "custom":
            return {
                "name": "Custom",
                "icon": "C",
                "description": "Custom preset",
                "prompt": "run custom prompt",
                "ai_prompt": "run ai custom prompt",
                "on_failure": "skip",
            }
        if key == "secondary":
            return {
                "name": "Secondary",
                "icon": "",
                "description": "",
                "prompt": "secondary prompt",
                "ai_prompt": "",
                "on_failure": "skip",
            }
        return None


def test_load_presets_from_catalog(monkeypatch) -> None:
    monkeypatch.setattr("codex_manager.prompts.catalog.get_catalog", lambda: _CatalogStub())
    loaded = presets_module._load_presets()
    assert set(loaded.keys()) == {"custom", "secondary"}
    assert loaded["custom"]["prompt"] == "run custom prompt"


def test_load_presets_falls_back_when_catalog_load_fails(monkeypatch) -> None:
    def raise_error():
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr("codex_manager.prompts.catalog.get_catalog", raise_error)
    loaded = presets_module._load_presets()
    assert "testing" in loaded
    assert loaded["testing"]["name"] == "Testing"
    assert "strategic_product_maximization" in loaded
    assert (
        loaded["strategic_product_maximization"]["name"]
        == "Strategic Product Maximization"
    )
    assert loaded is not presets_module._FALLBACK_PRESETS


def test_getters_use_cache_and_return_expected_fields(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_load():
        calls["count"] += 1
        return {
            "custom": {
                "name": "Custom",
                "icon": "C",
                "description": "desc",
                "prompt": "direct prompt",
                "ai_prompt": "ai prompt",
            }
        }

    monkeypatch.setattr(presets_module, "_presets_cache", None)
    monkeypatch.setattr(presets_module, "_load_presets", fake_load)

    assert presets_module.get_preset("custom")["name"] == "Custom"
    assert presets_module.get_prompt("custom") == "direct prompt"
    assert presets_module.get_prompt("custom", ai_decides=True) == "ai prompt"
    assert presets_module.get_prompt("missing") == ""
    listed = presets_module.list_presets()
    assert listed == [{"key": "custom", "name": "Custom", "icon": "C", "description": "desc"}]
    assert calls["count"] == 1
