"""Tests for prompt catalog loading, accessors, and persistence."""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import codex_manager.prompts.catalog as catalog_module
from codex_manager.prompts.catalog import PromptCatalog, _deep_merge, _load_yaml


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_load_yaml_success_and_failure(caplog, tmp_path: Path) -> None:
    good = _write(tmp_path / "good.yaml", "a: 1\nb:\n  - two\n")
    assert _load_yaml(good) == {"a": 1, "b": ["two"]}

    bad = _write(tmp_path / "bad.yaml", "a: [1, 2\n")
    with caplog.at_level("WARNING"):
        assert _load_yaml(bad) == {}
    assert "Failed to load" in caplog.text


def test_deep_merge_recursively_overrides_values() -> None:
    base = {"a": 1, "nested": {"x": 1, "y": 2}, "keep": {"k": "v"}}
    override = {"nested": {"y": 99, "z": 3}, "new": 7, "keep": 9}
    merged = _deep_merge(base, override)
    assert merged == {
        "a": 1,
        "nested": {"x": 1, "y": 99, "z": 3},
        "keep": 9,
        "new": 7,
    }


def test_catalog_accessors_all_prompts_and_overrides(monkeypatch, tmp_path: Path) -> None:
    builtin = _write(
        tmp_path / "builtin.yaml",
        """
pipeline:
  ideation:
    name: Ideation
    description: Generate ideas
    prompt: " builtin ideation "
presets:
  testing:
    name: Testing
    icon: T
    description: Test stuff
    prompt: " builtin prompt "
    ai_prompt: " builtin ai prompt "
    on_failure: retry
  docs:
    name: Docs
    prompt: " docs prompt "
  simple:
    prompt: " simple prompt "
brain:
  plan_step:
    system: " plan system "
scientist:
  theorize:
    name: Theorize
    description: Make hypotheses
    prompt: " theory prompt "
optimizer:
  evaluate_prompt:
    system: " eval system "
""".strip(),
    )
    user = _write(
        tmp_path / "user.yaml",
        """
pipeline:
  ideation:
    prompt: " user ideation "
presets:
  testing:
    prompt: " user testing prompt "
  docs:
    description: Write docs
brain:
  plan_step:
    system: " user plan system "
""".strip(),
    )
    extra = _write(
        tmp_path / "extra.yaml",
        """
presets:
  testing:
    ai_prompt: " extra ai prompt "
  docs:
    ai_prompt: " docs ai prompt "
""".strip(),
    )

    monkeypatch.setattr(catalog_module, "_BUILTIN_YAML", builtin)
    monkeypatch.setattr(catalog_module, "_USER_OVERRIDE", user)
    catalog = PromptCatalog(extra_path=extra)

    assert catalog.pipeline("ideation") == "user ideation"
    assert catalog.pipeline("missing") == ""
    assert catalog.pipeline_meta("ideation") == {
        "name": "Ideation",
        "description": "Generate ideas",
    }
    assert catalog.list_pipeline_phases() == ["ideation"]

    assert catalog.preset("testing") == "user testing prompt"
    assert catalog.preset("testing", ai_decides=True) == "extra ai prompt"
    assert catalog.preset("simple", ai_decides=True) == "simple prompt"
    assert catalog.preset("missing") == ""

    assert catalog.preset_meta("docs") == {
        "key": "docs",
        "name": "Docs",
        "icon": "",
        "description": "Write docs",
        "on_failure": "skip",
    }
    assert {entry["key"] for entry in catalog.list_presets()} == {"testing", "docs", "simple"}

    docs = catalog.preset_detail("docs")
    assert docs == {
        "name": "Docs",
        "icon": "",
        "description": "Write docs",
        "prompt": "docs prompt",
        "ai_prompt": "docs ai prompt",
        "on_failure": "skip",
    }
    assert catalog.preset_detail("missing") is None

    assert catalog.brain("plan_step") == "user plan system"
    assert catalog.scientist("theorize") == "theory prompt"
    assert catalog.scientist_meta("theorize") == {
        "name": "Theorize",
        "description": "Make hypotheses",
    }
    assert catalog.optimizer("evaluate_prompt") == "eval system"
    assert isinstance(catalog.raw, dict)

    paths = {entry["path"] for entry in catalog.all_prompts()}
    assert "pipeline.ideation" in paths
    assert "presets.testing.prompt" in paths
    assert "presets.testing.ai_prompt" in paths
    assert "presets.docs.prompt" in paths
    assert "presets.docs.ai_prompt" in paths
    assert "brain.plan_step" in paths
    assert "scientist.theorize" in paths


def test_catalog_reload_reloads_from_disk(monkeypatch, tmp_path: Path) -> None:
    builtin = _write(
        tmp_path / "builtin.yaml",
        """
pipeline:
  ideation:
    prompt: old
""".strip(),
    )
    user = tmp_path / "user.yaml"
    monkeypatch.setattr(catalog_module, "_BUILTIN_YAML", builtin)
    monkeypatch.setattr(catalog_module, "_USER_OVERRIDE", user)

    catalog = PromptCatalog()
    assert catalog.pipeline("ideation") == "old"

    _write(
        builtin,
        """
pipeline:
  ideation:
    prompt: new
""".strip(),
    )
    catalog.reload()
    assert catalog.pipeline("ideation") == "new"


def test_catalog_save_yaml_and_json_fallback(monkeypatch, tmp_path: Path) -> None:
    builtin = _write(
        tmp_path / "builtin.yaml",
        """
pipeline:
  ideation:
    prompt: "hello"
""".strip(),
    )
    user = tmp_path / "user.yaml"
    monkeypatch.setattr(catalog_module, "_BUILTIN_YAML", builtin)
    monkeypatch.setattr(catalog_module, "_USER_OVERRIDE", user)
    catalog = PromptCatalog()

    yaml_target = tmp_path / "saved.yaml"
    assert catalog.save(yaml_target) == yaml_target
    assert "pipeline:" in yaml_target.read_text(encoding="utf-8")

    real_import = builtins.__import__

    def fake_import(name: str, globals_=None, locals_=None, fromlist=(), level: int = 0):
        if name == "yaml":
            raise ImportError("yaml intentionally unavailable")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    json_target = tmp_path / "saved.json"
    assert catalog.save(json_target) == json_target
    saved = json.loads(json_target.read_text(encoding="utf-8"))
    assert saved["pipeline"]["ideation"]["prompt"] == "hello"


def test_get_catalog_returns_singleton(monkeypatch, tmp_path: Path) -> None:
    builtin = _write(
        tmp_path / "builtin.yaml",
        """
pipeline:
  ideation:
    prompt: singleton
""".strip(),
    )
    user = tmp_path / "missing.yaml"
    monkeypatch.setattr(catalog_module, "_BUILTIN_YAML", builtin)
    monkeypatch.setattr(catalog_module, "_USER_OVERRIDE", user)
    monkeypatch.setattr(catalog_module, "_default_catalog", None)

    first = catalog_module.get_catalog()
    second = catalog_module.get_catalog()
    assert first is second
    assert first.pipeline("ideation") == "singleton"


def test_builtin_catalog_includes_strategic_product_maximization(
    monkeypatch, tmp_path: Path
) -> None:
    # Isolate from user-local prompt overrides for deterministic assertions.
    monkeypatch.setattr(catalog_module, "_USER_OVERRIDE", tmp_path / "missing.yaml")
    catalog = PromptCatalog()

    detail = catalog.preset_detail("strategic_product_maximization")
    assert detail is not None
    assert detail["name"] == "Strategic Product Maximization"
    assert "STRATEGIC PRODUCT MAXIMIZATION MODE" in detail["prompt"]
