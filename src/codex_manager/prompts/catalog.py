"""Centralized prompt catalog — single source of truth for all prompts.

Loads prompts from ``templates.yaml`` (next to this module) and provides
typed accessors for every subsystem: pipeline phases, GUI presets, brain
system prompts, and scientist prompts.

Supports a user-override file at ``~/.codex_manager/prompt_overrides.yaml``
that is merged on top of the built-in defaults.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BUILTIN_YAML = Path(__file__).resolve().parent / "templates.yaml"
_USER_OVERRIDE = Path.home() / ".codex_manager" / "prompt_overrides.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning an empty dict on failure."""
    try:
        import yaml  # type: ignore[import-untyped]

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins)."""
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


class PromptCatalog:
    """Loads and serves prompts from the YAML catalog.

    Usage::

        catalog = PromptCatalog()
        prompt = catalog.pipeline("ideation")
        preset = catalog.preset("testing")
        system = catalog.brain("plan_step")
    """

    def __init__(self, extra_path: Path | None = None) -> None:
        self._data: dict[str, Any] = {}
        self._load(extra_path)

    def _load(self, extra_path: Path | None = None) -> None:
        """Load built-in templates, then merge user overrides."""
        self._data = _load_yaml(_BUILTIN_YAML)

        # Merge user overrides
        if _USER_OVERRIDE.exists():
            overrides = _load_yaml(_USER_OVERRIDE)
            if overrides:
                self._data = _deep_merge(self._data, overrides)
                logger.info("Loaded prompt overrides from %s", _USER_OVERRIDE)

        # Merge extra file (e.g., project-specific overrides)
        if extra_path and extra_path.exists():
            extra = _load_yaml(extra_path)
            if extra:
                self._data = _deep_merge(self._data, extra)
                logger.info("Loaded extra prompts from %s", extra_path)

    def reload(self) -> None:
        """Re-read all YAML files from disk."""
        self._load()

    # ── Pipeline prompts ─────────────────────────────────────────

    def pipeline(self, phase: str) -> str:
        """Return the prompt for a pipeline phase (ideation, implementation, etc.)."""
        phases = self._data.get("pipeline", {})
        entry = phases.get(phase, {})
        return (entry.get("prompt") or "").strip()

    def pipeline_meta(self, phase: str) -> dict[str, str]:
        """Return metadata (name, description) for a pipeline phase."""
        phases = self._data.get("pipeline", {})
        entry = phases.get(phase, {})
        return {
            "name": entry.get("name", phase),
            "description": entry.get("description", ""),
        }

    def list_pipeline_phases(self) -> list[str]:
        """Return all defined pipeline phase keys."""
        return list(self._data.get("pipeline", {}).keys())

    # ── Preset prompts (GUI job types) ───────────────────────────

    def preset(self, key: str, *, ai_decides: bool = False) -> str:
        """Return the prompt for a GUI preset job type."""
        presets = self._data.get("presets", {})
        entry = presets.get(key, {})
        field = "ai_prompt" if ai_decides else "prompt"
        return (entry.get(field) or entry.get("prompt") or "").strip()

    def preset_meta(self, key: str) -> dict[str, Any]:
        """Return full metadata for a preset (name, icon, description, etc.)."""
        presets = self._data.get("presets", {})
        entry = presets.get(key, {})
        return {
            "key": key,
            "name": entry.get("name", key),
            "icon": entry.get("icon", ""),
            "description": entry.get("description", ""),
            "on_failure": entry.get("on_failure", "skip"),
        }

    def list_presets(self) -> list[dict[str, Any]]:
        """Return a summary list of all presets (for the GUI)."""
        presets = self._data.get("presets", {})
        return [
            {
                "key": k,
                "name": v.get("name", k),
                "icon": v.get("icon", ""),
                "description": v.get("description", ""),
            }
            for k, v in presets.items()
        ]

    def preset_detail(self, key: str) -> dict[str, Any] | None:
        """Return full preset data (for API responses)."""
        presets = self._data.get("presets", {})
        entry = presets.get(key)
        if not entry:
            return None
        return {
            "name": entry.get("name", key),
            "icon": entry.get("icon", ""),
            "description": entry.get("description", ""),
            "prompt": (entry.get("prompt") or "").strip(),
            "ai_prompt": (entry.get("ai_prompt") or "").strip(),
            "on_failure": entry.get("on_failure", "skip"),
        }

    # ── Brain prompts ────────────────────────────────────────────

    def brain(self, key: str) -> str:
        """Return a brain system prompt by key (plan_step, evaluate_step, etc.)."""
        brains = self._data.get("brain", {})
        entry = brains.get(key, {})
        return (entry.get("system") or "").strip()

    # ── Scientist prompts ────────────────────────────────────────

    def scientist(self, key: str) -> str:
        """Return a scientist prompt by key (theorize, experiment, skeptic, analyze)."""
        sci = self._data.get("scientist", {})
        entry = sci.get(key, {})
        return (entry.get("prompt") or "").strip()

    def scientist_meta(self, key: str) -> dict[str, str]:
        """Return metadata for a scientist phase."""
        sci = self._data.get("scientist", {})
        entry = sci.get(key, {})
        return {
            "name": entry.get("name", key),
            "description": entry.get("description", ""),
        }

    # ── Optimizer prompts ────────────────────────────────────────

    def optimizer(self, key: str) -> str:
        """Return an optimizer meta-prompt."""
        opt = self._data.get("optimizer", {})
        entry = opt.get(key, {})
        return (entry.get("system") or "").strip()

    # ── Raw access ───────────────────────────────────────────────

    @property
    def raw(self) -> dict[str, Any]:
        """Direct access to the full parsed data."""
        return self._data

    def all_prompts(self) -> list[dict[str, str]]:
        """Return a flat list of all prompts with their paths and content.

        Useful for the optimizer to iterate over every prompt.
        """
        results: list[dict[str, str]] = []

        for phase, entry in self._data.get("pipeline", {}).items():
            if isinstance(entry, dict) and entry.get("prompt"):
                results.append({
                    "path": f"pipeline.{phase}",
                    "name": entry.get("name", phase),
                    "content": entry["prompt"].strip(),
                })

        for key, entry in self._data.get("presets", {}).items():
            if isinstance(entry, dict):
                if entry.get("prompt"):
                    results.append({
                        "path": f"presets.{key}.prompt",
                        "name": f"{entry.get('name', key)} (preset)",
                        "content": entry["prompt"].strip(),
                    })
                if entry.get("ai_prompt"):
                    results.append({
                        "path": f"presets.{key}.ai_prompt",
                        "name": f"{entry.get('name', key)} (AI decides)",
                        "content": entry["ai_prompt"].strip(),
                    })

        for key, entry in self._data.get("brain", {}).items():
            if isinstance(entry, dict) and entry.get("system"):
                results.append({
                    "path": f"brain.{key}",
                    "name": f"Brain: {key}",
                    "content": entry["system"].strip(),
                })

        for key, entry in self._data.get("scientist", {}).items():
            if isinstance(entry, dict) and entry.get("prompt"):
                results.append({
                    "path": f"scientist.{key}",
                    "name": f"Scientist: {entry.get('name', key)}",
                    "content": entry["prompt"].strip(),
                })

        return results

    # ── Saving (for the optimizer) ───────────────────────────────

    def save(self, path: Path | None = None) -> Path:
        """Write the current catalog back to YAML.

        Writes to the user override file by default, preserving the
        built-in templates.yaml unchanged.
        """
        target = path or _USER_OVERRIDE
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            import yaml  # type: ignore[import-untyped]

            target.write_text(
                yaml.dump(self._data, default_flow_style=False, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
        except ImportError:
            import json

            target.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info("Saved prompt catalog to %s", target)
        return target


# Module-level singleton for convenience
_default_catalog: PromptCatalog | None = None


def get_catalog() -> PromptCatalog:
    """Return the module-level singleton catalog (lazy-loaded)."""
    global _default_catalog
    if _default_catalog is None:
        _default_catalog = PromptCatalog()
    return _default_catalog
