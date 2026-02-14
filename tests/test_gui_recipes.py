"""Tests for GUI built-in recipe catalog helpers."""

from __future__ import annotations

from codex_manager.gui import recipes as recipes_module


def test_autopilot_default_contains_new_features_step_prompt() -> None:
    recipe = recipes_module.get_recipe("autopilot_default")
    assert recipe is not None

    steps = recipe.get("steps", [])
    assert isinstance(steps, list)
    new_features = next((s for s in steps if s.get("name") == "02 New Features"), None)
    assert new_features is not None
    prompt = str(new_features.get("custom_prompt", ""))
    assert "Identify the highest-impact features" in prompt
    assert "effort-to-value ratio" in prompt


def test_get_recipe_returns_deep_copy() -> None:
    first = recipes_module.get_recipe("strategic")
    second = recipes_module.get_recipe("strategic")
    assert first is not None
    assert second is not None
    assert first is not second

    first_steps = first.get("steps", [])
    second_steps = second.get("steps", [])
    assert isinstance(first_steps, list)
    assert isinstance(second_steps, list)

    first_steps[0]["job_type"] = "custom"
    assert second_steps[0]["job_type"] == "strategic_product_maximization"


def test_recipe_summaries_and_steps_map_expose_known_default() -> None:
    summaries = recipes_module.list_recipe_summaries()
    by_id = {entry["id"]: entry for entry in summaries}
    assert recipes_module.DEFAULT_RECIPE_ID in by_id
    assert by_id["autopilot_default"]["step_count"] == 7

    mapped = recipes_module.recipe_steps_map()
    assert recipes_module.DEFAULT_RECIPE_ID in mapped
    assert len(mapped["autopilot_default"]) == 7
