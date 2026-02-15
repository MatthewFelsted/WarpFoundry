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


def test_todo_wishlist_autopilot_recipe_exists_and_uses_terminate_signal() -> None:
    recipe = recipes_module.get_recipe("todo_wishlist_autopilot")
    assert recipe is not None

    steps = recipe.get("steps", [])
    assert isinstance(steps, list)
    assert len(steps) == 2
    assert steps[0]["name"] == "01 Build To-Do/Wishlist"
    assert steps[1]["loop_count"] == 20
    assert "Always make the backlog more useful" in str(steps[0]["custom_prompt"])
    assert "Do not invent work unrelated" in str(steps[0]["custom_prompt"])
    assert "[TERMINATE_STEP]" in str(steps[1]["custom_prompt"])


def test_feature_dream_autopilot_recipe_exists_and_loops_until_done() -> None:
    recipe = recipes_module.get_recipe("feature_dream_autopilot")
    assert recipe is not None

    steps = recipe.get("steps", [])
    assert isinstance(steps, list)
    assert len(steps) == 2
    assert steps[0]["name"] == "01 Dream Up Features"
    assert steps[0]["job_type"] == "feature_discovery"
    assert ".codex_manager/owner/FEATURE_DREAMS.md" in str(steps[0]["custom_prompt"])
    assert "Write/update `.codex_manager/owner/FEATURE_DREAMS.md` directly" in str(
        steps[0]["custom_prompt"]
    )
    assert steps[1]["name"] == "02 Implement Dreamed Features"
    assert steps[1]["loop_count"] == 30
    assert "If `.codex_manager/owner/FEATURE_DREAMS.md` does not exist" in str(
        steps[1]["custom_prompt"]
    )
    assert "[TERMINATE_STEP]" in str(steps[1]["custom_prompt"])
