"""Tests for GUI built-in recipe catalog helpers."""

from __future__ import annotations

from pathlib import Path

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


def test_custom_recipe_save_list_and_delete_roundtrip(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    recipe, created, path = recipes_module.save_custom_recipe(
        repo,
        {
            "id": "my_custom_recipe",
            "name": "My Custom Recipe",
            "description": "repo-specific",
            "steps": [
                {"job_type": "implementation"},
                {"job_type": "testing", "on_failure": "abort"},
            ],
        },
    )
    assert created is True
    assert path.is_file()
    assert recipe["id"] == "my_custom_recipe"
    assert recipe["source"] == "custom"

    detail = recipes_module.get_recipe("my_custom_recipe", repo=repo)
    assert detail is not None
    assert detail["source"] == "custom"
    assert len(detail["steps"]) == 2

    summaries = recipes_module.list_recipe_summaries(repo=repo)
    custom_summary = next((row for row in summaries if row["id"] == "my_custom_recipe"), None)
    assert custom_summary is not None
    assert custom_summary["source"] == "custom"
    assert custom_summary["step_count"] == 2

    steps_map = recipes_module.recipe_steps_map(repo=repo)
    assert "my_custom_recipe" in steps_map

    deleted, delete_path = recipes_module.delete_custom_recipe(repo, "my_custom_recipe")
    assert deleted is True
    assert delete_path == path
    assert recipes_module.get_recipe("my_custom_recipe", repo=repo) is None


def test_custom_recipe_import_and_export_bundle(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    summary, path = recipes_module.import_custom_recipes(
        repo,
        {
            "recipes": [
                {
                    "id": "alpha_recipe",
                    "name": "Alpha",
                    "steps": [{"job_type": "implementation"}],
                },
                {
                    "id": "beta_recipe",
                    "name": "Beta",
                    "steps": [{"job_type": "testing"}],
                },
            ]
        },
    )
    assert summary["imported"] == 2
    assert summary["created"] == 2
    assert path.is_file()

    exported_all = recipes_module.export_custom_recipes(repo)
    rows = exported_all.get("recipes", [])
    assert isinstance(rows, list)
    assert len(rows) == 2

    exported_one = recipes_module.export_custom_recipes(repo, recipe_id="alpha_recipe")
    one_rows = exported_one.get("recipes", [])
    assert isinstance(one_rows, list)
    assert len(one_rows) == 1
    assert one_rows[0]["id"] == "alpha_recipe"


def test_custom_recipe_rejects_builtin_recipe_id(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    try:
        recipes_module.save_custom_recipe(
            repo,
            {
                "id": "autopilot_default",
                "name": "Invalid",
                "steps": [{"job_type": "implementation"}],
            },
        )
    except ValueError as exc:
        assert "reserved" in str(exc)
    else:
        raise AssertionError("Expected ValueError for built-in recipe id collision")
