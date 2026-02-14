"""Built-in Easy-mode recipe catalog shared by GUI and CLI surfaces."""

from __future__ import annotations

from copy import deepcopy

DEFAULT_RECIPE_ID = "autopilot_default"

_RECIPE_DATA: dict[str, dict[str, object]] = {
    "autopilot_default": {
        "name": "Autopilot Default",
        "description": "7-step full sweep with scripted prompts and commits",
        "sequence": (
            "Easy Improvements -> New Features -> GUI/UX Polish -> Code Quality -> "
            "Bug Hunt -> Performance -> Open Ended"
        ),
        "steps": [
            {
                "name": "01 Easy Improvements",
                "job_type": "feature_discovery",
                "loop_count": 1,
                "prompt_mode": "custom",
                "custom_prompt": (
                    "How can this project be improved? Dream up some improvements that you can "
                    "make, and then implement them."
                ),
            },
            {
                "name": "02 New Features",
                "job_type": "implementation",
                "loop_count": 1,
                "prompt_mode": "custom",
                "custom_prompt": (
                    "Analyze this codebase thoroughly. Identify the highest-impact features "
                    "that are missing or would significantly enhance the user experience. Pick "
                    "the top 3, rank them by effort-to-value ratio, and implement the best one "
                    "end-to-end. Commit with a clear message explaining what you added and why."
                ),
            },
            {
                "name": "03 GUI UX Polish",
                "job_type": "implementation",
                "loop_count": 1,
                "prompt_mode": "custom",
                "custom_prompt": (
                    "Review every aspect of this project's UI/UX with a critical eye: layout, "
                    "spacing, typography, color, responsiveness, accessibility, animations, "
                    "empty states, error states, and loading states. Identify the biggest "
                    "visual and interaction weaknesses. Fix the top issues, making the "
                    "interface feel polished and professional. Commit your changes with "
                    "before/after descriptions."
                ),
            },
            {
                "name": "04 Code Quality Architecture",
                "job_type": "refactoring",
                "loop_count": 1,
                "prompt_mode": "custom",
                "custom_prompt": (
                    "Audit this codebase for code smells, anti-patterns, duplicated logic, "
                    "poor naming, missing error handling, and architectural weaknesses. "
                    "Prioritize the changes that most improve maintainability and reliability. "
                    "Refactor the worst offenders. Commit with clear explanations of each "
                    "improvement."
                ),
            },
            {
                "name": "05 Bug Hunt Hardening",
                "job_type": "bug_hunting",
                "loop_count": 1,
                "prompt_mode": "custom",
                "custom_prompt": (
                    "Stress-test this codebase by reading every code path critically. Find edge "
                    "cases, unhandled errors, race conditions, security issues, and silent "
                    "failures. Fix the most impactful bugs you find. Add defensive checks where "
                    "needed. Commit each fix separately with a description of the bug and how "
                    "you verified the fix."
                ),
            },
            {
                "name": "06 Performance",
                "job_type": "performance",
                "loop_count": 1,
                "prompt_mode": "custom",
                "custom_prompt": (
                    "Profile this codebase for performance bottlenecks: unnecessary re-renders, "
                    "expensive computations, unoptimized queries, large bundle sizes, missing "
                    "caching, and slow operations. Identify the biggest wins and implement them. "
                    "Commit with measurable descriptions of what improved and why."
                ),
            },
            {
                "name": "07 Open Ended",
                "job_type": "strategic_product_maximization",
                "loop_count": 1,
                "prompt_mode": "custom",
                "custom_prompt": (
                    "You have full autonomy. Study this entire codebase, understand its purpose "
                    "and users, then make the single highest-leverage improvement you can find, "
                    "whether that's a new feature, a UX fix, a refactor, a bug fix, or a "
                    "performance win. Explain your reasoning for why this was the most valuable "
                    "change, then implement it fully. Commit with a detailed message."
                ),
            },
        ],
    },
    "strategic": {
        "name": "Strategic Product Max",
        "description": "Highest leverage product wins with validation",
        "sequence": (
            "Strategic Product Maximization x2 -> Implementation x2 -> Testing x1 -> "
            "Bug Hunting x1"
        ),
        "steps": [
            {"job_type": "strategic_product_maximization", "loop_count": 2},
            {"job_type": "implementation", "loop_count": 2},
            {"job_type": "testing", "loop_count": 1},
            {"job_type": "bug_hunting", "loop_count": 1},
        ],
    },
    "improve": {
        "name": "Improve Everything",
        "description": "Discovery, implementation, testing, bug hunting, refactoring",
        "sequence": (
            "Feature Discovery x2 -> Implementation x2 -> Testing x1 -> Bug Hunting x1 -> "
            "Refactoring x1"
        ),
        "steps": [
            {"job_type": "feature_discovery", "loop_count": 2},
            {"job_type": "implementation", "loop_count": 2},
            {"job_type": "testing", "loop_count": 1},
            {"job_type": "bug_hunting", "loop_count": 1},
            {"job_type": "refactoring", "loop_count": 1},
        ],
    },
    "fix": {
        "name": "Fix and Stabilize",
        "description": "Bug hunting, testing, refactoring",
        "sequence": "Bug Hunting x3 -> Testing x2 -> Refactoring x1",
        "steps": [
            {"job_type": "bug_hunting", "loop_count": 3},
            {"job_type": "testing", "loop_count": 2},
            {"job_type": "refactoring", "loop_count": 1},
        ],
    },
    "polish": {
        "name": "Polish and Document",
        "description": "Refactor, performance, docs, security audit",
        "sequence": "Refactoring x2 -> Performance x1 -> Documentation x2 -> Security Audit x1",
        "steps": [
            {"job_type": "refactoring", "loop_count": 2},
            {"job_type": "performance", "loop_count": 1},
            {"job_type": "documentation", "loop_count": 2},
            {"job_type": "security_audit", "loop_count": 1},
        ],
    },
    "test": {
        "name": "Test Coverage",
        "description": "Testing and bug hunting focus",
        "sequence": "Testing x3 -> Bug Hunting x2 -> Testing x2",
        "steps": [
            {"job_type": "testing", "loop_count": 3},
            {"job_type": "bug_hunting", "loop_count": 2},
            {"job_type": "testing", "loop_count": 2},
        ],
    },
}


def get_recipe(recipe_id: str) -> dict[str, object] | None:
    """Return a deep-copied recipe payload by id, or None when absent."""
    recipe = _RECIPE_DATA.get(recipe_id)
    if recipe is None:
        return None
    payload = deepcopy(recipe)
    payload["id"] = recipe_id
    return payload


def list_recipes() -> list[dict[str, object]]:
    """Return all recipes with full step details."""
    recipes: list[dict[str, object]] = []
    for recipe_id in _RECIPE_DATA:
        recipe = get_recipe(recipe_id)
        if recipe is not None:
            recipes.append(recipe)
    return recipes


def list_recipe_summaries() -> list[dict[str, object]]:
    """Return lightweight recipe summaries."""
    summaries: list[dict[str, object]] = []
    for recipe_id in _RECIPE_DATA:
        recipe = _RECIPE_DATA[recipe_id]
        steps = recipe.get("steps", [])
        summaries.append(
            {
                "id": recipe_id,
                "name": str(recipe.get("name", recipe_id)),
                "description": str(recipe.get("description", "")),
                "sequence": str(recipe.get("sequence", "")),
                "step_count": len(steps) if isinstance(steps, list) else 0,
            }
        )
    return summaries


def recipe_steps_map() -> dict[str, list[dict[str, object]]]:
    """Return recipes in the frontend shape: recipe-id -> step list."""
    payload: dict[str, list[dict[str, object]]] = {}
    for recipe_id, recipe in _RECIPE_DATA.items():
        steps = recipe.get("steps", [])
        payload[recipe_id] = deepcopy(steps) if isinstance(steps, list) else []
    return payload
