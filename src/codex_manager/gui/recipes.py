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
            "Strategic Product Maximization x2 -> Implementation x2 -> Testing x1 -> Bug Hunting x1"
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
    "todo_wishlist_autopilot": {
        "name": "To-Do/Wishlist Autopilot",
        "description": "Generate a practical to-do list, then implement until complete",
        "sequence": "Create To-Do/Wishlist -> Implement open items until done",
        "steps": [
            {
                "name": "01 Build To-Do/Wishlist",
                "job_type": "implementation",
                "loop_count": 1,
                "prompt_mode": "custom",
                "custom_prompt": (
                    "Create or refine `.codex_manager/owner/TODO_WISHLIST.md` as a high-signal "
                    "execution backlog for this exact repository. Use markdown checklist items "
                    "(`- [ ] ...`) grouped by priority. Preserve existing unchecked items unless "
                    "you are clarifying wording, removing duplicates, or reordering for better "
                    "execution flow (no large rewrites).\n\n"
                    "Always make the backlog more useful in this run:\n"
                    "1. Inspect the current repository state and infer missing opportunities "
                    "(features, UX, reliability, test coverage, performance, security, DX, docs).\n"
                    "2. Add net-new, repo-specific improvements when the list is thin, stale, or "
                    "too generic.\n"
                    "3. Ensure each open item is concrete and implementable in one focused pass.\n"
                    "4. Avoid vague goals; include clear completion intent for each item.\n\n"
                    "Quality bar:\n"
                    "- Prioritize by value-to-effort and execution order.\n"
                    "- Keep the final list actionable and non-redundant.\n"
                    "- If there are no useful open items, propose at least 5 high-value new ones.\n"
                    "- Do not invent work unrelated to improving this repository."
                ),
            },
            {
                "name": "02 Implement Open Wishlist Items",
                "job_type": "implementation",
                "loop_count": 20,
                "prompt_mode": "custom",
                "custom_prompt": (
                    "Read `.codex_manager/owner/TODO_WISHLIST.md` and implement the first "
                    "unchecked item (`- [ ] ...`) end-to-end. After implementation, mark it "
                    "done as `- [x] ...` and add a short completion note. Repeat this process "
                    "for each repetition. If there are no unchecked items left, output exactly "
                    "`[TERMINATE_STEP]` on its own line and do not make code changes."
                ),
            },
        ],
    },
    "feature_dream_autopilot": {
        "name": "Feature Dream Autopilot",
        "description": "Dream up high-value features, then implement all until complete",
        "sequence": "Dream Up Feature List -> Implement features until done",
        "steps": [
            {
                "name": "01 Dream Up Features",
                "job_type": "implementation",
                "loop_count": 1,
                "prompt_mode": "custom",
                "custom_prompt": (
                    "Create or refine `.codex_manager/owner/FEATURE_DREAMS.md` as a feature-only "
                    "execution list for this repository using markdown checkboxes (`- [ ]`). "
                    "Preserve existing unchecked feature items unless deduplicating, clarifying, "
                    "or reordering by value/effort.\n\n"
                    "In this step, dream up meaningful repo-specific features and upgrades:\n"
                    "1. Analyze current capabilities and identify missing user value.\n"
                    "2. Propose 5-15 concrete, implementable feature ideas with clear outcomes.\n"
                    "3. Prioritize by impact-to-effort and execution order.\n"
                    "4. Avoid vague items; each item must be actionable in one focused implementation pass.\n\n"
                    "If a useful feature list already exists, improve quality and fill strategic gaps "
                    "instead of rewriting everything."
                ),
            },
            {
                "name": "02 Implement Dreamed Features",
                "job_type": "implementation",
                "loop_count": 30,
                "prompt_mode": "custom",
                "custom_prompt": (
                    "Read `.codex_manager/owner/FEATURE_DREAMS.md` and implement the first unchecked "
                    "feature item (`- [ ] ...`) end-to-end. After implementation, mark it done as "
                    "`- [x] ...` and add a short completion note.\n\n"
                    "Repeat this process for each repetition. Continue until there are no unchecked "
                    "feature items left.\n\n"
                    "If there are no unchecked items remaining, output exactly `[TERMINATE_STEP]` "
                    "on its own line and do not make code changes."
                ),
            },
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
