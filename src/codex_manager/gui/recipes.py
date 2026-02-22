"""Built-in and per-repository Easy-mode recipe catalog helpers."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

DEFAULT_RECIPE_ID = "autopilot_default"
_CUSTOM_RECIPE_FILENAME = "CUSTOM_RECIPES.json"
_RECIPE_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62})$")
_ALLOWED_PROMPT_MODES = {"preset", "ai_decides", "custom"}
_ALLOWED_ON_FAILURE = {"skip", "retry", "abort"}
_ALLOWED_AGENTS = {"codex", "claude_code", "auto"}
_MAX_RECIPE_STEPS = 50
_MAX_NAME_CHARS = 120
_MAX_DESCRIPTION_CHARS = 280
_MAX_SEQUENCE_CHARS = 420


def _safe_int(value: object, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _coerce_repo_path(repo: Path | str | None) -> Path | None:
    if repo is None:
        return None
    raw = str(repo).strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def custom_recipes_path(repo: Path | str) -> Path:
    """Return the custom-recipe JSON path for a repository."""
    root = _coerce_repo_path(repo)
    if root is None:
        raise ValueError("repo path is required")
    return root / ".codex_manager" / "owner" / _CUSTOM_RECIPE_FILENAME


def _slugify_recipe_id(raw_value: object) -> str:
    raw = str(raw_value or "").strip().lower()
    if not raw:
        return ""
    slug = re.sub(r"[^a-z0-9_-]+", "_", raw)
    slug = re.sub(r"_+", "_", slug).strip("_-")
    return slug[:63]


def _recipe_summary(recipe: dict[str, object]) -> dict[str, object]:
    steps = recipe.get("steps", [])
    return {
        "id": str(recipe.get("id") or "").strip(),
        "name": str(recipe.get("name") or "").strip(),
        "description": str(recipe.get("description") or "").strip(),
        "sequence": str(recipe.get("sequence") or "").strip(),
        "step_count": len(steps) if isinstance(steps, list) else 0,
        "source": str(recipe.get("source") or "builtin").strip() or "builtin",
    }


def _normalize_recipe_step(row: dict[str, object], *, index: int) -> dict[str, object]:
    name = str(row.get("name") or "").strip()
    job_type = str(row.get("job_type") or "").strip()
    if not job_type:
        raise ValueError(f"step #{index} is missing job_type")
    if not name:
        name = job_type.replace("_", " ").strip().title() or f"Step {index}"

    prompt_mode_raw = str(row.get("prompt_mode") or "").strip().lower()
    custom_prompt = str(row.get("custom_prompt") or "")
    prompt_mode = prompt_mode_raw or ("custom" if custom_prompt.strip() else "preset")
    if prompt_mode not in _ALLOWED_PROMPT_MODES:
        allowed = ", ".join(sorted(_ALLOWED_PROMPT_MODES))
        raise ValueError(f"step #{index} has invalid prompt_mode '{prompt_mode}' (expected: {allowed})")

    on_failure = str(row.get("on_failure") or "skip").strip().lower() or "skip"
    if on_failure not in _ALLOWED_ON_FAILURE:
        allowed = ", ".join(sorted(_ALLOWED_ON_FAILURE))
        raise ValueError(f"step #{index} has invalid on_failure '{on_failure}' (expected: {allowed})")

    agent = str(row.get("agent") or "auto").strip().lower() or "auto"
    if agent not in _ALLOWED_AGENTS:
        allowed = ", ".join(sorted(_ALLOWED_AGENTS))
        raise ValueError(f"step #{index} has invalid agent '{agent}' (expected: {allowed})")

    normalized: dict[str, object] = {
        "name": name[: _MAX_NAME_CHARS],
        "job_type": job_type,
        "prompt_mode": prompt_mode,
        "custom_prompt": custom_prompt,
        "on_failure": on_failure,
        "max_retries": _safe_int(row.get("max_retries"), 1, 1, 50),
        "loop_count": _safe_int(row.get("loop_count"), 1, 1, 10_000),
        "enabled": bool(row.get("enabled", True)),
        "agent": agent,
    }

    cua_provider = str(row.get("cua_provider") or "").strip().lower()
    if cua_provider:
        normalized["cua_provider"] = cua_provider
    cua_target_url = str(row.get("cua_target_url") or "").strip()
    if cua_target_url:
        normalized["cua_target_url"] = cua_target_url
    return normalized


def _default_sequence_from_steps(steps: list[dict[str, object]]) -> str:
    labels: list[str] = []
    for idx, step in enumerate(steps, start=1):
        label = str(step.get("name") or step.get("job_type") or f"Step {idx}").strip()
        if not label:
            label = f"Step {idx}"
        labels.append(label)
        if len(labels) >= 8:
            break
    suffix = " -> ..." if len(steps) > len(labels) else ""
    return " -> ".join(labels) + suffix


def _normalize_recipe_payload(
    row: dict[str, object],
    *,
    allow_builtin_id: bool,
) -> dict[str, object]:
    raw_name = str(row.get("name") or "").strip()
    recipe_id = _slugify_recipe_id(row.get("id") or raw_name)
    if not recipe_id or not _RECIPE_ID_RE.fullmatch(recipe_id):
        raise ValueError(
            "recipe id must be 1-63 chars (lowercase letters, digits, '_' or '-') and start with a letter/digit"
        )
    if not allow_builtin_id and recipe_id in _RECIPE_DATA:
        raise ValueError(f"recipe id '{recipe_id}' is reserved by a built-in recipe")

    name = raw_name or recipe_id.replace("_", " ").replace("-", " ").title()
    name = name[: _MAX_NAME_CHARS]

    description = str(row.get("description") or "").strip()[: _MAX_DESCRIPTION_CHARS]
    steps_raw = row.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise ValueError("recipe must include a non-empty steps array")
    if len(steps_raw) > _MAX_RECIPE_STEPS:
        raise ValueError(f"recipe has too many steps (max {_MAX_RECIPE_STEPS})")

    steps: list[dict[str, object]] = []
    for idx, step in enumerate(steps_raw, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"step #{idx} must be an object")
        steps.append(_normalize_recipe_step(step, index=idx))

    sequence = str(row.get("sequence") or "").strip()[: _MAX_SEQUENCE_CHARS]
    if not sequence:
        sequence = _default_sequence_from_steps(steps)

    return {
        "id": recipe_id,
        "name": name,
        "description": description,
        "sequence": sequence,
        "steps": steps,
    }


def _custom_recipes_from_json_value(
    value: object,
    *,
    strict: bool,
) -> list[dict[str, object]]:
    rows = value.get("recipes") if isinstance(value, dict) else value
    if rows is None:
        rows = []
    if not isinstance(rows, list):
        if strict:
            raise ValueError("custom recipe store must be a JSON array or {'recipes': [...]} object")
        return []

    recipes: list[dict[str, object]] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            if strict:
                raise ValueError(f"custom recipe entry #{idx} must be an object")
            continue
        try:
            recipes.append(_normalize_recipe_payload(row, allow_builtin_id=False))
        except ValueError:
            if strict:
                raise
    return recipes


def _load_custom_recipes(repo: Path | str, *, strict: bool) -> list[dict[str, object]]:
    path = custom_recipes_path(repo)
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        if strict:
            raise ValueError(f"could not parse custom recipe file '{path}': {exc}") from exc
        return []
    return _custom_recipes_from_json_value(raw, strict=strict)


def _write_custom_recipes(repo: Path | str, recipes: list[dict[str, object]]) -> Path:
    path = custom_recipes_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "recipes": recipes,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)
    return path


def _with_source(recipe: dict[str, object], source: str) -> dict[str, object]:
    payload = deepcopy(recipe)
    payload["source"] = source
    return payload

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
                "job_type": "feature_discovery",
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
                    "instead of rewriting everything.\n\n"
                    "Important output rule:\n"
                    "- Write/update `.codex_manager/owner/FEATURE_DREAMS.md` directly in the repository.\n"
                    "- Do not return only analysis text without updating that file."
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
                    "If `.codex_manager/owner/FEATURE_DREAMS.md` does not exist, first create it with "
                    "at least 5 concrete feature items for this repository, then implement the first item.\n\n"
                    "Repeat this process for each repetition. Continue until there are no unchecked "
                    "feature items left.\n\n"
                    "If there are no unchecked items remaining, output exactly `[TERMINATE_STEP]` "
                    "on its own line and do not make code changes."
                ),
            },
        ],
    },
}


def _get_builtin_recipe(recipe_id: str) -> dict[str, object] | None:
    recipe = _RECIPE_DATA.get(recipe_id)
    if recipe is None:
        return None
    payload = deepcopy(recipe)
    payload["id"] = recipe_id
    return _with_source(payload, "builtin")


def _list_custom_recipes(repo: Path | str, *, strict: bool) -> list[dict[str, object]]:
    custom = _load_custom_recipes(repo, strict=strict)
    custom.sort(key=lambda item: (str(item.get("name") or "").lower(), str(item.get("id") or "")))
    return [_with_source(row, "custom") for row in custom]


def get_recipe(recipe_id: str, *, repo: Path | str | None = None) -> dict[str, object] | None:
    """Return one recipe by id, including per-repo custom recipes when provided."""
    recipe_key = str(recipe_id or "").strip()
    if not recipe_key:
        return None

    builtin = _get_builtin_recipe(recipe_key)
    if builtin is not None:
        return builtin

    repo_path = _coerce_repo_path(repo)
    if repo_path is None:
        return None
    for recipe in _list_custom_recipes(repo_path, strict=False):
        if str(recipe.get("id") or "") == recipe_key:
            return recipe
    return None


def list_custom_recipes(repo: Path | str) -> list[dict[str, object]]:
    """Return all custom recipes for a repository."""
    return _list_custom_recipes(repo, strict=False)


def list_recipes(*, repo: Path | str | None = None) -> list[dict[str, object]]:
    """Return built-in recipes plus optional per-repo custom recipes."""
    recipes: list[dict[str, object]] = []
    for recipe_id in _RECIPE_DATA:
        recipe = _get_builtin_recipe(recipe_id)
        if recipe is not None:
            recipes.append(recipe)

    repo_path = _coerce_repo_path(repo)
    if repo_path is not None:
        recipes.extend(_list_custom_recipes(repo_path, strict=False))
    return recipes


def list_recipe_summaries(*, repo: Path | str | None = None) -> list[dict[str, object]]:
    """Return lightweight recipe summaries (built-in + optional custom)."""
    return [_recipe_summary(recipe) for recipe in list_recipes(repo=repo)]


def recipe_steps_map(*, repo: Path | str | None = None) -> dict[str, list[dict[str, object]]]:
    """Return recipes in the frontend shape: recipe-id -> step list."""
    payload: dict[str, list[dict[str, object]]] = {}
    for recipe in list_recipes(repo=repo):
        recipe_id = str(recipe.get("id") or "").strip()
        if not recipe_id:
            continue
        steps = recipe.get("steps", [])
        payload[recipe_id] = deepcopy(steps) if isinstance(steps, list) else []
    return payload


def save_custom_recipe(
    repo: Path | str,
    recipe_payload: dict[str, object],
) -> tuple[dict[str, object], bool, Path]:
    """Create or update one custom recipe for a repository."""
    repo_path = _coerce_repo_path(repo)
    if repo_path is None:
        raise ValueError("repo path is required")
    if not isinstance(recipe_payload, dict):
        raise ValueError("recipe must be a JSON object")

    normalized = _normalize_recipe_payload(recipe_payload, allow_builtin_id=False)
    existing = _load_custom_recipes(repo_path, strict=True)

    created = True
    for idx, row in enumerate(existing):
        if str(row.get("id") or "") == str(normalized.get("id") or ""):
            existing[idx] = normalized
            created = False
            break
    if created:
        existing.append(normalized)

    existing.sort(key=lambda item: (str(item.get("name") or "").lower(), str(item.get("id") or "")))
    path = _write_custom_recipes(repo_path, existing)
    return _with_source(normalized, "custom"), created, path


def delete_custom_recipe(repo: Path | str, recipe_id: str) -> tuple[bool, Path]:
    """Delete one custom recipe by id. Returns ``(deleted, store_path)``."""
    repo_path = _coerce_repo_path(repo)
    if repo_path is None:
        raise ValueError("repo path is required")
    normalized_id = _slugify_recipe_id(recipe_id)
    if not normalized_id:
        return False, custom_recipes_path(repo_path)

    existing = _load_custom_recipes(repo_path, strict=True)
    remaining = [row for row in existing if str(row.get("id") or "") != normalized_id]
    deleted = len(remaining) != len(existing)
    path = _write_custom_recipes(repo_path, remaining) if deleted else custom_recipes_path(repo_path)
    return deleted, path


def import_custom_recipes(
    repo: Path | str,
    payload: object,
    *,
    replace: bool = False,
) -> tuple[dict[str, object], Path]:
    """Import custom recipes from JSON payload (single recipe, list, or {'recipes':[]})."""
    repo_path = _coerce_repo_path(repo)
    if repo_path is None:
        raise ValueError("repo path is required")

    incoming = payload
    if isinstance(incoming, str):
        try:
            incoming = json.loads(incoming)
        except json.JSONDecodeError as exc:
            raise ValueError(f"import payload is not valid JSON: {exc}") from exc

    if isinstance(incoming, dict) and ("id" in incoming or "steps" in incoming) and "recipes" not in incoming:
        rows: object = [incoming]
    else:
        rows = incoming
    if isinstance(rows, dict):
        rows = rows.get("recipes")
    if not isinstance(rows, list):
        raise ValueError("import payload must be a recipe object, recipe list, or {'recipes': [...]} object")
    if not rows:
        raise ValueError("import payload does not contain any recipes")

    normalized_rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"import recipe #{idx} must be an object")
        normalized = _normalize_recipe_payload(row, allow_builtin_id=False)
        recipe_id = str(normalized.get("id") or "")
        if recipe_id in seen_ids:
            raise ValueError(f"duplicate recipe id in import payload: {recipe_id}")
        seen_ids.add(recipe_id)
        normalized_rows.append(normalized)

    existing = [] if replace else _load_custom_recipes(repo_path, strict=True)
    merged_by_id = {str(item.get("id") or ""): item for item in existing}
    created = 0
    updated = 0
    for item in normalized_rows:
        recipe_id = str(item.get("id") or "")
        if recipe_id in merged_by_id:
            updated += 1
        else:
            created += 1
        merged_by_id[recipe_id] = item

    merged = list(merged_by_id.values())
    merged.sort(key=lambda item: (str(item.get("name") or "").lower(), str(item.get("id") or "")))
    path = _write_custom_recipes(repo_path, merged)
    summary = {
        "imported": len(normalized_rows),
        "created": created,
        "updated": updated,
        "total_custom_recipes": len(merged),
        "replaced_existing": bool(replace),
    }
    return summary, path


def export_custom_recipes(
    repo: Path | str,
    *,
    recipe_id: str = "",
) -> dict[str, object]:
    """Export all custom recipes, or one custom recipe when ``recipe_id`` is provided."""
    repo_path = _coerce_repo_path(repo)
    if repo_path is None:
        raise ValueError("repo path is required")

    custom = _load_custom_recipes(repo_path, strict=True)
    selected_id = _slugify_recipe_id(recipe_id)
    if selected_id:
        for row in custom:
            if str(row.get("id") or "") == selected_id:
                return {"version": 1, "recipes": [row]}
        raise ValueError(f"custom recipe not found: {recipe_id}")
    return {"version": 1, "recipes": custom}
