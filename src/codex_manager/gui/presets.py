"""Preset job types with intelligent prompt templates.

Each preset has:
- A *direct* prompt (clear instructions for Codex)
- An *ai_decides* prompt (Codex analyses the repo and chooses what to do)

Prompts are loaded from the centralized prompt catalog (``prompts/templates.yaml``).
The catalog is the single source of truth — edits to this file's ``PRESETS`` dict
are used only as fallbacks if the catalog fails to load.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Fallback presets (used if YAML catalog is unavailable) ────────

_FALLBACK_PRESETS: dict[str, dict] = {
    "feature_discovery": {
        "name": "Feature Discovery",
        "icon": "\U0001f4a1",
        "description": "Analyze codebase and suggest high-impact improvements",
        "prompt": (
            "Analyze this repository thoroughly. Identify the 3 most impactful features "
            "or improvements that would meaningfully enhance the codebase. Consider: "
            "missing functionality, developer experience, performance, and code quality. "
            "Then implement the single most impactful improvement."
        ),
        "ai_prompt": (
            "You are a senior software architect reviewing this repository. Study its "
            "purpose, architecture, dependencies, and current state. Identify the single "
            "highest-impact improvement you could make. Explain your reasoning briefly, "
            "then implement the change."
        ),
        "on_failure": "skip",
    },
    "strategic_product_maximization": {
        "name": "Strategic Product Maximization",
        "icon": "\U0001f4c8",
        "description": "Prioritize and implement the highest product-leverage change",
        "prompt": (
            "Operate in STRATEGIC PRODUCT MAXIMIZATION MODE for this repository. "
            "Rank 3-5 opportunities by impact, effort, and risk; pick the single "
            "highest-leverage change; then implement it end-to-end with production-"
            "quality code, tests, and documentation updates."
        ),
        "ai_prompt": (
            "You are a product-minded principal engineer. Diagnose bottlenecks in "
            "user value, reliability, and adoption; select the top impact-to-effort "
            "improvement; and implement it fully (code + tests + docs)."
        ),
        "on_failure": "skip",
    },
    "implementation": {
        "name": "Implementation",
        "icon": "\U0001f528",
        "description": "Build new features and implement improvements",
        "prompt": (
            "Review any TODOs, FIXMEs, or incomplete features in the codebase. "
            "Implement the most critical missing functionality. Write clean, "
            "well-documented code that follows the project's existing patterns."
        ),
        "ai_prompt": (
            "As an expert developer, examine this repository for incomplete or missing "
            "features. Determine what would add the most value, then implement it with "
            "production-quality code including error handling and documentation."
        ),
        "on_failure": "skip",
    },
    "testing": {
        "name": "Testing",
        "icon": "\U0001f9ea",
        "description": "Write and improve test coverage",
        "prompt": (
            "Analyze the test coverage of this repository. Write comprehensive tests "
            "for untested or under-tested code. Focus on edge cases, error paths, "
            "integration points, and boundary conditions. Use the project's existing "
            "test framework and conventions."
        ),
        "ai_prompt": (
            "You are a QA engineer specializing in test design. Analyze this "
            "repository's test suite for gaps. Identify the most critical untested "
            "code paths, then write the most impactful tests to address those gaps."
        ),
        "on_failure": "skip",
    },
    "bug_hunting": {
        "name": "Bug Hunting",
        "icon": "\U0001f41b",
        "description": "Find and fix bugs, edge cases, and issues",
        "prompt": (
            "Search this repository for bugs, race conditions, off-by-one errors, "
            "null pointer issues, resource leaks, and edge cases. Fix any issues "
            "found. Verify each fix doesn't break existing functionality."
        ),
        "ai_prompt": (
            "You are a bug bounty hunter reviewing this codebase. Use your expertise "
            "to find subtle bugs that might cause issues in production. Analyze "
            "control flow, data handling, and error paths. Fix the most critical bug."
        ),
        "on_failure": "skip",
    },
    "refactoring": {
        "name": "Refactoring",
        "icon": "\u267b\ufe0f",
        "description": "Clean up and restructure code for maintainability",
        "prompt": (
            "Identify code smells, duplicated logic, overly complex functions, and "
            "poor abstractions. Refactor for clarity, maintainability, and performance. "
            "Keep behavior identical — ensure all existing tests still pass."
        ),
        "ai_prompt": (
            "As a senior developer focused on code quality, identify the most impactful "
            "refactoring opportunity in this codebase. Look for long functions, "
            "duplicated code, poor naming, tight coupling, or missing abstractions. "
            "Refactor while preserving all behavior."
        ),
        "on_failure": "skip",
    },
    "documentation": {
        "name": "Documentation",
        "icon": "\U0001f4dd",
        "description": "Update docs, docstrings, and comments",
        "prompt": (
            "Review all source files for missing or outdated documentation. Update "
            "docstrings, add inline comments for complex logic, and ensure the README "
            "accurately reflects the current state. Follow the project's style."
        ),
        "ai_prompt": (
            "You are a technical writer reviewing this codebase. Find the most poorly "
            "documented areas and improve them. Focus on public APIs, complex "
            "algorithms, and configuration. Make the code self-documenting."
        ),
        "on_failure": "skip",
    },
    "performance": {
        "name": "Performance",
        "icon": "\u26a1",
        "description": "Optimize speed, memory, and efficiency",
        "prompt": (
            "Analyze this codebase for performance bottlenecks: unnecessary "
            "allocations, O(n^2) algorithms, synchronous I/O that could be async, "
            "missing caching, and redundant computations. Implement the most "
            "impactful optimization."
        ),
        "ai_prompt": (
            "You are a performance engineer. Profile this codebase mentally and "
            "identify the biggest performance win available. Consider algorithmic "
            "complexity, I/O patterns, caching, and data structures. Implement it."
        ),
        "on_failure": "skip",
    },
    "security_audit": {
        "name": "Security Audit",
        "icon": "\U0001f512",
        "description": "Find and fix security vulnerabilities",
        "prompt": (
            "Audit this repository for security vulnerabilities: injection attacks, "
            "authentication/authorization bypass, sensitive data exposure, insecure "
            "defaults, and dependency vulnerabilities. Fix any issues found."
        ),
        "ai_prompt": (
            "You are a security researcher performing a code audit. Examine this "
            "codebase for OWASP Top 10 vulnerabilities and common security "
            "anti-patterns. Identify the most critical issue and implement a fix."
        ),
        "on_failure": "skip",
    },
    "visual_test": {
        "name": "Visual Testing (CUA)",
        "icon": "\U0001f441\ufe0f",
        "description": "Use a Computer-Using Agent to visually test the application UI",
        "prompt": (
            "Visually inspect the application UI. Navigate through the main views, "
            "test interactive elements (buttons, forms, dropdowns), and report any "
            "visual bugs, broken layouts, or usability issues you find."
        ),
        "ai_prompt": (
            "You are a quality assurance specialist performing visual testing on a "
            "web application. Use the browser to navigate, interact with elements, "
            "and verify the UI works correctly. Report issues with severity levels."
        ),
        "on_failure": "skip",
        "is_cua": True,
    },
}


def _load_presets() -> dict[str, dict]:
    """Load presets from the prompt catalog, falling back to hardcoded defaults."""
    try:
        from codex_manager.prompts.catalog import get_catalog

        catalog = get_catalog()
        presets: dict[str, dict] = {}

        for key in catalog.raw.get("presets", {}):
            detail = catalog.preset_detail(key)
            if detail:
                presets[key] = detail

        if presets:
            return presets
    except Exception as exc:
        logger.debug("Could not load presets from catalog: %s", exc)

    return dict(_FALLBACK_PRESETS)


# Module-level PRESETS dict — loaded lazily on first access
_presets_cache: dict[str, dict] | None = None


def _get_presets() -> dict[str, dict]:
    global _presets_cache
    if _presets_cache is None:
        _presets_cache = _load_presets()
    return _presets_cache


# Public API — maintain backward compatibility
PRESETS = _FALLBACK_PRESETS  # static reference for imports that access PRESETS directly


def get_preset(key: str) -> dict | None:
    """Return a preset by key, or None."""
    return _get_presets().get(key)


def get_prompt(key: str, *, ai_decides: bool = False) -> str:
    """Return the prompt text for a preset job type."""
    preset = _get_presets().get(key)
    if not preset:
        return ""
    return preset["ai_prompt"] if ai_decides else preset["prompt"]


def list_presets() -> list[dict]:
    """Return a summary list of all presets (for the GUI)."""
    return [
        {
            "key": k,
            "name": v.get("name", k),
            "icon": v.get("icon", ""),
            "description": v.get("description", ""),
        }
        for k, v in _get_presets().items()
    ]
