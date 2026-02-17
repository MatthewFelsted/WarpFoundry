"""Encoding hygiene regression checks."""

from __future__ import annotations

from pathlib import Path

from codex_manager.encoding_hygiene import (
    normalize_mojibake_text,
    scan_paths_for_mojibake,
)


def test_normalize_mojibake_text_replaces_known_signatures() -> None:
    raw = "prefix \u00c3\u00a2\u00e2\u20ac\u009d\u00e2\u201a\u00ac and \u00e2\u2022\u0090 suffix"
    normalized, replacements = normalize_mojibake_text(raw)
    assert replacements >= 2
    assert "\u00c3\u00a2\u00e2\u20ac\u009d\u00e2\u201a\u00ac" not in normalized
    assert "\u00e2\u2022\u0090" not in normalized
    assert "-" in normalized


def test_repo_has_no_known_mojibake_signatures() -> None:
    roots = [Path("src"), Path("docs")]
    issues = scan_paths_for_mojibake(roots)
    assert issues == []
