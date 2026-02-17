"""Additional coverage for encoding hygiene scan/normalize helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from codex_manager.encoding_hygiene import (
    normalize_paths_in_place,
    scan_paths_for_mojibake,
    scan_text_for_mojibake,
)

pytestmark = pytest.mark.unit


def test_scan_text_for_mojibake_counts_signatures_and_c1_controls() -> None:
    text = "bad:\u00e2\u20ac\u009d ok \u0085"

    counts = scan_text_for_mojibake(text)

    assert counts["\u00e2\u20ac\u009d"] == 1
    assert counts["C1_CONTROL_RANGE"] == 2


def test_scan_paths_for_mojibake_skips_missing_roots_and_non_utf8_files(tmp_path: Path) -> None:
    root = tmp_path / "scan-root"
    root.mkdir(parents=True, exist_ok=True)
    good = root / "file.md"
    bad_bytes = root / "binary-like.txt"
    good.write_text("prefix \u00e2\u20ac\u009d suffix", encoding="utf-8")
    bad_bytes.write_bytes(b"\xff\xfe\xfd")

    missing = tmp_path / "missing-root"
    issues = scan_paths_for_mojibake([missing, root])

    assert any(issue.path == good and issue.signature == "\u00e2\u20ac\u009d" for issue in issues)
    assert all(issue.path != bad_bytes for issue in issues)


def test_normalize_paths_in_place_updates_text_files_and_ignores_binary_suffixes(tmp_path: Path) -> None:
    root = tmp_path / "normalize-root"
    root.mkdir(parents=True, exist_ok=True)
    text_file = root / "content.md"
    image_file = root / "icon.png"
    text_file.write_text("alpha \u00e2\u20ac\u201d omega \u0085", encoding="utf-8")
    original_image = b"\x89PNG\r\n\x1a\nbinary-data-\xe2\x80\x94"
    image_file.write_bytes(original_image)

    changed = normalize_paths_in_place([root])

    assert changed == [text_file]
    assert "\u00e2\u20ac\u201d" not in text_file.read_text(encoding="utf-8")
    assert "\u0085" not in text_file.read_text(encoding="utf-8")
    assert image_file.read_bytes() == original_image


def test_normalize_paths_in_place_skips_unreadable_utf8_files(tmp_path: Path) -> None:
    root = tmp_path / "skip-root"
    root.mkdir(parents=True, exist_ok=True)
    unreadable = root / "bad.txt"
    unreadable.write_bytes(b"\xff\xfe\xfd")

    changed = normalize_paths_in_place([root])

    assert changed == []
