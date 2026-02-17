"""Repository text-encoding hygiene helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Deterministic replacements for the mojibake signatures repeatedly observed
# in repository source/templates.
MOJIBAKE_REPLACEMENTS: dict[str, str] = {
    "\u00e2\u2022\u0090": "-",
    "\u00c3\u00a2\u00e2\u20ac\u009d\u00e2\u201a\u00ac": "-",
    "\u00e2\u20ac\u009d": '"',
    "\u00e2\u20ac\u201c": "-",
    "\u00e2\u20ac\u201d": "--",
    "\u00e2\u20ac\u2122": "'",
    "\u00c2\u00a0": " ",
}


@dataclass(frozen=True, slots=True)
class HygieneIssue:
    """A single encoding-hygiene violation found in a file."""

    path: Path
    signature: str
    occurrences: int


def normalize_mojibake_text(text: str) -> tuple[str, int]:
    """Return normalized text and the number of replacements applied."""
    updated = text
    replacements = 0
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        if bad not in updated:
            continue
        count = updated.count(bad)
        updated = updated.replace(bad, good)
        replacements += count
    c1_controls = [ch for ch in updated if "\u0080" <= ch <= "\u009f"]
    if c1_controls:
        updated = "".join(ch for ch in updated if not ("\u0080" <= ch <= "\u009f"))
        replacements += len(c1_controls)
    return updated, replacements


def scan_text_for_mojibake(text: str) -> dict[str, int]:
    """Return mojibake-signature counts in ``text``."""
    counts: dict[str, int] = {}
    for bad in MOJIBAKE_REPLACEMENTS:
        count = text.count(bad)
        if count > 0:
            counts[bad] = count
    c1_count = sum(1 for ch in text if "\u0080" <= ch <= "\u009f")
    if c1_count > 0:
        counts["C1_CONTROL_RANGE"] = c1_count
    return counts


def _iter_text_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".ico",
            ".svg",
            ".woff",
            ".woff2",
            ".ttf",
            ".eot",
            ".pdf",
        }:
            continue
        files.append(path)
    return files


def scan_paths_for_mojibake(roots: list[Path]) -> list[HygieneIssue]:
    """Scan paths and return detected mojibake/c1-control issues."""
    issues: list[HygieneIssue] = []
    for root in roots:
        if not root.exists():
            continue
        for path in _iter_text_files(root):
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            for signature, count in scan_text_for_mojibake(text).items():
                issues.append(HygieneIssue(path=path, signature=signature, occurrences=count))
    return issues


def normalize_paths_in_place(roots: list[Path]) -> list[Path]:
    """Normalize mojibake signatures in-place and return modified file paths."""
    changed: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in _iter_text_files(root):
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            normalized, replacements = normalize_mojibake_text(text)
            if replacements <= 0 or normalized == text:
                continue
            path.write_text(normalized, encoding="utf-8")
            changed.append(path)
    return changed
