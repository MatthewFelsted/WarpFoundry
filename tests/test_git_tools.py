"""Unit tests for git_tools helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from codex_manager.git_tools import diff_numstat_entries, reset_to_ref, revert_all


def test_revert_all_preserves_codex_manager_artifacts(tmp_path: Path):
    with patch("codex_manager.git_tools._run_git") as run_git:
        revert_all(tmp_path)

    assert run_git.call_count == 2
    first = run_git.call_args_list[0].args
    second = run_git.call_args_list[1].args

    assert first[:3] == ("checkout", "--", ".")
    assert second[:4] == ("clean", "-fd", "-e", ".codex_manager/")


def test_diff_numstat_entries_parses_text_and_binary_rows(tmp_path: Path):
    sample = (
        "12\t3\tsrc/app.py\n"
        "-\t-\tassets/logo.png\n"
        "0\t7\tdocs/readme.md\n"
    )
    with patch(
        "codex_manager.git_tools._run_git",
        return_value=SimpleNamespace(stdout=sample),
    ):
        rows = diff_numstat_entries(tmp_path)

    assert rows == [
        {"path": "src/app.py", "insertions": 12, "deletions": 3},
        {"path": "assets/logo.png", "insertions": None, "deletions": None},
        {"path": "docs/readme.md", "insertions": 0, "deletions": 7},
    ]


def test_reset_to_ref_preserves_codex_manager_artifacts(tmp_path: Path):
    with patch("codex_manager.git_tools._run_git") as run_git:
        reset_to_ref(tmp_path, "abc1234")

    assert run_git.call_count == 2
    first = run_git.call_args_list[0].args
    second = run_git.call_args_list[1].args

    assert first[:3] == ("reset", "--hard", "abc1234")
    assert second[:4] == ("clean", "-fd", "-e", ".codex_manager/")
