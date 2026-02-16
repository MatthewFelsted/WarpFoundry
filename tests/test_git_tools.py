"""Unit tests for git_tools helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from codex_manager.git_tools import (
    diff_numstat_entries,
    pending_numstat,
    pending_numstat_entries,
    reset_to_ref,
    revert_all,
    summarize_numstat_entries,
)


def test_revert_all_preserves_codex_manager_artifacts(tmp_path: Path):
    with patch("codex_manager.git_tools._run_git") as run_git:
        revert_all(tmp_path)

    assert run_git.call_count == 2
    first = run_git.call_args_list[0].args
    second = run_git.call_args_list[1].args

    assert first[:3] == ("checkout", "--", ".")
    assert second[:4] == ("clean", "-fd", "-e", ".codex_manager/")


def test_diff_numstat_entries_parses_text_and_binary_rows(tmp_path: Path):
    sample = "12\t3\tsrc/app.py\n-\t-\tassets/logo.png\n0\t7\tdocs/readme.md\n"
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


def test_summarize_numstat_entries_ignores_binary_line_counts():
    rows = [
        {"path": "src/app.py", "insertions": 12, "deletions": 3},
        {"path": "assets/logo.png", "insertions": None, "deletions": None},
        {"path": "README.md", "insertions": 5, "deletions": 2},
    ]

    assert summarize_numstat_entries(rows) == (3, 17, 5)


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (repo / "tracked_a.txt").write_text("a\n", encoding="utf-8")
    (repo / "tracked_b.txt").write_text("u\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True, text=True)


def test_pending_numstat_entries_include_staged_unstaged_and_untracked(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)

    (repo / "tracked_a.txt").write_text("a\nb\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "tracked_a.txt"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (repo / "tracked_b.txt").write_text("u\nv\n", encoding="utf-8")
    (repo / "new_notes.md").write_text("first\nsecond\n", encoding="utf-8")

    rows = pending_numstat_entries(repo)
    paths = {str(row.get("path")) for row in rows}
    assert "tracked_a.txt" in paths
    assert "tracked_b.txt" in paths
    assert "new_notes.md" in paths

    files_changed, ins, dels = pending_numstat(repo)
    assert files_changed == 3
    assert ins == 4
    assert dels == 0


def test_pending_numstat_entries_marks_untracked_binary_files(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    binary = repo / "blob.bin"
    binary.write_bytes(b"\x00\x01\x02\x03")

    rows = pending_numstat_entries(repo)
    row = next((item for item in rows if str(item.get("path")) == "blob.bin"), None)
    assert row is not None
    assert row["insertions"] is None
    assert row["deletions"] is None
