from __future__ import annotations

from pathlib import Path

from codex_manager.managed_artifacts import (
    capture_artifact_snapshot,
    merge_eval_result_with_artifact_delta,
    summarize_artifact_delta,
)
from codex_manager.schemas import EvalResult, TestOutcome


def test_summarize_artifact_delta_tracks_owner_markdown_changes(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    owner_dir = repo / ".codex_manager" / "owner"
    owner_dir.mkdir(parents=True, exist_ok=True)

    before = capture_artifact_snapshot(repo)
    target = owner_dir / "TODO_WISHLIST.md"
    target.write_text("- [ ] First item\n- [ ] Second item\n", encoding="utf-8")

    entries, ins, dels = summarize_artifact_delta(repo, before)

    assert ins == 2
    assert dels == 0
    assert len(entries) == 1
    assert entries[0]["path"] == ".codex_manager/owner/TODO_WISHLIST.md"
    assert entries[0]["insertions"] == 2
    assert entries[0]["deletions"] == 0


def test_merge_eval_result_with_artifact_delta_updates_eval_totals():
    eval_result = EvalResult(
        test_outcome=TestOutcome.SKIPPED,
        test_summary="skipped",
        test_exit_code=0,
        files_changed=0,
        net_lines_changed=0,
        changed_files=[],
    )
    artifact_entries = [
        {
            "path": ".codex_manager/owner/FEATURE_DREAMS.md",
            "insertions": 5,
            "deletions": 1,
            "source": "managed_artifact",
        }
    ]

    merged = merge_eval_result_with_artifact_delta(eval_result, artifact_entries)

    assert merged == {"files_added": 1, "insertions": 5, "deletions": 1}
    assert eval_result.files_changed == 1
    assert eval_result.net_lines_changed == 4
    assert eval_result.changed_files[0]["path"] == ".codex_manager/owner/FEATURE_DREAMS.md"
