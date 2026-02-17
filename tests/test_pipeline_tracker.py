"""Tests for markdown log tracker initialization, queries, and context assembly."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from codex_manager.pipeline.tracker import LogTracker


class _LedgerStub:
    def __init__(self, response: str = "ledger context") -> None:
        self.response = response
        self.calls: list[tuple[list[str] | None, int]] = []

    def get_context_for_prompt(self, *, categories, max_items):
        self.calls.append((categories, max_items))
        return self.response


def test_initialize_creates_templates_and_preserves_existing_content(tmp_path: Path) -> None:
    tracker = LogTracker(tmp_path / "repo")
    tracker.initialize()

    for filename in (
        "WISHLIST.md",
        "TESTPLAN.md",
        "ERRORS.md",
        "EXPERIMENTS.md",
        "PROGRESS.md",
        "RESEARCH.md",
    ):
        assert tracker.path_for(filename).exists()

    tracker.write("WISHLIST.md", "custom content")
    tracker.initialize()
    assert tracker.read("WISHLIST.md") == "custom content"


def test_read_write_append_and_path_for(tmp_path: Path) -> None:
    tracker = LogTracker(tmp_path / "repo")
    assert tracker.read("missing.md") == ""

    tracker.write("TESTPLAN.md", "first")
    tracker.append("TESTPLAN.md", "second")
    assert tracker.read("TESTPLAN.md") == "first\nsecond"
    assert tracker.path_for("TESTPLAN.md").name == "TESTPLAN.md"


def test_count_helpers_and_next_item_id(tmp_path: Path) -> None:
    tracker = LogTracker(tmp_path / "repo")
    tracker.write(
        "WISHLIST.md",
        "\n".join(
            [
                "- [WISH-001] Add docs  **Status**: pending",
                "- [WISH-002] Add tests **Status**: done",
                "- [WISH-010] Refactor  **Status**: Pending",
            ]
        ),
    )

    assert tracker.count_items("WISHLIST.md", status="pending") == 2
    assert tracker.count_all_items("WISHLIST.md") == {"pending": 2, "done": 1}
    assert tracker.next_item_id("WISHLIST.md", "WISH") == "WISH-011"


def test_get_context_for_phase_builds_expected_sections(tmp_path: Path) -> None:
    tracker = LogTracker(tmp_path / "repo")
    tracker.write("WISHLIST.md", "wishlist content")
    tracker.write("TESTPLAN.md", "testplan content")
    tracker.write("ERRORS.md", "errors content")
    tracker.write("EXPERIMENTS.md", "experiments content")
    tracker.write("RESEARCH.md", "research content")
    tracker.write("PROGRESS.md", "progress content")

    ledger = _LedgerStub(response="ledger block")
    debugging = tracker.get_context_for_phase("debugging", ledger=ledger)
    assert "ledger block" in debugging
    assert "## Current ERRORS.md" in debugging
    assert ledger.calls[0][0] == ["error", "bug", "observation"]
    assert ledger.calls[0][1] == 25

    testing = tracker.get_context_for_phase("testing", ledger=ledger)
    assert "## Current TESTPLAN.md" in testing
    assert "## Recent PROGRESS.md" in testing

    progress_review = tracker.get_context_for_phase("progress_review")
    assert "## Current WISHLIST.md" in progress_review
    assert "## Current TESTPLAN.md" in progress_review
    assert "## Current ERRORS.md" in progress_review
    assert "## Current EXPERIMENTS.md" in progress_review
    assert "## Current RESEARCH.md" in progress_review

    assert tracker.get_context_for_phase("not-a-phase") == ""


def test_get_context_for_phase_requests_wishlist_categories(tmp_path: Path) -> None:
    tracker = LogTracker(tmp_path / "repo")
    tracker.write("WISHLIST.md", "wishlist content")

    ledger = _LedgerStub(response="ledger block")
    ideation = tracker.get_context_for_phase("ideation", ledger=ledger)
    prioritization = tracker.get_context_for_phase("prioritization", ledger=ledger)
    implementation = tracker.get_context_for_phase("implementation", ledger=ledger)

    assert "ledger block" in ideation
    assert "ledger block" in prioritization
    assert "ledger block" in implementation
    assert ledger.calls[0][0] == ["suggestion", "wishlist", "feature"]
    assert ledger.calls[1][0] == ["todo", "feature", "suggestion", "wishlist"]
    assert ledger.calls[2][0] == ["todo", "feature", "suggestion", "wishlist"]
    assert all(call[1] == 25 for call in ledger.calls)


def test_log_phase_result_appends_status_entry(tmp_path: Path) -> None:
    tracker = LogTracker(tmp_path / "repo")
    tracker.log_phase_result("testing", iteration=2, success=False, summary="Tests failed")
    tracker.log_phase_result("debugging", iteration=3, success=True, summary="Fixed issues")

    progress = tracker.read("PROGRESS.md")
    assert "[FAILED] testing (iteration 2)" in progress
    assert "[SUCCESS] debugging (iteration 3)" in progress
    assert "Tests failed" in progress
    assert "Fixed issues" in progress


def test_read_recovers_non_utf8_log_and_rewrites_utf8(tmp_path: Path) -> None:
    tracker = LogTracker(tmp_path / "repo")
    tracker.initialize()

    progress = tracker.path_for("PROGRESS.md")
    progress.write_bytes(b"alpha\x97omega")

    text = tracker.read("PROGRESS.md")
    assert "alpha" in text
    assert "omega" in text
    assert "\u2014" in text

    # Recovered file is rewritten as UTF-8 so future reads do not fail.
    assert b"\x97" not in progress.read_bytes()
    assert progress.read_text(encoding="utf-8") == text

    tracker.append("PROGRESS.md", "tail")
    assert "tail" in tracker.read("PROGRESS.md")


def test_initialize_normalizes_markdown_logs_once_and_records_event(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    logs_dir = repo / ".codex_manager" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "PROGRESS.md").write_bytes(b"alpha\x97omega")

    tracker = LogTracker(repo)
    tracker.initialize()

    progress = tracker.read("PROGRESS.md")
    assert "alpha" in progress
    assert "omega" in progress
    assert "\u2014" in progress
    assert b"\x97" not in tracker.path_for("PROGRESS.md").read_bytes()
    assert (logs_dir / ".encoding_normalized_v1").exists()
    assert "encoding-recovery" in tracker.read("ERRORS.md")


def test_append_is_thread_safe_under_concurrent_writes(tmp_path: Path) -> None:
    tracker = LogTracker(tmp_path / "repo")
    tracker.initialize()

    workers = 12
    entries_per_worker = 20
    start_barrier = threading.Barrier(workers)

    def append_entries(worker: int) -> None:
        start_barrier.wait(timeout=5)
        for idx in range(entries_per_worker):
            tracker.append("PROGRESS.md", f"worker-{worker}-entry-{idx}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(append_entries, range(workers)))

    content = tracker.read("PROGRESS.md")
    for worker in range(workers):
        for idx in range(entries_per_worker):
            assert f"worker-{worker}-entry-{idx}" in content


def test_append_rotates_markdown_logs_when_threshold_exceeded(tmp_path: Path) -> None:
    tracker = LogTracker(
        tmp_path / "repo",
        markdown_rotate_bytes=220,
        markdown_max_archives=2,
    )
    tracker.initialize()

    for idx in range(40):
        tracker.append("PROGRESS.md", f"entry-{idx} " + ("x" * 60))

    archive_dir = tracker.path_for("PROGRESS.md").parent / "archive" / "tracker"
    archives = list(archive_dir.glob("PROGRESS-*"))
    assert archives
    assert len(archives) <= 2
    assert "entry-" in tracker.read("PROGRESS.md")


def test_initialize_migrates_legacy_error_log_to_canonical_path(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    legacy = repo / ".codex_manager" / "ERRORS.md"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("legacy warning\n", encoding="utf-8")

    tracker = LogTracker(repo)
    tracker.initialize()

    assert not legacy.exists()
    canonical = tracker.read("ERRORS.md")
    assert "legacy warning" in canonical
    archived = list((tracker.path_for("ERRORS.md").parent / "archive" / "legacy").glob("ERRORS-*"))
    assert archived


def test_initialize_science_creates_expected_files(tmp_path: Path) -> None:
    tracker = LogTracker(tmp_path / "repo")
    tracker.initialize_science()

    science_dir = tracker.science_dir()
    assert tracker.path_for("SCIENTIST_REPORT.md").exists()
    assert (science_dir / "README.md").exists()
    assert (science_dir / "TRIALS.jsonl").exists()
    assert (science_dir / "EVIDENCE.md").exists()
    assert (science_dir / "HYPOTHESES.md").exists()
    assert (science_dir / "ANALYSIS.md").exists()
    assert (science_dir / "EXPERIMENTS_LATEST.md").exists()
    assert (science_dir / "prompts").is_dir()
    assert (science_dir / "outputs").is_dir()
    assert (science_dir / "snapshots").is_dir()


def test_initialize_creates_agent_protocol_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    tracker = LogTracker(repo)
    tracker.initialize()

    protocol = repo / ".codex_manager" / "AGENT_PROTOCOL.md"
    assert protocol.exists()
    assert "Agent Protocol" in protocol.read_text(encoding="utf-8")


def test_science_helpers_write_jsonl_and_artifacts(tmp_path: Path) -> None:
    tracker = LogTracker(tmp_path / "repo")
    tracker.initialize_science()

    tracker.write_science("EVIDENCE.md", "first line")
    tracker.append_science("EVIDENCE.md", "second line")
    assert (
        tracker.science_path_for("EVIDENCE.md").read_text(encoding="utf-8")
        == "first line\nsecond line"
    )

    tracker.append_science_jsonl("TRIALS.jsonl", {"phase": "experiment", "success": True})
    lines = (
        tracker.science_path_for("TRIALS.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["phase"] == "experiment"
    assert payload["success"] is True

    artifact = tracker.save_science_artifact(
        "prompts", "Cycle 1 / Experiment", "prompt text", suffix=".txt"
    )
    assert artifact.exists()
    assert "Cycle-1-Experiment" in artifact.name
    assert artifact.read_text(encoding="utf-8") == "prompt text"


def test_get_context_for_phase_includes_scientist_report_for_implementation(tmp_path: Path) -> None:
    tracker = LogTracker(tmp_path / "repo")
    tracker.write("WISHLIST.md", "wishlist content")
    tracker.write("SCIENTIST_REPORT.md", "science report content")

    implementation_ctx = tracker.get_context_for_phase("implementation")

    assert "## Current SCIENTIST_REPORT.md" in implementation_ctx
    assert "science report content" in implementation_ctx


def test_get_recovered_backlog_context_collects_cross_run_sources(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    tracker = LogTracker(repo)
    tracker.initialize()
    tracker.write("WISHLIST.md", "- [ ] Keep unresolved wishlist work in scope\n")

    owner_dir = repo / ".codex_manager" / "owner"
    owner_dir.mkdir(parents=True, exist_ok=True)
    (owner_dir / "TODO_WISHLIST.md").write_text(
        "- [ ] Keep archive ideas alive\n",
        encoding="utf-8",
    )
    (owner_dir / "GENERAL_REQUEST_HISTORY.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "request": "Preserve experiment findings for future runs.",
                        "status": "considered",
                    }
                ),
                json.dumps(
                    {
                        "request": "Already shipped request.",
                        "status": "implemented",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    tracker.path_for("HISTORY.jsonl").write_text(
        json.dumps(
            {
                "event": "phase_result",
                "level": "warn",
                "summary": "Implementation failed to finish all tasks.",
                "context": {"success": False, "phase": "implementation", "cycle": 3},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    archive_dir = repo / ".codex_manager" / "output_history" / "20260217T010101Z"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "RUN_SUMMARY.md").write_text(
        "- [ ] Revisit archived trial notes\n",
        encoding="utf-8",
    )

    recovered = tracker.get_recovered_backlog_context(max_items=12)

    assert "## Recovered Pending Backlog (Cross-Run)" in recovered
    assert "[logs/WISHLIST.md] Keep unresolved wishlist work in scope" in recovered
    assert "[owner/TODO_WISHLIST.md] Keep archive ideas alive" in recovered
    assert "owner/GENERAL_REQUEST_HISTORY.jsonl:considered" in recovered
    assert "Follow up failed phase 'implementation'" in recovered
    assert "output_history/20260217T010101Z/RUN_SUMMARY.md" in recovered


def test_get_context_for_self_improvement_phase_includes_recovered_backlog(tmp_path: Path) -> None:
    tracker = LogTracker(tmp_path / "repo")
    tracker.initialize()
    tracker.write("WISHLIST.md", "- [ ] Recover pending items before restart\n")

    context = tracker.get_context_for_phase("apply_upgrades_and_restart")

    assert "## Recovered Pending Backlog (Cross-Run)" in context
    assert "Recover pending items before restart" in context
