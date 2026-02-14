"""Unit tests for schemas module."""

from __future__ import annotations

import json
from pathlib import Path

from codex_manager.schemas import (
    CodexEvent,
    EvalResult,
    EventKind,
    LoopState,
    RoundRecord,
    RunResult,
    TestOutcome,
    UsageInfo,
)


class TestRunResult:
    def test_defaults(self):
        r = RunResult()
        assert r.success is False
        assert r.exit_code == -1
        assert r.final_message == ""
        assert r.events == []

    def test_with_events(self):
        ev = CodexEvent(kind=EventKind.AGENT_MESSAGE, text="hello")
        r = RunResult(success=True, exit_code=0, events=[ev])
        assert len(r.events) == 1
        assert r.events[0].text == "hello"


class TestEvalResult:
    def test_defaults(self):
        e = EvalResult()
        assert e.test_outcome == TestOutcome.ERROR
        assert e.net_lines_changed == 0

    def test_serialization(self):
        e = EvalResult(test_outcome=TestOutcome.PASSED, net_lines_changed=42)
        data = json.loads(e.model_dump_json())
        assert data["test_outcome"] == "passed"
        assert data["net_lines_changed"] == 42


class TestLoopState:
    def test_save_and_load(self, tmp_path: Path):
        state = LoopState(
            repo_path=str(tmp_path),
            goal="test goal",
            mode="dry-run",
            max_rounds=3,
        )
        state.save()

        loaded = LoopState.load(tmp_path)
        assert loaded is not None
        assert loaded.goal == "test goal"
        assert loaded.max_rounds == 3

    def test_load_missing(self, tmp_path: Path):
        assert LoopState.load(tmp_path) is None

    def test_round_record_defaults(self):
        r = RoundRecord(round_number=1)
        assert r.round_number == 1
        assert r.commit_sha is None
        assert r.skipped is False


class TestUsageInfo:
    def test_defaults(self):
        u = UsageInfo()
        assert u.total_tokens == 0
        assert u.model is None

    def test_with_values(self):
        u = UsageInfo(input_tokens=100, output_tokens=200, total_tokens=300, model="o3")
        assert u.total_tokens == 300
