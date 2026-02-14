"""Unit tests for the loop module."""

from __future__ import annotations

import pytest

import codex_manager.loop as loop_module
from codex_manager.loop import (
    ImprovementLoop,
    _is_low_impact,
    build_round_prompt,
    default_budget_policy,
    default_stop_policy,
)
from codex_manager.schemas import (
    EvalResult,
    LoopState,
    RoundRecord,
    RunResult,
    StopReason,
    TestOutcome,
)


class TestIsLowImpact:
    def test_below_threshold(self):
        er = EvalResult(net_lines_changed=5, files_changed=1)
        assert _is_low_impact(er) is True

    def test_above_threshold(self):
        er = EvalResult(net_lines_changed=50, files_changed=5)
        assert _is_low_impact(er) is False

    def test_exact_threshold(self):
        er = EvalResult(net_lines_changed=20, files_changed=2)
        assert _is_low_impact(er) is False  # >= threshold


class TestBuildRoundPrompt:
    def test_first_round(self):
        prompt = build_round_prompt("fix bugs", 1, None)
        assert "fix bugs" in prompt
        assert "Round 1" in prompt

    def test_with_previous(self):
        prev = RoundRecord(
            round_number=1,
            eval_result=EvalResult(
                test_outcome=TestOutcome.FAILED,
                test_summary="2 failed",
                diff_stat="1 file changed",
            ),
            run_result=RunResult(errors=["timeout"]),
        )
        prompt = build_round_prompt("fix bugs", 2, prev)
        assert "Previous round" in prompt
        assert "failed" in prompt.lower()


class TestDefaultStopPolicy:
    def _make_state(self, rounds: list[RoundRecord]) -> LoopState:
        return LoopState(repo_path="/tmp/test", goal="test", rounds=rounds)

    def test_returns_none_when_few_rounds(self):
        state = self._make_state([RoundRecord(round_number=1)])
        assert default_stop_policy(state) is None

    def test_stops_on_consecutive_failures(self):
        rounds = [
            RoundRecord(round_number=i, run_result=RunResult(success=False))
            for i in range(1, 4)
        ]
        state = self._make_state(rounds)
        assert default_stop_policy(state) == StopReason.CONSECUTIVE_FAILURES

    def test_stops_on_low_impact_convergence(self):
        rounds = [
            RoundRecord(
                round_number=i,
                run_result=RunResult(success=True, exit_code=0),
                eval_result=EvalResult(
                    test_outcome=TestOutcome.PASSED,
                    net_lines_changed=3,
                    files_changed=1,
                ),
            )
            for i in range(1, 3)
        ]
        state = self._make_state(rounds)
        assert default_stop_policy(state) == StopReason.TESTS_PASS_LOW_IMPACT

    def test_continues_when_high_impact(self):
        rounds = [
            RoundRecord(
                round_number=i,
                run_result=RunResult(success=True, exit_code=0),
                eval_result=EvalResult(
                    test_outcome=TestOutcome.PASSED,
                    net_lines_changed=100,
                    files_changed=10,
                ),
            )
            for i in range(1, 3)
        ]
        state = self._make_state(rounds)
        assert default_stop_policy(state) is None


class TestDefaultBudgetPolicy:
    def test_under_budget(self):
        state = LoopState(
            repo_path="/tmp/test",
            goal="test",
            total_input_tokens=100_000,
            total_output_tokens=100_000,
        )
        assert default_budget_policy(state) is None

    def test_over_budget(self):
        state = LoopState(
            repo_path="/tmp/test",
            goal="test",
            total_input_tokens=1_500_000,
            total_output_tokens=600_000,
        )
        assert default_budget_policy(state) == StopReason.BUDGET_EXHAUSTED


class TestImprovementLoopInit:
    @staticmethod
    def _make_repo(tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        return repo

    def test_rejects_unknown_mode(self, tmp_path):
        repo = self._make_repo(tmp_path)
        with pytest.raises(ValueError, match="mode must be 'dry-run' or 'apply'"):
            ImprovementLoop(repo_path=repo, goal="goal", mode="invalid")

    def test_rejects_non_positive_rounds(self, tmp_path):
        repo = self._make_repo(tmp_path)
        with pytest.raises(ValueError, match="max_rounds must be >= 1"):
            ImprovementLoop(repo_path=repo, goal="goal", max_rounds=0)

    def test_normalizes_mode_input(self, tmp_path):
        repo = self._make_repo(tmp_path)
        loop = ImprovementLoop(repo_path=repo, goal="goal", mode=" APPLY ")
        assert loop.mode == "apply"

    def test_apply_mode_init_sets_git_identity_before_creating_branch(
        self, tmp_path, monkeypatch
    ):
        repo = self._make_repo(tmp_path)
        call_order: list[str] = []

        monkeypatch.setattr(
            loop_module,
            "ensure_git_identity",
            lambda _repo: call_order.append("ensure"),
        )
        monkeypatch.setattr(
            loop_module,
            "create_branch",
            lambda _repo: (call_order.append("branch"), "codex-manager/test")[1],
        )

        loop = ImprovementLoop(repo_path=repo, goal="goal", mode="apply")
        state = loop._init_state()

        assert call_order == ["ensure", "branch"]
        assert state.branch_name == "codex-manager/test"
