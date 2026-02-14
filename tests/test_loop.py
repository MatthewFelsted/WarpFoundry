"""Unit tests for the loop module."""

from __future__ import annotations

from codex_manager.loop import (
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
