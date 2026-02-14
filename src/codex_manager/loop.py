"""Improvement-loop orchestrator.

The :class:`ImprovementLoop` drives repeated Codex CLI invocations against a
target repository, evaluating after each round and stopping when diminishing
returns are detected.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from pathlib import Path

from codex_manager.codex_cli import CodexRunner
from codex_manager.eval_tools import RepoEvaluator
from codex_manager.git_tools import (
    commit_all,
    create_branch,
    generate_commit_message,
    is_clean,
    revert_all,
)
from codex_manager.schemas import (
    EvalResult,
    LoopState,
    RoundRecord,
    RunResult,
    StopReason,
    TestOutcome,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BUDGET_MAX_TOKENS: int = 2_000_000
"""Maximum total tokens (input + output) before the budget policy stops the loop."""

LOW_IMPACT_LINE_THRESHOLD: int = 20
"""Net lines-changed below which a round is considered low-impact."""

# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------

def default_stop_policy(state: LoopState) -> StopReason | None:
    """Decide whether to stop the loop based on recent rounds.

    Stop when:
    1. Tests pass AND no errors AND last 2 rounds had "low impact".
    2. Three consecutive Codex failures.
    """
    rounds = state.rounds
    if len(rounds) < 2:
        return None

    # --- consecutive failures ---
    recent_failures = 0
    for r in reversed(rounds):
        if not r.run_result.success:
            recent_failures += 1
        else:
            break
    if recent_failures >= 3:
        return StopReason.CONSECUTIVE_FAILURES

    # --- low impact convergence ---
    last_two = rounds[-2:]
    all_pass = all(r.eval_result.test_outcome == TestOutcome.PASSED for r in last_two)
    all_low_impact = all(_is_low_impact(r.eval_result) for r in last_two)
    no_errors = all(len(r.run_result.errors) == 0 for r in last_two)

    if all_pass and all_low_impact and no_errors:
        return StopReason.TESTS_PASS_LOW_IMPACT

    return None


def default_budget_policy(state: LoopState) -> StopReason | None:
    """Return a stop reason if the token budget has been exhausted.

    Default budget: 2 000 000 total tokens across all rounds.
    """
    total = state.total_input_tokens + state.total_output_tokens
    if total >= DEFAULT_BUDGET_MAX_TOKENS:
        return StopReason.BUDGET_EXHAUSTED
    return None


def _is_low_impact(er: EvalResult, threshold: int = LOW_IMPACT_LINE_THRESHOLD) -> bool:
    """Return True when a round's diff is below the low-impact threshold."""
    return abs(er.net_lines_changed) < threshold and er.files_changed <= 2


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_round_prompt(
    goal: str,
    round_number: int,
    previous: RoundRecord | None,
) -> str:
    """Construct the prompt sent to Codex for a given round.

    Includes context from the previous round so Codex can iterate
    meaningfully.
    """
    parts: list[str] = [
        f"## Goal\n{goal}\n",
        f"## Round {round_number}",
    ]

    if previous is not None:
        parts.append("### Previous round context")
        if previous.eval_result.test_summary:
            parts.append(f"Test result: {previous.eval_result.test_outcome.value}")
            parts.append(f"```\n{previous.eval_result.test_summary[:1000]}\n```")
        if previous.eval_result.diff_stat:
            parts.append(f"Diff stat:\n```\n{previous.eval_result.diff_stat[:500]}\n```")
        if previous.run_result.errors:
            parts.append(
                "Errors from last run:\n" + "\n".join(f"- {e[:200]}" for e in previous.run_result.errors[:5])
            )
        parts.append("")

    parts.append(
        "Please analyse the repository and make improvements towards the goal. "
        "Focus on the highest-impact changes first."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class ImprovementLoop:
    """Orchestrates repeated Codex runs with evaluation and stopping logic.

    Parameters
    ----------
    repo_path:
        Absolute path to the target git repository.
    goal:
        Human-language description of the desired improvement.
    mode:
        ``"dry-run"`` (read-only) or ``"apply"`` (Codex can modify files and
        a branch is created).
    runner:
        A :class:`CodexRunner` instance (created automatically if omitted).
    evaluator:
        A :class:`RepoEvaluator` instance (created automatically if omitted).
    stop_policy:
        Callable that receives the current :class:`LoopState` and returns a
        :class:`StopReason` or ``None`` to continue.
    budget_policy:
        Callable that checks token/cost budgets.
    """

    def __init__(
        self,
        repo_path: str | Path,
        goal: str,
        *,
        mode: str = "dry-run",
        max_rounds: int = 10,
        runner: CodexRunner | None = None,
        evaluator: RepoEvaluator | None = None,
        stop_policy: Callable[[LoopState], StopReason | None] | None = None,
        budget_policy: Callable[[LoopState], StopReason | None] | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        if not self.repo_path.is_dir():
            raise FileNotFoundError(f"Repository path does not exist: {self.repo_path}")
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"Not a git repository (no .git found): {self.repo_path}")

        normalized_mode = (mode or "").strip().lower()
        if normalized_mode not in {"dry-run", "apply"}:
            raise ValueError("mode must be 'dry-run' or 'apply'")
        try:
            parsed_max_rounds = int(max_rounds)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_rounds must be a positive integer") from exc
        if parsed_max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")

        self.goal = goal
        self.mode = normalized_mode
        self.max_rounds = parsed_max_rounds

        self.runner = runner or CodexRunner()
        self.evaluator = evaluator or RepoEvaluator()
        self.stop_policy = stop_policy or default_stop_policy
        self.budget_policy = budget_policy or default_budget_policy

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> LoopState:
        """Execute the improvement loop and return the final state."""
        state = self._init_state()
        logger.info(
            "Starting improvement loop: goal=%r, mode=%s, max_rounds=%d",
            self.goal,
            self.mode,
            self.max_rounds,
        )
        if self.mode == "dry-run":
            logger.warning(
                "SAFE MODE ACTIVE: dry-run mode is enabled; repository edits will be reverted."
            )

        for round_num in range(1, self.max_rounds + 1):
            state.current_round = round_num
            logger.info("──── Round %d / %d ────", round_num, self.max_rounds)

            # Build prompt with context from previous round
            prev = state.rounds[-1] if state.rounds else None
            prompt = build_round_prompt(self.goal, round_num, prev)

            # --- Codex run ---
            run_result = self._run_codex(prompt)
            logger.info(
                "Codex finished (exit=%d, success=%s, msg=%s...)",
                run_result.exit_code,
                run_result.success,
                run_result.final_message[:80] if run_result.final_message else "<empty>",
            )

            # --- Evaluate ---
            eval_result = self.evaluator.evaluate(self.repo_path)
            logger.info(
                "Eval: tests=%s, files_changed=%d, net_lines=%d",
                eval_result.test_outcome.value,
                eval_result.files_changed,
                eval_result.net_lines_changed,
            )

            # --- Commit (apply mode) or revert (dry-run) ---
            commit_sha = None
            if self.mode == "apply" and not is_clean(self.repo_path):
                msg = generate_commit_message(
                    round_num, prompt, eval_result.test_outcome.value
                )
                commit_sha = commit_all(self.repo_path, msg)
                logger.info("Committed %s", commit_sha)
            elif self.mode == "dry-run" and not is_clean(self.repo_path):
                revert_all(self.repo_path)
                logger.info("Reverted changes (dry-run mode)")

            # --- Record ---
            record = RoundRecord(
                round_number=round_num,
                prompt=prompt,
                run_result=run_result,
                eval_result=eval_result,
                commit_sha=commit_sha,
            )
            state.rounds.append(record)

            # Accumulate usage
            state.total_input_tokens += run_result.usage.input_tokens
            state.total_output_tokens += run_result.usage.output_tokens

            # Persist state after each round
            state.save()

            # --- Stop checks ---
            stop = self.stop_policy(state)
            if stop is not None:
                state.stop_reason = stop
                logger.info("Stopping: %s", stop.value)
                break

            budget_stop = self.budget_policy(state)
            if budget_stop is not None:
                state.stop_reason = budget_stop
                logger.info("Stopping: %s", budget_stop.value)
                break
        else:
            state.stop_reason = StopReason.MAX_ROUNDS

        state.finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
        state.save()
        logger.info("Loop finished: %s (%d rounds)", state.stop_reason, len(state.rounds))
        return state

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_state(self) -> LoopState:
        """Initialise (or resume) loop state."""
        # Try to resume existing state
        existing = LoopState.load(self.repo_path)
        if existing is not None and existing.stop_reason is None:
            logger.info("Resuming existing loop (round %d)", existing.current_round)
            return existing

        state = LoopState(
            repo_path=str(self.repo_path),
            goal=self.goal,
            mode=self.mode,
            max_rounds=self.max_rounds,
        )

        # In apply mode, create a branch
        if self.mode == "apply":
            branch = create_branch(self.repo_path)
            state.branch_name = branch

        return state

    def _run_codex(self, prompt: str) -> RunResult:
        """Invoke the Codex runner with mode-appropriate flags."""
        return self.runner.run(
            self.repo_path,
            prompt,
            full_auto=(self.mode == "apply"),
        )
