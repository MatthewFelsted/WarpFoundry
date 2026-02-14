"""Regression tests for pipeline orchestrator behavior."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import ClassVar

import codex_manager.pipeline.orchestrator as orchestrator_module
from codex_manager.brain.manager import BrainDecision
from codex_manager.git_tools import diff_numstat
from codex_manager.pipeline.orchestrator import PipelineOrchestrator
from codex_manager.pipeline.phases import PhaseConfig, PhaseResult, PipelineConfig, PipelinePhase
from codex_manager.schemas import EvalResult, RunResult, TestOutcome, UsageInfo


def _init_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True, capture_output=True, text=True)
    (repo / "README.md").write_text("init\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True, text=True)


def _head_sha(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class _WriteRunner:
    name = "StubCodex"

    def __init__(self, *args, **kwargs):
        pass

    def run(self, repo_path, prompt, *, full_auto=False, extra_args=None):
        repo = Path(repo_path)
        (repo / "leaked.txt").write_text("leak\n", encoding="utf-8")
        return RunResult(
            success=True,
            exit_code=0,
            usage=UsageInfo(input_tokens=1, output_tokens=1, total_tokens=2),
        )


class _CommitRunner:
    name = "StubCodex"

    def __init__(self, *args, **kwargs):
        pass

    def run(self, repo_path, prompt, *, full_auto=False, extra_args=None):
        repo = Path(repo_path)
        target = repo / "src" / "pipeline_agent_commit.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("def pipeline_added():\n    return 'ok'\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "commit", "-m", "pipeline runner-authored commit"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return RunResult(
            success=True,
            exit_code=0,
            final_message="Committed one pipeline change.",
            usage=UsageInfo(input_tokens=3, output_tokens=4, total_tokens=7),
        )


class _NoopRunner:
    name = "StubCodex"

    def __init__(self, *args, **kwargs):
        pass

    def run(self, repo_path, prompt, *, full_auto=False, extra_args=None):
        return RunResult(
            success=True,
            exit_code=0,
            usage=UsageInfo(input_tokens=1, output_tokens=1, total_tokens=2),
        )


class _CaptureAutoRunner:
    name = "StubCodex"
    full_auto_calls: ClassVar[list[bool]] = []

    def __init__(self, *args, **kwargs):
        pass

    def run(self, repo_path, prompt, *, full_auto=False, extra_args=None):
        self.__class__.full_auto_calls.append(bool(full_auto))
        return RunResult(
            success=True,
            exit_code=0,
            usage=UsageInfo(input_tokens=1, output_tokens=1, total_tokens=2),
        )


class _TerminateIterationRunner:
    name = "StubCodex"
    calls: ClassVar[int] = 0

    def __init__(self, *args, **kwargs):
        pass

    def run(self, repo_path, prompt, *, full_auto=False, extra_args=None):
        self.__class__.calls += 1
        return RunResult(
            success=True,
            exit_code=0,
            final_message="No more useful work for this phase.\n[TERMINATE_STEP]\n",
            usage=UsageInfo(input_tokens=1, output_tokens=1, total_tokens=2),
        )


class _FollowUpRunner:
    name = "StubCodex"
    calls: ClassVar[int] = 0

    def __init__(self, *args, **kwargs):
        pass

    def run(self, repo_path, prompt, *, full_auto=False, extra_args=None):
        self.__class__.calls += 1
        return RunResult(
            success=True,
            exit_code=0,
            final_message="ok\n",
            usage=UsageInfo(input_tokens=1, output_tokens=1, total_tokens=2),
        )


class _FollowUpBrain:
    def __init__(self, config):
        self.config = config
        self.enabled = bool(config.enabled)
        self._decisions = 0

    def plan_step(self, goal, step_name, base_prompt, history_summary="", ledger_context=""):
        return base_prompt

    def evaluate_step(
        self,
        step_name,
        success,
        test_outcome,
        files_changed,
        net_lines,
        errors,
        goal,
        ledger_context="",
    ):
        self._decisions += 1
        if self._decisions == 1:
            return BrainDecision(
                action="follow_up",
                reasoning="Need one focused follow-up before continuing.",
                follow_up_prompt="Run one focused follow-up pass.",
                severity="medium",
            )
        return BrainDecision(action="continue", reasoning="Continue.")

    def assess_progress(self, goal, total_loops, history_summary):
        return BrainDecision(action="continue", reasoning="continue")


class _ContinueBrain:
    def __init__(self, config):
        self.config = config
        self.enabled = bool(config.enabled)

    def plan_step(self, goal, step_name, base_prompt, history_summary="", ledger_context=""):
        return base_prompt

    def evaluate_step(
        self,
        step_name,
        success,
        test_outcome,
        files_changed,
        net_lines,
        errors,
        goal,
        ledger_context="",
    ):
        return BrainDecision(action="continue", reasoning="Continue.")

    def assess_progress(self, goal, total_loops, history_summary):
        return BrainDecision(action="continue", reasoning="continue")


class _ScienceRunner:
    name = "StubCodex"

    def __init__(self, *args, **kwargs):
        pass

    def run(self, repo_path, prompt, *, full_auto=False, extra_args=None):
        repo = Path(repo_path)
        marker = repo / "README.md"
        prev = marker.read_text(encoding="utf-8") if marker.exists() else ""
        marker.write_text(prev + "x\n", encoding="utf-8")

        final = ""
        if "Phase: skeptic" in prompt:
            final = (
                "SKEPTIC_VERDICT: supported\n"
                "SKEPTIC_CONFIDENCE: high\n"
                "SKEPTIC_RATIONALE: independent replication confirmed the effect.\n"
            )

        return RunResult(
            success=True,
            exit_code=0,
            usage=UsageInfo(input_tokens=3, output_tokens=2, total_tokens=5),
            final_message=final,
        )


class _SpyEvaluator:
    instances: ClassVar[list[_SpyEvaluator]] = []

    def __init__(self, test_cmd=None, timeout=300, skip_tests=False):
        self.test_cmd = test_cmd
        self.skip_tests = skip_tests
        _SpyEvaluator.instances.append(self)

    def evaluate(self, repo_path):
        return EvalResult(
            test_outcome=TestOutcome.SKIPPED,
            test_summary="skipped",
            test_exit_code=0,
            files_changed=0,
            net_lines_changed=0,
        )


def test_default_phase_order_inherits_global_agent():
    cfg = PipelineConfig(agent="claude_code")
    phases = cfg.get_phase_order()
    assert phases
    assert all(p.agent == "claude_code" for p in phases)


def test_dry_run_commit_phase_reverts_changes(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setattr(orchestrator_module, "CodexRunner", _WriteRunner)
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        test_cmd="",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.COMMIT,
                iterations=1,
                custom_prompt="Create a throwaway edit.",
            )
        ],
    )
    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.stop_reason == "max_cycles_reached"
    assert not (repo / "leaked.txt").exists()
    files_changed, _, _ = diff_numstat(repo)
    assert files_changed == 0


def test_dry_run_rolls_back_agent_authored_commit(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    baseline_head = _head_sha(repo)

    monkeypatch.setattr(orchestrator_module, "CodexRunner", _CommitRunner)
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        test_cmd="",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                custom_prompt="Implement and commit one feature.",
            )
        ],
    )
    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.results
    result = state.results[0]
    assert result.success is True
    assert result.commit_sha is None
    assert result.files_changed >= 1
    assert any(
        str(item.get("path", "")).endswith("src/pipeline_agent_commit.py")
        for item in result.changed_files
    )

    assert _head_sha(repo) == baseline_head
    assert not (repo / "src" / "pipeline_agent_commit.py").exists()
    files_changed, _, _ = diff_numstat(repo)
    assert files_changed == 0


def test_improvement_check_uses_full_cycle_results_with_iterations(tmp_path: Path):
    cfg = PipelineConfig(
        improvement_threshold=60.0,
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=3,
                custom_prompt="noop",
            )
        ],
    )
    orch = PipelineOrchestrator(repo_path=tmp_path, config=cfg)
    orch.state.total_cycles_completed = 2
    orch.state.results = [
        PhaseResult(
            cycle=1,
            phase="ideation",
            iteration=1,
            success=True,
            test_outcome="passed",
            files_changed=10,
            net_lines_changed=100,
        ),
        PhaseResult(
            cycle=1,
            phase="ideation",
            iteration=2,
            success=True,
            test_outcome="passed",
            files_changed=10,
            net_lines_changed=100,
        ),
        PhaseResult(
            cycle=1,
            phase="ideation",
            iteration=3,
            success=True,
            test_outcome="passed",
            files_changed=10,
            net_lines_changed=100,
        ),
        PhaseResult(
            cycle=2,
            phase="ideation",
            iteration=1,
            success=True,
            test_outcome="passed",
            files_changed=10,
            net_lines_changed=100,
        ),
        PhaseResult(
            cycle=2,
            phase="ideation",
            iteration=2,
            success=True,
            test_outcome="passed",
            files_changed=10,
            net_lines_changed=100,
        ),
        PhaseResult(
            cycle=2,
            phase="ideation",
            iteration=3,
            success=True,
            test_outcome="passed",
            files_changed=0,
            net_lines_changed=0,
        ),
    ]

    stop = orch._check_stop_conditions(cfg, start_time=time.monotonic())
    assert stop is None
    assert orch.state.improvement_pct > cfg.improvement_threshold


def test_empty_test_command_skips_tests(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    _SpyEvaluator.instances.clear()
    monkeypatch.setattr(orchestrator_module, "CodexRunner", _NoopRunner)
    monkeypatch.setattr(orchestrator_module, "RepoEvaluator", _SpyEvaluator)
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        test_cmd="",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                custom_prompt="noop",
            )
        ],
    )
    PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert _SpyEvaluator.instances
    assert _SpyEvaluator.instances[0].skip_tests is True
    assert _SpyEvaluator.instances[0].test_cmd is None


def test_quoted_test_command_is_parsed(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    _SpyEvaluator.instances.clear()
    monkeypatch.setattr(orchestrator_module, "CodexRunner", _NoopRunner)
    monkeypatch.setattr(orchestrator_module, "RepoEvaluator", _SpyEvaluator)
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        test_cmd='pytest -k "slow suite" -q',
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                custom_prompt="noop",
            )
        ],
    )
    PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert _SpyEvaluator.instances
    assert _SpyEvaluator.instances[0].skip_tests is False
    assert _SpyEvaluator.instances[0].test_cmd == ["pytest", "-k", "slow suite", "-q"]


def test_pipeline_phase_execution_is_non_interactive(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    _CaptureAutoRunner.full_auto_calls.clear()
    monkeypatch.setattr(orchestrator_module, "CodexRunner", _CaptureAutoRunner)
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        test_cmd="",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                custom_prompt="noop",
            )
        ],
    )

    PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert _CaptureAutoRunner.full_auto_calls
    assert all(_CaptureAutoRunner.full_auto_calls)


def test_phase_terminate_tag_skips_remaining_iterations(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    _TerminateIterationRunner.calls = 0
    monkeypatch.setattr(orchestrator_module, "CodexRunner", _TerminateIterationRunner)
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        test_cmd="",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=3,
                custom_prompt="Run ideation once and stop repeats if no-op.",
            )
        ],
    )

    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.results
    assert len(state.results) == 1
    assert state.results[0].terminate_repeats is True
    assert _TerminateIterationRunner.calls == 1


def test_brain_follow_up_executes_extra_phase_and_logs(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    _FollowUpRunner.calls = 0
    monkeypatch.setattr(orchestrator_module, "CodexRunner", _FollowUpRunner)
    monkeypatch.setattr(orchestrator_module, "BrainManager", _FollowUpBrain)
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        test_cmd="",
        brain_enabled=True,
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                custom_prompt="Run ideation.",
            )
        ],
    )
    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.stop_reason == "max_cycles_reached"
    assert _FollowUpRunner.calls == 2
    assert state.total_phases_completed == 2
    assert len(state.results) == 2

    brain_log = repo / ".codex_manager" / "logs" / "BRAIN.md"
    assert brain_log.exists()
    content = brain_log.read_text(encoding="utf-8")
    assert "follow_up_started" in content
    assert "follow_up_finished" in content


def test_pipeline_writes_history_log_entries(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setattr(orchestrator_module, "CodexRunner", _NoopRunner)
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        test_cmd="",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                custom_prompt="Run ideation.",
            )
        ],
    )
    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.stop_reason == "max_cycles_reached"
    assert state.total_phases_completed == 1

    history_md = repo / ".codex_manager" / "logs" / "HISTORY.md"
    history_jsonl = repo / ".codex_manager" / "logs" / "HISTORY.jsonl"
    assert history_md.exists()
    assert history_jsonl.exists()

    markdown = history_md.read_text(encoding="utf-8")
    assert "run_started" in markdown
    assert "phase_result" in markdown
    assert "run_finished" in markdown

    lines = [ln for ln in history_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 3
    events = [json.loads(ln).get("event") for ln in lines]
    assert "run_started" in events
    assert "phase_result" in events
    assert "run_finished" in events


def test_brain_logbook_init_failure_does_not_abort_pipeline(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    class _FailingLogbook:
        def __init__(self, _repo):
            pass

        def initialize(self):
            raise OSError("disk denied")

    monkeypatch.setattr(orchestrator_module, "CodexRunner", _NoopRunner)
    monkeypatch.setattr(orchestrator_module, "BrainManager", _ContinueBrain)
    monkeypatch.setattr(orchestrator_module, "BrainLogbook", _FailingLogbook)
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        test_cmd="",
        brain_enabled=True,
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                custom_prompt="Run ideation.",
            )
        ],
    )
    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.stop_reason == "max_cycles_reached"
    assert state.total_phases_completed == 1
    assert len(state.results) == 1


def test_preflight_fails_fast_when_codex_binary_missing(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    monkeypatch.setattr(PipelineOrchestrator, "_has_codex_auth", staticmethod(lambda: True))

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        codex_binary="definitely-not-a-real-codex-binary",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                custom_prompt="noop",
            )
        ],
    )
    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.stop_reason == "preflight_failed"
    assert state.total_phases_completed == 0


def test_preflight_rejects_directory_path_as_codex_binary(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    fake_binary_dir = tmp_path / "not-a-binary"
    fake_binary_dir.mkdir()
    monkeypatch.setattr(PipelineOrchestrator, "_has_codex_auth", staticmethod(lambda: True))

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        codex_binary=str(fake_binary_dir),
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                custom_prompt="noop",
            )
        ],
    )
    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.stop_reason == "preflight_failed"
    assert state.total_phases_completed == 0


def test_science_experiment_records_evidence_and_marks_inconclusive(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    _SpyEvaluator.instances.clear()
    monkeypatch.setattr(orchestrator_module, "CodexRunner", _NoopRunner)
    monkeypatch.setattr(orchestrator_module, "RepoEvaluator", _SpyEvaluator)
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        science_enabled=True,
        test_cmd="",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.EXPERIMENT,
                iterations=1,
                custom_prompt="Run one controlled experiment.",
            )
        ],
    )
    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.results
    result = state.results[0]
    assert result.phase == "experiment"
    assert result.success is False
    assert "Science verdict=inconclusive" in result.error_message

    science_dir = repo / ".codex_manager" / "logs" / "scientist"
    assert (science_dir / "TRIALS.jsonl").exists()
    assert (science_dir / "EVIDENCE.md").exists()
    assert (science_dir / "EXPERIMENTS_LATEST.md").exists()

    lines = (science_dir / "TRIALS.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    payload = json.loads(lines[-1])
    assert payload["phase"] == "experiment"
    assert payload["verdict"] == "inconclusive"


def test_record_science_trial_rolls_back_failed_experiment(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    cfg = PipelineConfig(mode="apply", max_cycles=1, test_cmd="")
    orch = PipelineOrchestrator(repo_path=repo, config=cfg)
    orch.tracker.initialize()
    orch.tracker.write(
        "EXPERIMENTS.md",
        (
            "### [EXP-001] Improve reliability\n"
            "- **Status**: testing\n"
            "- **Hypothesis**: If we change X then Y improves\n"
            "- **Success Criteria**: tests do not regress\n"
            "- **Confidence**: medium\n"
        ),
    )

    # Make repo dirty so rollback has work to do.
    (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    assert (repo / "dirty.txt").exists()

    baseline = EvalResult(
        test_outcome=TestOutcome.SKIPPED,
        test_summary="baseline",
        test_exit_code=0,
        files_changed=0,
        net_lines_changed=0,
    )
    result = PhaseResult(
        cycle=1,
        phase=PipelinePhase.EXPERIMENT.value,
        iteration=1,
        success=True,
        test_outcome="skipped",
        test_summary="post",
        test_exit_code=0,
        files_changed=0,
        net_lines_changed=0,
        agent_used="codex",
        agent_final_message="no measurable changes",
    )

    trial = orch._record_science_trial(
        cycle=1,
        phase=PipelinePhase.EXPERIMENT,
        iteration=1,
        prompt="experiment prompt",
        result=result,
        baseline_eval=baseline,
        repo=repo,
        baseline_repo_clean=True,
        selected_hypothesis={
            "id": "EXP-001",
            "title": "Improve reliability",
            "status": "testing",
            "hypothesis": "If we change X then Y improves",
            "success_criteria": "tests do not regress",
            "confidence": "medium",
            "block": "",
        },
    )

    assert trial["verdict"] == "inconclusive"
    assert trial["rollback_action"] == "reverted"
    assert result.science_rolled_back is True
    assert result.success is False
    assert "Science verdict=inconclusive" in result.error_message
    assert not (repo / "dirty.txt").exists()


def test_record_science_trial_ids_are_unique(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    cfg = PipelineConfig(mode="apply", max_cycles=1, test_cmd="")
    orch = PipelineOrchestrator(repo_path=repo, config=cfg)
    orch.tracker.initialize()
    orch.tracker.write(
        "EXPERIMENTS.md",
        (
            "### [EXP-001] Improve reliability\n"
            "- **Status**: testing\n"
            "- **Hypothesis**: If we change X then Y improves\n"
            "- **Success Criteria**: tests do not regress\n"
            "- **Confidence**: high\n"
        ),
    )

    baseline = EvalResult(
        test_outcome=TestOutcome.SKIPPED,
        test_summary="baseline",
        test_exit_code=0,
        files_changed=0,
        net_lines_changed=0,
    )
    result_a = PhaseResult(
        cycle=1,
        phase=PipelinePhase.EXPERIMENT.value,
        iteration=1,
        success=True,
        test_outcome="skipped",
        test_summary="post",
        test_exit_code=0,
        files_changed=1,
        net_lines_changed=5,
        agent_used="codex",
    )
    result_b = PhaseResult(
        cycle=1,
        phase=PipelinePhase.EXPERIMENT.value,
        iteration=2,
        success=True,
        test_outcome="skipped",
        test_summary="post",
        test_exit_code=0,
        files_changed=1,
        net_lines_changed=6,
        agent_used="codex",
    )

    trial_a = orch._record_science_trial(
        cycle=1,
        phase=PipelinePhase.EXPERIMENT,
        iteration=1,
        prompt="experiment prompt A",
        result=result_a,
        baseline_eval=baseline,
        repo=repo,
        baseline_repo_clean=True,
        selected_hypothesis={
            "id": "EXP-001",
            "title": "Improve reliability",
            "status": "testing",
            "hypothesis": "If we change X then Y improves",
            "success_criteria": "tests do not regress",
            "confidence": "high",
            "block": "",
        },
    )
    trial_b = orch._record_science_trial(
        cycle=1,
        phase=PipelinePhase.EXPERIMENT,
        iteration=2,
        prompt="experiment prompt B",
        result=result_b,
        baseline_eval=baseline,
        repo=repo,
        baseline_repo_clean=True,
        selected_hypothesis={
            "id": "EXP-001",
            "title": "Improve reliability",
            "status": "testing",
            "hypothesis": "If we change X then Y improves",
            "success_criteria": "tests do not regress",
            "confidence": "high",
            "block": "",
        },
    )

    assert trial_a["trial_id"] != trial_b["trial_id"]
    assert trial_a["experiment_id"] != trial_b["experiment_id"]
    assert result_a.science_trial_id != result_b.science_trial_id
    assert result_a.science_experiment_id != result_b.science_experiment_id


def test_science_auto_commit_runs_only_after_supported_skeptic(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    commit_calls: list[str] = []
    monkeypatch.setattr(orchestrator_module, "CodexRunner", _ScienceRunner)
    monkeypatch.setattr(orchestrator_module, "create_branch", lambda repo_path: "science/test")
    monkeypatch.setattr(
        orchestrator_module,
        "commit_all",
        lambda repo_path, msg: (commit_calls.append(msg) or "deadbeef"),
    )
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    cfg = PipelineConfig(
        mode="apply",
        max_cycles=1,
        science_enabled=True,
        auto_commit=True,
        test_cmd="",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.EXPERIMENT,
                iterations=1,
                custom_prompt="Run controlled experiment.",
            ),
            PhaseConfig(
                phase=PipelinePhase.SKEPTIC,
                iterations=1,
                custom_prompt="Challenge and replicate.",
            ),
        ],
    )
    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.results
    assert any(r.phase == "experiment" for r in state.results)
    assert any(r.phase == "skeptic" for r in state.results)
    assert len(commit_calls) == 1


def test_non_strict_budget_stops_after_phase_boundary(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setattr(orchestrator_module, "CodexRunner", _NoopRunner)
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=3,
        max_total_tokens=3,
        strict_token_budget=False,
        test_cmd="",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=2,
                custom_prompt="noop",
            ),
            PhaseConfig(
                phase=PipelinePhase.TESTING,
                iterations=1,
                custom_prompt="noop",
            ),
        ],
    )
    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.stop_reason == "budget_exhausted"
    assert all(r.phase == "ideation" for r in state.results)
    assert len(state.results) == 2


def test_unlimited_cycle_logs_and_completion_messages_are_ascii_safe(
    monkeypatch, tmp_path: Path
):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setattr(orchestrator_module, "CodexRunner", _NoopRunner)
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    captured_logs: list[str] = []
    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=3,
        unlimited=True,
        max_total_tokens=1,
        strict_token_budget=True,
        test_cmd="",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                custom_prompt="noop",
            )
        ],
    )
    state = PipelineOrchestrator(
        repo_path=repo,
        config=cfg,
        log_callback=lambda _level, message: captured_logs.append(message),
    ).run()

    assert state.stop_reason == "budget_exhausted"
    assert ("=" * 20) + " Cycle 1 / inf " + ("=" * 20) in captured_logs
    assert any(
        msg.startswith("Pipeline finished - budget_exhausted")
        for msg in captured_logs
    )
    for msg in captured_logs:
        msg.encode("ascii")


def test_orchestrator_source_has_no_known_mojibake_sequences() -> None:
    source_text = Path(orchestrator_module.__file__).read_text(encoding="utf-8")

    assert "\u00e2\u20ac\u201d" not in source_text  # "â€”"
    assert "\u00e2\u201d\x81" not in source_text    # corrupted line separator
    assert "\u00e2\u02c6\u017e" not in source_text  # "âˆž"
    assert "\u00ce\u201d" not in source_text        # "Î”"
    assert not any("\u0080" <= ch <= "\u009f" for ch in source_text)
