"""Regression tests for pipeline orchestrator behavior."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import ClassVar

import codex_manager.pipeline.orchestrator as orchestrator_module
import codex_manager.preflight as preflight_module
import codex_manager.research.deep_research as deep_research_module
from codex_manager.brain.manager import BrainDecision
from codex_manager.git_tools import diff_numstat
from codex_manager.pipeline.orchestrator import PipelineOrchestrator
from codex_manager.pipeline.phases import PhaseConfig, PhaseResult, PipelineConfig, PipelinePhase
from codex_manager.schemas import EvalResult, RunResult, TestOutcome, UsageInfo


def _init_git_repo(repo: Path) -> None:
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
    (repo / "README.md").write_text("init\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True, text=True
    )


def _head_sha(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _attach_bare_origin(repo: Path, tmp_path: Path) -> Path:
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", str(remote)],
        check=True,
        capture_output=True,
        text=True,
    )
    branch_result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    branch = branch_result.stdout.strip() or "master"
    subprocess.run(
        ["git", "remote", "add", "origin", str(remote)],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "push", "--set-upstream", "origin", branch],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return remote


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


class _OwnerDocRunner:
    name = "StubCodex"

    def __init__(self, *args, **kwargs):
        pass

    def run(self, repo_path, prompt, *, full_auto=False, extra_args=None):
        repo = Path(repo_path)
        target = repo / ".codex_manager" / "owner" / "TODO_WISHLIST.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("- [ ] Add one high-value improvement.\n", encoding="utf-8")
        return RunResult(
            success=True,
            exit_code=0,
            final_message="Created owner todo document.",
            usage=UsageInfo(input_tokens=2, output_tokens=3, total_tokens=5),
        )


class _PhaseWriteRunner:
    name = "StubCodex"
    calls: ClassVar[int] = 0

    def __init__(self, *args, **kwargs):
        pass

    def run(self, repo_path, prompt, *, full_auto=False, extra_args=None):
        self.__class__.calls += 1
        repo = Path(repo_path)
        target = repo / "README.md"
        existing = target.read_text(encoding="utf-8") if target.exists() else ""
        target.write_text(existing + f"edit {self.__class__.calls}\n", encoding="utf-8")
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


class _FailingChangedEvaluator:
    def __init__(self, test_cmd=None, timeout=300, skip_tests=False):
        self.test_cmd = test_cmd
        self.skip_tests = skip_tests

    def evaluate(self, repo_path):
        return EvalResult(
            test_outcome=TestOutcome.FAILED,
            test_summary="tests failed",
            test_exit_code=1,
            files_changed=1,
            net_lines_changed=4,
            changed_files=[{"path": "README.md", "insertions": 4, "deletions": 0}],
        )


class _StubUrlopenResponse:
    def __init__(self, body: bytes = b"ok"):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._body


def test_default_phase_order_inherits_global_agent():
    cfg = PipelineConfig(agent="claude_code")
    phases = cfg.get_phase_order()
    assert phases
    assert all(p.agent == "claude_code" for p in phases)


def test_default_science_order_runs_before_implementation():
    cfg = PipelineConfig(science_enabled=True)
    phase_order = [p.phase for p in cfg.get_phase_order()]

    assert phase_order.index(PipelinePhase.THEORIZE) < phase_order.index(
        PipelinePhase.IMPLEMENTATION
    )
    assert phase_order.index(PipelinePhase.ANALYZE) < phase_order.index(
        PipelinePhase.IMPLEMENTATION
    )


def test_pipeline_log_queue_drops_oldest_entries_on_overflow(caplog, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    cfg = PipelineConfig(mode="dry-run", max_cycles=1, test_cmd="")
    orch = PipelineOrchestrator(repo_path=repo, config=cfg, log_queue_maxsize=100)

    with caplog.at_level(logging.WARNING):
        for idx in range(160):
            orch._log("info", f"overflow event {idx}")

    assert orch.log_queue.qsize() == 100
    assert orch._log_queue_drops == 60

    events, replay_gap = orch.get_log_events_since(1, limit=500)
    assert replay_gap is True
    assert len(events) == 100
    assert int(events[0]["id"]) > 1
    assert int(events[-1]["id"]) == orch._next_log_event_id
    assert any("Pipeline log queue overflow" in record.getMessage() for record in caplog.records)


def test_default_deep_research_order_runs_before_prioritization():
    cfg = PipelineConfig(deep_research_enabled=True)
    phase_order = [p.phase for p in cfg.get_phase_order()]

    assert phase_order.index(PipelinePhase.DEEP_RESEARCH) < phase_order.index(
        PipelinePhase.PRIORITIZATION
    )


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
                test_policy="full",
                custom_prompt="noop",
            )
        ],
    )
    PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert _SpyEvaluator.instances
    assert _SpyEvaluator.instances[0].skip_tests is False
    assert _SpyEvaluator.instances[0].test_cmd == ["pytest", "-k", "slow suite", "-q"]


def test_smoke_policy_uses_smoke_test_command(monkeypatch, tmp_path: Path):
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
        test_cmd="pytest -q",
        smoke_test_cmd="pytest -q -m smoke",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IMPLEMENTATION,
                iterations=1,
                test_policy="smoke",
                custom_prompt="noop",
            )
        ],
    )
    PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert _SpyEvaluator.instances
    assert _SpyEvaluator.instances[0].skip_tests is False
    assert _SpyEvaluator.instances[0].test_cmd == ["pytest", "-q", "-m", "smoke"]


def test_smoke_policy_falls_back_to_full_test_command(monkeypatch, tmp_path: Path):
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
        test_cmd="pytest -q",
        smoke_test_cmd="",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IMPLEMENTATION,
                iterations=1,
                test_policy="smoke",
                custom_prompt="noop",
            )
        ],
    )
    PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert _SpyEvaluator.instances
    assert _SpyEvaluator.instances[0].skip_tests is False
    assert _SpyEvaluator.instances[0].test_cmd == ["pytest", "-q"]


def test_skip_policy_overrides_configured_test_commands(monkeypatch, tmp_path: Path):
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
        test_cmd="pytest -q",
        smoke_test_cmd="pytest -q -m smoke",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.TESTING,
                iterations=1,
                test_policy="skip",
                custom_prompt="noop",
            )
        ],
    )
    PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert _SpyEvaluator.instances
    assert _SpyEvaluator.instances[0].skip_tests is True
    assert _SpyEvaluator.instances[0].test_cmd is None


def test_implementation_phase_noop_is_validation_failure(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

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
                phase=PipelinePhase.IMPLEMENTATION,
                iterations=1,
                custom_prompt="Implement one change.",
            )
        ],
    )
    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.results
    result = state.results[0]
    assert result.agent_success is True
    assert result.validation_success is False
    assert result.tests_passed is False
    assert result.success is False
    assert "no repository changes detected" in result.error_message


def test_non_mutating_phase_noop_can_still_succeed(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

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
                custom_prompt="Brainstorm only.",
            )
        ],
    )
    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.results
    result = state.results[0]
    assert result.agent_success is True
    assert result.validation_success is True
    assert result.tests_passed is False
    assert result.success is True


def test_pipeline_counts_owner_artifacts_and_persists_debug_log(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setattr(orchestrator_module, "CodexRunner", _OwnerDocRunner)
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
                phase=PipelinePhase.IMPLEMENTATION,
                iterations=1,
                custom_prompt="Create owner TODO markdown.",
            )
        ],
    )
    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.results
    result = state.results[0]
    assert result.success is True
    assert result.validation_success is True
    assert result.files_changed >= 1
    assert result.net_lines_changed > 0
    assert any(
        str(entry.get("path", "")).startswith(".codex_manager/owner/")
        for entry in result.changed_files
    )

    debug_path = repo / ".codex_manager" / "logs" / "PIPELINE_DEBUG.jsonl"
    assert debug_path.exists()
    debug_lines = [line for line in debug_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert debug_lines
    payload = json.loads(debug_lines[-1])
    assert payload["phase"] == "implementation"
    assert payload["metrics"]["files_changed"] >= 1


def test_pipeline_prompt_logging_is_metadata_only_by_default(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setattr(orchestrator_module, "CodexRunner", _NoopRunner)
    monkeypatch.setattr(orchestrator_module, "RepoEvaluator", _SpyEvaluator)
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    secret = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz123456"
    captured_logs: list[str] = []
    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        test_cmd="",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                custom_prompt=f"Analyze this repo with api_key={secret}.",
            )
        ],
    )

    PipelineOrchestrator(
        repo_path=repo,
        config=cfg,
        log_callback=lambda _level, message: captured_logs.append(message),
    ).run()

    joined = "\n".join(captured_logs)
    assert "Prompt metadata: " in joined
    assert secret not in joined

    debug_path = repo / ".codex_manager" / "logs" / "PIPELINE_DEBUG.jsonl"
    debug_lines = [line for line in debug_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(debug_lines[-1])
    assert payload["prompt_preview"].startswith("[metadata-only] ")
    assert payload["prompt_metadata"]["redaction_hits"] >= 1
    assert payload["prompt_length"] == payload["prompt_metadata"]["length_chars"]
    assert payload["prompt_sha256"] == payload["prompt_metadata"]["sha256"]
    assert secret not in json.dumps(payload)


def test_pipeline_prompt_logging_debug_opt_in_shows_raw_prompt(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setenv("CODEX_MANAGER_PROMPT_DEBUG", "1")
    monkeypatch.setattr(orchestrator_module, "CodexRunner", _NoopRunner)
    monkeypatch.setattr(orchestrator_module, "RepoEvaluator", _SpyEvaluator)
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    secret = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz123456"
    captured_logs: list[str] = []
    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        test_cmd="",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                custom_prompt=f"Use api_key={secret} for troubleshooting.",
            )
        ],
    )

    PipelineOrchestrator(
        repo_path=repo,
        config=cfg,
        log_callback=lambda _level, message: captured_logs.append(message),
    ).run()

    assert any(secret in message for message in captured_logs)

    debug_path = repo / ".codex_manager" / "logs" / "PIPELINE_DEBUG.jsonl"
    debug_lines = [line for line in debug_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(debug_lines[-1])
    assert secret in payload["prompt_preview"]


def test_pipeline_missing_validation_command_warning_logs_once(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setattr(orchestrator_module, "CodexRunner", _NoopRunner)
    monkeypatch.setattr(orchestrator_module, "RepoEvaluator", _SpyEvaluator)
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    captured_logs: list[str] = []
    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=2,
        stop_on_convergence=False,
        test_cmd="",
        smoke_test_cmd="",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IMPLEMENTATION,
                iterations=1,
                custom_prompt="Implement one change.",
            )
        ],
    )

    PipelineOrchestrator(
        repo_path=repo,
        config=cfg,
        log_callback=lambda _level, message: captured_logs.append(message),
    ).run()

    joined = "\n".join(captured_logs)
    expected = "No validation command is configured."
    assert joined.count(expected) == 1
    assert "no matching command is configured; tests will be skipped." not in joined


def test_native_deep_research_logs_provider_prompt_previews(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    def _fake_run_native_deep_research(*, repo_path, topic, project_context, settings):
        return deep_research_module.DeepResearchRunResult(
            ok=True,
            topic=topic,
            providers=[
                deep_research_module.DeepResearchProviderResult(
                    provider="openai",
                    ok=True,
                    summary="Use atomic writes for state persistence.",
                    sources=["https://docs.python.org/3/library/pathlib.html"],
                    input_tokens=21,
                    output_tokens=34,
                    estimated_cost_usd=0.0012,
                )
            ],
            merged_summary="Atomic state writes and explicit fail-fast guards are recommended.",
            merged_sources=["https://docs.python.org/3/library/pathlib.html"],
            total_input_tokens=21,
            total_output_tokens=34,
            total_estimated_cost_usd=0.0012,
            governance_warnings=[],
            filtered_source_count=0,
            provider_prompt_previews={
                "openai": (
                    "You are running deep technical research for a software project.\n"
                    f"Topic: {topic}\n\n"
                    "Project context:\n"
                    f"{project_context}"
                )
            },
        )

    monkeypatch.setattr(
        orchestrator_module, "run_native_deep_research", _fake_run_native_deep_research
    )

    cfg = PipelineConfig(
        mode="dry-run",
        deep_research_enabled=True,
        deep_research_native_enabled=True,
        deep_research_providers="openai",
        test_cmd="",
    )
    orch = PipelineOrchestrator(repo_path=repo, config=cfg)
    result = orch._execute_native_deep_research(
        topic="Repository hardening gaps after WISH-001, WISH-002, and WISH-008",
        cycle=1,
        iteration=1,
        phase_context="## Context\nWISH-008 done; verify remaining auth/redaction gaps.",
    )

    assert result.success is True
    assert result.phase == "deep_research"
    assert result.agent_used == "deep_research:openai"

    debug_path = repo / ".codex_manager" / "logs" / "PIPELINE_DEBUG.jsonl"
    assert debug_path.exists()
    debug_lines = [line for line in debug_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert debug_lines
    payload = json.loads(debug_lines[-1])
    assert payload["phase"] == "deep_research"
    assert payload["runner"] == "deep_research:openai"

    native_payload = payload["native_deep_research"]
    assert native_payload["provider_prompt_previews"]["openai"].startswith("[metadata-only] ")
    assert native_payload["provider_prompt_metadata"]["openai"]["length_chars"] > 0
    assert native_payload["project_context_preview"].startswith("[metadata-only] ")
    assert native_payload["result"]["ok"] is True
    assert payload["prompt_length"] == payload["prompt_metadata"]["length_chars"]
    assert payload["prompt_sha256"] == payload["prompt_metadata"]["sha256"]


def test_pipeline_phase_marks_failed_tests_as_failure(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setattr(orchestrator_module, "CodexRunner", _NoopRunner)
    monkeypatch.setattr(orchestrator_module, "RepoEvaluator", _FailingChangedEvaluator)
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        test_cmd="pytest -q",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IMPLEMENTATION,
                iterations=1,
                custom_prompt="Implement and validate.",
            )
        ],
    )
    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.results
    result = state.results[0]
    assert result.agent_success is True
    assert result.validation_success is False
    assert result.tests_passed is False
    assert result.success is False
    assert "tests=failed" in result.error_message


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


def test_auto_commit_per_phase_commits_after_each_eligible_phase(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    _PhaseWriteRunner.calls = 0
    commit_calls: list[str] = []
    real_commit_all = orchestrator_module.commit_all

    monkeypatch.setattr(orchestrator_module, "CodexRunner", _PhaseWriteRunner)
    monkeypatch.setattr(
        orchestrator_module,
        "commit_all",
        lambda repo_path, msg: commit_calls.append(msg) or real_commit_all(repo_path, msg),
    )
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    cfg = PipelineConfig(
        mode="apply",
        max_cycles=1,
        auto_commit=True,
        commit_frequency="per_phase",
        test_cmd="",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IMPLEMENTATION,
                iterations=1,
                custom_prompt="Implement something.",
            ),
            PhaseConfig(
                phase=PipelinePhase.DEBUGGING,
                iterations=1,
                custom_prompt="Fix what is broken.",
            ),
        ],
    )
    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.stop_reason == "max_cycles_reached"
    assert len(commit_calls) == 2


def test_auto_commit_per_cycle_commits_once_at_cycle_end(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    _PhaseWriteRunner.calls = 0
    commit_calls: list[str] = []
    real_commit_all = orchestrator_module.commit_all

    monkeypatch.setattr(orchestrator_module, "CodexRunner", _PhaseWriteRunner)
    monkeypatch.setattr(
        orchestrator_module,
        "commit_all",
        lambda repo_path, msg: commit_calls.append(msg) or real_commit_all(repo_path, msg),
    )
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    cfg = PipelineConfig(
        mode="apply",
        max_cycles=1,
        auto_commit=True,
        commit_frequency="per_cycle",
        test_cmd="",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IMPLEMENTATION,
                iterations=1,
                custom_prompt="Implement something.",
            ),
            PhaseConfig(
                phase=PipelinePhase.DEBUGGING,
                iterations=1,
                custom_prompt="Fix what is broken.",
            ),
        ],
    )
    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.stop_reason == "max_cycles_reached"
    assert len(commit_calls) == 1
    assert any("pipeline-cycle-1" in msg for msg in commit_calls)


def test_auto_commit_manual_skips_non_commit_phase_auto_commits(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    _PhaseWriteRunner.calls = 0
    commit_calls: list[str] = []
    real_commit_all = orchestrator_module.commit_all

    monkeypatch.setattr(orchestrator_module, "CodexRunner", _PhaseWriteRunner)
    monkeypatch.setattr(
        orchestrator_module,
        "commit_all",
        lambda repo_path, msg: commit_calls.append(msg) or real_commit_all(repo_path, msg),
    )
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    cfg = PipelineConfig(
        mode="apply",
        max_cycles=1,
        auto_commit=True,
        commit_frequency="manual",
        test_cmd="",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IMPLEMENTATION,
                iterations=1,
                custom_prompt="Implement something.",
            ),
            PhaseConfig(
                phase=PipelinePhase.DEBUGGING,
                iterations=1,
                custom_prompt="Fix what is broken.",
            ),
        ],
    )
    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.stop_reason == "max_cycles_reached"
    assert len(commit_calls) == 0
    files_changed, _, _ = diff_numstat(repo)
    assert files_changed > 0


def test_pr_aware_mode_runs_on_feature_branch_and_auto_pushes(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    remote = _attach_bare_origin(repo, tmp_path)

    _PhaseWriteRunner.calls = 0
    monkeypatch.setattr(orchestrator_module, "CodexRunner", _PhaseWriteRunner)
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    feature_branch = "feature/pr-aware-sync"
    cfg = PipelineConfig(
        mode="apply",
        max_cycles=1,
        auto_commit=True,
        commit_frequency="per_phase",
        pr_aware_enabled=True,
        pr_feature_branch=feature_branch,
        pr_remote="origin",
        pr_auto_push=True,
        pr_sync_description=False,
        test_cmd="",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IMPLEMENTATION,
                iterations=1,
                custom_prompt="Implement something for PR sync mode.",
            )
        ],
    )
    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.stop_reason == "max_cycles_reached"
    branch_result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_result.stdout.strip() == feature_branch

    pushed = subprocess.run(
        ["git", "ls-remote", "--heads", str(remote), feature_branch],
        check=True,
        capture_output=True,
        text=True,
    )
    assert feature_branch in pushed.stdout
    assert state.pr_aware.get("enabled") is True
    assert state.pr_aware.get("branch") == feature_branch
    assert state.pr_aware.get("last_pushed_head")


def test_pr_aware_mode_syncs_pull_request_description(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    patched_bodies: list[str] = []

    def _stub_setup_pr(self: PipelineOrchestrator, *, repo: Path, config: PipelineConfig) -> None:
        self._pr_aware_state = {
            "enabled": True,
            "branch": "feature/pr-sync",
            "remote": "origin",
            "base_branch": "main",
            "remote_url": "https://github.com/example/demo.git",
            "owner": "example",
            "repo_name": "demo",
            "token": "ghp_test_token",
            "auto_push": False,
            "sync_description": True,
            "upstream_configured": True,
            "pull_number": 42,
            "pull_request_url": "https://github.com/example/demo/pull/42",
            "last_pushed_head": "",
            "last_push_error": "",
            "last_sync_error": "",
            "last_body_digest": "",
        }
        self._refresh_pr_aware_state()

    def _stub_github_api_request(self, *, method: str, path: str, token: str, payload=None):
        if method.upper() == "PATCH" and path.endswith("/pulls/42"):
            if isinstance(payload, dict):
                patched_bodies.append(str(payload.get("body") or ""))
            return {}, ""
        return {}, ""

    monkeypatch.setattr(orchestrator_module, "CodexRunner", _NoopRunner)
    monkeypatch.setattr(PipelineOrchestrator, "_setup_pr_aware_mode", _stub_setup_pr)
    monkeypatch.setattr(PipelineOrchestrator, "_github_api_request", _stub_github_api_request)
    monkeypatch.setattr(PipelineOrchestrator, "_preflight_issues", lambda *_a, **_k: [])

    cfg = PipelineConfig(
        mode="apply",
        max_cycles=1,
        pr_aware_enabled=True,
        pr_auto_push=False,
        pr_sync_description=True,
        test_cmd="",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                custom_prompt="Collect one idea.",
            )
        ],
    )
    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.stop_reason == "max_cycles_reached"
    assert patched_bodies
    latest = patched_bodies[-1]
    assert "WarpFoundry Pipeline Summary" in latest
    assert "Stop reason: `max_cycles_reached`" in latest


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


def test_pipeline_run_completion_webhooks_emit_payloads(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    captured_requests: list[dict[str, object]] = []

    def _fake_urlopen(request_obj, timeout=0):
        payload = {}
        if request_obj.data:
            payload = json.loads(request_obj.data.decode("utf-8"))
        captured_requests.append(
            {
                "url": request_obj.full_url,
                "method": request_obj.get_method(),
                "timeout": timeout,
                "payload": payload,
            }
        )
        return _StubUrlopenResponse()

    monkeypatch.setattr(orchestrator_module, "CodexRunner", _NoopRunner)
    monkeypatch.setattr(orchestrator_module, "urlopen", _fake_urlopen)
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        test_cmd="",
        run_completion_webhooks=[
            "https://hooks.slack.com/services/T000/B000/XXX",
            "https://discord.com/api/webhooks/123/token",
            "https://example.com/webhooks/run-complete",
        ],
        run_completion_webhook_timeout_seconds=7,
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
    assert state.run_id
    assert len(captured_requests) == 3
    assert all(item["method"] == "POST" for item in captured_requests)
    assert all(int(item["timeout"]) == 7 for item in captured_requests)

    by_url = {str(item["url"]): item for item in captured_requests}

    generic = by_url["https://example.com/webhooks/run-complete"]["payload"]
    assert generic["event"] == "warpfoundry.pipeline.run.completed"
    assert generic["repo_path"] == str(repo.resolve())
    assert generic["run_id"] == state.run_id
    assert generic["stop_reason"] == "max_cycles_reached"
    assert generic["status"] == "success"
    assert generic["tests"]["skipped"] == 1
    assert "history_jsonl" in generic["artifact_links"]
    assert "outputs_dir" in generic["artifact_links"]

    slack = by_url["https://hooks.slack.com/services/T000/B000/XXX"]["payload"]
    assert "text" in slack
    assert state.run_id in slack["text"]
    assert "Stop reason" in slack["text"]

    discord = by_url["https://discord.com/api/webhooks/123/token"]["payload"]
    assert "content" in discord
    assert state.run_id in discord["content"]
    assert "Artifacts:" in discord["content"]


def test_preflight_rejects_invalid_run_completion_webhook_url(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    monkeypatch.setattr(PipelineOrchestrator, "_binary_exists", staticmethod(lambda _binary: True))
    monkeypatch.setattr(PipelineOrchestrator, "_has_codex_auth", staticmethod(lambda: True))
    monkeypatch.setattr(PipelineOrchestrator, "_has_claude_auth", staticmethod(lambda: True))

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        run_completion_webhooks=["not-a-url"],
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                custom_prompt="noop",
            )
        ],
    )
    orch = PipelineOrchestrator(repo_path=repo, config=cfg)
    issues = orch._preflight_issues(cfg, repo.resolve())

    assert any("Invalid run-completion webhook URL" in issue for issue in issues)


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


def test_preflight_rejects_placeholder_codex_auth_env(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setattr(preflight_module.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(PipelineOrchestrator, "_binary_exists", staticmethod(lambda _binary: True))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-your-key-here")
    monkeypatch.delenv("CODEX_API_KEY", raising=False)

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                custom_prompt="noop",
            )
        ],
    )
    orch = PipelineOrchestrator(repo_path=repo, config=cfg)
    issues = orch._preflight_issues(cfg, repo.resolve())

    assert any("Codex auth not detected" in issue for issue in issues)


def test_collect_required_agents_resolves_auto_phase_to_global_agent(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    cfg = PipelineConfig(
        agent="claude_code",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                agent="auto",
                custom_prompt="noop",
            )
        ],
    )
    orch = PipelineOrchestrator(repo_path=repo, config=cfg)

    assert orch._collect_required_agents(cfg) == {"claude_code"}


def test_preflight_openai_image_provider_accepts_codex_auth(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    monkeypatch.setattr(PipelineOrchestrator, "_binary_exists", staticmethod(lambda _binary: True))
    monkeypatch.setattr(PipelineOrchestrator, "_has_codex_auth", staticmethod(lambda: True))

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        image_generation_enabled=True,
        image_provider="openai",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                custom_prompt="noop",
            )
        ],
    )
    orch = PipelineOrchestrator(repo_path=repo, config=cfg)
    issues = orch._preflight_issues(cfg, repo.resolve())

    assert not any("Image generation (openai provider)" in issue for issue in issues)


def test_preflight_rejects_placeholder_google_deep_research_auth(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    monkeypatch.setattr(PipelineOrchestrator, "_binary_exists", staticmethod(lambda _binary: True))
    monkeypatch.setattr(PipelineOrchestrator, "_has_codex_auth", staticmethod(lambda: True))
    monkeypatch.setenv("GOOGLE_API_KEY", "your-key-here")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        deep_research_enabled=True,
        deep_research_native_enabled=True,
        deep_research_providers="google",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                custom_prompt="noop",
            )
        ],
    )
    orch = PipelineOrchestrator(repo_path=repo, config=cfg)
    issues = orch._preflight_issues(cfg, repo.resolve())

    assert any("placeholder" in issue.lower() and "GOOGLE_API_KEY" in issue for issue in issues)


def test_preflight_rejects_placeholder_openai_cua_auth(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    monkeypatch.setattr(PipelineOrchestrator, "_binary_exists", staticmethod(lambda _binary: True))
    monkeypatch.setattr(PipelineOrchestrator, "_has_codex_auth", staticmethod(lambda: True))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-your-key-here")
    monkeypatch.delenv("CODEX_API_KEY", raising=False)

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        cua_enabled=True,
        cua_provider="openai",
    )
    orch = PipelineOrchestrator(repo_path=repo, config=cfg)
    issues = orch._preflight_issues(cfg, repo.resolve())

    assert any("placeholder" in issue.lower() and "OPENAI_API_KEY" in issue for issue in issues)


def test_preflight_fails_fast_when_repo_worktree_is_dirty(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    (repo / "DIRTY_PREFLIGHT.txt").write_text("dirty\n", encoding="utf-8")

    monkeypatch.setattr(PipelineOrchestrator, "_has_codex_auth", staticmethod(lambda: True))
    monkeypatch.setattr(PipelineOrchestrator, "_binary_exists", staticmethod(lambda _binary: True))

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=1,
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                custom_prompt="noop",
            )
        ],
    )
    orch = PipelineOrchestrator(repo_path=repo, config=cfg)
    issues = orch._preflight_issues(cfg, repo.resolve())
    assert any("worktree is dirty" in issue.lower() for issue in issues)

    state = orch.run()

    assert state.stop_reason == "preflight_failed"
    assert state.total_phases_completed == 0


def test_pipeline_log_init_failure_finishes_cleanly(monkeypatch, tmp_path: Path):
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
                custom_prompt="noop",
            )
        ],
    )

    orch = PipelineOrchestrator(repo_path=repo, config=cfg)

    def _raise_log_init() -> None:
        raise OSError("logs readonly")

    monkeypatch.setattr(orch.tracker, "initialize", _raise_log_init)

    state = orch.run()
    assert state.running is False
    assert state.stop_reason is not None
    assert state.stop_reason.startswith("error:")
    assert "logs readonly" in state.stop_reason
    assert state.finished_at


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
    report_file = repo / ".codex_manager" / "logs" / "SCIENTIST_REPORT.md"
    assert (science_dir / "TRIALS.jsonl").exists()
    assert (science_dir / "EVIDENCE.md").exists()
    assert (science_dir / "EXPERIMENTS_LATEST.md").exists()
    assert report_file.exists()
    assert "Science Trial Timeline" in report_file.read_text(encoding="utf-8")

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
        agent_success=True,
        validation_success=True,
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
        agent_success=True,
        validation_success=True,
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
        agent_success=True,
        validation_success=True,
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
        lambda repo_path, msg: commit_calls.append(msg) or "deadbeef",
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


def test_unlimited_cycle_logs_and_completion_messages_are_ascii_safe(monkeypatch, tmp_path: Path):
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
    assert any(msg.startswith("Pipeline finished - budget_exhausted") for msg in captured_logs)
    for msg in captured_logs:
        msg.encode("ascii")


def test_self_improvement_phase_requests_restart_and_writes_checkpoint(
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

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=2,
        test_cmd="",
        self_improvement_enabled=True,
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                custom_prompt="Run ideation.",
            ),
            PhaseConfig(
                phase=PipelinePhase.APPLY_UPGRADES_AND_RESTART,
                iterations=1,
                custom_prompt="Prepare restart checkpoint.",
            ),
            PhaseConfig(
                phase=PipelinePhase.TESTING,
                iterations=1,
                custom_prompt="Run tests.",
            ),
        ],
    )
    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.stop_reason == "self_restart_requested"
    assert state.restart_required is True
    assert state.resume_cycle == 1
    assert state.resume_phase_index == 2
    assert any(r.phase == "apply_upgrades_and_restart" for r in state.results)

    checkpoint = Path(state.restart_checkpoint_path)
    assert checkpoint.is_file()

    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["repo_path"] == str(repo.resolve())
    assert payload["resume_cycle"] == 1
    assert payload["resume_phase_index"] == 2
    assert payload["config"]["self_improvement_enabled"] is True


def test_self_improvement_phase_recovers_backlog_into_progress_and_vector_memory(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    owner_dir = repo / ".codex_manager" / "owner"
    owner_dir.mkdir(parents=True, exist_ok=True)
    (owner_dir / "TODO_WISHLIST.md").write_text(
        "- [ ] Carry forward unfinished archive task\n",
        encoding="utf-8",
    )
    archive_dir = repo / ".codex_manager" / "output_history" / "20260217T020202Z"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "RUN_SUMMARY.md").write_text(
        "- [ ] Validate archived experiment outcome\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(orchestrator_module, "CodexRunner", _NoopRunner)
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=2,
        test_cmd="",
        self_improvement_enabled=True,
        vector_memory_enabled=True,
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                custom_prompt="Run ideation.",
            ),
            PhaseConfig(
                phase=PipelinePhase.APPLY_UPGRADES_AND_RESTART,
                iterations=1,
                custom_prompt="Prepare restart checkpoint.",
            ),
            PhaseConfig(
                phase=PipelinePhase.TESTING,
                iterations=1,
                custom_prompt="Run tests.",
            ),
        ],
    )
    state = PipelineOrchestrator(repo_path=repo, config=cfg).run()

    assert state.stop_reason == "self_restart_requested"
    progress = (repo / ".codex_manager" / "logs" / "PROGRESS.md").read_text(encoding="utf-8")
    assert "Recovered Backlog Snapshot" in progress
    assert "Carry forward unfinished archive task" in progress

    vector_events = repo / ".codex_manager" / "memory" / "vector_events.jsonl"
    assert vector_events.exists()
    rows = [
        json.loads(line)
        for line in vector_events.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        isinstance(row.get("metadata"), dict)
        and row["metadata"].get("category") == "recovered_backlog"
        for row in rows
    )


def test_resume_cycle_and_phase_index_starts_from_checkpoint_position(
    monkeypatch, tmp_path: Path
):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    class _RecordPromptRunner:
        name = "StubCodex"
        prompts: ClassVar[list[str]] = []

        def __init__(self, *args, **kwargs):
            pass

        def run(self, repo_path, prompt, *, full_auto=False, extra_args=None):
            self.__class__.prompts.append(prompt)
            return RunResult(
                success=True,
                exit_code=0,
                usage=UsageInfo(input_tokens=1, output_tokens=1, total_tokens=2),
            )

    _RecordPromptRunner.prompts.clear()
    monkeypatch.setattr(orchestrator_module, "CodexRunner", _RecordPromptRunner)
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_preflight_issues",
        lambda self, config, repo_path: [],
    )

    cfg = PipelineConfig(
        mode="dry-run",
        max_cycles=2,
        test_cmd="",
        phases=[
            PhaseConfig(
                phase=PipelinePhase.IDEATION,
                iterations=1,
                custom_prompt="Run ideation.",
            ),
            PhaseConfig(
                phase=PipelinePhase.TESTING,
                iterations=1,
                custom_prompt="Run tests.",
            ),
        ],
    )

    state = PipelineOrchestrator(
        repo_path=repo,
        config=cfg,
        resume_cycle=2,
        resume_phase_index=1,
    ).run()

    assert state.stop_reason == "max_cycles_reached"
    assert state.total_cycles_completed == 2
    assert len(state.results) == 1
    assert state.results[0].cycle == 2
    assert state.results[0].phase == "testing"
    assert _RecordPromptRunner.prompts
    assert "Pipeline Cycle 2, Phase: testing" in _RecordPromptRunner.prompts[0]


def test_orchestrator_source_has_no_known_mojibake_sequences() -> None:
    source_text = Path(orchestrator_module.__file__).read_text(encoding="utf-8")

    assert "\u00e2\u20ac\u201d" not in source_text  # "â€”"
    assert "\u00e2\u201d\x81" not in source_text  # corrupted line separator
    assert "\u00e2\u02c6\u017e" not in source_text  # "âˆž"
    assert "\u00ce\u201d" not in source_text  # "Î”"
    assert not any("\u0080" <= ch <= "\u009f" for ch in source_text)
