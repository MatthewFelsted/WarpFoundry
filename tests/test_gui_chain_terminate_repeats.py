"""Tests for repeat-skipping behavior in chain execution."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import ClassVar

import codex_manager.gui.chain as chain_module
from codex_manager.brain.manager import BrainDecision
from codex_manager.git_tools import diff_numstat
from codex_manager.gui.chain import ChainExecutor
from codex_manager.gui.models import ChainConfig, TaskStep
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


class _TerminateRunner:
    name = "StubCodex"
    calls: ClassVar[int] = 0

    def __init__(self, *args, **kwargs):
        pass

    def run(self, repo_path, prompt, *, full_auto=False, extra_args=None):
        self.__class__.calls += 1
        return RunResult(
            success=True,
            exit_code=0,
            final_message="No additional work for repeats.\n[TERMINATE_STEP]\n",
            usage=UsageInfo(input_tokens=1, output_tokens=1, total_tokens=2),
        )


class _OutputRunner:
    name = "StubCodex"

    def __init__(self, *args, **kwargs):
        pass

    def run(self, repo_path, prompt, *, full_auto=False, extra_args=None):
        return RunResult(
            success=True,
            exit_code=0,
            final_message="Captured concrete findings for the next step to consume.",
            usage=UsageInfo(input_tokens=3, output_tokens=5, total_tokens=8),
        )


class _PlaceholderRunner:
    name = "StubCodex"

    def __init__(self, *args, **kwargs):
        pass

    def run(self, repo_path, prompt, *, full_auto=False, extra_args=None):
        return RunResult(
            success=True,
            exit_code=0,
            final_message="Working in `C:\\repo` now. Share the task you want implemented and I'll proceed.",
            usage=UsageInfo(input_tokens=1, output_tokens=1, total_tokens=2),
        )


class _NoopEvaluator:
    instances: ClassVar[list[_NoopEvaluator]] = []

    def __init__(self, test_cmd=None, skip_tests=False):
        self.test_cmd = test_cmd
        self.skip_tests = skip_tests
        self.__class__.instances.append(self)

    def evaluate(self, repo_path):
        return EvalResult(
            test_outcome=TestOutcome.SKIPPED,
            test_summary="skipped",
            test_exit_code=0,
            files_changed=0,
            net_lines_changed=0,
        )


class _FailingHistoryLogbook:
    def __init__(self, _repo: Path):
        pass

    def initialize(self) -> None:
        raise OSError("history denied")


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
        return BrainDecision(action="continue", reasoning="ok", severity="low")

    def assess_progress(self, goal, total_loops, history_summary):
        return BrainDecision(action="continue", reasoning="ok")


class _FollowUpBrainNeedsInput:
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
        return BrainDecision(
            action="follow_up",
            reasoning="Need missing context before implementation.",
            follow_up_prompt=(
                "Please provide: (1) exact requirements, (2) target files, and (3) test command."
            ),
            severity="high",
        )

    def assess_progress(self, goal, total_loops, history_summary):
        return BrainDecision(action="continue", reasoning="ok")


class _ChangedEvaluator:
    def __init__(self, test_cmd=None, skip_tests=False):
        self.test_cmd = test_cmd
        self.skip_tests = skip_tests

    def evaluate(self, repo_path):
        return EvalResult(
            test_outcome=TestOutcome.SKIPPED,
            test_summary="skipped",
            test_exit_code=0,
            files_changed=2,
            net_lines_changed=11,
            changed_files=[
                {"path": "src/app.py", "insertions": 7, "deletions": 1},
                {"path": "tests/test_app.py", "insertions": 5, "deletions": 0},
            ],
        )


class _FailingChangedEvaluator:
    def __init__(self, test_cmd=None, skip_tests=False):
        self.test_cmd = test_cmd
        self.skip_tests = skip_tests

    def evaluate(self, repo_path):
        return EvalResult(
            test_outcome=TestOutcome.FAILED,
            test_summary="tests failed",
            test_exit_code=1,
            files_changed=1,
            net_lines_changed=3,
            changed_files=[
                {"path": "src/app.py", "insertions": 3, "deletions": 0},
            ],
        )


class _CommitRunner:
    name = "StubCodex"

    def __init__(self, *args, **kwargs):
        pass

    def run(self, repo_path, prompt, *, full_auto=False, extra_args=None):
        repo = Path(repo_path)
        target = repo / "src" / "agent_commit_feature.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("def added_by_runner():\n    return 'ok'\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "commit", "-m", "runner-authored feature commit"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return RunResult(
            success=True,
            exit_code=0,
            final_message="Implemented a feature and committed it.",
            usage=UsageInfo(input_tokens=5, output_tokens=9, total_tokens=14),
        )


def test_chain_skips_remaining_repeats_after_terminate_tag(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    _TerminateRunner.calls = 0
    monkeypatch.setattr(chain_module, "CodexRunner", _TerminateRunner)
    monkeypatch.setattr(chain_module, "RepoEvaluator", _NoopEvaluator)
    monkeypatch.setattr(chain_module, "ensure_git_identity", lambda _repo: None)

    config = ChainConfig(
        name="Repeat skip test",
        repo_path=str(repo),
        mode="dry-run",
        max_loops=1,
        test_cmd="",
        steps=[
            TaskStep(
                name="Implementation",
                job_type="implementation",
                prompt_mode="custom",
                custom_prompt="Do the step once and signal if repeats are unnecessary.",
                loop_count=3,
                enabled=True,
                agent="codex",
            )
        ],
    )

    executor = ChainExecutor()
    executor.config = config
    executor.state.running = True
    executor._run_loop()

    assert _TerminateRunner.calls == 1
    assert executor.state.total_steps_completed == 1
    assert executor.state.results
    assert executor.state.results[0].terminate_repeats is True


def test_chain_parses_quoted_test_cmd(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    _TerminateRunner.calls = 0
    _NoopEvaluator.instances.clear()
    monkeypatch.setattr(chain_module, "CodexRunner", _TerminateRunner)
    monkeypatch.setattr(chain_module, "RepoEvaluator", _NoopEvaluator)
    monkeypatch.setattr(chain_module, "ensure_git_identity", lambda _repo: None)

    config = ChainConfig(
        name="Quoted test cmd",
        repo_path=str(repo),
        mode="dry-run",
        max_loops=1,
        test_cmd='pytest -k "slow suite" -q',
        steps=[
            TaskStep(
                name="Implementation",
                job_type="implementation",
                prompt_mode="custom",
                custom_prompt="Run once.",
                loop_count=1,
                enabled=True,
                agent="codex",
            )
        ],
    )

    executor = ChainExecutor()
    executor.config = config
    executor.state.running = True
    executor._run_loop()

    assert _NoopEvaluator.instances
    evaluator = _NoopEvaluator.instances[0]
    assert evaluator.skip_tests is False
    assert evaluator.test_cmd == ["pytest", "-k", "slow suite", "-q"]


def test_chain_file_instructions_include_terminate_tag_contract(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    config = ChainConfig(
        repo_path=str(repo),
        steps=[TaskStep(name="Implementation", job_type="implementation", enabled=True)],
    )
    executor = ChainExecutor()
    step = config.steps[0]

    prompt = executor._append_file_instructions("Base prompt", repo, config, step, 0)
    assert "Repository Scope (Strict)" in prompt
    assert str(repo) in prompt
    assert "task labels only" in prompt
    assert "[TERMINATE_STEP]" in prompt
    assert "skip" in prompt.lower()


def test_chain_brain_goal_anchors_to_repo_path(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    config = ChainConfig(
        name="Discover Chain",
        repo_path=str(repo),
        steps=[TaskStep(name="Implementation", job_type="implementation", enabled=True)],
    )

    goal = ChainExecutor._brain_goal(config, repo)
    assert str(repo) in goal
    assert "Improve repository" in goal


def test_prepare_output_dir_cleans_existing_entries(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    out_dir = repo / ".codex_manager" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stale.md").write_text("stale\n", encoding="utf-8")
    nested = out_dir / "nested"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "old.txt").write_text("old\n", encoding="utf-8")

    executor = ChainExecutor()
    executor.output_dir = out_dir
    executor._prepare_output_dir()

    assert out_dir.exists()
    assert not any(out_dir.iterdir())


def test_agent_debug_log_path_env_controls_logging(monkeypatch, tmp_path: Path):
    log_path = tmp_path / "agent-debug.jsonl"
    monkeypatch.setenv("CODEX_MANAGER_DEBUG_LOG_PATH", str(log_path))

    chain_module._agent_log(
        location="tests.chain",
        message="debug marker",
        data={"case": "env_path"},
        hypothesis_id="H-test",
    )

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert lines
    payload = json.loads(lines[0])
    assert payload["location"] == "tests.chain"
    assert payload["message"] == "debug marker"
    assert payload["hypothesisId"] == "H-test"
    assert payload["data"] == {"case": "env_path"}


def test_step_output_filename_falls_back_for_empty_slug():
    step = TaskStep(name="***", job_type="")
    assert ChainExecutor._step_output_filename(step) == "step.md"


def test_chain_runtime_state_tracks_run_loop_limits(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setattr(chain_module, "CodexRunner", _TerminateRunner)
    monkeypatch.setattr(chain_module, "RepoEvaluator", _NoopEvaluator)
    monkeypatch.setattr(chain_module, "ensure_git_identity", lambda _repo: None)

    _TerminateRunner.calls = 0
    config = ChainConfig(
        name="Loop labels",
        repo_path=str(repo),
        mode="dry-run",
        max_loops=1,
        unlimited=False,
        test_cmd="",
        steps=[
            TaskStep(
                name="Implementation",
                job_type="implementation",
                prompt_mode="custom",
                custom_prompt="Run once.",
                loop_count=1,
                enabled=True,
                agent="codex",
            )
        ],
    )
    executor = ChainExecutor()
    executor.config = config
    executor.state.running = True
    executor._run_loop()

    assert executor.state.run_max_loops == 1
    assert executor.state.run_unlimited is False


def test_chain_branch_creation_failure_finalizes_state(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setattr(chain_module, "CodexRunner", _TerminateRunner)
    monkeypatch.setattr(chain_module, "RepoEvaluator", _NoopEvaluator)
    monkeypatch.setattr(chain_module, "ensure_git_identity", lambda _repo: None)

    def _raise_branch(_repo: Path) -> str:
        raise RuntimeError("branch denied")

    monkeypatch.setattr(chain_module, "create_branch", _raise_branch)

    config = ChainConfig(
        name="Branch failure finalization",
        repo_path=str(repo),
        mode="apply",
        max_loops=1,
        test_cmd="",
        steps=[
            TaskStep(
                name="Implementation",
                job_type="implementation",
                prompt_mode="custom",
                custom_prompt="noop",
                loop_count=1,
                enabled=True,
                agent="codex",
            )
        ],
    )

    executor = ChainExecutor()
    executor.config = config
    executor.state.running = True
    executor._run_loop()

    assert executor.state.running is False
    assert executor.state.stop_reason == "branch_creation_failed"
    assert executor.state.finished_at
    assert executor.state.elapsed_seconds >= 0


def test_chain_history_logbook_init_failure_is_non_fatal(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setattr(chain_module, "CodexRunner", _TerminateRunner)
    monkeypatch.setattr(chain_module, "RepoEvaluator", _NoopEvaluator)
    monkeypatch.setattr(chain_module, "HistoryLogbook", _FailingHistoryLogbook)
    monkeypatch.setattr(chain_module, "ensure_git_identity", lambda _repo: None)

    _TerminateRunner.calls = 0
    config = ChainConfig(
        name="History init fallback",
        repo_path=str(repo),
        mode="dry-run",
        max_loops=1,
        stop_on_convergence=False,
        test_cmd="",
        steps=[
            TaskStep(
                name="Implementation",
                job_type="implementation",
                prompt_mode="custom",
                custom_prompt="noop",
                loop_count=1,
                enabled=True,
                agent="codex",
            )
        ],
    )

    executor = ChainExecutor()
    executor.config = config
    executor.state.running = True
    executor._run_loop()

    assert executor.state.running is False
    assert executor.state.stop_reason == "max_loops_reached"
    assert executor.state.total_steps_completed == 1
    assert _TerminateRunner.calls == 1
    assert executor.state.finished_at


def test_chain_persists_memory_and_reuses_it_in_future_prompts(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setattr(chain_module, "CodexRunner", _OutputRunner)
    monkeypatch.setattr(chain_module, "RepoEvaluator", _NoopEvaluator)
    monkeypatch.setattr(chain_module, "ensure_git_identity", lambda _repo: None)

    config = ChainConfig(
        name="Memory persistence",
        repo_path=str(repo),
        mode="dry-run",
        max_loops=1,
        test_cmd="",
        steps=[
            TaskStep(
                name="Implementation",
                job_type="implementation",
                prompt_mode="custom",
                custom_prompt="Create concrete findings.",
                loop_count=1,
                enabled=True,
                agent="codex",
            )
        ],
    )
    executor = ChainExecutor()
    executor.config = config
    executor.state.running = True
    executor._run_loop()

    memory_path = repo / ".codex_manager" / "memory" / "chain_step_memory.jsonl"
    assert memory_path.exists()
    entries = [
        json.loads(line)
        for line in memory_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert entries
    assert any(
        "Captured concrete findings" in str(entry.get("output_excerpt", "")) for entry in entries
    )
    assert any(str(entry.get("output_file", "")).endswith("Implementation.md") for entry in entries)

    executor2 = ChainExecutor()
    executor2._initialize_step_memory(repo)
    prompt = executor2._resolve_prompt(config.steps[0], loop_num=2, step_idx=0)
    assert "Recent Chain Memory" in prompt
    assert "Captured concrete findings" in prompt


def test_chain_ignores_placeholder_status_output(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setattr(chain_module, "CodexRunner", _PlaceholderRunner)
    monkeypatch.setattr(chain_module, "RepoEvaluator", _NoopEvaluator)
    monkeypatch.setattr(chain_module, "ensure_git_identity", lambda _repo: None)

    config = ChainConfig(
        name="Placeholder output",
        repo_path=str(repo),
        mode="dry-run",
        max_loops=1,
        test_cmd="",
        steps=[
            TaskStep(
                name="Implementation",
                job_type="implementation",
                prompt_mode="custom",
                custom_prompt="",
                loop_count=1,
                enabled=True,
                agent="codex",
            )
        ],
    )
    executor = ChainExecutor()
    executor.config = config
    executor.state.running = True
    executor._run_loop()

    out_file = repo / ".codex_manager" / "outputs" / "Implementation.md"
    if out_file.exists():
        content = out_file.read_text(encoding="utf-8")
        assert "Share the task you want implemented" not in content

    memory_path = repo / ".codex_manager" / "memory" / "chain_step_memory.jsonl"
    if memory_path.exists():
        assert "Share the task you want implemented" not in memory_path.read_text(encoding="utf-8")


def test_chain_marks_implementation_noop_as_failed_validation(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setattr(chain_module, "CodexRunner", _OutputRunner)
    monkeypatch.setattr(chain_module, "RepoEvaluator", _NoopEvaluator)
    monkeypatch.setattr(chain_module, "ensure_git_identity", lambda _repo: None)

    config = ChainConfig(
        name="Implementation no-op guard",
        repo_path=str(repo),
        mode="dry-run",
        max_loops=1,
        test_cmd="",
        steps=[
            TaskStep(
                name="Implementation",
                job_type="implementation",
                prompt_mode="custom",
                custom_prompt="Implement one concrete feature.",
                loop_count=1,
                enabled=True,
                agent="codex",
            )
        ],
    )

    executor = ChainExecutor()
    executor.config = config
    executor.state.running = True
    executor._run_loop()

    assert executor.state.results
    result = executor.state.results[0]
    assert result.agent_success is True
    assert result.validation_success is False
    assert result.tests_passed is False
    assert result.success is False
    assert "no repository changes detected" in result.error_message


def test_chain_keeps_discovery_noop_as_success(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setattr(chain_module, "CodexRunner", _OutputRunner)
    monkeypatch.setattr(chain_module, "RepoEvaluator", _NoopEvaluator)
    monkeypatch.setattr(chain_module, "ensure_git_identity", lambda _repo: None)

    config = ChainConfig(
        name="Discovery no-op allowed",
        repo_path=str(repo),
        mode="dry-run",
        max_loops=1,
        test_cmd="",
        steps=[
            TaskStep(
                name="Discovery",
                job_type="feature_discovery",
                prompt_mode="custom",
                custom_prompt="List and rank opportunities.",
                loop_count=1,
                enabled=True,
                agent="codex",
            )
        ],
    )

    executor = ChainExecutor()
    executor.config = config
    executor.state.running = True
    executor._run_loop()

    assert executor.state.results
    result = executor.state.results[0]
    assert result.agent_success is True
    assert result.validation_success is True
    assert result.tests_passed is False
    assert result.success is True


def test_chain_marks_step_failed_when_tests_fail(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setattr(chain_module, "CodexRunner", _OutputRunner)
    monkeypatch.setattr(chain_module, "RepoEvaluator", _FailingChangedEvaluator)
    monkeypatch.setattr(chain_module, "ensure_git_identity", lambda _repo: None)

    config = ChainConfig(
        name="Test failure propagates",
        repo_path=str(repo),
        mode="dry-run",
        max_loops=1,
        test_cmd="pytest -q",
        steps=[
            TaskStep(
                name="Implementation",
                job_type="implementation",
                prompt_mode="custom",
                custom_prompt="Implement and validate.",
                loop_count=1,
                enabled=True,
                agent="codex",
            )
        ],
    )

    executor = ChainExecutor()
    executor.config = config
    executor.state.running = True
    executor._run_loop()

    assert executor.state.results
    result = executor.state.results[0]
    assert result.agent_success is True
    assert result.validation_success is False
    assert result.tests_passed is False
    assert result.success is False
    assert "tests=failed" in result.error_message


def test_chain_stops_early_when_latest_loop_has_no_progress(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setattr(chain_module, "CodexRunner", _PlaceholderRunner)
    monkeypatch.setattr(chain_module, "RepoEvaluator", _NoopEvaluator)
    monkeypatch.setattr(chain_module, "ensure_git_identity", lambda _repo: None)

    config = ChainConfig(
        name="No progress guard",
        repo_path=str(repo),
        mode="dry-run",
        max_loops=3,
        stop_on_convergence=True,
        test_cmd="",
        steps=[
            TaskStep(
                name="Implementation",
                job_type="implementation",
                prompt_mode="custom",
                custom_prompt="",
                loop_count=1,
                enabled=True,
                agent="codex",
            )
        ],
    )
    executor = ChainExecutor()
    executor.config = config
    executor.state.running = True
    executor._run_loop()

    assert executor.state.stop_reason == "no_progress_detected"
    assert executor.state.total_loops_completed == 1


def test_chain_stops_when_brain_follow_up_requires_human_input(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setattr(chain_module, "CodexRunner", _PlaceholderRunner)
    monkeypatch.setattr(chain_module, "RepoEvaluator", _NoopEvaluator)
    monkeypatch.setattr(chain_module, "BrainManager", _FollowUpBrainNeedsInput)
    monkeypatch.setattr(chain_module, "ensure_git_identity", lambda _repo: None)

    config = ChainConfig(
        name="Brain asks for input",
        repo_path=str(repo),
        mode="dry-run",
        max_loops=2,
        stop_on_convergence=True,
        test_cmd="",
        brain_enabled=True,
        steps=[
            TaskStep(
                name="Implementation",
                job_type="implementation",
                prompt_mode="custom",
                custom_prompt="Implement one concrete feature.",
                loop_count=1,
                enabled=True,
                agent="codex",
            )
        ],
    )
    executor = ChainExecutor()
    executor.config = config
    executor.state.running = True
    executor._run_loop()

    assert executor.state.stop_reason == "brain_needs_input"
    assert executor.state.total_steps_completed == 1
    assert len(executor.state.results) == 1


def test_chain_appends_execution_evidence_for_file_changes(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setattr(chain_module, "CodexRunner", _OutputRunner)
    monkeypatch.setattr(chain_module, "RepoEvaluator", _ChangedEvaluator)
    monkeypatch.setattr(chain_module, "ensure_git_identity", lambda _repo: None)

    config = ChainConfig(
        name="Evidence block",
        repo_path=str(repo),
        mode="dry-run",
        max_loops=1,
        test_cmd="",
        steps=[
            TaskStep(
                name="Implementation",
                job_type="implementation",
                prompt_mode="custom",
                custom_prompt="Make one meaningful change.",
                loop_count=1,
                enabled=True,
                agent="codex",
            )
        ],
    )
    executor = ChainExecutor()
    executor.config = config
    executor.state.running = True
    executor._run_loop()

    out_file = repo / ".codex_manager" / "outputs" / "Implementation.md"
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "Execution Evidence" in content
    assert "Changed files:" in content
    assert "src/app.py" in content
    assert "tests/test_app.py" in content


def test_chain_counts_agent_authored_commit_deltas(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    monkeypatch.setattr(chain_module, "CodexRunner", _CommitRunner)
    monkeypatch.setattr(chain_module, "RepoEvaluator", _NoopEvaluator)
    monkeypatch.setattr(chain_module, "ensure_git_identity", lambda _repo: None)

    config = ChainConfig(
        name="Agent commit accounting",
        repo_path=str(repo),
        mode="apply",
        max_loops=1,
        test_cmd="",
        steps=[
            TaskStep(
                name="Implementation",
                job_type="implementation",
                prompt_mode="custom",
                custom_prompt="Implement and commit one feature.",
                loop_count=1,
                enabled=True,
                agent="codex",
            )
        ],
    )
    executor = ChainExecutor()
    executor.config = config
    executor.state.running = True
    executor._run_loop()

    assert executor.state.results
    result = executor.state.results[0]
    assert result.success is True
    assert result.files_changed >= 1
    assert result.net_lines_changed != 0
    assert result.commit_sha
    assert any(
        str(item.get("path", "")).endswith("src/agent_commit_feature.py")
        for item in result.changed_files
    )


def test_chain_dry_run_rolls_back_agent_authored_commit(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    baseline_head = _head_sha(repo)

    monkeypatch.setattr(chain_module, "CodexRunner", _CommitRunner)
    monkeypatch.setattr(chain_module, "RepoEvaluator", _NoopEvaluator)
    monkeypatch.setattr(chain_module, "ensure_git_identity", lambda _repo: None)

    config = ChainConfig(
        name="Dry-run commit rollback",
        repo_path=str(repo),
        mode="dry-run",
        max_loops=1,
        test_cmd="",
        steps=[
            TaskStep(
                name="Implementation",
                job_type="implementation",
                prompt_mode="custom",
                custom_prompt="Implement and commit one feature.",
                loop_count=1,
                enabled=True,
                agent="codex",
            )
        ],
    )
    executor = ChainExecutor()
    executor.config = config
    executor.state.running = True
    executor._run_loop()

    assert executor.state.results
    result = executor.state.results[0]
    assert result.success is True
    assert result.commit_sha is None
    assert result.files_changed >= 1
    assert any(
        str(item.get("path", "")).endswith("src/agent_commit_feature.py")
        for item in result.changed_files
    )

    assert _head_sha(repo) == baseline_head
    assert not (repo / "src" / "agent_commit_feature.py").exists()
    files_changed, _, _ = diff_numstat(repo)
    assert files_changed == 0


def test_chain_file_instructions_include_primary_handoff_file(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    config = ChainConfig(
        repo_path=str(repo),
        steps=[
            TaskStep(name="Discover", job_type="feature_discovery", enabled=True),
            TaskStep(name="Implement", job_type="implementation", enabled=True),
        ],
    )
    executor = ChainExecutor()
    first_step = config.steps[0]
    second_step = config.steps[1]

    first_output = executor._step_output_path(repo, first_step)
    first_output.parent.mkdir(parents=True, exist_ok=True)
    first_output.write_text("Initial findings\n", encoding="utf-8")

    prompt = executor._append_file_instructions("Base prompt", repo, config, second_step, 1)
    first_rel = str(executor._step_output_relpath(first_step)).replace("\\", "/")
    assert "Step Handoff" in prompt
    assert first_rel in prompt


def test_chain_brain_logbook_init_failure_does_not_abort_run(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    class _FailingLogbook:
        def __init__(self, _repo):
            pass

        def initialize(self):
            raise OSError("disk denied")

    monkeypatch.setattr(chain_module, "CodexRunner", _TerminateRunner)
    monkeypatch.setattr(chain_module, "RepoEvaluator", _NoopEvaluator)
    monkeypatch.setattr(chain_module, "BrainManager", _ContinueBrain)
    monkeypatch.setattr(chain_module, "BrainLogbook", _FailingLogbook)
    monkeypatch.setattr(chain_module, "ensure_git_identity", lambda _repo: None)

    _TerminateRunner.calls = 0
    config = ChainConfig(
        name="Brain logbook fallback",
        repo_path=str(repo),
        mode="dry-run",
        max_loops=1,
        test_cmd="",
        brain_enabled=True,
        steps=[
            TaskStep(
                name="Implementation",
                job_type="implementation",
                prompt_mode="custom",
                custom_prompt="Run once.",
                loop_count=1,
                enabled=True,
                agent="codex",
            )
        ],
    )

    executor = ChainExecutor()
    executor.config = config
    executor.state.running = True
    executor._run_loop()

    assert _TerminateRunner.calls == 1
    assert executor.state.total_steps_completed == 1
    assert executor.state.results
