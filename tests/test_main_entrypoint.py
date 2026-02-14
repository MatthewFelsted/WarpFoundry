"""Tests for CLI entrypoint dispatch and command handlers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import codex_manager.__main__ as main_module


def test_main_entrypoint_source_is_ascii_safe() -> None:
    source_text = Path(main_module.__file__).read_text(encoding="utf-8")
    assert source_text.isascii()
    assert not any("\u0080" <= ch <= "\u009f" for ch in source_text)


def test_main_prints_help_when_repo_or_goal_missing(capsys) -> None:
    rc = main_module.main([])
    captured = capsys.readouterr()
    assert rc == 1
    assert "Tip: run 'python -m codex_manager gui'" in captured.err


def test_main_dispatches_subcommands(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_pipeline(args):
        calls["pipeline_repo"] = args.repo
        return 10

    def fake_strategic(args):
        calls["strategic_repo"] = args.repo
        return 14

    def fake_optimize(args):
        calls["opt_model"] = args.model
        return 11

    def fake_visual(args):
        calls["visual_provider"] = args.provider
        return 12

    def fake_list():
        calls["listed"] = True
        return 13

    def fake_doctor(args):
        calls["doctor_repo"] = args.repo
        return 15

    monkeypatch.setattr(main_module, "_run_pipeline", fake_pipeline)
    monkeypatch.setattr(main_module, "_run_strategic", fake_strategic)
    monkeypatch.setattr(main_module, "_optimize_prompts", fake_optimize)
    monkeypatch.setattr(main_module, "_visual_test", fake_visual)
    monkeypatch.setattr(main_module, "_list_prompts", fake_list)
    monkeypatch.setattr(main_module, "_run_doctor", fake_doctor)

    repo = tmp_path / "repo"
    repo.mkdir()
    assert main_module.main(["pipeline", "--repo", str(repo)]) == 10
    assert main_module.main(["strategic", "--repo", str(repo)]) == 14
    assert main_module.main(["optimize-prompts", "--model", "gpt-x"]) == 11
    assert main_module.main(["visual-test", "--provider", "anthropic"]) == 12
    assert main_module.main(["list-prompts"]) == 13
    assert main_module.main(["doctor", "--repo", str(repo)]) == 15

    assert calls["pipeline_repo"] == str(repo)
    assert calls["strategic_repo"] == str(repo)
    assert calls["opt_model"] == "gpt-x"
    assert calls["visual_provider"] == "anthropic"
    assert calls["listed"] is True
    assert calls["doctor_repo"] == str(repo)


def test_main_dispatches_gui(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_gui_main(*, port: int, open_browser: bool) -> None:
        calls["port"] = port
        calls["open_browser"] = open_browser

    monkeypatch.setattr("codex_manager.gui.main", fake_gui_main)
    assert main_module.main(["gui", "--port", "6111", "--no-browser"]) == 0
    assert calls == {"port": 6111, "open_browser": False}


def test_main_loop_mode_invalid_repo_path(capsys, tmp_path: Path) -> None:
    missing = tmp_path / "missing-repo"
    rc = main_module.main(["--repo", str(missing), "--goal", "Improve tests"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "repo path does not exist" in captured.err


def test_main_loop_mode_success_with_stubbed_components(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    class StubRunner:
        calls: ClassVar[list[tuple[str, int]]] = []

        def __init__(self, codex_binary: str, timeout: int):
            self.__class__.calls.append((codex_binary, timeout))

    class StubEvaluator:
        calls: ClassVar[list[list[str] | None]] = []

        def __init__(self, test_cmd=None):
            self.__class__.calls.append(test_cmd)

    class StubState:
        goal = "Improve docs"
        mode = "dry-run"
        rounds: ClassVar[list[object]] = []
        max_rounds = 2
        stop_reason = "max_rounds"
        branch_name = None
        total_input_tokens = 120
        total_output_tokens = 30

        def state_path(self) -> Path:
            return Path("state.json")

    class StubLoop:
        init_kwargs: ClassVar[dict[str, object]] = {}

        def __init__(self, **kwargs):
            self.__class__.init_kwargs = kwargs

        def run(self):
            return StubState()

    monkeypatch.setattr("codex_manager.codex_cli.CodexRunner", StubRunner)
    monkeypatch.setattr("codex_manager.eval_tools.RepoEvaluator", StubEvaluator)
    monkeypatch.setattr("codex_manager.loop.ImprovementLoop", StubLoop)
    monkeypatch.setattr(main_module, "_print_cli_preflight_guard", lambda **_kwargs: True)

    rc = main_module.main(
        [
            "--repo",
            str(repo),
            "--goal",
            "Improve docs",
            "--rounds",
            "2",
            "--test-cmd",
            "pytest -q",
            "--codex-bin",
            "codex-test",
            "--timeout",
            "45",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert StubRunner.calls == [("codex-test", 45)]
    assert StubEvaluator.calls == [["pytest", "-q"]]
    assert StubLoop.init_kwargs["repo_path"] == repo.resolve()
    assert StubLoop.init_kwargs["goal"] == "Improve docs"
    assert "Codex Manager - Run Summary" in captured.out


def test_run_goal_loop_parses_quoted_test_command(monkeypatch, capsys, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    class StubRunner:
        def __init__(self, codex_binary: str, timeout: int):
            self.codex_binary = codex_binary
            self.timeout = timeout

    class StubEvaluator:
        calls: ClassVar[list[list[str] | None]] = []

        def __init__(self, test_cmd=None):
            self.__class__.calls.append(test_cmd)

    class StubState:
        goal = "Goal"
        mode = "dry-run"
        rounds: ClassVar[list[object]] = []
        max_rounds = 1
        stop_reason = "max_rounds"
        branch_name = None
        total_input_tokens = 1
        total_output_tokens = 1

        def state_path(self) -> Path:
            return Path("state.json")

    class StubLoop:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self):
            return StubState()

    monkeypatch.setattr("codex_manager.codex_cli.CodexRunner", StubRunner)
    monkeypatch.setattr("codex_manager.eval_tools.RepoEvaluator", StubEvaluator)
    monkeypatch.setattr("codex_manager.loop.ImprovementLoop", StubLoop)
    monkeypatch.setattr(main_module, "_print_cli_preflight_guard", lambda **_kwargs: True)

    rc = main_module._run_goal_loop(
        repo_path=str(repo),
        goal="Goal",
        mode="dry-run",
        rounds=1,
        test_cmd='pytest -k "slow suite" -q',
        codex_bin="codex",
        timeout=30,
    )
    capsys.readouterr()

    assert rc == 0
    assert StubEvaluator.calls == [["pytest", "-k", "slow suite", "-q"]]


def test_run_strategic_uses_builtin_goal(monkeypatch, capsys, tmp_path: Path) -> None:
    repo = tmp_path / "discover-chain"
    repo.mkdir()

    class StubRunner:
        calls: ClassVar[list[tuple[str, int]]] = []

        def __init__(self, codex_binary: str, timeout: int):
            self.__class__.calls.append((codex_binary, timeout))

    class StubEvaluator:
        calls: ClassVar[list[list[str] | None]] = []

        def __init__(self, test_cmd=None):
            self.__class__.calls.append(test_cmd)

    class StubState:
        goal = "Strategic goal"
        mode = "dry-run"
        rounds: ClassVar[list[object]] = []
        max_rounds = 3
        stop_reason = "max_rounds"
        branch_name = None
        total_input_tokens = 12
        total_output_tokens = 3

        def state_path(self) -> Path:
            return Path("state.json")

    class StubLoop:
        init_kwargs: ClassVar[dict[str, object]] = {}

        def __init__(self, **kwargs):
            self.__class__.init_kwargs = kwargs

        def run(self):
            return StubState()

    monkeypatch.setattr("codex_manager.codex_cli.CodexRunner", StubRunner)
    monkeypatch.setattr("codex_manager.eval_tools.RepoEvaluator", StubEvaluator)
    monkeypatch.setattr("codex_manager.loop.ImprovementLoop", StubLoop)
    monkeypatch.setattr(main_module, "_print_cli_preflight_guard", lambda **_kwargs: True)

    args = argparse.Namespace(
        repo=str(repo),
        mode="dry-run",
        rounds=3,
        test_cmd="pytest -q",
        codex_bin="codex-strat",
        timeout=50,
        goal_extra="Prioritize user retention instrumentation.",
    )
    rc = main_module._run_strategic(args)
    captured = capsys.readouterr()

    assert rc == 0
    assert StubRunner.calls == [("codex-strat", 50)]
    assert StubEvaluator.calls == [["pytest", "-q"]]
    goal = str(StubLoop.init_kwargs["goal"])
    assert "STRATEGIC PRODUCT MAXIMIZATION MODE" in goal
    assert "Discover Chain" in goal
    assert "Prioritize user retention instrumentation." in goal
    assert "Codex Manager - Run Summary" in captured.out


def test_run_pipeline_validates_repo_and_runs_orchestrator(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    missing_args = argparse.Namespace(
        repo=str(tmp_path / "missing"),
        mode="dry-run",
        cycles=1,
        science=False,
        brain=False,
        brain_model="gpt-5.2",
        agent="codex",
        test_cmd=None,
        timeout=60,
        max_tokens=1000,
        max_time=10,
        local_only=False,
        skip_preflight=False,
    )
    assert main_module._run_pipeline(missing_args) == 1

    repo = tmp_path / "repo"
    repo.mkdir()

    class StubPipeline:
        last_instance: StubPipeline | None = None

        def __init__(self, repo_path: Path, config):
            self.repo_path = repo_path
            self.config = config
            self.__class__.last_instance = self

        def run(self):
            return SimpleNamespace(
                total_cycles_completed=1,
                total_phases_completed=2,
                stop_reason="max_cycles_reached",
                total_tokens=321,
                elapsed_seconds=4.2,
                results=[
                    SimpleNamespace(
                        phase="ideation",
                        iteration=1,
                        success=True,
                        test_outcome="passed",
                        files_changed=1,
                        net_lines_changed=5,
                    )
                ],
            )

    monkeypatch.setattr("codex_manager.pipeline.PipelineOrchestrator", StubPipeline)
    monkeypatch.setattr(main_module, "_print_cli_preflight_guard", lambda **_kwargs: True)
    args = argparse.Namespace(
        repo=str(repo),
        mode="apply",
        cycles=3,
        science=True,
        brain=True,
        brain_model="gpt-5.2",
        agent="codex",
        test_cmd="pytest -q",
        timeout=120,
        max_tokens=5000,
        max_time=60,
        local_only=True,
        skip_preflight=False,
    )
    rc = main_module._run_pipeline(args)
    captured = capsys.readouterr()

    assert rc == 0
    assert StubPipeline.last_instance is not None
    assert StubPipeline.last_instance.repo_path == repo.resolve()
    assert StubPipeline.last_instance.config.test_cmd == "pytest -q"
    assert StubPipeline.last_instance.config.science_enabled is True
    assert "AI Manager - Pipeline Summary" in captured.out


def test_run_pipeline_preflight_failure_stops_before_orchestrator(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(main_module, "_print_cli_preflight_guard", lambda **_kwargs: False)

    args = argparse.Namespace(
        repo=str(repo),
        mode="dry-run",
        cycles=1,
        science=False,
        brain=False,
        brain_model="gpt-5.2",
        agent="codex",
        test_cmd=None,
        timeout=60,
        max_tokens=1000,
        max_time=10,
        local_only=False,
        skip_preflight=False,
    )

    assert main_module._run_pipeline(args) == 1


def test_run_pipeline_preflight_uses_selected_agent(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    captured_guard_args: dict[str, object] = {}

    def _fake_guard(**kwargs):
        captured_guard_args.update(kwargs)
        return True

    class StubPipeline:
        def __init__(self, repo_path: Path, config):
            self.repo_path = repo_path
            self.config = config

        def run(self):
            return SimpleNamespace(
                total_cycles_completed=0,
                total_phases_completed=0,
                stop_reason="done",
                total_tokens=0,
                elapsed_seconds=0.0,
                results=[],
            )

    monkeypatch.setattr(main_module, "_print_cli_preflight_guard", _fake_guard)
    monkeypatch.setattr("codex_manager.pipeline.PipelineOrchestrator", StubPipeline)

    args = argparse.Namespace(
        repo=str(repo),
        mode="dry-run",
        cycles=1,
        science=False,
        brain=False,
        brain_model="gpt-5.2",
        agent="claude_code",
        test_cmd=None,
        timeout=60,
        max_tokens=1000,
        max_time=10,
        local_only=False,
        skip_preflight=False,
    )

    assert main_module._run_pipeline(args) == 0
    assert captured_guard_args["repo"] == repo.resolve()
    assert captured_guard_args["agents"] == ["claude_code"]


def test_optimize_prompts_local_only_uses_ollama_model(monkeypatch, capsys) -> None:
    calls: dict[str, object] = {}

    class StubOptimizer:
        def __init__(self, *, model: str, threshold: float):
            calls["model"] = model
            calls["threshold"] = threshold

        def optimize_all(self, *, dry_run: bool):
            calls["dry_run"] = dry_run
            return []

        def summary(self) -> str:
            return "summary text"

    monkeypatch.setattr("codex_manager.prompts.PromptOptimizer", StubOptimizer)
    monkeypatch.setattr(
        "codex_manager.brain.connector.get_default_ollama_model",
        lambda: "ollama:test-model",
    )

    args = argparse.Namespace(model="gpt-5.2", threshold=8.0, dry_run=True, local_only=True)
    rc = main_module._optimize_prompts(args)
    captured = capsys.readouterr()

    assert rc == 0
    assert calls == {"model": "ollama:test-model", "threshold": 8.0, "dry_run": True}
    assert "Local-only mode: using ollama:test-model" in captured.out
    assert "summary text" in captured.out


def test_visual_test_uses_session_runner_and_formats_output(monkeypatch, capsys) -> None:
    result = SimpleNamespace(
        success=True,
        total_steps=3,
        duration_seconds=12,
        error="",
        observations=[
            SimpleNamespace(
                severity="major",
                element="Login button",
                actual="Overlaps text",
                expected="No overlap",
                recommendation="Adjust margin",
            )
        ],
        summary="All key workflows pass\nOBSERVATION|ignored|ignored",
        screenshots_saved=["shot1.png", "shot2.png"],
    )
    monkeypatch.setattr("codex_manager.cua.session.run_cua_session_sync", lambda _config: result)

    args = argparse.Namespace(
        provider="openai",
        task="",
        url="http://localhost:5088",
        max_steps=5,
        timeout=60,
        headed=False,
    )
    rc = main_module._visual_test(args)
    captured = capsys.readouterr()

    assert rc == 0
    assert "CUA Session Complete" in captured.out
    assert "Success:    True" in captured.out
    assert "Login button" in captured.out
    assert "All key workflows pass" in captured.out


def test_list_prompts_uses_catalog(monkeypatch, capsys) -> None:
    class StubCatalog:
        def all_prompts(self):
            return [
                {
                    "path": "pipeline.ideation",
                    "name": "Ideation",
                    "content": "Prompt text for ideation",
                }
            ]

    monkeypatch.setattr("codex_manager.prompts.catalog.get_catalog", lambda: StubCatalog())
    rc = main_module._list_prompts()
    captured = capsys.readouterr()
    assert rc == 0
    assert "Prompt Catalog" in captured.out
    assert "[pipeline.ideation]" in captured.out


def test_run_goal_loop_preflight_failure_stops_before_runner(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(main_module, "_print_cli_preflight_guard", lambda **_kwargs: False)

    rc = main_module._run_goal_loop(
        repo_path=str(repo),
        goal="Goal",
        mode="dry-run",
        rounds=1,
        test_cmd="pytest -q",
        codex_bin="codex",
        timeout=30,
    )
    capsys.readouterr()
    assert rc == 1


def test_run_doctor_json_mode(monkeypatch, capsys) -> None:
    class FakeReport:
        ready = False

        def to_dict(self) -> dict[str, object]:
            return {"ready": False, "summary": {"pass": 1, "warn": 0, "fail": 1}}

    monkeypatch.setattr(main_module, "build_preflight_report", lambda **_kwargs: FakeReport())
    args = argparse.Namespace(
        repo="",
        agents="codex",
        codex_bin="codex",
        claude_bin="claude",
        json=True,
    )
    rc = main_module._run_doctor(args)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["ready"] is False
