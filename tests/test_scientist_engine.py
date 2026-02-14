"""Tests for ScientistEngine compatibility wrapper behavior."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import codex_manager.scientist.engine as scientist_engine_module
from codex_manager.pipeline.phases import PhaseResult, PipelineState
from codex_manager.scientist.engine import ScientistEngine


class _StubOrchestrator:
    calls: ClassVar[list[object]] = []

    def __init__(self, repo_path, config, catalog=None):
        self.repo_path = Path(repo_path)
        self.config = config
        self.catalog = catalog

    def run(self):
        self.__class__.calls.append(self.config)
        return PipelineState(
            results=[
                PhaseResult(
                    cycle=1,
                    phase="experiment",
                    iteration=1,
                    success=True,
                    test_outcome="skipped",
                    science_trial_id="trial-exp",
                    science_experiment_id="exp-001",
                    science_hypothesis_id="EXP-001",
                    science_verdict="supported",
                    science_verdict_rationale="no regression and measurable repo delta observed",
                ),
                PhaseResult(
                    cycle=1,
                    phase="skeptic",
                    iteration=1,
                    success=True,
                    test_outcome="skipped",
                    science_trial_id="trial-skeptic",
                    science_experiment_id="exp-001",
                    science_hypothesis_id="EXP-001",
                    science_verdict="supported",
                    science_verdict_rationale="replication confirmed",
                ),
            ]
        )


def test_scientist_engine_run_cycle_uses_pipeline_runtime(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    _StubOrchestrator.calls.clear()
    monkeypatch.setattr(scientist_engine_module, "PipelineOrchestrator", _StubOrchestrator)

    engine = ScientistEngine(repo_path=repo, mode="dry-run")
    results = engine.run_cycle()

    assert _StubOrchestrator.calls
    cfg = _StubOrchestrator.calls[-1]
    assert cfg.science_enabled is True
    assert cfg.auto_commit is False
    assert [p.phase.value for p in cfg.phases] == ["theorize", "experiment", "skeptic", "analyze"]

    assert len(results) == 1
    assert results[0].hypothesis_id == "EXP-001"
    assert results[0].conclusion == "supported"
    assert results[0].success is True
