"""Autonomous improvement pipeline.

The pipeline automates the full development lifecycle:

    Ideation → Prioritization → Implementation → Testing → Debugging → Commit → Science

Each phase operates on structured markdown log files (WISHLIST.md, TESTPLAN.md,
ERRORS.md, EXPERIMENTS.md, PROGRESS.md) that serve as persistent context and
progress trackers.

Usage::

    from codex_manager.pipeline import PipelineOrchestrator, PipelineConfig

    pipeline = PipelineOrchestrator(
        repo_path="/path/to/repo",
        config=PipelineConfig(mode="apply"),
    )
    state = pipeline.run()
"""

from codex_manager.pipeline.orchestrator import PipelineOrchestrator
from codex_manager.pipeline.phases import PhaseConfig, PipelineConfig, PipelinePhase
from codex_manager.pipeline.tracker import LogTracker

__all__ = [
    "LogTracker",
    "PhaseConfig",
    "PipelineConfig",
    "PipelineOrchestrator",
    "PipelinePhase",
]
