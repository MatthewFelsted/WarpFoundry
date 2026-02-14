"""Scientist engine compatibility wrapper.

This module keeps the public ``ScientistEngine`` API, but routes execution
through the pipeline Scientist runtime so there is a single authoritative
science path (the orchestrator).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codex_manager.eval_tools import RepoEvaluator
from codex_manager.pipeline.orchestrator import PipelineOrchestrator
from codex_manager.pipeline.phases import PhaseConfig, PipelineConfig, PipelinePhase
from codex_manager.pipeline.tracker import LogTracker
from codex_manager.prompts.catalog import PromptCatalog, get_catalog

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """A testable hypothesis about what could improve the project."""

    id: str
    title: str
    hypothesis: str
    success_criteria: str
    experiment_design: str
    estimated_cost: str
    confidence: str
    status: str = "proposed"
    baseline: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    conclusion: str = ""
    analysis: str = ""


@dataclass
class ExperimentResult:
    """Result of running a single experiment."""

    hypothesis_id: str
    success: bool
    baseline_metrics: dict[str, Any]
    post_metrics: dict[str, Any]
    conclusion: str
    analysis: str
    changes_kept: bool
    duration_seconds: float = 0.0


class ScientistEngine:
    """Compatibility wrapper over the pipeline science runtime.

    The orchestrator remains the canonical implementation. This class exists to
    preserve the public ScientistEngine API for callers.
    """

    def __init__(
        self,
        repo_path: str | Path,
        brain_model: str = "gpt-5.2",
        catalog: PromptCatalog | None = None,
        max_experiments: int = 3,
        local_only: bool = False,
        mode: str = "apply",
        agent: str = "codex",
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.brain_model = brain_model
        self.catalog = catalog or get_catalog()
        self.max_experiments = max(1, int(max_experiments))
        self.local_only = local_only
        self.mode = mode
        self.agent = agent
        self.tracker = LogTracker(self.repo_path)
        self.evaluator = RepoEvaluator()

    def collect_baseline(self) -> dict[str, Any]:
        """Collect baseline repo metrics."""
        eval_result = self.evaluator.evaluate(self.repo_path)
        return {
            "test_outcome": eval_result.test_outcome.value,
            "test_summary": eval_result.test_summary[:500],
            "test_exit_code": eval_result.test_exit_code,
            "files_changed": eval_result.files_changed,
            "net_lines_changed": eval_result.net_lines_changed,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        }

    def generate_hypotheses(self, count: int = 3) -> list[Hypothesis]:
        """Run theorize once and parse hypotheses from EXPERIMENTS.md."""
        self._run_science_phases(
            [
                PhaseConfig(
                    phase=PipelinePhase.THEORIZE,
                    iterations=1,
                    agent=self.agent,
                )
            ]
        )
        return self._parse_hypotheses_from_markdown(
            self.tracker.read("EXPERIMENTS.md"),
            limit=max(1, int(count)),
        )

    def run_experiment(self, hypothesis: Hypothesis) -> ExperimentResult:
        """Run one targeted experiment + skeptic pass for a hypothesis."""
        experiment_prompt = (
            f"{self.catalog.scientist('experiment')}\n\n"
            f"Target Hypothesis ID: {hypothesis.id}\n"
            f"Hypothesis: {hypothesis.hypothesis}\n"
            f"Success Criteria: {hypothesis.success_criteria}\n"
            "Only evaluate this hypothesis in this trial.\n"
        ).strip()
        skeptic_prompt = (
            f"{self.catalog.scientist('skeptic')}\n\n"
            f"Target Hypothesis ID: {hypothesis.id}\n"
            "Challenge and replicate the latest experiment for this hypothesis only.\n"
        ).strip()

        state = self._run_science_phases(
            [
                PhaseConfig(
                    phase=PipelinePhase.EXPERIMENT,
                    iterations=1,
                    agent=self.agent,
                    custom_prompt=experiment_prompt,
                ),
                PhaseConfig(
                    phase=PipelinePhase.SKEPTIC,
                    iterations=1,
                    agent=self.agent,
                    custom_prompt=skeptic_prompt,
                ),
            ]
        )
        return self._to_experiment_results(state.results)[0] if state.results else ExperimentResult(
            hypothesis_id=hypothesis.id,
            success=False,
            baseline_metrics={},
            post_metrics={},
            conclusion="inconclusive",
            analysis="No experiment result produced.",
            changes_kept=False,
            duration_seconds=0.0,
        )

    def analyze_findings(self) -> str:
        """Run analyze phase once and return the final message."""
        state = self._run_science_phases(
            [
                PhaseConfig(
                    phase=PipelinePhase.ANALYZE,
                    iterations=1,
                    agent=self.agent,
                )
            ]
        )
        for result in reversed(state.results):
            if result.phase == PipelinePhase.ANALYZE.value:
                return (result.agent_final_message or "").strip()
        return ""

    def run_cycle(self) -> list[ExperimentResult]:
        """Run theorize -> experiment -> skeptic -> analyze."""
        state = self._run_science_phases(
            [
                PhaseConfig(
                    phase=PipelinePhase.THEORIZE,
                    iterations=1,
                    agent=self.agent,
                ),
                PhaseConfig(
                    phase=PipelinePhase.EXPERIMENT,
                    iterations=self.max_experiments,
                    agent=self.agent,
                ),
                PhaseConfig(
                    phase=PipelinePhase.SKEPTIC,
                    iterations=self.max_experiments,
                    agent=self.agent,
                ),
                PhaseConfig(
                    phase=PipelinePhase.ANALYZE,
                    iterations=1,
                    agent=self.agent,
                ),
            ]
        )
        return self._to_experiment_results(state.results)

    def _run_science_phases(self, phases: list[PhaseConfig]):
        """Run a one-cycle pipeline restricted to the provided science phases."""
        config = PipelineConfig(
            mode=self.mode,
            max_cycles=1,
            agent=self.agent,
            science_enabled=True,
            brain_enabled=False,
            brain_model=self.brain_model,
            local_only=self.local_only,
            auto_commit=False,
            phases=phases,
        )
        orchestrator = PipelineOrchestrator(
            repo_path=self.repo_path,
            config=config,
            catalog=self.catalog,
        )
        return orchestrator.run()

    @staticmethod
    def _parse_hypotheses_from_markdown(markdown: str, *, limit: int) -> list[Hypothesis]:
        """Parse hypothesis blocks from ``EXPERIMENTS.md`` into ``Hypothesis`` objects."""
        if not markdown.strip():
            return []
        pattern = re.compile(
            r"^### \[(EXP-[0-9]{3,})\]\s*(.*?)(?:\r?\n)(.*?)(?=^### \[(?:EXP-[0-9]{3,})\]|\Z)",
            flags=re.MULTILINE | re.DOTALL,
        )
        items: list[Hypothesis] = []
        for match in pattern.finditer(markdown):
            hyp_id = match.group(1).strip()
            title = match.group(2).strip()
            body = match.group(3).strip()

            def _field(label: str, default: str = "", body_text: str = body) -> str:
                fm = re.search(
                    rf"^- \*\*{re.escape(label)}\*\*:\s*(.+)$",
                    body_text,
                    flags=re.IGNORECASE | re.MULTILINE,
                )
                return fm.group(1).strip() if fm else default

            items.append(
                Hypothesis(
                    id=hyp_id,
                    title=title or "Untitled",
                    hypothesis=_field("Hypothesis"),
                    success_criteria=_field("Success Criteria"),
                    experiment_design=_field("Experiment Design"),
                    estimated_cost=_field("Estimated Cost", "unknown"),
                    confidence=_field("Confidence", "medium").lower(),
                    status=_field("Status", "proposed").lower(),
                )
            )
            if len(items) >= limit:
                break
        return items

    def _to_experiment_results(self, phase_results: list[Any]) -> list[ExperimentResult]:
        """Convert pipeline phase outputs into public ``ExperimentResult`` records."""
        trials = self._read_trial_records()
        by_trial_id = {str(t.get("trial_id", "")): t for t in trials}
        skeptic_by_experiment: dict[str, Any] = {}
        for result in phase_results:
            if result.phase == PipelinePhase.SKEPTIC.value and result.science_experiment_id:
                skeptic_by_experiment[result.science_experiment_id] = result

        out: list[ExperimentResult] = []
        for result in phase_results:
            if result.phase != PipelinePhase.EXPERIMENT.value:
                continue

            skeptic = skeptic_by_experiment.get(result.science_experiment_id)
            final_verdict = result.science_verdict or "inconclusive"
            analysis_parts = [result.science_verdict_rationale or ""]
            if skeptic is not None:
                skeptic_verdict = skeptic.science_verdict or "inconclusive"
                analysis_parts.append(skeptic.science_verdict_rationale or "")
                if skeptic_verdict != "supported":
                    final_verdict = skeptic_verdict
                elif final_verdict == "supported":
                    final_verdict = "supported"
            if final_verdict not in {"supported", "refuted", "inconclusive"}:
                final_verdict = "inconclusive"

            trial_payload = by_trial_id.get(result.science_trial_id, {})
            baseline_metrics = trial_payload.get("baseline", {})
            post_metrics = trial_payload.get("post", {})

            out.append(
                ExperimentResult(
                    hypothesis_id=result.science_hypothesis_id or "",
                    success=final_verdict == "supported",
                    baseline_metrics=baseline_metrics if isinstance(baseline_metrics, dict) else {},
                    post_metrics=post_metrics if isinstance(post_metrics, dict) else {},
                    conclusion=final_verdict,
                    analysis=" ".join(p for p in analysis_parts if p).strip(),
                    changes_kept=(final_verdict == "supported" and not result.science_rolled_back),
                    duration_seconds=float(result.duration_seconds or 0.0),
                )
            )
        return out

    def _read_trial_records(self) -> list[dict[str, Any]]:
        """Load structured scientist trial records from ``TRIALS.jsonl``."""
        trials_path = self.tracker.science_path_for("TRIALS.jsonl")
        if not trials_path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in trials_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records
