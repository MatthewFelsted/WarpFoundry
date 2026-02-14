"""User-facing guidance for chain/pipeline stop reasons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StopSeverity = Literal["info", "warn", "error"]
RunMode = Literal["chain", "pipeline"]


@dataclass(frozen=True)
class GuidanceTemplate:
    """Template for rendering actionable run-stop guidance."""

    label: str
    severity: StopSeverity
    summary: str
    next_steps: tuple[str, ...]


_SHARED_TEMPLATES: dict[str, GuidanceTemplate] = {
    "user_stopped": GuidanceTemplate(
        label="Stopped by user",
        severity="warn",
        summary="The run was manually stopped before completing all planned work.",
        next_steps=(
            "Review the latest results to decide where to resume.",
            "Start a new run when ready.",
        ),
    ),
    "brain_escalation": GuidanceTemplate(
        label="Brain escalation",
        severity="error",
        summary="The brain layer requested human intervention for a blocking issue.",
        next_steps=(
            "Check the live log and .codex_manager/ERRORS.md for the escalation reason.",
            "Fix the blocker, then restart the run.",
        ),
    ),
    "brain_requested_stop": GuidanceTemplate(
        label="Brain requested stop",
        severity="info",
        summary="The brain layer determined this run should stop instead of continuing.",
        next_steps=(
            "Review the latest outputs and decide whether to continue with a new run.",
            "If needed, provide a more specific goal and rerun.",
        ),
    ),
    "brain_converged": GuidanceTemplate(
        label="Brain convergence",
        severity="info",
        summary="The brain layer assessed that goals were sufficiently achieved.",
        next_steps=(
            "Review results and tests, then switch to apply mode if you are satisfied.",
            "Raise quality bars or refine goals for another pass.",
        ),
    ),
    "budget_exhausted": GuidanceTemplate(
        label="Token budget exhausted",
        severity="warn",
        summary="The configured token budget was reached.",
        next_steps=(
            "Increase the token budget for a longer run.",
            "Narrow scope to a single high-impact objective.",
        ),
    ),
    "max_time_reached": GuidanceTemplate(
        label="Maximum runtime reached",
        severity="warn",
        summary="The run reached the configured time limit.",
        next_steps=(
            "Increase the max time limit if you want a longer run.",
            "Split work into smaller runs with clearer goals.",
        ),
    ),
    "diminishing_returns": GuidanceTemplate(
        label="Diminishing returns detected",
        severity="info",
        summary="Recent iterations improved less than your configured threshold.",
        next_steps=(
            "Review changes and keep the best result.",
            "Lower the improvement threshold to continue exploring.",
        ),
    ),
    "convergence_detected": GuidanceTemplate(
        label="Convergence detected",
        severity="info",
        summary="Recent iterations were low impact and stable, so the run stopped.",
        next_steps=(
            "Switch to apply mode if you are satisfied with current quality.",
            "Raise loop limits or tighten goals for deeper improvements.",
        ),
    ),
    "branch_creation_failed": GuidanceTemplate(
        label="Branch creation failed",
        severity="error",
        summary="The run could not create a working git branch.",
        next_steps=(
            "Confirm the repository is writable and git is healthy.",
            "Check git logs/status, then retry.",
        ),
    ),
}

_CHAIN_TEMPLATES: dict[str, GuidanceTemplate] = {
    "no_config": GuidanceTemplate(
        label="Missing run configuration",
        severity="error",
        summary="The chain started without a valid configuration payload.",
        next_steps=(
            "Reload the page and start the chain again.",
            "If this repeats, check browser/network logs for malformed requests.",
        ),
    ),
    "step_failed_abort": GuidanceTemplate(
        label="Step failed (abort policy)",
        severity="error",
        summary="A step failed and was configured to abort on failure.",
        next_steps=(
            "Inspect the failed step output and error log entry.",
            "Change that step to retry/skip or fix the root cause, then rerun.",
        ),
    ),
    "max_loops_reached": GuidanceTemplate(
        label="Maximum loops reached",
        severity="warn",
        summary="The chain completed the configured number of loops.",
        next_steps=(
            "Increase Max Loops (or enable Unlimited mode) for deeper passes.",
            "Review current results to choose the next targeted run.",
        ),
    ),
}

_PIPELINE_TEMPLATES: dict[str, GuidanceTemplate] = {
    "preflight_failed": GuidanceTemplate(
        label="Preflight checks failed",
        severity="error",
        summary="The pipeline failed startup validation before execution began.",
        next_steps=(
            "Open Setup Diagnostics and resolve all failed checks.",
            "Retry the pipeline once diagnostics show ready.",
        ),
    ),
    "phase_failed_abort": GuidanceTemplate(
        label="Phase failed (abort policy)",
        severity="error",
        summary="A phase failed and was configured to abort on failure.",
        next_steps=(
            "Inspect phase logs and the latest error details.",
            "Adjust on-failure policy or fix the issue, then rerun.",
        ),
    ),
    "max_cycles_reached": GuidanceTemplate(
        label="Maximum cycles reached",
        severity="warn",
        summary="The pipeline completed the configured number of cycles.",
        next_steps=(
            "Increase Max Cycles (or enable Unlimited mode) to continue.",
            "Review pipeline logs and results to choose the next focus area.",
        ),
    ),
}


def _error_guidance(stop_reason: str) -> dict[str, object]:
    detail = stop_reason.partition(":")[2].strip()
    summary = (
        f"Runtime error: {detail[:240]}"
        if detail
        else "The run stopped due to an unexpected runtime error."
    )
    return {
        "code": stop_reason,
        "label": "Unexpected runtime error",
        "severity": "error",
        "summary": summary,
        "next_steps": [
            "Inspect the latest error log entry for stack trace details.",
            "Fix the underlying issue, then rerun.",
        ],
    }


def _fallback_guidance(stop_reason: str) -> dict[str, object]:
    label = stop_reason.replace("_", " ").strip().capitalize() or "Run finished"
    return {
        "code": stop_reason,
        "label": label,
        "severity": "warn",
        "summary": "Run finished with an unrecognized stop reason.",
        "next_steps": [
            "Check live logs and output files for more context.",
            "Rerun with refined configuration if needed.",
        ],
    }


def _template_to_dict(stop_reason: str, template: GuidanceTemplate) -> dict[str, object]:
    return {
        "code": stop_reason,
        "label": template.label,
        "severity": template.severity,
        "summary": template.summary,
        "next_steps": list(template.next_steps),
    }


def get_stop_guidance(stop_reason: str | None, *, mode: RunMode) -> dict[str, object] | None:
    """Return user-facing guidance for a run stop reason."""
    reason = (stop_reason or "").strip()
    if not reason:
        return None

    if reason.lower().startswith("error:"):
        return _error_guidance(reason)

    template = _SHARED_TEMPLATES.get(reason)
    if template is None:
        mode_templates = _CHAIN_TEMPLATES if mode == "chain" else _PIPELINE_TEMPLATES
        template = mode_templates.get(reason)
    if template is None:
        return _fallback_guidance(reason)
    return _template_to_dict(reason, template)
