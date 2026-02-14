from __future__ import annotations

from codex_manager.gui.stop_guidance import get_stop_guidance


def test_stop_guidance_handles_runtime_errors() -> None:
    guidance = get_stop_guidance("error: boom", mode="chain")
    assert guidance
    assert guidance["label"] == "Unexpected runtime error"
    assert guidance["severity"] == "error"
    assert "boom" in guidance["summary"]


def test_stop_guidance_returns_mode_specific_templates() -> None:
    chain_guidance = get_stop_guidance("max_loops_reached", mode="chain")
    pipeline_guidance = get_stop_guidance("max_cycles_reached", mode="pipeline")
    assert chain_guidance
    assert pipeline_guidance
    assert chain_guidance["label"] == "Maximum loops reached"
    assert pipeline_guidance["label"] == "Maximum cycles reached"


def test_stop_guidance_includes_no_progress_template() -> None:
    guidance = get_stop_guidance("no_progress_detected", mode="chain")
    assert guidance
    assert guidance["label"] == "No progress detected"
    assert guidance["severity"] == "info"


def test_stop_guidance_includes_brain_needs_input_template() -> None:
    guidance = get_stop_guidance("brain_needs_input", mode="chain")
    assert guidance
    assert guidance["label"] == "Brain needs user input"
    assert guidance["severity"] == "warn"


def test_stop_guidance_falls_back_for_unknown_codes() -> None:
    guidance = get_stop_guidance("mystery_reason", mode="chain")
    assert guidance
    assert guidance["label"] == "Mystery reason"
    assert guidance["severity"] == "warn"
