"""Tests for agent control tag parsing helpers."""

from __future__ import annotations

from codex_manager.agent_signals import (
    TERMINATE_STEP_TAG,
    contains_terminate_step_signal,
    terminate_step_instruction,
)


def test_contains_terminate_step_signal_recognizes_tag_variants() -> None:
    assert contains_terminate_step_signal(TERMINATE_STEP_TAG)
    assert contains_terminate_step_signal("[Terminate Step]")
    assert contains_terminate_step_signal("Done.\n[terminate-step]\nNothing else to do.")


def test_contains_terminate_step_signal_ignores_plain_text_without_tag() -> None:
    assert not contains_terminate_step_signal("terminate step if needed")
    assert not contains_terminate_step_signal("No-op")


def test_terminate_step_instruction_mentions_canonical_tag() -> None:
    text = terminate_step_instruction("step repeat")
    assert TERMINATE_STEP_TAG in text
    assert "skip" in text.lower()

