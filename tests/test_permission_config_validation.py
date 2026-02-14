"""Validation tests for Codex permission configuration fields."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from codex_manager.gui.models import (
    DANGER_CONFIRMATION_PHRASE,
    ChainConfig,
    PipelineGUIConfig,
)
from codex_manager.pipeline.phases import PipelineConfig


class TestPermissionConfigValidation:
    def test_timeout_defaults_to_unlimited(self):
        chain = ChainConfig()
        pipe_gui = PipelineGUIConfig()
        pipe = PipelineConfig()
        assert chain.timeout_per_step == 0
        assert pipe_gui.timeout_per_phase == 0
        assert pipe.timeout_per_phase == 0

    def test_chain_config_rejects_invalid_sandbox_mode(self):
        with pytest.raises(ValidationError):
            ChainConfig(codex_sandbox_mode="invalid-mode")

    def test_chain_config_rejects_invalid_approval_policy(self):
        with pytest.raises(ValidationError):
            ChainConfig(codex_approval_policy="prompt-me")

    def test_pipeline_gui_config_rejects_invalid_sandbox_mode(self):
        with pytest.raises(ValidationError):
            PipelineGUIConfig(codex_sandbox_mode="full-trust")

    def test_pipeline_config_rejects_invalid_approval_policy(self):
        with pytest.raises(ValidationError):
            PipelineConfig(codex_approval_policy="always")

    def test_pipeline_gui_config_rejects_invalid_commit_frequency(self):
        with pytest.raises(ValidationError):
            PipelineGUIConfig(commit_frequency="every_step")

    def test_pipeline_config_rejects_invalid_commit_frequency(self):
        with pytest.raises(ValidationError):
            PipelineConfig(commit_frequency="every_step")

    def test_chain_config_rejects_invalid_reasoning_effort(self):
        with pytest.raises(ValidationError):
            ChainConfig(codex_reasoning_effort="ultra")

    def test_pipeline_gui_config_rejects_invalid_reasoning_effort(self):
        with pytest.raises(ValidationError):
            PipelineGUIConfig(codex_reasoning_effort="ultra")

    def test_pipeline_config_rejects_invalid_reasoning_effort(self):
        with pytest.raises(ValidationError):
            PipelineConfig(codex_reasoning_effort="ultra")

    def test_valid_values_are_accepted(self):
        cfg = ChainConfig(
            codex_sandbox_mode="danger-full-access",
            codex_approval_policy="on-failure",
            codex_reasoning_effort="medium",
            codex_bypass_approvals_and_sandbox=True,
            codex_danger_confirmation=DANGER_CONFIRMATION_PHRASE,
        )
        assert cfg.codex_sandbox_mode == "danger-full-access"
        assert cfg.codex_approval_policy == "on-failure"
        assert cfg.codex_reasoning_effort == "medium"
        assert cfg.codex_bypass_approvals_and_sandbox is True

    @pytest.mark.parametrize(
        "cls",
        [ChainConfig, PipelineGUIConfig, PipelineConfig],
    )
    def test_bypass_requires_exact_confirmation_phrase(self, cls):
        with pytest.raises(ValidationError):
            cls(codex_bypass_approvals_and_sandbox=True)
