"""Data models for task chains, pipeline, and chain execution state."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

CodexSandboxMode = Literal["read-only", "workspace-write", "danger-full-access"]
CodexApprovalPolicy = Literal["untrusted", "on-failure", "on-request", "never"]
CodexReasoningEffort = Literal["inherit", "low", "medium", "high", "xhigh"]
CommitFrequency = Literal["per_phase", "per_cycle", "manual"]
DependencyInstallPolicy = Literal["disallow", "project_only", "allow_system"]
ImageProvider = Literal["openai", "google"]
VectorMemoryBackend = Literal["chroma"]
DeepResearchProviders = Literal["openai", "google", "both"]
DANGER_CONFIRMATION_PHRASE = "I UNDERSTAND"


def _default_image_model(provider: ImageProvider) -> str:
    if provider == "google":
        return "nano-banana"
    return "gpt-image-1"


class TaskStep(BaseModel):
    """A single step in a task chain."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    job_type: str = "feature_discovery"
    prompt_mode: str = "preset"  # "preset" | "ai_decides" | "custom"
    custom_prompt: str = ""
    on_failure: str = "skip"  # "skip" | "retry" | "abort"
    max_retries: int = 1
    loop_count: int = 1  # how many times to run this step before advancing
    enabled: bool = True
    agent: str = "codex"  # "codex" | "claude_code" | "auto"
    # CUA-specific (only used when job_type == "visual_test")
    cua_target_url: str = ""
    cua_provider: str = ""  # "" = use chain default; "openai" | "anthropic"


class ChainConfig(BaseModel):
    """Full configuration for a task chain."""

    name: str = "Untitled Chain"
    repo_path: str = ""
    mode: str = "dry-run"
    steps: list[TaskStep] = Field(default_factory=list)
    max_loops: int = 3
    unlimited: bool = False
    improvement_threshold: float = 1.0  # percent - stop when improvement drops below this
    max_time_minutes: int = 120
    max_total_tokens: int = 2_000_000
    # If True, stop immediately when token budget is reached mid-loop.
    strict_token_budget: bool = False
    stop_on_convergence: bool = True
    test_cmd: str = ""  # empty = skip tests; set to e.g. "python -m pytest -q" for code repos
    codex_binary: str = "codex"
    claude_binary: str = "claude"
    codex_sandbox_mode: CodexSandboxMode = "workspace-write"
    codex_approval_policy: CodexApprovalPolicy = "never"
    codex_reasoning_effort: CodexReasoningEffort = "xhigh"
    codex_bypass_approvals_and_sandbox: bool = False
    codex_danger_confirmation: str = ""
    allow_path_creation: bool = True
    dependency_install_policy: DependencyInstallPolicy = "project_only"
    image_generation_enabled: bool = False
    image_provider: ImageProvider = "openai"
    image_model: str = "gpt-image-1"
    vector_memory_enabled: bool = False
    vector_memory_backend: VectorMemoryBackend = "chroma"
    vector_memory_collection: str = ""
    vector_memory_top_k: int = 8
    # Inactivity timeout in seconds. 0 disables timeout.
    timeout_per_step: int = 0
    parallel_execution: bool = False  # run independent steps concurrently

    # Brain (thinking layer)
    brain_enabled: bool = False
    brain_model: str = "gpt-5.2"

    # Local-only mode — force all AI calls through Ollama
    local_only: bool = False

    # CUA (Computer-Using Agent) visual testing
    cua_enabled: bool = False
    cua_provider: str = "anthropic"  # "openai" | "anthropic"
    cua_target_url: str = ""
    cua_task: str = ""
    cua_headless: bool = True

    @model_validator(mode="after")
    def _validate_danger_confirmation(self) -> ChainConfig:
        """Require explicit confirmation text when dangerous bypass mode is enabled."""
        if (
            self.codex_bypass_approvals_and_sandbox
            and self.codex_danger_confirmation.strip() != DANGER_CONFIRMATION_PHRASE
        ):
            raise ValueError(
                "codex_danger_confirmation must be exactly "
                f"'{DANGER_CONFIRMATION_PHRASE}' when bypass is enabled"
            )
        if not self.image_model.strip():
            self.image_model = _default_image_model(self.image_provider)
        self.vector_memory_top_k = min(30, max(1, int(self.vector_memory_top_k or 8)))
        return self


class StepResult(BaseModel):
    """Recorded result of executing one step."""

    loop_number: int = 0
    step_index: int = 0
    step_name: str = ""
    job_type: str = ""
    agent_used: str = "codex"  # which agent ran this step
    prompt_used: str = ""
    terminate_repeats: bool = False
    agent_success: bool = False
    validation_success: bool = False
    tests_passed: bool = False
    success: bool = False
    test_outcome: str = "error"
    files_changed: int = 0
    net_lines_changed: int = 0
    changed_files: list[dict[str, Any]] = Field(default_factory=list)
    commit_sha: str | None = None
    error_message: str = ""
    duration_seconds: float = 0.0
    output_chars: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class ChainState(BaseModel):
    """Runtime state of a chain execution."""

    running: bool = False
    paused: bool = False
    run_max_loops: int = 0
    run_unlimited: bool = False
    current_loop: int = 0
    current_step: int = 0
    current_step_name: str = ""
    current_step_started_at_epoch_ms: int = 0
    total_steps_completed: int = 0
    total_loops_completed: int = 0
    results: list[StepResult] = Field(default_factory=list)
    stop_reason: str | None = None
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    started_at: str | None = None
    finished_at: str | None = None
    improvement_pct: float = 100.0  # latest computed improvement %
    last_log_epoch_ms: int = 0
    last_log_level: str = ""
    last_log_message: str = ""


# ── Pipeline GUI config ──────────────────────────────────────────


class PipelinePhaseGUI(BaseModel):
    """A pipeline phase as configured from the GUI."""

    phase: str = "ideation"
    enabled: bool = True
    iterations: int = 1
    agent: str = "codex"
    on_failure: str = "skip"  # "skip" | "retry" | "abort"
    custom_prompt: str = ""  # if non-empty, overrides catalog prompt for this phase


class PipelineGUIConfig(BaseModel):
    """Pipeline configuration submitted from the web GUI.

    Stop-condition fields mirror :class:`ChainConfig` for consistency.
    """

    repo_path: str = ""
    mode: str = "dry-run"
    max_cycles: int = 3
    unlimited: bool = False
    agent: str = "codex"
    science_enabled: bool = False
    brain_enabled: bool = False
    brain_model: str = "gpt-5.2"
    phases: list[PipelinePhaseGUI] = Field(default_factory=list)

    # Stop conditions (consistent with ChainConfig)
    improvement_threshold: float = 1.0
    max_time_minutes: int = 240
    max_total_tokens: int = 5_000_000
    # If True, stop immediately when token budget is reached mid-cycle.
    strict_token_budget: bool = False
    stop_on_convergence: bool = True

    # Advanced settings (consistent with ChainConfig)
    test_cmd: str = ""  # empty = skip tests; set to e.g. "python -m pytest -q" for code repos
    codex_binary: str = "codex"
    claude_binary: str = "claude"
    codex_sandbox_mode: CodexSandboxMode = "workspace-write"
    codex_approval_policy: CodexApprovalPolicy = "never"
    codex_reasoning_effort: CodexReasoningEffort = "xhigh"
    codex_bypass_approvals_and_sandbox: bool = False
    codex_danger_confirmation: str = ""
    allow_path_creation: bool = True
    dependency_install_policy: DependencyInstallPolicy = "project_only"
    image_generation_enabled: bool = False
    image_provider: ImageProvider = "openai"
    image_model: str = "gpt-image-1"
    vector_memory_enabled: bool = False
    vector_memory_backend: VectorMemoryBackend = "chroma"
    vector_memory_collection: str = ""
    vector_memory_top_k: int = 8
    deep_research_enabled: bool = False
    deep_research_providers: DeepResearchProviders = "both"
    deep_research_max_age_hours: int = 168
    deep_research_dedupe: bool = True
    deep_research_native_enabled: bool = False
    deep_research_retry_attempts: int = 2
    deep_research_daily_quota: int = 8
    deep_research_max_provider_tokens: int = 12000
    deep_research_budget_usd: float = 5.0
    deep_research_openai_model: str = "gpt-5.2"
    deep_research_google_model: str = "gemini-3-pro-preview"
    self_improvement_enabled: bool = False
    self_improvement_auto_restart: bool = False
    # Inactivity timeout in seconds. 0 disables timeout.
    timeout_per_phase: int = 0

    # Git settings
    auto_commit: bool = True
    commit_frequency: CommitFrequency = "per_phase"

    # Local-only mode — force all AI calls through Ollama
    local_only: bool = False

    # CUA (Computer-Using Agent) visual testing
    cua_enabled: bool = False
    cua_provider: str = "anthropic"  # "openai" | "anthropic"
    cua_target_url: str = ""
    cua_task: str = ""
    cua_headless: bool = True

    @model_validator(mode="after")
    def _validate_danger_confirmation(self) -> PipelineGUIConfig:
        """Require explicit confirmation text when dangerous bypass mode is enabled."""
        if (
            self.codex_bypass_approvals_and_sandbox
            and self.codex_danger_confirmation.strip() != DANGER_CONFIRMATION_PHRASE
        ):
            raise ValueError(
                "codex_danger_confirmation must be exactly "
                f"'{DANGER_CONFIRMATION_PHRASE}' when bypass is enabled"
            )
        if not self.image_model.strip():
            self.image_model = _default_image_model(self.image_provider)
        self.vector_memory_top_k = min(30, max(1, int(self.vector_memory_top_k or 8)))
        self.deep_research_max_age_hours = max(1, int(self.deep_research_max_age_hours or 168))
        self.deep_research_retry_attempts = min(
            6, max(1, int(self.deep_research_retry_attempts or 2))
        )
        self.deep_research_daily_quota = min(100, max(1, int(self.deep_research_daily_quota or 8)))
        self.deep_research_max_provider_tokens = min(
            64000,
            max(512, int(self.deep_research_max_provider_tokens or 12000)),
        )
        self.deep_research_budget_usd = max(0.0, float(self.deep_research_budget_usd or 5.0))
        self.deep_research_openai_model = (
            str(self.deep_research_openai_model or "gpt-5.2").strip() or "gpt-5.2"
        )
        self.deep_research_google_model = (
            str(self.deep_research_google_model or "gemini-3-pro-preview").strip()
            or "gemini-3-pro-preview"
        )
        return self
