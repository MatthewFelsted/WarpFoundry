"""Pipeline phase definitions and configuration.

Each phase in the autonomous pipeline has:
- A prompt (loaded from the prompt catalog)
- A target log file it reads from and writes to
- An iteration count (how many times to loop within the phase)
- Enable/disable toggles
"""

from __future__ import annotations

from enum import Enum
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


class PipelinePhase(str, Enum):
    """The phases of the autonomous improvement pipeline."""

    IDEATION = "ideation"
    PRIORITIZATION = "prioritization"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    DEBUGGING = "debugging"
    COMMIT = "commit"
    DEEP_RESEARCH = "deep_research"
    APPLY_UPGRADES_AND_RESTART = "apply_upgrades_and_restart"
    PROGRESS_REVIEW = "progress_review"

    # Visual testing phase (CUA — computer-using agent)
    VISUAL_TEST = "visual_test"

    # Scientist phases (optional, advanced)
    THEORIZE = "theorize"
    EXPERIMENT = "experiment"
    SKEPTIC = "skeptic"
    ANALYZE = "analyze"


# Map phases to their primary log file
PHASE_LOG_FILES: dict[PipelinePhase, str] = {
    PipelinePhase.IDEATION: "WISHLIST.md",
    PipelinePhase.PRIORITIZATION: "WISHLIST.md",
    PipelinePhase.IMPLEMENTATION: "WISHLIST.md",
    PipelinePhase.TESTING: "TESTPLAN.md",
    PipelinePhase.DEBUGGING: "ERRORS.md",
    PipelinePhase.COMMIT: "PROGRESS.md",
    PipelinePhase.DEEP_RESEARCH: "RESEARCH.md",
    PipelinePhase.APPLY_UPGRADES_AND_RESTART: "PROGRESS.md",
    PipelinePhase.PROGRESS_REVIEW: "PROGRESS.md",
    PipelinePhase.VISUAL_TEST: "TESTPLAN.md",
    PipelinePhase.THEORIZE: "EXPERIMENTS.md",
    PipelinePhase.EXPERIMENT: "EXPERIMENTS.md",
    PipelinePhase.SKEPTIC: "EXPERIMENTS.md",
    PipelinePhase.ANALYZE: "EXPERIMENTS.md",
}

# Default iteration counts per phase
DEFAULT_ITERATIONS: dict[PipelinePhase, int] = {
    PipelinePhase.IDEATION: 3,  # Generate ideas 3 times
    PipelinePhase.PRIORITIZATION: 1,  # Prioritize once
    PipelinePhase.IMPLEMENTATION: 5,  # Implement top 5 bundles
    PipelinePhase.TESTING: 2,  # Design + run tests twice
    PipelinePhase.DEBUGGING: 3,  # Debug up to 3 rounds
    PipelinePhase.COMMIT: 1,  # One commit pass
    PipelinePhase.DEEP_RESEARCH: 1,  # One deep-research sweep
    PipelinePhase.APPLY_UPGRADES_AND_RESTART: 1,  # One checkpoint/restart pass
    PipelinePhase.PROGRESS_REVIEW: 1,  # Review once
    PipelinePhase.VISUAL_TEST: 1,  # Visual test once per cycle
    PipelinePhase.THEORIZE: 2,  # Generate hypotheses twice
    PipelinePhase.EXPERIMENT: 3,  # Run up to 3 experiments
    PipelinePhase.SKEPTIC: 1,  # Independent challenge/replication pass
    PipelinePhase.ANALYZE: 1,  # Analyze once
}

# Default phase execution order
DEFAULT_PHASE_ORDER: list[PipelinePhase] = [
    PipelinePhase.IDEATION,
    PipelinePhase.PRIORITIZATION,
    PipelinePhase.IMPLEMENTATION,
    PipelinePhase.TESTING,
    PipelinePhase.DEBUGGING,
    PipelinePhase.COMMIT,
    PipelinePhase.PROGRESS_REVIEW,
]

# CUA visual test phase (appended when enabled)
CUA_PHASES: list[PipelinePhase] = [
    PipelinePhase.VISUAL_TEST,
]

# Science phases are inserted near the start of the lifecycle when enabled
SCIENCE_PHASES: list[PipelinePhase] = [
    PipelinePhase.THEORIZE,
    PipelinePhase.EXPERIMENT,
    PipelinePhase.SKEPTIC,
    PipelinePhase.ANALYZE,
]

# Self-improvement restart checkpoint phase (optional)
SELF_IMPROVEMENT_PHASES: list[PipelinePhase] = [
    PipelinePhase.APPLY_UPGRADES_AND_RESTART,
]

# Deep-research phase (optional, runs before prioritization by default)
DEEP_RESEARCH_PHASES: list[PipelinePhase] = [
    PipelinePhase.DEEP_RESEARCH,
]


class PhaseConfig(BaseModel):
    """Configuration for a single pipeline phase."""

    phase: PipelinePhase
    enabled: bool = True
    iterations: int = 1
    agent: str = "codex"  # "codex" | "claude_code" | "auto"
    on_failure: str = "skip"  # "skip" | "retry" | "abort"
    max_retries: int = 2
    custom_prompt: str = ""  # if non-empty, overrides catalog prompt for this phase

    @classmethod
    def defaults_for(
        cls,
        phase: PipelinePhase,
        *,
        agent: str = "codex",
    ) -> PhaseConfig:
        """Return the default config for a phase."""
        return cls(
            phase=phase,
            iterations=DEFAULT_ITERATIONS.get(phase, 1),
            agent=agent,
        )


class PipelineConfig(BaseModel):
    """Full configuration for the autonomous pipeline.

    Mirrors :class:`ChainConfig` stop-condition fields for consistency.
    """

    # Core settings
    mode: str = "dry-run"  # "dry-run" | "apply"
    max_cycles: int = 3  # How many full pipeline cycles to run
    unlimited: bool = False  # loop until diminishing returns
    agent: str = "codex"
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
    deep_research_openai_model: str = "gpt-5.3"
    deep_research_google_model: str = "gemini-3-pro-preview"
    self_improvement_enabled: bool = False
    self_improvement_auto_restart: bool = False
    # Inactivity timeout in seconds. 0 disables timeout.
    timeout_per_phase: int = 0
    test_cmd: str = "python -m pytest -q"

    # Phase configuration
    phases: list[PhaseConfig] = Field(default_factory=list)

    # Science mode
    science_enabled: bool = False

    # CUA (Computer-Using Agent) visual testing
    cua_enabled: bool = False
    cua_provider: str = "openai"  # "openai" | "anthropic"
    cua_target_url: str = ""  # URL to visually test
    cua_task: str = ""  # What to test / verify visually
    cua_headless: bool = True

    # Brain (thinking layer)
    brain_enabled: bool = False
    brain_model: str = "gpt-5.3"

    # Local-only mode — force all AI calls through Ollama
    local_only: bool = False

    # Stop conditions (consistent with ChainConfig)
    max_total_tokens: int = 5_000_000
    # If True, stop immediately when token budget is reached mid-cycle.
    strict_token_budget: bool = False
    max_time_minutes: int = 240
    stop_on_convergence: bool = True
    improvement_threshold: float = 1.0  # percent - stop when improvement drops below this

    # Git settings
    auto_commit: bool = True  # commit after implementation + debugging phases
    commit_frequency: CommitFrequency = "per_phase"

    @model_validator(mode="after")
    def _validate_danger_confirmation(self) -> PipelineConfig:
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
            str(self.deep_research_openai_model or "gpt-5.3").strip() or "gpt-5.3"
        )
        self.deep_research_google_model = (
            str(self.deep_research_google_model or "gemini-3-pro-preview").strip()
            or "gemini-3-pro-preview"
        )
        return self

    def get_phase_order(self) -> list[PhaseConfig]:
        """Return phases in execution order, respecting enable flags."""
        if self.phases:
            return [p for p in self.phases if p.enabled]

        # Build default phase list
        order = list(DEFAULT_PHASE_ORDER)
        if self.science_enabled:
            # Scientist mode runs before implementation so findings can inform
            # prioritization and code changes in the same cycle.
            try:
                insert_at = order.index(PipelinePhase.IMPLEMENTATION)
            except ValueError:
                insert_at = 0
            order[insert_at:insert_at] = list(SCIENCE_PHASES)
        if self.deep_research_enabled:
            try:
                insert_at = order.index(PipelinePhase.PRIORITIZATION)
            except ValueError:
                insert_at = 1 if order else 0
            order[insert_at:insert_at] = list(DEEP_RESEARCH_PHASES)
        if self.cua_enabled:
            order.extend(CUA_PHASES)
        if self.self_improvement_enabled:
            order.extend(SELF_IMPROVEMENT_PHASES)

        # When per-phase overrides are not provided, inherit the global agent.
        return [PhaseConfig.defaults_for(p, agent=self.agent) for p in order]


class PhaseResult(BaseModel):
    """Result of executing a single phase iteration."""

    cycle: int = 0
    phase: str
    iteration: int
    agent_success: bool = False
    validation_success: bool = False
    tests_passed: bool = False
    success: bool = False
    test_outcome: str = "skipped"
    test_summary: str = ""
    test_exit_code: int = -1
    files_changed: int = 0
    net_lines_changed: int = 0
    changed_files: list[dict[str, Any]] = Field(default_factory=list)
    commit_sha: str | None = None
    error_message: str = ""
    duration_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    prompt_used: str = ""
    agent_final_message: str = ""
    terminate_repeats: bool = False
    agent_used: str = "codex"
    science_trial_id: str = ""
    science_experiment_id: str = ""
    science_hypothesis_id: str = ""
    science_hypothesis_title: str = ""
    science_confidence: str = ""
    science_verdict: str = ""
    science_verdict_rationale: str = ""
    science_tradeoff_deltas: dict[str, int] = Field(default_factory=dict)
    science_rolled_back: bool = False


class PipelineState(BaseModel):
    """Full runtime state of a pipeline execution."""

    running: bool = False
    paused: bool = False
    current_cycle: int = 0
    current_phase: str = ""
    current_iteration: int = 0
    current_phase_started_at_epoch_ms: int = 0
    total_cycles_completed: int = 0
    total_phases_completed: int = 0
    results: list[PhaseResult] = Field(default_factory=list)
    successes: int = 0
    failures: int = 0
    stop_reason: str | None = None
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    started_at: str | None = None
    finished_at: str | None = None
    improvement_pct: float = 100.0
    last_log_epoch_ms: int = 0
    last_log_level: str = ""
    last_log_message: str = ""
    restart_required: bool = False
    restart_checkpoint_path: str = ""
    resume_cycle: int = 0
    resume_phase_index: int = 0

    def to_summary(self, *, since_results: int | None = None) -> dict[str, Any]:
        """Return a summary dict for API responses.

        When ``since_results`` is provided, returns only new result rows as
        ``results_delta`` to keep polling payloads small.
        """
        total_results = len(self.results)
        payload: dict[str, Any] = {
            "running": self.running,
            "paused": self.paused,
            "current_cycle": self.current_cycle,
            "current_phase": self.current_phase,
            "current_iteration": self.current_iteration,
            "current_phase_started_at_epoch_ms": self.current_phase_started_at_epoch_ms,
            "total_cycles": self.total_cycles_completed,
            "total_phases": self.total_phases_completed,
            "total_results": total_results,
            "stop_reason": self.stop_reason,
            "total_tokens": self.total_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "improvement_pct": self.improvement_pct,
            "successes": self.successes,
            "failures": self.failures,
            "last_log_epoch_ms": self.last_log_epoch_ms,
            "last_log_level": self.last_log_level,
            "last_log_message": self.last_log_message,
            "restart_required": self.restart_required,
            "restart_checkpoint_path": self.restart_checkpoint_path,
            "resume_cycle": self.resume_cycle,
            "resume_phase_index": self.resume_phase_index,
        }
        if since_results is None:
            payload["results"] = [r.model_dump() for r in self.results]
            return payload

        offset = min(max(0, since_results), total_results)
        payload["results_delta"] = [r.model_dump() for r in self.results[offset:]]
        return payload
