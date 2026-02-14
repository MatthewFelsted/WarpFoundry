"""BrainManager — the AI thinking layer that sits above Codex.

The brain:
1. Plans and refines prompts before Codex executes them
2. Evaluates results after each step and decides follow-up actions
3. Handles errors by reasoning about them and proposing fixes
4. Escalates to the user only when it cannot resolve issues itself
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from codex_manager.brain.connector import (
    DEFAULT_LEAD_MODEL,
    connect,
    get_default_ollama_model,
)

logger = logging.getLogger(__name__)


@dataclass
class BrainDecision:
    """Result of the brain's analysis."""

    action: str = "continue"  # continue | follow_up | retry | escalate | skip | stop
    refined_prompt: str = ""
    reasoning: str = ""
    follow_up_prompt: str = ""
    severity: str = "low"  # low | medium | high | critical
    human_message: str = ""  # only set when action == "escalate"


@dataclass
class BrainConfig:
    """Configuration for the brain layer."""

    enabled: bool = False
    model: str = DEFAULT_LEAD_MODEL
    local_only: bool = False  # when True, force all AI calls through Ollama
    temperature: float = 0.3  # lower = more deterministic for planning
    max_output_tokens: int = 4096
    auto_fix_errors: bool = True  # try to fix errors before escalating
    escalation_threshold: int = 3  # consecutive failures before escalating
    timeout: float = 120.0

    def __post_init__(self) -> None:
        """When local_only is set, override the model to use Ollama."""
        if self.local_only and not self.model.startswith("ollama:") and not self.model.startswith("ollama/"):
            self.model = get_default_ollama_model()
            logger.info("Local-only mode: brain model overridden to %s", self.model)


class BrainManager:
    """Orchestrates AI-powered thinking for the manager loop.

    When enabled, the brain intercepts key decision points in the chain:
    - Before each step: refines the prompt for best results
    - After each step: evaluates whether follow-up is needed
    - On errors: reasons about the cause and proposes fixes
    - Between loops: assesses overall progress

    Parameters
    ----------
    config:
        A :class:`BrainConfig` controlling model, temperature, etc.
    """

    def __init__(self, config: BrainConfig | None = None) -> None:
        self.config = config or BrainConfig()
        self._consecutive_failures = 0

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    # ------------------------------------------------------------------
    # Pre-step: refine the prompt
    # ------------------------------------------------------------------

    def plan_step(
        self,
        goal: str,
        step_name: str,
        base_prompt: str,
        history_summary: str = "",
        ledger_context: str = "",
    ) -> str:
        """Refine a step's prompt using the brain model.

        Returns the refined prompt, or the original if the brain is
        disabled or fails. When ledger_context is provided (open errors,
        observations, etc.), the brain can prioritize addressing those items.
        """
        if not self.enabled:
            return base_prompt

        system = (
            "You are an expert software engineering manager. Your job is to refine "
            "a prompt that will be sent to an AI coding agent (Codex). Make the prompt "
            "more specific, actionable, and likely to succeed. Keep it concise.\n\n"
            "Rules:\n"
            "- Output ONLY the refined prompt text, nothing else\n"
            "- Preserve the intent of the original prompt\n"
            "- Add specific guidance based on the history if available\n"
            "- If open items (ledger) are provided, consider prioritizing them\n"
            "- Avoid repeating work that was already done successfully\n"
        )

        user_msg = (
            f"## Goal\n{goal}\n\n"
            f"## Current Step: {step_name}\n\n"
            f"## Original Prompt\n{base_prompt}\n\n"
        )
        if ledger_context:
            user_msg += f"## Open Items (Knowledge Ledger)\n{ledger_context}\n\n"
        if history_summary:
            user_msg += f"## Recent History\n{history_summary}\n\n"

        user_msg += "## Refined Prompt\n"

        try:
            result = self._call(system + "\n\n" + user_msg)
            if result and len(result.strip()) > 20:
                logger.info("[brain] Refined prompt (%d chars → %d chars)", len(base_prompt), len(result))
                return result.strip()
        except Exception as exc:
            logger.warning("[brain] Failed to refine prompt: %s", exc)

        return base_prompt

    # ------------------------------------------------------------------
    # Post-step: evaluate and decide
    # ------------------------------------------------------------------

    def evaluate_step(
        self,
        step_name: str,
        success: bool,
        test_outcome: str,
        files_changed: int,
        net_lines: int,
        errors: list[str],
        goal: str,
        ledger_context: str = "",
    ) -> BrainDecision:
        """Evaluate a step's result and decide what to do next.

        Returns a :class:`BrainDecision` with the recommended action.
        When ledger_context is provided, the brain can consider open items
        when suggesting follow-up actions.
        """
        if not self.enabled:
            if success:
                self._consecutive_failures = 0
                return BrainDecision(action="continue", reasoning="Step succeeded")
            self._consecutive_failures += 1
            return BrainDecision(action="skip", reasoning="Step failed (brain disabled)")

        prompt = (
            "You are evaluating the result of an AI coding step.\n\n"
            f"Step: {step_name}\n"
            f"Success: {success}\n"
            f"Test outcome: {test_outcome}\n"
            f"Files changed: {files_changed}\n"
            f"Net lines changed: {net_lines:+d}\n"
            f"Errors: {'; '.join(errors[:3]) if errors else 'none'}\n"
            f"Goal: {goal}\n\n"
        )
        if ledger_context:
            prompt += f"Open items (knowledge ledger):\n{ledger_context[:1500]}\n\n"
        prompt += (
            "Respond with a JSON object (no markdown):\n"
            '{"action": "continue|follow_up|retry|skip|stop", '
            '"reasoning": "brief explanation", '
            '"follow_up_prompt": "prompt for follow-up if action is follow_up", '
            '"severity": "low|medium|high|critical"}\n'
        )

        try:
            raw = self._call(prompt)
            decision = self._parse_decision(raw)
            if decision.action == "continue":
                self._consecutive_failures = 0
            elif decision.action in ("retry", "follow_up"):
                self._consecutive_failures += 1
            return decision
        except Exception as exc:
            logger.warning("[brain] Evaluation failed: %s", exc)
            self._consecutive_failures += 1 if not success else 0
            return BrainDecision(
                action="continue" if success else "skip",
                reasoning=f"Brain evaluation failed: {exc}",
            )

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def handle_error(
        self,
        error_text: str,
        step_name: str,
        context: str = "",
    ) -> BrainDecision:
        """Analyse an error and decide how to proceed.

        The brain will try to:
        1. Understand the root cause
        2. Suggest a fix that Codex can execute
        3. Or escalate to the human if the issue is beyond automated repair
        """
        if not self.enabled:
            return BrainDecision(
                action="skip" if self._consecutive_failures < self.config.escalation_threshold else "escalate",
                reasoning="Brain disabled; skipping error",
                severity="medium",
            )

        self._consecutive_failures += 1

        if self._consecutive_failures >= self.config.escalation_threshold:
            return BrainDecision(
                action="escalate",
                reasoning=f"{self._consecutive_failures} consecutive failures",
                severity="high",
                human_message=(
                    f"The AI manager has encountered {self._consecutive_failures} consecutive "
                    f"failures. Last error in '{step_name}':\n\n{error_text[:500]}\n\n"
                    "Please review and resolve before resuming."
                ),
            )

        prompt = (
            "You are a senior engineer debugging an error from an AI coding tool.\n\n"
            f"Step: {step_name}\n"
            f"Error:\n{error_text[:2000]}\n\n"
        )
        if context:
            prompt += f"Context:\n{context[:1000]}\n\n"

        prompt += (
            "Respond with JSON (no markdown):\n"
            '{"action": "retry|follow_up|skip|escalate", '
            '"reasoning": "root cause analysis", '
            '"follow_up_prompt": "if action is follow_up, the prompt to fix the issue", '
            '"severity": "low|medium|high|critical", '
            '"human_message": "if escalating, what to tell the user"}\n'
        )

        try:
            raw = self._call(prompt)
            return self._parse_decision(raw)
        except Exception as exc:
            logger.warning("[brain] Error analysis failed: %s", exc)
            return BrainDecision(
                action="skip",
                reasoning=f"Brain error analysis failed: {exc}",
                severity="medium",
            )

    # ------------------------------------------------------------------
    # Progress assessment (between loops)
    # ------------------------------------------------------------------

    def assess_progress(
        self,
        goal: str,
        total_loops: int,
        history_summary: str,
    ) -> BrainDecision:
        """Assess overall progress and decide whether to continue looping."""
        if not self.enabled:
            return BrainDecision(action="continue")

        prompt = (
            "You are assessing the progress of an AI-driven code improvement loop.\n\n"
            f"Goal: {goal}\n"
            f"Loops completed: {total_loops}\n\n"
            f"History:\n{history_summary[:3000]}\n\n"
            "Questions to answer:\n"
            "1. Has meaningful progress been made toward the goal?\n"
            "2. Is there still significant work remaining?\n"
            "3. Should the loop continue, or has it reached diminishing returns?\n\n"
            "Respond with JSON (no markdown):\n"
            '{"action": "continue|stop", '
            '"reasoning": "assessment of progress and remaining work", '
            '"refined_prompt": "if continuing, any adjustments to focus"}\n'
        )

        try:
            raw = self._call(prompt)
            return self._parse_decision(raw)
        except Exception as exc:
            logger.warning("[brain] Progress assessment failed: %s", exc)
            return BrainDecision(action="continue", reasoning=f"Assessment failed: {exc}")

    # ------------------------------------------------------------------
    # Ask a freeform question
    # ------------------------------------------------------------------

    def ask(self, question: str, context: str = "") -> str:
        """Ask the brain a freeform question and return the answer."""
        if not self.enabled:
            return ""
        prompt = question
        if context:
            prompt = f"Context:\n{context[:3000]}\n\nQuestion: {question}"
        try:
            return self._call(prompt)
        except Exception as exc:
            logger.warning("[brain] Question failed: %s", exc)
            return f"[Brain error: {exc}]"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call(self, prompt: str) -> str:
        """Make a single call to the brain model."""
        result = connect(
            self.config.model,
            prompt,
            text_only=True,
            per_request_timeout=self.config.timeout,
            max_output_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            operation="brain",
            disable_cache=True,
        )
        return str(result).strip()

    @staticmethod
    def _parse_decision(raw: str) -> BrainDecision:
        """Parse a JSON decision from the brain's response."""
        import json

        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re

            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                return BrainDecision(action="continue", reasoning=f"Could not parse: {text[:200]}")

        return BrainDecision(
            action=data.get("action", "continue"),
            refined_prompt=data.get("refined_prompt", ""),
            reasoning=data.get("reasoning", ""),
            follow_up_prompt=data.get("follow_up_prompt", ""),
            severity=data.get("severity", "low"),
            human_message=data.get("human_message", ""),
        )

    def reset_failure_count(self) -> None:
        """Reset the consecutive failure counter (e.g. after human intervention)."""
        self._consecutive_failures = 0
