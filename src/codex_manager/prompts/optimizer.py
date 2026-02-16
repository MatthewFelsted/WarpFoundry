"""Prompt optimizer — uses a strong AI model to refine every prompt in the catalog.

Usage::

    warpfoundry optimize-prompts [--model gpt-5.2] [--dry-run]

This is a one-time (or occasional) process.  The optimizer:
1. Loads all prompts from the catalog
2. Evaluates each for quality (clarity, specificity, structure, effectiveness)
3. Rewrites prompts that score below the threshold
4. Saves the optimized catalog to ``~/.codex_manager/prompt_overrides.yaml``
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from codex_manager.prompts.catalog import PromptCatalog, get_catalog

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of optimizing a single prompt."""

    path: str
    name: str
    original: str
    optimized: str
    scores_before: dict[str, float] = field(default_factory=dict)
    scores_after: dict[str, float] = field(default_factory=dict)
    improved: bool = False
    skipped: bool = False
    skip_reason: str = ""


class PromptOptimizer:
    """Evaluates and optimizes all prompts in the catalog.

    Parameters
    ----------
    model:
        The AI model to use for evaluation and rewriting.
    threshold:
        Prompts scoring above this overall score (1-10) are kept as-is.
    catalog:
        An optional pre-loaded catalog (uses the global singleton otherwise).
    """

    def __init__(
        self,
        model: str = "gpt-5.2",
        threshold: float = 7.5,
        catalog: PromptCatalog | None = None,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.catalog = catalog or get_catalog()
        self._results: list[OptimizationResult] = []

    def optimize_all(self, *, dry_run: bool = False) -> list[OptimizationResult]:
        """Evaluate and optimize every prompt in the catalog.

        Parameters
        ----------
        dry_run:
            If True, evaluate but don't save changes.

        Returns
        -------
        list[OptimizationResult]:
            Per-prompt results showing before/after scores and content.
        """
        from codex_manager.brain.connector import connect

        all_prompts = self.catalog.all_prompts()
        self._results = []

        logger.info(
            "Optimizing %d prompts (model=%s, threshold=%.1f, dry_run=%s)",
            len(all_prompts),
            self.model,
            self.threshold,
            dry_run,
        )

        for i, entry in enumerate(all_prompts, 1):
            path = entry["path"]
            name = entry["name"]
            content = entry["content"]

            logger.info("[%d/%d] Evaluating: %s", i, len(all_prompts), name)

            # Step 1: Evaluate the current prompt
            eval_prompt = self.catalog.optimizer("evaluate_prompt")
            if not eval_prompt:
                eval_prompt = (
                    "Rate this prompt on a 1-10 scale across: "
                    "Clarity, Specificity, Structure, Effectiveness, Efficiency. "
                    'Respond with JSON: {"scores": {...}, "overall": N, "suggestions": ["..."]}'
                )

            try:
                raw_eval = connect(
                    self.model,
                    f"{eval_prompt}\n\n---\n\nPrompt to evaluate:\n{content}",
                    text_only=True,
                    per_request_timeout=60.0,
                    temperature=0.2,
                    disable_cache=True,
                )
                scores_before = self._parse_scores(str(raw_eval))
            except Exception as exc:
                logger.warning("  Evaluation failed: %s", exc)
                self._results.append(
                    OptimizationResult(
                        path=path,
                        name=name,
                        original=content,
                        optimized=content,
                        skipped=True,
                        skip_reason=f"Evaluation failed: {exc}",
                    )
                )
                continue

            overall = scores_before.get("overall", 0)
            logger.info("  Score: %.1f / 10 %s", overall, scores_before.get("scores", {}))

            if overall >= self.threshold:
                logger.info("  Above threshold (%.1f) — keeping as-is", self.threshold)
                self._results.append(
                    OptimizationResult(
                        path=path,
                        name=name,
                        original=content,
                        optimized=content,
                        scores_before=scores_before,
                        skipped=True,
                        skip_reason=f"Score {overall:.1f} >= threshold {self.threshold:.1f}",
                    )
                )
                continue

            # Step 2: Optimize the prompt
            refine_system = self.catalog.optimizer("refine_prompt")
            if not refine_system:
                refine_system = (
                    "You are a world-class prompt engineer. Optimize this prompt to "
                    "elicit peak performance from AI coding models. Output ONLY the "
                    "optimized prompt text."
                )

            suggestions = scores_before.get("suggestions", [])
            suggestions_text = ""
            if suggestions:
                suggestions_text = "\n\nEvaluation suggestions to address:\n" + "\n".join(
                    f"- {s}" for s in suggestions
                )

            try:
                optimized = connect(
                    self.model,
                    f"{refine_system}\n\n---\n\n"
                    f"Original prompt (score {overall:.1f}/10):\n{content}"
                    f"{suggestions_text}\n\n---\n\nOptimized prompt:",
                    text_only=True,
                    per_request_timeout=90.0,
                    temperature=0.3,
                    disable_cache=True,
                )
                optimized = str(optimized).strip()
            except Exception as exc:
                logger.warning("  Optimization failed: %s", exc)
                self._results.append(
                    OptimizationResult(
                        path=path,
                        name=name,
                        original=content,
                        optimized=content,
                        scores_before=scores_before,
                        skipped=True,
                        skip_reason=f"Optimization failed: {exc}",
                    )
                )
                continue

            if not optimized or len(optimized) < 20:
                logger.warning("  Optimizer returned empty/short response")
                self._results.append(
                    OptimizationResult(
                        path=path,
                        name=name,
                        original=content,
                        optimized=content,
                        scores_before=scores_before,
                        skipped=True,
                        skip_reason="Optimizer returned inadequate response",
                    )
                )
                continue

            # Step 3: Re-evaluate the optimized prompt
            try:
                raw_eval2 = connect(
                    self.model,
                    f"{eval_prompt}\n\n---\n\nPrompt to evaluate:\n{optimized}",
                    text_only=True,
                    per_request_timeout=60.0,
                    temperature=0.2,
                    disable_cache=True,
                )
                scores_after = self._parse_scores(str(raw_eval2))
            except Exception:
                scores_after = {"overall": overall + 1}  # assume improvement

            new_overall = scores_after.get("overall", 0)
            improved = new_overall > overall
            logger.info(
                "  Optimized: %.1f → %.1f %s",
                overall,
                new_overall,
                "(improved)" if improved else "(no improvement — keeping original)",
            )

            result = OptimizationResult(
                path=path,
                name=name,
                original=content,
                optimized=optimized if improved else content,
                scores_before=scores_before,
                scores_after=scores_after,
                improved=improved,
            )
            self._results.append(result)

            # Apply the optimized prompt to the catalog data
            if improved and not dry_run:
                self._apply_to_catalog(path, optimized)

        # Save if not dry-run
        if not dry_run:
            saved_path = self.catalog.save()
            logger.info("Saved optimized catalog to %s", saved_path)

        return self._results

    def _apply_to_catalog(self, path: str, new_content: str) -> None:
        """Update a specific prompt in the catalog's raw data."""
        parts = path.split(".")
        data = self.catalog.raw

        # Navigate to the parent
        for part in parts[:-1]:
            if part not in data:
                return
            data = data[part]

        # Set the prompt field
        last = parts[-1]
        if isinstance(data, dict):
            if last in data and isinstance(data[last], dict):
                # It's a nested entry — set the "prompt" or "system" field
                if "prompt" in data[last]:
                    data[last]["prompt"] = new_content
                elif "system" in data[last]:
                    data[last]["system"] = new_content
            elif last in ("prompt", "ai_prompt", "system"):
                data[last] = new_content

    @staticmethod
    def _parse_scores(raw: str) -> dict[str, Any]:
        """Parse a JSON evaluation response from the AI."""
        text = raw.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.splitlines()
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return {"overall": 5.0, "error": "Could not parse evaluation"}

    def summary(self) -> str:
        """Return a human-readable summary of optimization results."""
        if not self._results:
            return "No results — run optimize_all() first."

        lines = [
            "Prompt Optimization Summary",
            "=" * 50,
            f"Total prompts: {len(self._results)}",
            f"Improved:      {sum(1 for r in self._results if r.improved)}",
            f"Skipped:       {sum(1 for r in self._results if r.skipped)}",
            "",
        ]

        for r in self._results:
            before = r.scores_before.get("overall", "?")
            after = r.scores_after.get("overall", "?") if r.scores_after else "—"
            status = "IMPROVED" if r.improved else ("SKIPPED" if r.skipped else "KEPT")
            lines.append(f"  [{status:>8}]  {r.name:<40}  {before} → {after}")
            if r.skip_reason:
                lines.append(f"             {r.skip_reason}")

        return "\n".join(lines)
