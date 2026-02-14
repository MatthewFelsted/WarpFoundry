"""Scientist module — hypothesis-driven self-improvement engine.

The scientist applies the scientific method to software improvement:

1. **Theorize** — Observe the codebase and form testable hypotheses
2. **Experiment** — Run controlled experiments (change one variable, measure)
3. **Skeptic** — Independently challenge and replicate claims
4. **Analyze** — Synthesize results into actionable insights

`ScientistEngine` is a compatibility wrapper that delegates to the pipeline
science runtime, keeping one authoritative implementation path.
"""

from codex_manager.scientist.engine import ScientistEngine

__all__ = ["ScientistEngine"]
