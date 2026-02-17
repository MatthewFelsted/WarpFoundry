# Agent Protocol

This file defines shared run-time coordination rules for all agents.

## Core Rules

1. Stay aligned to repository goals and current phase scope.
2. Reuse existing project context before proposing new work.
3. Avoid duplicate work when prior research or memory already covers a topic.
4. When uncertain, state assumptions explicitly in outputs/logs.
5. Keep edits incremental, testable, and reversible.
6. Avoid absolute marketing/legal guarantees unless verified by trusted sources.

## Coordination Contract

- Inputs:
  - `.codex_manager/logs/*.md` phase logs
  - `.codex_manager/ledger/*` open-item ledger
  - `.codex_manager/memory/*` long-term memory / research cache
- Outputs:
  - Update phase logs with concise, structured findings
  - Reference IDs for ledger/memory items when reusing prior context
  - Record decision rationale in `PROGRESS.md`

## Safety + Quality

- Prefer high-signal, directly actionable recommendations.
- Avoid speculative changes unrelated to project improvement.
- If blocked by missing credentials/tools, log the blocker and next step.
- For research-backed claims, include HTTPS source URLs from credible domains.
- Flag low-trust or policy-blocked sources for owner review.
