# Agent Protocol

This document defines shared execution standards for Codex Manager components and any cooperating agents.

## Source-of-Truth Locations

- Runtime logs: `.codex_manager/logs/*.md`
- Knowledge ledger: `.codex_manager/ledger/*`
- Long-term memory and research cache: `.codex_manager/memory/*`
- Foundational project setup artifacts: `.codex_manager/foundation/*`

## Behavioral Rules

1. Stay aligned to the active repository objective and current phase scope.
2. Reuse prior context before generating new work; avoid duplicate effort.
3. Prefer actionable outputs with clear implementation implications.
4. Make assumptions explicit whenever evidence is incomplete.
5. Keep changes incremental, testable, and reversible.
6. Log blockers and next steps instead of silently failing.
7. Avoid absolute marketing/legal guarantees unless verified and approved.

## Coordination Contract

- Input contract:
  - Read current phase log and relevant cross-phase logs.
  - Read open items from the knowledge ledger.
  - Read vector memory/research hits when available.
- Output contract:
  - Write concise, structured updates to phase logs.
  - Reference related ledger/memory IDs when reusing findings.
  - Record key decisions and rationale in `PROGRESS.md`.

## Quality Gate

- Recommendations must be repository-relevant and implementable.
- Avoid speculative churn that does not move project quality forward.
- When uncertain, propose small safe experiments with measurable outcomes.
- For research-backed recommendations, include HTTPS sources and flag low-trust domains.
