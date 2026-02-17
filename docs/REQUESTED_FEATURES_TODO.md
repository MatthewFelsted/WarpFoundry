# Requested Features TODO

Captured on: 2026-02-15

## Implemented in this pass

- Optional per-repo vector memory plumbing (Chroma backend) with pipeline integration.
- Optional deep-research pipeline phase with cache-aware dedupe and `RESEARCH.md` logging.
- Native provider deep-research execution path (OpenAI/Google) with retries, daily quota, and budget controls.
- Deep-research governance filtering for source URLs (HTTPS/domain policy checks).
- Cross-surface vector-memory retrieval/ingestion for chain mode and scientist artifacts.
- Self-improvement UI labeling now explicitly scoped to this program and project display name variable.
- New Project flow now supports foundational prompt artifacts and AI-assisted prompt refinement endpoint.
- One-time foundational bootstrap execution chain with autorun and resume status endpoints.
- Added `Marketing Mode` and `Monetization Mode` presets.
- Replaced chain-tab Owner Decision Board usage with a repo-wide Idea Generator (free-tier behavior) that scans repository files and uses an independent model selection path.
- Added monetization governance checks for claim quality and citation/source policy warnings.
- Added licensing/commercial packaging templates in New Project flow (`docs/*` + `.codex_manager/business/*`).
- Added shared agent protocol artifacts (`docs/AGENT_PROTOCOL.md` and runtime protocol file).
- Added a scheduled model/provider watchdog with persisted snapshots, diffs, and API controls.
- Added watchdog alert API and in-GUI alert banner with actions (`run now`, guide link, mute).
- Added optional legal-review checkpoint state for licensing/pricing docs, with sign-off APIs and draft warnings.
- Added GUI-managed governance source-domain policy controls (allow/deny lists) with persisted settings.
- Added watchdog remediation playbooks with one-click migration suggestions in the alert banner.

## Remaining follow-ups (backlog)

- None at the moment from this request batch.
