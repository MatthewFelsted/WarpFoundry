# Outputs and Artifacts

Use this guide to understand what WarpFoundry writes into your target repository and how to inspect it quickly.

## Fast path in the GUI

1. Set the repository path in Chain Builder or Pipeline.
2. Run at least one step/phase.
3. In the Execution panel, open **Step Outputs** to browse generated files.
4. When a run stops, read the **Why it stopped** card for next actions.

You can also open this guide in-app from the header via **Outputs**.

## Chain mode artifacts

Inside the target repository:

- `.codex_manager/outputs/*.md`: Per-step outputs named from step titles.
- `.codex_manager/ERRORS.md`: Aggregated chain errors for the current run family.
- `.codex_manager/state.json`: Run state snapshot.

## Pipeline mode artifacts

Inside the target repository:

- `.codex_manager/logs/WISHLIST.md`: Candidate improvements and planning notes.
- `.codex_manager/logs/TESTPLAN.md`: Test strategy and execution notes.
- `.codex_manager/logs/ERRORS.md`: Pipeline failure log.
- `.codex_manager/logs/EXPERIMENTS.md`: Experiment tracking notes.
- `.codex_manager/logs/RESEARCH.md`: Deep-research findings and cache reuse notes (when deep-research mode is enabled).
- `.codex_manager/logs/SCIENTIST_REPORT.md`: Scientist dashboard with evidence tables, TODO checklist, and implementation rollout status.
- `.codex_manager/logs/PROGRESS.md`: Progress summaries across phases/cycles.
- `.codex_manager/logs/BRAIN.md`: Brain decisions (when brain mode is enabled).
- `.codex_manager/logs/HISTORY.md`: Historical run notes.
- `.codex_manager/logs/scientist/*`: Scientist loop artifacts (if enabled).
- `.codex_manager/memory/vector_events.jsonl`: Long-term memory journal for vector retrieval context.
- `.codex_manager/memory/deep_research_cache.jsonl`: Deep-research cache entries for dedupe.
- `.codex_manager/owner/decision_board.json`: Owner approve/hold/deny decisions for monetization outputs.
- `.codex_manager/business/licensing_profile.json`: Licensing/commercial profile (if enabled in New Project flow).
- `.codex_manager/business/legal_review.json`: Legal review checkpoint/sign-off state for licensing/pricing publication readiness.
- `docs/LICENSING_STRATEGY.md`: Draft OSS/commercial licensing strategy template.
- `docs/COMMERCIAL_OFFERING.md`: Draft commercial packaging template.
- `docs/PRICING_TIERS.md`: Draft tiered pricing template (optional).

Outside target repositories (home-level shared telemetry):

- `~/.codex_manager/watchdog/config.json`: Model-watchdog schedule/config.
- `~/.codex_manager/watchdog/state.json`: Last run status and next due timing.
- `~/.codex_manager/watchdog/model_catalog_latest.json`: Latest provider model catalog snapshot.
- `~/.codex_manager/watchdog/model_catalog_history.ndjson`: Historical catalog snapshots with diffs.
- `~/.codex_manager/governance/source_policy.json`: GUI-managed source-domain allow/deny policy for governance checks.

## Reading outputs effectively

- Prioritize files touched in the last run by timestamp.
- Start with errors first (`ERRORS.md`) before reviewing feature outputs.
- Compare dry-run outputs against apply-mode intent before committing changes.
- Keep step names explicit so output filenames are easy to track.
- In the Pipeline tab, set **Repository Path** before opening log tabs so logs can be read even when no pipeline is currently running.

## Common confusion points

- "No outputs listed": confirm repo path is set and at least one step has finished.
- "Missing logs": pipeline logs appear only after relevant phases run; empty files mean that phase has not written entries yet.
- "Unexpected filename": output filenames are generated from step names and may be slugified.

