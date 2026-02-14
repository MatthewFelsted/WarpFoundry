# Outputs and Artifacts

Use this guide to understand what Codex Manager writes into your target repository and how to inspect it quickly.

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
- `.codex_manager/logs/SCIENTIST_REPORT.md`: Scientist dashboard with evidence tables, TODO checklist, and implementation rollout status.
- `.codex_manager/logs/PROGRESS.md`: Progress summaries across phases/cycles.
- `.codex_manager/logs/BRAIN.md`: Brain decisions (when brain mode is enabled).
- `.codex_manager/logs/HISTORY.md`: Historical run notes.
- `.codex_manager/logs/scientist/*`: Scientist loop artifacts (if enabled).

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
