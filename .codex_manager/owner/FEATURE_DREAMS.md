# Feature Dreams

Execution order: top to bottom. Keep this list feature-only and implementation-ready.

## P0 - Highest Value / Lowest Effort

- [x] [M] Add GitHub Actions integration: generate a repo-ready workflow that runs `warpfoundry pipeline` on push/PR and uploads `.codex_manager/logs` plus run summaries as artifacts. (Completed: added `warpfoundry github-actions --repo <path>` workflow generator with artifact upload support.)
- [x] [M] Add signed commits/tags support (GPG or SSH signing) with setup UI, key validation, and pre-push blocking when signing is misconfigured. (Completed: added Git Sync Signing modal + `/api/git/signing` setup/validation endpoints and push guard that blocks misconfigured signing before `git push`.)
- [x] [S] Add CLI checkpoint resume support: `warpfoundry pipeline --resume-checkpoint <path>` plus `warpfoundry pipeline --resume-state` for headless recovery without GUI restart. (Completed: added pipeline resume flags, checkpoint payload loading/validation, resume-state default lookup, repo mismatch guard, and checkpoint-consume cleanup in CLI mode with docs/tests coverage.)
- [x] [S] Add run-completion webhooks (Slack/Discord/generic HTTP): send success/failure payloads with repo, run id, stop reason, tests, tokens, and artifact links. (Completed: added pipeline run-completion webhook delivery with Slack/Discord formatting plus generic JSON payloads, CLI + GUI config wiring, payload includes run id/stop reason/tests/tokens/artifact links, and preflight URL validation.)
- [x] [M] Add "Promote Last Dry-Run to Apply" flow that copies the last dry-run config, shows diff/test summary, and starts apply mode with one confirmation. (Completed: added dry-run promotion preview/start APIs, run diff-summary aggregation in run-comparison history, and a one-click Pipeline UI action that confirms using test+diff summary before starting apply mode from the promoted config.)

## P1 - Product Leverage

- [x] [M] Add per-run cost analytics in Run Comparison: estimate USD by provider/model from token usage, rank runs by cost/performance, and flag budget outliers. (Completed: run-comparison aggregation now computes per-run/model estimated USD cost from history token usage, emits lowest-cost + best-cost-efficiency badges/IDs, flags budget outliers, and surfaces cost columns/details in the GUI.)
- [x] [M] Add custom Autopilot recipe editor: create/save/edit/delete per-repo recipes and import/export recipe JSON for sharing. (Completed: added per-repo custom recipe persistence + validation, CRUD/import/export APIs, dynamic Easy-mode custom recipe cards, and a GUI recipe-management modal for JSON editing/import/export.)
- [x] [S] Add one-click artifact bundle export: zip selected outputs/logs/config/history for a run and expose download path in GUI plus API. (Completed: added run-comparison artifact bundle export/download APIs with include toggles, run-scoped history/config capture, zip bundles under `.codex_manager/output_history/artifact_bundles`, and per-run "Export Bundle" UI action.)
- [ ] [M] Add scheduled pipeline runs per workspace repo (daily/weekly plus branch/mode/cycles) with next-run visibility and skip-on-dirty guardrails.
- [x] [M] Add Feature Dreams suggestion context upload (parity with TODO Wishlist): allow optional context files and include "files scanned/context used" transparency in results. (Completed: Feature Dreams modal now supports context-file uploads with client-side limits, suggestion API normalizes and passes context files into prompt context, and responses report `context_files_used` for transparency.)

## P2 - Advanced Features

- [ ] [M] Add transient failure resilience policy for chain/pipeline: classify retryable errors, apply exponential backoff, and optionally fail over between Codex and Claude before marking failure.
- [ ] [M] Add CUA visual regression mode: capture baseline screenshots, compare future runs with configurable diff threshold, and log failed comparisons with saved diff images.
