# Feature Dreams

Execution order: top to bottom. Keep this list feature-only and implementation-ready.

## P0 - Highest Value / Lowest Effort

- [x] [M] Add GitHub Actions integration: generate a repo-ready workflow that runs `warpfoundry pipeline` on push/PR and uploads `.codex_manager/logs` plus run summaries as artifacts. (Completed: added `warpfoundry github-actions --repo <path>` workflow generator with artifact upload support.)
- [ ] [M] Add signed commits/tags support (GPG or SSH signing) with setup UI, key validation, and pre-push blocking when signing is misconfigured.
- [ ] [S] Add CLI checkpoint resume support: `warpfoundry pipeline --resume-checkpoint <path>` plus `warpfoundry pipeline --resume-state` for headless recovery without GUI restart.
- [ ] [S] Add run-completion webhooks (Slack/Discord/generic HTTP): send success/failure payloads with repo, run id, stop reason, tests, tokens, and artifact links.
- [ ] [M] Add "Promote Last Dry-Run to Apply" flow that copies the last dry-run config, shows diff/test summary, and starts apply mode with one confirmation.

## P1 - Product Leverage

- [ ] [M] Add per-run cost analytics in Run Comparison: estimate USD by provider/model from token usage, rank runs by cost/performance, and flag budget outliers.
- [ ] [M] Add custom Autopilot recipe editor: create/save/edit/delete per-repo recipes and import/export recipe JSON for sharing.
- [ ] [S] Add one-click artifact bundle export: zip selected outputs/logs/config/history for a run and expose download path in GUI plus API.
- [ ] [M] Add scheduled pipeline runs per workspace repo (daily/weekly plus branch/mode/cycles) with next-run visibility and skip-on-dirty guardrails.
- [ ] [M] Add Feature Dreams suggestion context upload (parity with TODO Wishlist): allow optional context files and include "files scanned/context used" transparency in results.

## P2 - Advanced Features

- [ ] [M] Add transient failure resilience policy for chain/pipeline: classify retryable errors, apply exponential backoff, and optionally fail over between Codex and Claude before marking failure.
- [ ] [M] Add CUA visual regression mode: capture baseline screenshots, compare future runs with configurable diff threshold, and log failed comparisons with saved diff images.
