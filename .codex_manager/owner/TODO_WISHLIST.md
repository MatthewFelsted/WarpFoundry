# codex_manager Owner Execution Backlog

Last updated: 2026-02-15
Source baseline: `.codex_manager/logs/WISHLIST.md` (WISH-001 to WISH-025)

This file tracks executable work only. Keep items concrete, repo-specific, and in strict ship order.

## Active Queue (execute top to bottom)

### P0 - Immediate correctness and safety

- [ ] [PR-01][WISH-007,WISH-002][S] Prompt and debug log hardening in chain and pipeline
  - Why now: `src/codex_manager/gui/chain.py` still logs `Prompt: ...` and many `[DEBUG]` lines by default.
  - Scope: `src/codex_manager/gui/chain.py`, `src/codex_manager/pipeline/orchestrator.py`, `src/codex_manager/prompt_logging.py`.
  - Done when: default logs are metadata-only for prompts and verbose debug lines require explicit opt-in.
  - Validate: add focused tests in `tests/test_gui_chain_terminate_repeats.py` and `tests/test_pipeline_orchestrator.py`.

- [ ] [PR-02][WISH-011][M] Implement true `on_failure=retry` semantics in pipeline phases
  - Why now: `PhaseConfig.max_retries` exists but pipeline flow still only uses skip/abort behavior.
  - Scope: `src/codex_manager/pipeline/orchestrator.py`, `src/codex_manager/pipeline/phases.py`.
  - Done when: retry attempts are bounded, attempt count is logged, and final phase status reflects retry exhaustion correctly.
  - Validate: extend `tests/test_pipeline_orchestrator.py` with retry success/failure/abort cases.

- [ ] [PR-03][WISH-020][S] Bound pipeline log queue to stop unbounded memory growth
  - Why now: `PipelineOrchestrator.log_queue` is currently unbounded while chain queue is bounded.
  - Scope: `src/codex_manager/pipeline/orchestrator.py`, `src/codex_manager/gui/app.py`.
  - Done when: queue has max size, oldest entries drop on overflow, and overflow warnings are emitted.
  - Validate: add overflow behavior tests in `tests/test_pipeline_orchestrator.py` and stream tests in `tests/test_gui_app_api.py`.

- [ ] [PR-04][WISH-006][M] Make ledger and tracker writes atomic and race-safe
  - Why now: `ledger.py` and `pipeline/tracker.py` still append/write directly without atomic replace strategy.
  - Scope: `src/codex_manager/ledger.py`, `src/codex_manager/pipeline/tracker.py`, shared file-write helper module.
  - Done when: write paths use temp-file + replace (or equivalent lock strategy) and no partial writes under concurrent access.
  - Validate: add stress tests in `tests/test_ledger.py` and `tests/test_pipeline_tracker.py`.

- [ ] [PR-05][WISH-016][S] Unify runtime errors into `.codex_manager/logs/ERRORS.md`
  - Why now: chain writes `.codex_manager/ERRORS.md` while pipeline uses `logs/ERRORS.md`.
  - Scope: `src/codex_manager/gui/chain.py`, `src/codex_manager/pipeline/tracker.py`, relevant read endpoints in `src/codex_manager/gui/app.py`.
  - Done when: both chain and pipeline append to a single canonical errors log with backward-compatible read behavior.
  - Validate: add cross-surface logging assertions in `tests/test_gui_chain_terminate_repeats.py` and `tests/test_gui_app_api.py`.

### P1 - Next wave after P0 closes

- [ ] [PR-06][WISH-003][L] Crash-safe pipeline state persistence and resume
  - Scope: persist cycle/phase cursor each iteration and resume non-self-improvement runs.
  - Key files: `src/codex_manager/pipeline/orchestrator.py`, `src/codex_manager/pipeline/phases.py`, `src/codex_manager/gui/app.py`.
  - Done when: pipeline can restart from saved cursor after process interruption with no duplicate phase writes.

- [ ] [PR-07][WISH-015][S] Strict numeric bounds validation for chain/pipeline config models
  - Scope: add `Field(ge/gt/...)` bounds on retries, loops, cycles, budgets, and timeout fields.
  - Key files: `src/codex_manager/gui/models.py`, `src/codex_manager/pipeline/phases.py`.
  - Done when: invalid numeric configs fail fast with clear validation errors.

- [ ] [PR-08][WISH-004][M] Phase-aware selective test execution
  - Scope: introduce evaluation policy (skip/smoke/full) by phase and change signal.
  - Key files: `src/codex_manager/pipeline/orchestrator.py`, `src/codex_manager/eval_tools.py`.
  - Done when: non-code phases do not pay full test cost and skip reasons are explicit in logs/results.

- [ ] [PR-09][WISH-013][L] Repo-safe parallel chain execution using isolated worktrees
  - Scope: remove shared-working-tree mutation in parallel mode and reconcile outputs deterministically.
  - Key files: `src/codex_manager/gui/chain.py`, `src/codex_manager/git_tools.py`.
  - Done when: parallel steps cannot race each other in one repo checkout and conflict policy is explicit.

### P2 - Strategic architecture and hygiene

- [ ] [PR-10][WISH-001,WISH-009][L] Gemini SDK migration and connector modularization
  - Scope: move off deprecated `google.generativeai`, then split provider logic behind a common adapter interface.
  - Key files: `src/codex_manager/brain/connector.py`, `src/codex_manager/brain/*`, `tests/test_brain_connector_env.py`.

- [ ] [PR-11][WISH-010][M] Unified retention policy for `.codex_manager` artifacts
  - Scope: central retention controls for logs, outputs, screenshots, and archives (not only chain output history).
  - Key files: `src/codex_manager/gui/chain.py`, `src/codex_manager/history_log.py`, `src/codex_manager/brain/logbook.py`, `src/codex_manager/pipeline/tracker.py`.

- [ ] [PR-12][WISH-024,WISH-023,WISH-025][M] Docstring contract cleanup + audit automation
  - Scope: fix stale connector docs, close missing public docstrings, and add regression audit command.
  - Key files: `src/codex_manager/brain/connector.py`, `src/codex_manager/gui/app.py`, `src/codex_manager/gui/chain.py`, `src/codex_manager/ledger.py`.

## Completed / Keep Closed (reopen only on regression)

- [x] [WISH-014] `/api/configs/load` path traversal hardening and config-name sanitization.
- [x] [WISH-019] Robust `test_cmd` parsing for quoting and Windows paths via `parse_test_command`.
- [x] [WISH-012] Dry-run rollback now covers commit-phase paths.
- [x] [WISH-017] `success` now includes validation/test outcomes (`agent_success`, `validation_success`, `tests_passed`).
- [x] [WISH-018] `commit_frequency` behavior implemented for `per_phase`, `per_cycle`, and `manual`.
- [x] [WISH-021] Pipeline log API can read logs even when no live executor is running.
- [x] [WISH-022] UTF-8 resilient log/doc reads with legacy-byte recovery and rewrite.
- [x] [WISH-008] Pipeline preflight checks for binaries, auth, repo writability, and optional deps.

## Completion Gate (before checking any active item)

- `python -m ruff check src tests`
- `python -m pytest -q`
- Update affected docs when behavior/flags/API shape changes (`README.md`, `docs/CLI_REFERENCE.md`, `docs/TROUBLESHOOTING.md`).
