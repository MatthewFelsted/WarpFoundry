# codex_manager Owner Execution Backlog

Last updated: 2026-02-23
Source baseline: `.codex_manager/logs/WISHLIST.md` (WISH-001 to WISH-050)

This file tracks executable work only. Keep items concrete, repo-specific, and in strict ship order.

## Active Queue (execute top to bottom)

### P0 - Immediate correctness, security, and run control

- [x] [PR-13][WISH-037][S] Add a repo-level mutating run lock shared by `/api/chain/start` and `/api/pipeline/start`; return `409` with lock owner metadata when busy. Done when chain and pipeline cannot run concurrently for the same resolved repo path and stale locks are cleared on finalization. Validate with new concurrent-start API tests in `tests/test_gui_app_api.py`. Completed 2026-02-23: added shared resolved-path run lock + owner metadata conflict payloads, hooked lock release to chain/pipeline finalization, and covered chain->pipeline/pipeline->chain conflicts plus finalization release in GUI API tests.
- [x] [PR-14][WISH-026][M] Wire `stop()` to hard-cancel active agent subprocesses (`terminate` then `kill` fallback). Done when stop requests complete quickly even during hung CLI calls and final state records cancellation explicitly. Validate with long-running subprocess cancellation tests in `tests/test_runner_common.py`, `tests/test_gui_chain_terminate_repeats.py`, and `tests/test_pipeline_orchestrator.py`. Completed 2026-02-23: added shared cancel-event subprocess termination (`terminate` + forced kill fallback), runner `stop()` hooks, chain/pipeline active-run cancellation wiring, and regression tests proving hung subprocess stop-cancel behavior plus explicit cancellation state/error propagation.
- [ ] [PR-15][WISH-036][S] Add wall-clock timeout controls (separate from inactivity timeout) for chain and pipeline runs. Done when noisy but overlong executions are forcibly ended with timeout type metadata (`inactivity` vs `wall_clock`). Validate with timeout regression tests in `tests/test_runner_common.py` and orchestration assertions in `tests/test_pipeline_orchestrator.py`.
- [ ] [PR-16][WISH-035][M] Enforce capability-contract policies at runtime (`allow_path_creation`, `dependency_install_policy`) instead of prompt-only guidance. Done when violations are detected from repo delta + `command_executions` and failed step/phase results include clear policy reason text. Validate with policy matrix tests in `tests/test_gui_chain_terminate_repeats.py` and `tests/test_pipeline_orchestrator.py`.
- [ ] [PR-17][WISH-047][S] Remove raw prompt text from git commit messages (`generate_commit_message`) and redact sensitive free-text fragments. Done when commit messages are metadata-based and tests prove prompt/secret strings do not leak into git history. Validate with unit tests in `tests/test_git_tools.py` plus existing commit-path tests.
- [ ] [PR-25][WISH-030][M] Protect mutating GUI APIs with same-session CSRF token + localhost origin checks. Done when cross-origin/browser-forged mutation requests fail with `403` and same-session GUI actions continue to work. Validate with API security tests in `tests/test_gui_app_api.py`.

### P1 - High-value reliability and UX correctness

- [ ] [PR-06][WISH-003][M] Persist and resume pipeline state after crashes/restarts (beyond self-improvement checkpoint handoff). Done when cycle/phase cursor and essential run metadata are atomically saved each iteration and resumed without duplicate phase side effects. Validate with crash/resume simulations in `tests/test_pipeline_orchestrator.py` and CLI resume tests in `tests/test_main_entrypoint.py`.
- [ ] [PR-18][WISH-029][S] Add configurable repo-delta validation (`requires_repo_delta`) to chain steps and pipeline phases. Done when artifact-only work can succeed where explicitly allowed and mutating phases still default to strict delta checks. Validate with chain/pipeline policy tests in `tests/test_gui_chain_terminate_repeats.py` and `tests/test_pipeline_orchestrator.py`.
- [ ] [PR-08][WISH-004][M] Extend selective test execution with change-sensitive gating (in addition to current phase policy). Done when non-code-only deltas skip expensive tests with explicit skip reasons and no regression in failure detection. Validate with evaluator/orchestrator tests in `tests/test_pipeline_orchestrator.py`.
- [ ] [PR-19][WISH-043][S] Fail `visual_test` steps/phases on configurable major/critical CUA findings. Done when success status incorporates observation severity threshold instead of transport success alone. Validate with severity-mix tests in `tests/test_cua_session.py`, `tests/test_gui_chain_terminate_repeats.py`, and `tests/test_pipeline_orchestrator.py`.
- [ ] [PR-20][WISH-045][S] Store CUA artifacts under repo-local `.codex_manager/logs/cua/<run_id>/` with a `session.json` manifest and stable relative links in logs. Done when chain/pipeline default to repo-local CUA outputs and manifests are produced for each run. Validate with path + manifest tests in `tests/test_cua_session.py` and `tests/test_pipeline_orchestrator.py`.
- [ ] [PR-21][WISH-034][S] Require exact standalone `[TERMINATE_STEP]` signal lines (no fuzzy substring matching). Done when quoted/explanatory mentions never trigger termination and accepted signals are explicitly audit-logged. Validate with parser tests in `tests/test_agent_signals.py` and integration checks in chain/pipeline suites.

### P2 - Architecture, maintainability, and DX

- [ ] [PR-07][WISH-015][S] Add strict `Field(...)` bounds for chain/pipeline numeric config fields and remove silent coercion surprises. Done when invalid ranges fail fast with consistent API/CLI error messaging. Validate with boundary tests in `tests/test_permission_config_validation.py` and `tests/test_gui_app_api.py`.
- [ ] [PR-09][WISH-013][M] Resolve `parallel_execution` semantics: implement isolated-worktree execution or explicitly disable/hide the option until true safe parallelism exists. Done when behavior is no longer misleading and shared-working-tree races are impossible by design. Validate with chain behavior tests in `tests/test_gui_chain_terminate_repeats.py` and UI/API config tests in `tests/test_gui_app_api.py`.
- [ ] [PR-10][WISH-001][M] Migrate Gemini support from deprecated `google.generativeai` to `google.genai`. Done when runtime/tests stop depending on deprecated SDK while preserving model listing, generation, and usage accounting behavior. Validate with provider/env tests in `tests/test_brain_connector_env.py`.
- [ ] [PR-24][WISH-009][M] Split `brain/connector.py` into provider adapters behind a shared interface after Gemini migration. Done when OpenAI/Anthropic/Gemini dispatch logic is modularized with adapter contract tests and unchanged external behavior. Validate with adapter and dispatch tests in `tests/test_brain_connector_env.py` (and new provider contract tests).
- [ ] [PR-12][WISH-024,WISH-023,WISH-025][M] Finish docstring contract cleanup and automate docstring-audit regression checks. Done when stale/missing public docstrings hit an explicit baseline and automated checks prevent regressions. Validate with audit tests under `tests/` and docs updates in `README.md`/`docs/TROUBLESHOOTING.md`.
- [ ] [PR-22][WISH-033][M] Extract one route family (chain/pipeline control) from `gui/app.py` into blueprints/services without changing API contracts. Done when extracted endpoints pass parity tests and `src/codex_manager/gui/app.py` shrinks measurably. Validate with `python -m pytest -q tests/test_gui_app_api.py`.
- [ ] [PR-23][WISH-032][S] Remove remaining mojibake artifacts in maintained source/template comments and add a regression scanner test. Done when known broken-byte signatures no longer appear in maintained source paths and the scanner prevents reintroduction. Validate with a new encoding-hygiene test module plus targeted `rg` checks.

## Completed / Keep Closed (reopen only on regression)

- [x] [PR-01][WISH-007,WISH-002] Prompt logging is metadata-only by default with explicit debug opt-in.
- [x] [PR-02][WISH-011] Pipeline now honors bounded `on_failure=retry` semantics end-to-end.
- [x] [PR-03][WISH-020] Pipeline log queue is bounded with drop-oldest + overflow warnings.
- [x] [PR-04][WISH-006] Ledger/tracker writes use atomic/race-safe file operations.
- [x] [PR-05][WISH-016] Runtime errors are unified under `.codex_manager/logs/ERRORS.md`.
- [x] [PR-11][WISH-010] Runtime artifact retention controls were centralized and enforced.
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
