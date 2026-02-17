# TEST PLAN - Coverage Audit and Prioritized Cases

## Coverage Audit (Current)

- Test suite status: `517 passed, 1 skipped`
- Overall coverage: `69%` (`src/codex_manager`)
- High-risk low-coverage areas still open:
  - `src/codex_manager/brain/connector.py` (29%)
  - `src/codex_manager/brain/manager.py` (31%)
  - `src/codex_manager/cua/openai_cua.py` (0%)
  - `src/codex_manager/cua/anthropic_cua.py` (0%)
  - `src/codex_manager/cua/session.py` (9%)
  - `src/codex_manager/pipeline/orchestrator.py` (60%)
  - `src/codex_manager/gui/app.py` (67%)

## Execution Optimization

- Ordering strategy:
  - `unit` tests first
  - `integration` tests second
  - `slow` tests last
- Implemented in `tests/conftest.py` via marker-aware collection ordering.
- Integration-heavy modules marked:
  - `tests/test_gui_app_api.py`
  - `tests/test_runner_integration.py`
  - `tests/test_history_logbook.py`
  - `tests/test_history_logbook_rotation.py`
- Slow-test marker is available for future live API tests (`@pytest.mark.slow`).

## P0 - Critical (Core Functionality, Data Integrity, Security)

- [x] `TC-P0-001` File lock identity is stable for path aliases
  - Purpose: Prevent per-path lock duplication that could corrupt concurrent writes.
  - Implemented: `tests/test_file_io.py::test_path_lock_reuses_same_lock_for_resolved_aliases`
- [x] `TC-P0-002` Atomic replace retries transient permission failures
  - Purpose: Validate resilient write behavior under file lock races.
  - Implemented: `tests/test_file_io.py::test_replace_file_with_retry_retries_permission_denied_then_succeeds`
- [x] `TC-P0-003` Atomic replace immediately raises non-permission OS errors
  - Purpose: Avoid masking real I/O failures.
  - Implemented: `tests/test_file_io.py::test_replace_file_with_retry_raises_non_permission_oserror`
- [x] `TC-P0-004` Atomic replace surfaces final permission error after max retries
  - Purpose: Ensure deterministic failure after retry budget is exhausted.
  - Implemented: `tests/test_file_io.py::test_replace_file_with_retry_raises_last_permission_error_after_max_retries`
- [x] `TC-P0-005` Atomic write removes temp files on failure
  - Purpose: Avoid temp-file leaks and stale artifacts.
  - Implemented: `tests/test_file_io.py::test_atomic_write_text_cleans_temp_file_when_replace_raises`
- [x] `TC-P0-006` Resilient read decodes legacy cp1252 and normalizes to UTF-8
  - Purpose: Protect log/ledger readability and long-term encoding hygiene.
  - Implemented: `tests/test_file_io.py::test_read_text_utf8_resilient_decodes_and_normalizes_legacy_text`
- [x] `TC-P0-007` Resilient read supports replacement fallback when all decoders fail
  - Purpose: Prevent hard failure on malformed bytes.
  - Implemented: `tests/test_file_io.py::test_read_text_utf8_resilient_uses_replacement_when_all_fallbacks_fail`
- [x] `TC-P0-008` Runtime retention honors byte budget by pruning oldest files first
  - Purpose: Keep artifact footprint bounded deterministically.
  - Implemented: `tests/test_artifact_retention_additional.py::test_cleanup_runtime_artifacts_prunes_oldest_files_to_satisfy_byte_budget`
- [x] `TC-P0-009` Runtime retention continues when a file cannot be unlinked
  - Purpose: Ensure cleanup is resilient to partial filesystem failures.
  - Implemented: `tests/test_artifact_retention_additional.py::test_cleanup_runtime_artifacts_continues_when_file_unlink_fails`
- [x] `TC-P0-010` Security path traversal guard for config loading
  - Purpose: Ensure `/api/configs/load` rejects traversal names.
  - Implemented (existing): `tests/test_gui_app_api.py::test_configs_load_rejects_invalid_names`

## P1 - High (Main Flows, Error Handling, Edge Cases)

- [x] `TC-P1-001` Retention is no-op when `.codex_manager` root is absent
  - Purpose: Avoid destructive behavior for uninitialized repos.
  - Implemented: `tests/test_artifact_retention_additional.py::test_cleanup_runtime_artifacts_noop_when_manager_root_missing`
- [x] `TC-P1-002` Retention preserves active empty directories
  - Purpose: Do not prune active run paths during cleanup.
  - Implemented: `tests/test_artifact_retention_additional.py::test_cleanup_runtime_artifacts_keeps_active_empty_directories`
- [x] `TC-P1-003` Mojibake scan counts signatures and C1 control range
  - Purpose: Verify signature detection used by hygiene workflows.
  - Implemented: `tests/test_encoding_hygiene_additional.py::test_scan_text_for_mojibake_counts_signatures_and_c1_controls`
- [x] `TC-P1-004` In-place normalization rewrites only text files and skips binary suffixes
  - Purpose: Prevent accidental corruption of binary assets.
  - Implemented: `tests/test_encoding_hygiene_additional.py::test_normalize_paths_in_place_updates_text_files_and_ignores_binary_suffixes`
- [x] `TC-P1-005` History log rotates when month changes and meta updates
  - Purpose: Ensure archival roll-forward works across month boundaries.
  - Implemented: `tests/test_history_logbook_rotation.py::test_rotate_if_month_changed_rotates_active_logs_and_updates_meta`
- [x] `TC-P1-006` History log ignores malformed metadata safely
  - Purpose: Harden against partial/corrupt meta writes.
  - Implemented: `tests/test_history_logbook_rotation.py::test_rotate_if_month_changed_ignores_malformed_meta`

## P2 - Medium (Integration Points, Boundary Conditions)

- [x] `TC-P2-001` Marker-aware run ordering for resource efficiency
  - Purpose: Reduce time-to-signal by running fast tests before integration/slow.
  - Implemented: `tests/conftest.py::pytest_collection_modifyitems`
- [ ] `TC-P2-002` Pipeline orchestrator cancellation/retry matrix expansion
  - Purpose: Cover currently sparse branches in orchestrator control flow.
  - Planned target: `tests/test_pipeline_orchestrator.py`
- [ ] `TC-P2-003` GUI app git-sync edge matrix (remote URL parsing, detached HEAD, branch locks)
  - Purpose: Increase confidence in repo sync safety paths.
  - Planned target: `tests/test_gui_app_api.py`
- [ ] `TC-P2-004` Vector store boundary tests (empty corpus, invalid metadata, failed persistence)
  - Purpose: Improve memory subsystem reliability.
  - Planned target: `tests/test_vector_memory.py`

## P3 - Low (Cosmetic, Docs, Style)

- [x] `TC-P3-001` Curated docs API mapping and retrieval endpoints
  - Implemented (existing): `tests/test_gui_app_api.py::test_docs_api_lists_curated_docs`
- [x] `TC-P3-002` Unknown docs key handling
  - Implemented (existing): `tests/test_gui_app_api.py::test_docs_api_rejects_unknown_doc_key`
- [ ] `TC-P3-003` Additional README/tutorial snippet validation smoke checks
  - Planned target: `tests/test_gui_app_api.py` and docs-based smoke tests

## Test Infrastructure and Fixtures

- Implemented infrastructure:
  - Marker registration and ordering hook: `tests/conftest.py`
  - Parametrized legacy-encoding cases: `tests/test_file_io.py`
  - Monkeypatch-based failure injection for I/O edge cases:
    - `Path.replace`, `Path.unlink`, `Path.read_bytes`, `atomic_write_text`, `_rotate_file`
  - Temp repo/file fixtures via `tmp_path`
- Future infrastructure needed:
  - Deterministic connector fakes for `brain.connector`/`brain.manager`
  - Optional live-provider contract tests marked `slow` and guarded by env vars

## Validation Commands

- Fast + integration regression run:
  - `pytest -q`
- Coverage audit:
  - `pytest --cov=src/codex_manager --cov-report=term-missing -q`
