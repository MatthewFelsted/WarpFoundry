# User-Friendly Improvements Roadmap

This backlog is prioritized for impact on first-time user success.

## P0 (highest impact)

1. Add guided "First Successful Run" wizard in GUI.
   - Why: users fail on repo/auth/preflight setup.
   - Where: extend existing onboarding and preflight UI.

2. Add docs links directly in header.
   - Why: users should not hunt for setup/troubleshooting.
   - Where: `README.md` and GUI top bar.

3. Replace stale "8 presets" references with actual preset count from catalog.
   - Why: avoid trust-breaking mismatch.

4. Add copy/paste safe default profiles.
   - Why: reduces advanced-setting confusion for non-expert users.

5. Add explicit auth diagnostics panel.
   - Why: immediate feedback for missing keys/login state.

## P1

1. Split giant frontend template into modular JS/CSS files.
   - Why: maintainability and lower regression risk.

2. Add "dry-run to apply" transition checklist in UI.
   - Why: users need clear safety handoff.

3. Add "why stopped" panel with actionable next step.
   - Why: stop reasons are shown but not always actionable.

4. Add output artifact explorer docs and in-app links.
   - Why: users often do not know where outputs/logs are written.

5. Add starter templates for common goals.
   - Why: better prompt quality for new users.

## P2

1. Add telemetry hooks (opt-in local JSON) for UX pain points.
   - Why: evidence-based UX improvements.

2. Add accessibility pass (keyboard nav, aria labels, contrast audit).
   - Why: broader usability.

3. Add in-app command copy buttons for common fixes.
   - Why: faster troubleshooting.

4. Add migration guide for old config formats.
   - Why: smoother upgrades.

## Acceptance criteria for documentation quality

- New user can complete first run in under 10 minutes.
- Every preflight error has a one-screen fix path.
- All CLI subcommands are documented with defaults.
- Output locations are documented for every mode.
