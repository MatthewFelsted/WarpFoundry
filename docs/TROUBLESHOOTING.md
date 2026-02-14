# Troubleshooting

Before diving into specific errors, run:

```bash
python -m codex_manager doctor --repo /path/to/repo
```

## Preflight: "Not a git repository"

Cause: target path is missing `.git`.

Fix:

```bash
cd /path/to/repo
git init
```

Or point Codex Manager to the correct repo.

## Preflight: "Repository is not writable"

Cause: permissions, OneDrive lock, read-only folder, or security policy.

Fix:

- Move repo to a writable local path.
- Ensure current user can write files.
- Close tools that lock files.

## Preflight: "Codex binary not found"

Cause: Codex CLI missing or not on PATH.

Fix:

- Install Codex CLI.
- Verify with `codex --version`.
- Set custom binary path in GUI advanced settings.

## Preflight: "Codex auth not detected"

Cause: no API key and no CLI auth profile.

Fix:

- Set `CODEX_API_KEY` or `OPENAI_API_KEY`.
- Or run `codex login`.

## Preflight: Claude auth/binary errors

Cause: Claude CLI not installed or auth missing.

Fix:

- Install Claude CLI.
- Set `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY`.
- Or log in with Claude CLI.

## CUA dependency errors

Cause: missing SDK/playwright components.

Fix:

```bash
pip install -e ".[cua]"
python -m playwright install
```

## `ERR_CONNECTION_REFUSED` in visual test

Cause: target URL is not running.

Fix:

- Start app first.
- Confirm URL and port.
- Retry visual test.

## Runs stop too early from token/time limits

Fix:

- Increase budget/time limits.
- Disable strict token budget if needed.
- Check convergence threshold and stop-on-convergence settings.

## No tests are running

Cause: validation command is blank in GUI chain/pipeline config.

Fix:

Set validation command, for example:

```bash
python -m pytest -q
```
