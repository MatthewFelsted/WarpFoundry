# Model Watchdog

The model watchdog is a background scheduler that snapshots provider model catalogs and key dependency versions so the app can detect deprecations and new model availability over time.

## What It Tracks

- Provider model lists (when provider credentials are available):
  - OpenAI
  - Anthropic
  - Google Gemini API
  - xAI
  - Local Ollama models
- Local dependency versions:
  - `warpfoundry` (and legacy `codex-manager`), `flask`, `pydantic`
  - `openai`, `anthropic`, `google-generativeai`, `chromadb`

## Default Schedule

- Default startup state: `disabled` (opt-in).
- When enabled, default interval: every `24` hours.
- Best-practice default cadence chosen for balance:
  - frequent enough to catch provider model lifecycle changes quickly,
  - infrequent enough to avoid unnecessary API traffic/cost.

## Storage

Files are stored in the user home config area:

- `~/.codex_manager/watchdog/config.json`
- `~/.codex_manager/watchdog/state.json`
- `~/.codex_manager/watchdog/model_catalog_latest.json`
- `~/.codex_manager/watchdog/model_catalog_history.ndjson`

## API Endpoints

- `GET /api/system/model-watchdog/status`
- `GET /api/system/model-watchdog/alerts`
  - Returns latest alert summary with actionable recommendation text.
- `POST /api/system/model-watchdog/run`
  - body: `{ "force": true }`
- `POST /api/system/model-watchdog/config`
  - optional fields:
    - `enabled`
    - `interval_hours`
    - `providers`
    - `request_timeout_seconds`
    - `auto_run_on_start`
    - `history_limit`

## Environment Overrides (First Run Defaults)

You can set default values before the first run creates config files:

- `CODEX_MANAGER_MODEL_WATCHDOG_ENABLED`
- `CODEX_MANAGER_MODEL_WATCHDOG_INTERVAL_HOURS`

After first launch, `config.json` becomes the source of truth.

## GUI Alerts

- The header includes an `Enable Model Watchdog` checkbox (unchecked by default on first run).
- The header now surfaces watchdog alerts in a dedicated banner when model removals
  or dependency drifts are detected.
- Banner actions:
  - `Apply migration suggestions`: one-click apply for recommended model ID replacements when removals are detected.
  - `Run now`: trigger an immediate watchdog run.
  - `Guide`: open this document in-app.
  - `Mute 1h`: temporarily hide banner notifications.
