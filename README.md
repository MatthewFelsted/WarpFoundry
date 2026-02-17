# WarpFoundry

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![GUI + CLI](https://img.shields.io/badge/Interface-GUI%20%2B%20CLI-0ea5e9)

WarpFoundry is a local orchestration layer for AI coding agents. It provides a web GUI and CLI workflows for repeatable repo improvement loops with preflight checks, run controls, and structured artifacts.

Compatibility note: the package/CLI name is `warpfoundry`, while source modules are currently under `src/codex_manager/`.

## Screenshots

### Chain Builder + execution panel
![Chain Builder + execution panel](screenshots/Screenshot%202026-02-17%20000344.png)

### Autopilot recipe selection
![Autopilot recipe selection](screenshots/Screenshot%202026-02-16%20235648.png)

### Pipeline controls (stop conditions, CUA, agent/brain)
![Pipeline controls](screenshots/Screenshot%202026-02-16%20235709.png)

### Pipeline phase configuration + Scientist mode
![Pipeline phases and Scientist mode](screenshots/Screenshot%202026-02-16%20235517.png)

### Live run + diagnostics view
![Live run and diagnostics](screenshots/Screenshot%202026-02-15%20211446.png)

### Step outputs + repository idea generator
![Step outputs and idea generator](screenshots/Screenshot%202026-02-16%20235821.png)

### Expert chain configuration view
![Expert chain configuration](screenshots/Screenshot%202026-02-17%20000313.png)

More screenshots: [`screenshots/`](screenshots)

## Verified Capabilities

These are based on the current codebase and CLI parser, not marketing copy:

- Web GUI with Chain Builder and Pipeline tabs (`warpfoundry gui`)
- Classic goal loop (`warpfoundry --repo <path> --goal "<goal>"`)
- Strategic shortcut (`warpfoundry strategic --repo <path>`)
- Autonomous pipeline (`warpfoundry pipeline --repo <path>`)
- Setup diagnostics with actionable next steps (`warpfoundry doctor --repo <path>`)
- Prompt tools (`list-prompts`, `optimize-prompts`, `list-recipes`)
- Optional visual UI testing via CUA (`warpfoundry visual-test --url <url>`)
- Built-in Autopilot recipes (8 total; see `warpfoundry list-recipes`)
- `dry-run` and `apply` execution modes

## Quick Start

```bash
cd <repo-root>
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[brain,cua,memory]"
python -m playwright install
```

Auth (minimum for Codex runs):

```bash
# PowerShell
$env:CODEX_API_KEY="sk-..."
# or $env:OPENAI_API_KEY="sk-..."
```

```bash
# Bash
export CODEX_API_KEY="sk-..."
# or export OPENAI_API_KEY="sk-..."
```

Run diagnostics first:

```bash
warpfoundry doctor --repo /path/to/repo
```

Start GUI:

```bash
warpfoundry gui
```

Or run a safe CLI loop:

```bash
warpfoundry strategic --repo /path/to/repo --mode dry-run --rounds 1
```

## Common Commands

```bash
warpfoundry gui
warpfoundry doctor --repo /path/to/repo
warpfoundry --repo /path/to/repo --goal "Improve tests and docs" --mode dry-run --rounds 2
warpfoundry strategic --repo /path/to/repo --focus reliability --focus ux
warpfoundry pipeline --repo /path/to/repo --mode dry-run --cycles 1
warpfoundry list-recipes
warpfoundry visual-test --url http://localhost:5088
```

Legacy/alternate entrypoints:

- `python -m warpfoundry`
- `python -m codex_manager`
- `codex-manager` (console script alias)

## Output Artifacts

WarpFoundry writes run artifacts into the target repository under `.codex_manager/`:

- `.codex_manager/outputs/*.md` (chain step outputs)
- `.codex_manager/state.json` (loop state)
- `.codex_manager/logs/WISHLIST.md`
- `.codex_manager/logs/TESTPLAN.md`
- `.codex_manager/logs/ERRORS.md`
- `.codex_manager/logs/EXPERIMENTS.md`
- `.codex_manager/logs/RESEARCH.md`
- `.codex_manager/logs/SCIENTIST_REPORT.md`
- `.codex_manager/logs/PROGRESS.md`
- `.codex_manager/logs/BRAIN.md`
- `.codex_manager/logs/HISTORY.md`
- `.codex_manager/logs/scientist/*`

See also: [`docs/OUTPUTS_AND_ARTIFACTS.md`](docs/OUTPUTS_AND_ARTIFACTS.md)

## Documentation

- [Quickstart](docs/QUICKSTART.md)
- [Tutorial](docs/TUTORIAL.md)
- [CLI Reference](docs/CLI_REFERENCE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Outputs and Artifacts](docs/OUTPUTS_AND_ARTIFACTS.md)
- [Agent Protocol](docs/AGENT_PROTOCOL.md)
- [Model Watchdog](docs/MODEL_WATCHDOG.md)
- [Licensing and Commercial](docs/LICENSING_AND_COMMERCIAL.md)
- [Requested Features Todo](docs/REQUESTED_FEATURES_TODO.md)
- [User Friendly Improvements](docs/USER_FRIENDLY_IMPROVEMENTS.md)

## Key Source Files

- CLI entrypoint: [`src/codex_manager/__main__.py`](src/codex_manager/__main__.py)
- Package shim for `python -m warpfoundry`: [`src/warpfoundry/__main__.py`](src/warpfoundry/__main__.py)
- GUI backend: [`src/codex_manager/gui/app.py`](src/codex_manager/gui/app.py)
- GUI template: [`src/codex_manager/gui/templates/index.html`](src/codex_manager/gui/templates/index.html)
- Chain runner: [`src/codex_manager/gui/chain.py`](src/codex_manager/gui/chain.py)
- Pipeline orchestrator: [`src/codex_manager/pipeline/orchestrator.py`](src/codex_manager/pipeline/orchestrator.py)
- Pipeline phases/config: [`src/codex_manager/pipeline/phases.py`](src/codex_manager/pipeline/phases.py)
- Preflight diagnostics: [`src/codex_manager/preflight.py`](src/codex_manager/preflight.py)
- Built-in recipes: [`src/codex_manager/gui/recipes.py`](src/codex_manager/gui/recipes.py)
- Prompt catalog templates: [`src/codex_manager/prompts/templates.yaml`](src/codex_manager/prompts/templates.yaml)

## Local Validation

```bash
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
```

## License

MIT (declared in [`pyproject.toml`](pyproject.toml)).
