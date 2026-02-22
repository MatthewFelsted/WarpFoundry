# WarpFoundry

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![GUI + CLI](https://img.shields.io/badge/Interface-GUI%20%2B%20CLI-0ea5e9)

**Software that upgrades itself — and everything else in your codebase.**
 
> A local AI control plane that turns coding agents into a repeatable, evidence-backed engineering system.

WarpFoundry is a self-improving AI framework. Point it at any repository and it will plan, execute, validate, and iterate on improvements autonomously — learning from each cycle to make the next one better. It ships with pre-made Autopilot recipes (structured prompt chains for common upgrade patterns). It orchestrates AI agents and coding tools into coherent, compounding engineering work.

It doesn't just run agents. It manages them: choosing what to do, verifying that it worked, capturing evidence, and deciding whether to keep going or stop. The result is AI-driven development that behaves like engineering, not guesswork. You can choose how long to run it. Choose the number of cycles, minutes, tokens, to run or stop when the program hits diminishing returns. You can select 'unlimited' mode, and it will run until you stop it.

**Continuous repo evolution at warp speed.** 

WarpFoundry is an AI Manager for coding agents: a local control plane that plans, runs, validates, and tracks autonomous repo work so your agents ship real progress, not random edits.

It orchestrates Codex (and optionally Claude Code) across iterative chains and autonomous pipelines, with Git-first safety rails, diagnostics, per-repo artifacts, and optional long-term vector memory.

---










## Continuous Improvement, On Autopilot

WarpFoundry is designed for **iterative upgrade cycles** - not "one prompt and pray."

A typical cycle looks like this:

1. **Preflight & diagnostics** (is the repo runnable, writable, correctly configured?)
2. **Plan** a structured change set (tasks, phases, stop reasons)
3. **Execute** changes through agent runs (chains or pipelines)
4. **Validate** with tests, lint, and sanity checks
5. **Capture evidence** (logs, outputs, experiment notes, progress reports)
6. **Decide**: stop, iterate, or escalate
7. **Repeat** until you hit a stop condition or diminishing returns

WarpFoundry can run this cycle on **any repo** - including **itself** - so improvements can compound over time rather than reset every session.

---


## Why Developers Use It

WarpFoundry is built for teams and solo devs who want agentic coding to feel like **engineering**: controlled, observable, reproducible.

- Turn AI coding agents into a repeatable engineering workflow, not one-off chats.
- Run autonomous improvement loops with clear stop reasons, logs, and evidence.
- Keep control with `dry-run` and `apply` modes, branch automation, and preflight checks.
- Reuse prior work through per-repo memory, deep-research caching, and structured outputs.
- Operate from one interface for planning, execution, troubleshooting, and iteration.


WarpFoundry is a local orchestration layer for AI coding agents. It provides a web GUI and CLI workflows for repeatable repo improvement loops with preflight checks, run controls, and structured artifacts.

Compatibility note: the package/CLI name is `warpfoundry`, while source modules are currently under `src/codex_manager/`.

## Screenshots

### Chain Builder + execution panel
![Live run and diagnostics](screenshots/Screenshot%202026-02-15%20211446.png)

### Autopilot recipe selection
![Pipeline phases and Scientist mode](screenshots/Screenshot%202026-02-16%20235517.png)

### Pipeline controls (stop conditions, CUA, agent/brain)
![Pipeline controls](screenshots/Screenshot%202026-02-16%20235709.png)

### Pipeline phase configuration + Scientist mode
![Autopilot recipe selection](screenshots/Screenshot%202026-02-16%20235648.png)

### Live run + diagnostics view
![Expert chain configuration](screenshots/Screenshot%202026-02-17%20000313.png)

### Step outputs + repository idea generator
![Chain Builder + execution panel](screenshots/Screenshot%202026-02-17%20000344.png)

### Expert chain configuration view
![Step outputs and idea generator](screenshots/Screenshot%202026-02-16%20235821.png)

More screenshots: [`screenshots/`](screenshots)

## Verified Capabilities

These are based on the current codebase and CLI parser, not marketing copy:

- Web GUI with Chain Builder and Pipeline tabs (`warpfoundry gui`)
- Classic goal loop (`warpfoundry --repo <path> --goal "<goal>"`)
- Strategic shortcut (`warpfoundry strategic --repo <path>`)
- Autonomous pipeline (`warpfoundry pipeline --repo <path>`)
- GitHub Actions workflow generation for pipeline CI (`warpfoundry github-actions --repo <path>`)
- Git Sync signing setup (GPG/SSH) with validation and push guardrails in GUI
- Pipeline "Promote Last Dry-Run to Apply" action with diff/test summary confirmation
- Run Comparison cost analytics (estimated USD by model/provider, lowest-cost + cost-efficiency badges, budget-outlier flagging)
- One-click run artifact bundle export (zip outputs/logs/config/history with per-run export/download from Run Comparison)
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
warpfoundry pipeline --repo /path/to/repo --resume-state
warpfoundry pipeline --repo /path/to/repo --webhook-url https://hooks.slack.com/services/T000/B000/XXX
warpfoundry github-actions --repo /path/to/repo --branch main
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

Apache 2.0 (declared in [`pyproject.toml`](pyproject.toml)).
