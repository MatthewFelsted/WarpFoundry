# Codex Manager

Codex Manager is a local Python app that orchestrates coding agents (Codex, and optionally Claude Code) to improve a target Git repository in iterative loops.

## What You Can Run

- Web GUI chain runner (interactive, configurable)
- Autonomous pipeline runner (phase-based cycles)
- Classic goal loop (`--repo` + `--goal`)
- Strategic loop shortcut (`strategic`)
- Setup diagnostics (`doctor`)
- CUA visual testing (`visual-test`)
- Prompt catalog inspection and optimization

## Docs Map

- `docs/QUICKSTART.md`
- `docs/OUTPUTS_AND_ARTIFACTS.md`
- `docs/TUTORIAL.md`
- `docs/CLI_REFERENCE.md`
- `docs/TROUBLESHOOTING.md`
- `docs/USER_FRIENDLY_IMPROVEMENTS.md`

## Quick Start

```bash
cd codex_manager
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

pip install -e ".[dev]"
```

Set auth:

```bash
# PowerShell
$env:CODEX_API_KEY="sk-..."

# Bash
export CODEX_API_KEY="sk-..."
```

Run setup diagnostics before the first CLI run:

```bash
python -m codex_manager doctor --repo /path/to/repo
```

Launch GUI:

```bash
python -m codex_manager gui
```

In the GUI header, use `Quickstart`, `Outputs`, `Troubleshooting`, and `CLI Reference` to open built-in docs without leaving the app.
Use the `Setup Diagnostics` panel under Repository Path (Chain and Pipeline tabs) to validate repo access, CLI binaries, and auth before you run.
Diagnostics now include prioritized `Next actions` with concrete commands so first-run fixes are one screen away.
When a run stops, the Execution panel now shows a `Why it stopped` card with actionable next steps.

## Entry Points

- GUI: `python -m codex_manager gui`
- Pipeline: `python -m codex_manager pipeline --repo <path>`
- Goal loop: `python -m codex_manager --repo <path> --goal "<goal>"`
- Strategic loop: `python -m codex_manager strategic --repo <path>`
- Setup diagnostics: `python -m codex_manager doctor --repo <path>`
- Visual test: `python -m codex_manager visual-test --url <url>`
- Prompt tools: `python -m codex_manager list-prompts`, `python -m codex_manager optimize-prompts`

## Strategic Product Maximization Mode

- GUI: choose recipe `Strategic Product Max` in Autopilot mode.
- Step type: `Strategic Product Maximization` is available in the chain builder.
- CLI shortcut example for Discover Chain:
  - `python -m codex_manager strategic --repo C:\path\to\discover-chain --rounds 6 --mode dry-run`

## Modes

- `dry-run`: evaluate changes, then revert
- `apply`: keep changes and commit based on runner config

## Where Files Are Written

Inside target repo:

- `.codex_manager/state.json`
- `.codex_manager/outputs/*.md`
- `.codex_manager/ERRORS.md` (chain mode)
- `.codex_manager/logs/WISHLIST.md`
- `.codex_manager/logs/TESTPLAN.md`
- `.codex_manager/logs/ERRORS.md`
- `.codex_manager/logs/EXPERIMENTS.md`
- `.codex_manager/logs/PROGRESS.md`
- `.codex_manager/logs/scientist/*`

## First-Run Checklist

- Target path exists and has `.git`
- Repo is writable
- Agent binaries are installed and on PATH
- Auth is present:
  - Codex: `CODEX_API_KEY` or `OPENAI_API_KEY`, or `codex login`
  - Claude: `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY`, or CLI login

## Local Validation

```bash
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
```

## License

MIT
