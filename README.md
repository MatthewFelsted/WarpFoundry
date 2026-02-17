# WarpFoundry

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![AI Agents](https://img.shields.io/badge/AI-Agents-blue)
![Vector Memory](https://img.shields.io/badge/Memory-ChromaDB-ff7a59)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)

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

> Tip: consider adding a screenshot/GIF here once published (GUI, Chain Builder, Autopilot, outputs). It dramatically improves GitHub conversion.

---


## Why Developers Use It

WarpFoundry is built for teams and solo devs who want agentic coding to feel like **engineering**: controlled, observable, reproducible.

- Turn AI coding agents into a repeatable engineering workflow, not one-off chats.
- Run autonomous improvement loops with clear stop reasons, logs, and evidence.
- Keep control with `dry-run` and `apply` modes, branch automation, and preflight checks.
- Reuse prior work through per-repo memory, deep-research caching, and structured outputs.
- Operate from one interface for planning, execution, troubleshooting, and iteration.

> CLI command is `warpfoundry`. Source checkouts can also use `python -m warpfoundry` (or legacy `python -m codex_manager`).

---

## What You Can Run

One codebase, multiple workflows:

- Web GUI chain runner (interactive, configurable)
- Autonomous pipeline runner (phase-based cycles)
- Classic goal loop (`--repo` + `--goal`)
- Strategic loop shortcut (`strategic`)
- Setup diagnostics (`doctor`)
- CUA visual testing (`visual-test`)
- Prompt catalog inspection and optimization

---

## Docs Map

If you want to understand how WarpFoundry works end-to-end, start here:

- `docs/QUICKSTART.md`
- `docs/OUTPUTS_AND_ARTIFACTS.md`
- `docs/TUTORIAL.md`
- `docs/CLI_REFERENCE.md`
- `docs/TROUBLESHOOTING.md`
- `docs/USER_FRIENDLY_IMPROVEMENTS.md`
- `docs/AGENT_PROTOCOL.md`
- `docs/MODEL_WATCHDOG.md`
- `docs/LICENSING_AND_COMMERCIAL.md`
- `docs/REQUESTED_FEATURES_TODO.md`

---

## Quick Start

```bash
cd <repo-root>
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

pip install -e ".[dev]"

# Optional: include vector-memory support (ChromaDB)
# pip install -e ".[memory]"
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
warpfoundry doctor --repo /path/to/repo
```

Launch GUI:

```bash
warpfoundry gui
```

On first launch, use the `Guided First Run` wizard to set repo path, run diagnostics, apply safe defaults, and start one dry-run chain automatically.

In the GUI header, use `Quickstart`, `Outputs`, `Troubleshooting`, and `CLI Reference` to open built-in docs without leaving the app.  
Use the `Setup Diagnostics` panel under Repository Path (Chain and Pipeline tabs) to validate repo access, CLI binaries, and auth before you run.  
Diagnostics now include prioritized `Next actions` with concrete commands so first-run fixes are one screen away.  
When a run stops, the Execution panel now shows a `Why it stopped` card with actionable next steps.

---

## Entry Points

- GUI: `warpfoundry gui`
- Pipeline: `warpfoundry pipeline --repo <path>`
- Goal loop: `warpfoundry --repo <path> --goal "<goal>"`
- Strategic loop: `warpfoundry strategic --repo <path>`
- Setup diagnostics: `warpfoundry doctor --repo <path>`
- Visual test: `warpfoundry visual-test --url <url>`
- Prompt tools: `warpfoundry list-prompts`, `warpfoundry optimize-prompts`, `warpfoundry list-recipes`

---

## Strategic Product Maximization Mode

- GUI: choose recipe `Strategic Product Max` in Autopilot mode.
- Step type: `Strategic Product Maximization` is available in the chain builder.
- CLI shortcut example for Discover Chain: `warpfoundry strategic --repo C:\path\to\discover-chain --rounds 6 --mode dry-run`

---

## Modes

- `dry-run`: evaluate changes, then revert
- `apply`: keep changes and commit based on runner config

---

## Where Files Are Written

WarpFoundry writes structured, reproducible artifacts so runs are explainable and repeatable.

Inside target repo:

- `.codex_manager/state.json`
- `.codex_manager/outputs/*.md`
- `.codex_manager/logs/WISHLIST.md`
- `.codex_manager/logs/TESTPLAN.md`
- `.codex_manager/logs/ERRORS.md` (canonical runtime error log for chain + pipeline)
- `.codex_manager/logs/EXPERIMENTS.md`
- `.codex_manager/logs/PROGRESS.md`
- `.codex_manager/logs/scientist/*`
- `.codex_manager/business/licensing_profile.json` (if licensing/commercial packaging is enabled on project creation)

Inside home config (cross-repo state):

- `~/.codex_manager/watchdog/*` (provider/dependency change snapshots)

---

## First-Run Checklist

- Target path exists and has `.git`
- Repo is writable
- Repo worktree is clean (or intentionally bypassed with `--skip-preflight`)
- Agent binaries are installed and on PATH
- Codex auth present: `CODEX_API_KEY` or `OPENAI_API_KEY`, or `codex login`
- Claude auth present: `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY`, or CLI login

---

## Local Validation

```bash
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
```

Text encoding standard: source/docs/templates should remain UTF-8 without mojibake signatures.
The `test_encoding_hygiene.py` test enforces this guardrail in CI.

---

## License

MIT

