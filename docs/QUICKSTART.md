# Quickstart

This is the shortest reliable path from install to first successful run.

## 1. Prerequisites

- Python 3.10+
- Git
- Codex CLI installed and on PATH
- A target repo with `.git`

## 2. Install

```bash
cd codex_manager
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
pip install -e ".[brain]"
pip install -e ".[cua]"
python -m playwright install
```

## 3. Set credentials

PowerShell:

```bash
$env:CODEX_API_KEY="sk-..."
$env:OPENAI_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

Bash:

```bash
export CODEX_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## 4. Run setup diagnostics (recommended)

```bash
python -m codex_manager doctor --repo /path/to/repo
```

If diagnostics report failures, fix those first (repo path, auth, binaries) before running loops.
Use the `Next actions` section to apply fixes in priority order (including copy/paste-ready commands).

## 5. Start GUI and run one chain

```bash
python -m codex_manager gui
```

Then in UI:

1. Set Repository Path.
2. Select `dry-run` for safety.
3. Pick recipe `Strategic Product Max` (or add one step like Implementation/Testing).
4. Set max loops to 1.
5. Click Start.

## 6. Verify outputs

Check target repo:

- `.codex_manager/outputs/`
- `.codex_manager/ERRORS.md`
- `.codex_manager/logs/` (if pipeline mode was used)

Tip: in the GUI header, open `Outputs` for a built-in guide to artifact locations and how to inspect them.

## 7. Strategic CLI run (optional)

```bash
python -m codex_manager strategic --repo /path/to/repo --mode dry-run --rounds 6
```

For Discover Chain specifically:

```bash
python -m codex_manager strategic --repo /path/to/discover-chain --mode dry-run --rounds 6
```

## 8. First generic CLI run (optional)

```bash
python -m codex_manager --repo /path/to/repo --goal "Improve docs and tests" --rounds 2 --mode dry-run
```

## 9. First pipeline run (optional)

```bash
python -m codex_manager pipeline --repo /path/to/repo --mode dry-run --cycles 1
```
