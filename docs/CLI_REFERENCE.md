# CLI Reference

Main entrypoint:

```bash
python -m codex_manager
```

## Global Goal Loop Options

| Option | Default | Meaning |
|---|---|---|
| `--repo PATH` | required for loop mode | Target Git repo path |
| `--goal TEXT` | required for loop mode | Natural-language objective |
| `--rounds N` | `10` | Max rounds |
| `--mode {dry-run,apply}` | `dry-run` | Safety mode |
| `--test-cmd CMD` | evaluator default | Validation command |
| `--codex-bin PATH` | `codex` | Codex CLI binary |
| `--timeout SECS` | `600` | Inactivity timeout per run (`0` disables) |
| `--skip-preflight` | false | Skip setup diagnostics before execution |
| `-v, --verbose` | off | Debug logging |

`--test-cmd` accepts quoted arguments and paths with spaces. On Windows, backslash paths
are preserved (for example `tests\test_api.py`), so targeted file runs work as expected.

Examples:

```bash
python -m codex_manager --repo /repo --goal "Run targeted tests" --test-cmd 'python -m pytest -k "slow suite" -q'
python -m codex_manager --repo C:\repo --goal "Run one test file" --test-cmd "python -m pytest tests\test_api.py -q"
```

## Subcommand: `gui`

```bash
python -m codex_manager gui [--port 5088] [--no-browser]
```

| Option | Default | Meaning |
|---|---|---|
| `--port` | `5088` | Web port |
| `--no-browser` | false | Do not auto-open browser |

## Subcommand: `pipeline`

```bash
python -m codex_manager pipeline --repo <path> [options]
```

| Option | Default |
|---|---|
| `--mode` | `dry-run` |
| `--cycles` | `3` |
| `--science` | false |
| `--brain` | false |
| `--brain-model` | `gpt-5.2` |
| `--agent` | `codex` (`codex` or `claude_code`) |
| `--codex-bin` | `codex` |
| `--claude-bin` | `claude` |
| `--test-cmd` | config default |
| `--timeout` | `600` |
| `--max-tokens` | `5000000` |
| `--max-time` | `240` minutes |
| `--local-only` | false |
| `--skip-preflight` | false |

## Subcommand: `strategic`

```bash
python -m codex_manager strategic --repo <path> [options]
```

Runs the classic loop with a built-in Strategic Product Maximization goal.
When the repo path name includes both `discover` and `chain`, extra Discover Chain focus is appended automatically.

| Option | Default |
|---|---|
| `--mode` | `dry-run` |
| `--rounds` | `6` |
| `--test-cmd` | evaluator default |
| `--codex-bin` | `codex` |
| `--timeout` | `600` |
| `--goal-extra` | empty |
| `--skip-preflight` | false |

## Subcommand: `optimize-prompts`

```bash
python -m codex_manager optimize-prompts [options]
```

| Option | Default |
|---|---|
| `--model` | `gpt-5.2` |
| `--threshold` | `7.5` |
| `--dry-run` | false |
| `--local-only` | false |

## Subcommand: `visual-test`

```bash
python -m codex_manager visual-test [options]
```

| Option | Default |
|---|---|
| `--url` | empty |
| `--task` | built-in visual QA task |
| `--provider` | `openai` |
| `--max-steps` | `30` |
| `--headed` | false (headless by default) |
| `--timeout` | `300` |

## Subcommand: `list-prompts`

```bash
python -m codex_manager list-prompts
```

No extra options.

## Subcommand: `list-recipes`

```bash
python -m codex_manager list-recipes [--recipe <id>]
```

Use this to inspect built-in Autopilot recipes and show full step prompts.

## Subcommand: `doctor`

```bash
python -m codex_manager doctor [options]
```

Runs setup diagnostics before a run. Returns exit code `0` when ready, `1` when any required checks fail.

| Option | Default |
|---|---|
| `--repo` | empty (optional) |
| `--agents` | `codex` |
| `--codex-bin` | `codex` |
| `--claude-bin` | `claude` |
| `--json` | false |

## Useful Examples

```bash
python -m codex_manager --repo /repo --goal "Increase test coverage" --rounds 3 --mode dry-run
python -m codex_manager strategic --repo /repo --mode dry-run --rounds 6
python -m codex_manager doctor --repo /repo --agents codex,claude_code
python -m codex_manager pipeline --repo /repo --mode apply --cycles 2 --brain --brain-model gpt-5.2
python -m codex_manager pipeline --repo /repo --agent claude_code --claude-bin /usr/local/bin/claude
python -m codex_manager optimize-prompts --dry-run
python -m codex_manager list-recipes --recipe autopilot_default
python -m codex_manager visual-test --url http://localhost:5088 --provider anthropic
```
