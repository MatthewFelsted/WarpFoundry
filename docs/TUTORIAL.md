# Tutorial: End-to-End Workflow

This tutorial walks through GUI chain, then pipeline, then optional visual test.

## Step 0: Prepare a safe branch

In your target repo:

```bash
git checkout -b codex-manager-tutorial
```

## Step 1: Run a GUI task chain

Start GUI:

```bash
python -m codex_manager gui
```

Configure:

1. Repository Path: your repo.
2. Mode: `dry-run`.
3. Add steps:
   - Feature Discovery
   - Implementation
   - Testing
4. Max loops: `2`
5. Validation command: `python -m pytest -q`
6. Start.

Observe:

- Live log panel
- Results table
- Token counter

## Step 2: Review generated artifacts

In target repo:

- `.codex_manager/outputs/*.md` for step outputs
- `.codex_manager/ERRORS.md` for chain warnings/errors

Check that proposed edits make sense before switching to apply mode.

## Step 3: Run strategic product loop (optional)

```bash
python -m codex_manager strategic --repo /path/to/repo --mode dry-run --rounds 6
```

If your target repo is Discover Chain, use its path and the command will add Discover Chain focus automatically.

## Step 4: Run autonomous pipeline

```bash
python -m codex_manager pipeline --repo /path/to/repo --mode dry-run --cycles 2
```

Pipeline phases include ideation, prioritization, implementation, testing, debugging, commit, and progress review. When Scientist Mode is enabled, science phases run before implementation so findings can drive code changes in the same cycle.

Inspect:

- `.codex_manager/logs/WISHLIST.md`
- `.codex_manager/logs/TESTPLAN.md`
- `.codex_manager/logs/ERRORS.md`
- `.codex_manager/logs/SCIENTIST_REPORT.md` (when Scientist Mode is enabled)
- `.codex_manager/logs/PROGRESS.md`

## Step 5: Enable apply mode

After dry-run looks good:

```bash
python -m codex_manager pipeline --repo /path/to/repo --mode apply --cycles 1
```

Then review with:

```bash
git log --oneline -n 5
git status
git diff HEAD~1
```

## Step 6: Run visual UI test (optional)

Start app under test first, then:

```bash
python -m codex_manager visual-test --url http://localhost:5088 --provider openai
```

If connection is refused, start the app and retry.

## Step 7: Close out

If satisfied:

```bash
git add -A
git commit -m "Apply codex-manager improvements"
```

If not, reset your tutorial branch as needed.
