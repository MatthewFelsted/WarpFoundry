# Feature Dreams

## P0 - Highest Value / Lowest Effort

- [x] Add GitHub auth setup UI (PAT + optional SSH key) with secure local storage and a "Test connection" action. _(Completed: Added a GitHub Auth modal, keyring-backed secret storage, and a backend connection test endpoint.)_
- [x] Add "Clone remote repo" flow: paste GitHub URL, choose destination, select default branch, and initialize `.codex_manager` in the cloned repo. _(Completed: Added Clone Repo modal + backend clone/branch APIs with remote branch selection and automatic `.codex_manager` scaffolding after clone.)_
- [x] Add repo sync controls in the header: "Fetch", "Pull", ahead/behind + dirty status, and one-click "Stash and Pull". _(Completed: Added header sync controls with live branch/tracking/ahead-behind/dirty status and new git sync APIs for fetch, pull, and stash+pull actions.)_
- [x] Add "Push" action with `--set-upstream` support, explicit auth/non-fast-forward errors, and guided recovery steps. _(Completed: Added header Push controls with optional set-upstream toggle, a new `/api/git/sync/push` endpoint, typed push failure classification (`auth`, `non_fast_forward`, `upstream_missing`), and API/UI recovery guidance messaging.)_
- [x] Add branch switcher UI (local + remote branches), branch creation, and dirty-worktree guardrails. _(Completed: Added branch list/switch/create APIs, header branch-switcher controls with local+remote branch options, and dirty-worktree guardrails with recovery guidance and explicit allow-dirty override.)_
- [x] Add "Open Pull Request" helper: after push, generate the GitHub PR URL and provide open/copy actions. _(Completed: Push success responses now include a GitHub PR compare URL when derivable from the remote, and header controls now provide `Open PR` and `Copy PR URL` actions.)_

## P1 - Product Leverage

- [x] Add commit workflow panel: stage/unstage files, commit message editor, commit action, and last-commit summary (hash/author/time). _(Completed: Added a commit workflow modal with stage/unstage per-file and bulk actions, commit message + commit API flow, and live last-commit summary metadata in the header git controls.)_
- [x] Add remote management UI: view/add/remove remotes, set default remote, and validate HTTPS/SSH URLs. _(Completed: Added a Remotes modal with list/add/remove/default actions, backend HTTPS/SSH URL validation, and `remote.pushDefault` integration for push defaults.)_
- [x] Add live "Sync status" widget: branch, tracking remote, ahead/behind counts, last fetch time, and manual refresh. _(Completed: Replaced the header sync pill with a live sync-status widget that displays branch/remote/ahead-behind/last-fetch metadata, added a manual Refresh action, and extended git sync status APIs to return tracking remote + last fetch timestamps.)_
- [x] Add GitHub repo metadata integration: detect repo from remote, show name/visibility/default branch, and link to GitHub page. _(Completed: Extended git sync status APIs with cached GitHub repo metadata detection from remotes (name/visibility/default branch/repo URL) and updated the header Sync Status widget to display metadata plus an `Open Repo` action.)_
- [x] Add credential troubleshooting assistant for common GitHub auth failures (PAT scopes, SSH known_hosts, key permissions). _(Completed: Added backend troubleshooting payloads for GitHub auth test + push auth failures, rendered a GitHub Auth modal assistant panel with targeted PAT/SSH guidance, and expanded push recovery steps to cover PAT scopes, known_hosts, and key-permission checks.)_
- [x] Add "Pre-flight before run" checks: clean/stash state, branch validation, remote reachability, and optional auto-pull. _(Completed: Added optional chain/pipeline git pre-flight with clean-or-auto-stash checks, branch/upstream/remote validation, optional ff-only auto-pull, and structured UI/API reporting.)_

## P2 - Advanced Features

- [x] Add PR-aware execution mode: run pipeline on a feature branch, auto-push updates, and keep PR description synced with run summary. _(Completed: Added pipeline PR-aware settings/UI, backend config + validation, orchestrator branch/auto-push automation, and GitHub PR description sync with live run summaries.)_
- [x] Add multi-repo workspace: manage multiple local repos with per-repo remote settings, recent runs, and quick pull/push/checkout actions. _(Completed: Added persisted workspace repo APIs + UI table for add/remove/activate flows, per-repo remote/default metadata, recent run summaries from `HISTORY.jsonl`, and row-level quick pull/push/checkout actions.)_
- [ ] Add GitHub Actions integration: generate a workflow to run Codex Manager pipelines on push/PR with logs/artifact upload.
- [ ] Add signed commits/tags support (GPG or SSH signing) with setup UI and pre-push configuration verification.
