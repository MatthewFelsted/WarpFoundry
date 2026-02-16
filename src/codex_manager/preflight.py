"""Shared preflight diagnostics for CLI, GUI, and pipeline entrypoints."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreflightCheck:
    """A single readiness check result."""

    category: str
    key: str
    label: str
    status: str
    detail: str
    hint: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "key": self.key,
            "label": self.label,
            "status": self.status,
            "detail": self.detail,
            "hint": self.hint,
        }


@dataclass(frozen=True)
class PreflightAction:
    """A prioritized, user-facing setup action."""

    key: str
    title: str
    detail: str
    command: str = ""
    severity: str = "required"

    def to_dict(self) -> dict[str, str]:
        return {
            "key": self.key,
            "title": self.title,
            "detail": self.detail,
            "command": self.command,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class AgentPreflightSpec:
    """Static check metadata for one supported agent implementation."""

    category: str
    binary_label: str
    binary_hint: str
    auth_label: str
    auth_env_vars: tuple[str, ...]
    auth_detector: Callable[[], bool]
    auth_detected_detail: str
    auth_missing_detail: str
    auth_hint: str


@dataclass(frozen=True)
class PreflightReport:
    """Structured diagnostics output for setup readiness."""

    requested_agents: list[str]
    checks: list[PreflightCheck]
    repo_path: str
    resolved_repo_path: str
    codex_binary: str = "codex"
    claude_binary: str = "claude"

    @property
    def summary(self) -> dict[str, int]:
        counts = {"pass": 0, "warn": 0, "fail": 0}
        for check in self.checks:
            if check.status in counts:
                counts[check.status] += 1
        return counts

    @property
    def ready(self) -> bool:
        return self.summary["fail"] == 0

    def failure_messages(self) -> list[str]:
        messages: list[str] = []
        for check in self.checks:
            if check.status != "fail":
                continue
            if check.hint:
                messages.append(f"{check.label}: {check.hint}")
            else:
                messages.append(f"{check.label}: {check.detail}")
        return messages

    @property
    def next_actions(self) -> list[PreflightAction]:
        return build_preflight_actions(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_path": self.repo_path,
            "resolved_repo_path": self.resolved_repo_path,
            "requested_agents": list(self.requested_agents),
            "codex_binary": self.codex_binary,
            "claude_binary": self.claude_binary,
            "checks": [c.to_dict() for c in self.checks],
            "summary": self.summary,
            "ready": self.ready,
            "next_actions": [a.to_dict() for a in self.next_actions],
        }


_PLACEHOLDER_SECRET_VALUES = {
    "sk-...",
    "sk-proj-...",
    "sk-ant-...",
    "api-key",
    "token",
    "token-here",
    "xxx",
    "your-key",
    "your key",
    "your_api_key",
    "your-api-key",
}
_PLACEHOLDER_SECRET_SUBSTRINGS = (
    "your-key-here",
    "your key here",
    "your_api_key_here",
    "your-openai-api-key",
    "your-anthropic-api-key",
    "replace-me",
    "replace_with",
    "change-me",
    "changeme",
    "placeholder",
    "set-me",
)


def _looks_like_placeholder_secret(value: str) -> bool:
    """Return True for obvious placeholder API-key text."""
    normalized = (value or "").strip().strip('"').strip("'").lower()
    if not normalized:
        return True
    if normalized in _PLACEHOLDER_SECRET_VALUES:
        return True
    if normalized.startswith("<") and normalized.endswith(">"):
        return True
    if normalized.endswith("..."):
        return True
    return any(token in normalized for token in _PLACEHOLDER_SECRET_SUBSTRINGS)


def _env_secret_present(var_names: tuple[str, ...]) -> bool:
    """Return True when any configured env var has a non-placeholder secret."""
    for name in var_names:
        raw = os.getenv(name)
        if raw is None:
            continue
        value = raw.strip()
        if not value:
            continue
        if _looks_like_placeholder_secret(value):
            continue
        return True
    return False


def _placeholder_env_vars(var_names: tuple[str, ...]) -> list[str]:
    """Return env-var names that are set to obvious placeholder key text."""
    placeholders: list[str] = []
    for name in var_names:
        raw = os.getenv(name)
        if raw is None:
            continue
        value = raw.strip()
        if not value:
            continue
        if _looks_like_placeholder_secret(value):
            placeholders.append(name)
    return placeholders


def binary_exists(binary: str) -> bool:
    """Return ``True`` when an executable exists for *binary*."""
    binary = os.path.expandvars(os.path.expanduser(str(binary or "").strip()))
    if not binary:
        return False
    try:
        candidate = Path(binary)
        if candidate.is_file():
            if os.name == "nt":
                # Windows executes only known script/binary extensions via CreateProcess.
                raw_pathext = os.getenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
                valid_exts = {ext.strip().lower() for ext in raw_pathext.split(";") if ext.strip()}
                if not valid_exts:
                    valid_exts = {".com", ".exe", ".bat", ".cmd"}
                return candidate.suffix.lower() in valid_exts
            return os.access(candidate, os.X_OK)
    except OSError:
        pass
    return shutil.which(binary) is not None


def has_codex_auth() -> bool:
    """Detect Codex/OpenAI auth in env vars or local auth files."""
    if _env_secret_present(("CODEX_API_KEY", "OPENAI_API_KEY")):
        return True
    home = Path.home()
    for path in (
        home / ".codex" / "auth.json",
        home / ".config" / "codex" / "auth.json",
    ):
        if path.exists():
            return True
    return False


def has_claude_auth() -> bool:
    """Detect Claude auth in env vars or local auth files."""
    if _env_secret_present(("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")):
        return True
    home = Path.home()
    for path in (
        home / ".claude.json",
        home / ".claude" / "auth.json",
        home / ".config" / "claude" / "auth.json",
        home / ".config" / "claude-code" / "auth.json",
    ):
        if path.exists():
            return True
    return False


_SUPPORTED_AGENT_SPECS: dict[str, AgentPreflightSpec] = {
    "codex": AgentPreflightSpec(
        category="codex",
        binary_label="Codex CLI binary available",
        binary_hint="Install Codex CLI or update --codex-bin.",
        auth_label="Codex authentication detected",
        auth_env_vars=("CODEX_API_KEY", "OPENAI_API_KEY"),
        auth_detector=lambda: has_codex_auth(),
        auth_detected_detail="Detected CODEX_API_KEY / OPENAI_API_KEY or Codex auth file.",
        auth_missing_detail="No Codex/OpenAI auth detected.",
        auth_hint="Set CODEX_API_KEY or OPENAI_API_KEY, or run 'codex login'.",
    ),
    "claude_code": AgentPreflightSpec(
        category="claude_code",
        binary_label="Claude Code CLI binary available",
        binary_hint="Install Claude Code CLI or update --claude-bin.",
        auth_label="Claude authentication detected",
        auth_env_vars=("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
        auth_detector=lambda: has_claude_auth(),
        auth_detected_detail="Detected ANTHROPIC_API_KEY / CLAUDE_API_KEY or Claude auth file.",
        auth_missing_detail="No Claude auth detected.",
        auth_hint="Set ANTHROPIC_API_KEY (or CLAUDE_API_KEY), or log in via Claude CLI.",
    ),
}


def _auth_failure_detail(placeholder_vars: list[str], missing_detail: str) -> str:
    if placeholder_vars:
        return "Detected placeholder value(s) in " + ", ".join(placeholder_vars) + "."
    return missing_detail


def _agent_binary_for_key(agent: str, *, codex_binary: str, claude_binary: str) -> str:
    if agent == "claude_code":
        return claude_binary
    return codex_binary


def _build_supported_agent_checks(
    *,
    agent: str,
    binary_name: str,
) -> list[PreflightCheck]:
    spec = _SUPPORTED_AGENT_SPECS[agent]
    binary_ok = binary_exists(binary_name)
    auth_ok = spec.auth_detector()
    placeholder_vars = _placeholder_env_vars(spec.auth_env_vars)

    return [
        PreflightCheck(
            category=spec.category,
            key="binary",
            label=spec.binary_label,
            status="pass" if binary_ok else "fail",
            detail=f"Configured binary: {binary_name}",
            hint=spec.binary_hint if not binary_ok else "",
        ),
        PreflightCheck(
            category=spec.category,
            key="auth",
            label=spec.auth_label,
            status="pass" if auth_ok else "fail",
            detail=(
                spec.auth_detected_detail
                if auth_ok
                else _auth_failure_detail(placeholder_vars, spec.auth_missing_detail)
            ),
            hint=spec.auth_hint if not auth_ok else "",
        ),
    ]


def repo_write_error(repo: Path) -> str | None:
    """Return a human-readable write failure for *repo*, if any."""
    try:
        probe_dir = repo / ".codex_manager"
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe_path = probe_dir / f".preflight-write-{uuid.uuid4().hex}.tmp"
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink(missing_ok=True)
        return None
    except Exception as exc:
        return f"Repository is not writable: {exc}"


def repo_worktree_counts(
    repo: Path,
    *,
    timeout_seconds: int = 15,
) -> tuple[int, int, int] | None:
    """Return (staged, unstaged, untracked) from ``git status --porcelain``.

    Returns ``None`` when git status cannot be queried (for example, malformed
    repositories created by unit tests that only contain a ``.git`` directory).
    """
    try:
        probe = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None

    if probe.returncode != 0:
        return None

    staged = 0
    unstaged = 0
    untracked = 0
    for raw_line in str(probe.stdout or "").splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith("??"):
            untracked += 1
            continue
        if line.startswith("!!"):
            continue
        x = line[0] if len(line) >= 1 else " "
        y = line[1] if len(line) >= 2 else " "
        if x not in {" ", "?"}:
            staged += 1
        if y != " ":
            unstaged += 1

    return staged, unstaged, untracked


def repo_worktree_check(repo: Path) -> tuple[str, str, str]:
    """Return (status, detail, hint) for repository worktree cleanliness."""
    counts = repo_worktree_counts(repo)
    if counts is None:
        return (
            "warn",
            "Skipped worktree cleanliness check (could not run git status).",
            "Run 'git -C \"<repo>\" status --short' to verify local changes before running.",
        )

    staged, unstaged, untracked = counts
    if not (staged or unstaged or untracked):
        return ("pass", "Worktree is clean.", "")

    return (
        "fail",
        "Worktree has local changes: "
        f"staged {staged}, unstaged {unstaged}, untracked {untracked}.",
        "Commit/stash/discard local changes first. "
        "Dry-run rollback can discard them, and apply mode can commit them.",
    )


def normalize_agent(agent: str) -> str:
    """Normalize agent names and map common aliases to canonical keys."""
    key = (agent or "codex").strip().lower()
    if key in {"", "auto"}:
        return "codex"
    if key in {"claude", "claude-code", "claude_code", "claudecode"}:
        return "claude_code"
    return key


def parse_agents(raw_agents: str | None) -> list[str]:
    """Parse comma/whitespace-delimited agents into a normalized unique list."""
    raw = (raw_agents or "").strip()
    if not raw:
        return ["codex"]
    keys: list[str] = []
    for token in re.split(r"[\s,;]+", raw):
        cleaned = token.strip().strip('"').strip("'").strip()
        if not cleaned:
            continue
        normalized = normalize_agent(cleaned)
        if normalized not in keys:
            keys.append(normalized)
    return keys or ["codex"]


def build_preflight_report(
    *,
    repo_path: str | Path | None,
    agents: Iterable[str],
    codex_binary: str = "codex",
    claude_binary: str = "claude",
) -> PreflightReport:
    """Build a structured readiness report for the requested repo and agents."""
    codex_binary = str(codex_binary or "codex").strip() or "codex"
    claude_binary = str(claude_binary or "claude").strip() or "claude"

    requested_agents: list[str] = []
    for agent in agents:
        normalized = normalize_agent(str(agent))
        if normalized not in requested_agents:
            requested_agents.append(normalized)
    if not requested_agents:
        requested_agents = ["codex"]

    checks: list[PreflightCheck] = []
    raw_repo = str(repo_path or "").strip()
    resolved_repo_path = raw_repo

    if not raw_repo:
        checks.append(
            PreflightCheck(
                category="repository",
                key="path",
                label="Repository path provided",
                status="warn",
                detail="No repository path set.",
                hint="Provide --repo to validate repository access.",
            )
        )
        checks.append(
            PreflightCheck(
                category="repository",
                key="git_repo",
                label="Git repository detected",
                status="warn",
                detail="Skipped until a repository path is provided.",
                hint="Select a folder that contains a .git directory.",
            )
        )
        checks.append(
            PreflightCheck(
                category="repository",
                key="writable",
                label="Repository is writable",
                status="warn",
                detail="Skipped until a repository path is provided.",
                hint="Use a writable local path.",
            )
        )
        checks.append(
            PreflightCheck(
                category="repository",
                key="clean_worktree",
                label="Repository worktree is clean",
                status="warn",
                detail="Skipped until a repository path is provided.",
                hint="Set --repo and run diagnostics again.",
            )
        )
    else:
        repo = Path(raw_repo)
        path_exists = repo.exists()
        exists = repo.is_dir()
        if path_exists:
            repo = repo.resolve()
            resolved_repo_path = str(repo)
        path_detail = f"Path found: {resolved_repo_path}"
        path_hint = ""
        if not path_exists:
            path_detail = f"Path not found: {resolved_repo_path}"
            path_hint = "Double-check the repository path."
        elif not exists:
            path_detail = f"Path exists but is not a directory: {resolved_repo_path}"
            path_hint = "Provide a directory path for the repository."
        checks.append(
            PreflightCheck(
                category="repository",
                key="path",
                label="Repository path exists",
                status="pass" if exists else "fail",
                detail=path_detail,
                hint=path_hint,
            )
        )
        if exists:
            has_git = (repo / ".git").exists()
            checks.append(
                PreflightCheck(
                    category="repository",
                    key="git_repo",
                    label="Git repository detected",
                    status="pass" if has_git else "fail",
                    detail=f".git {'found' if has_git else 'not found'} in {repo}",
                    hint="Initialize git (git init) or pick an existing git repository."
                    if not has_git
                    else "",
                )
            )
            if has_git:
                clean_status, clean_detail, clean_hint = repo_worktree_check(repo)
                checks.append(
                    PreflightCheck(
                        category="repository",
                        key="clean_worktree",
                        label="Repository worktree is clean",
                        status=clean_status,
                        detail=clean_detail,
                        hint=clean_hint,
                    )
                )
            else:
                checks.append(
                    PreflightCheck(
                        category="repository",
                        key="clean_worktree",
                        label="Repository worktree is clean",
                        status="warn",
                        detail="Skipped because no git repository was detected.",
                        hint="Initialize git first, then ensure the worktree is clean.",
                    )
                )
            write_err = repo_write_error(repo)
            checks.append(
                PreflightCheck(
                    category="repository",
                    key="writable",
                    label="Repository is writable",
                    status="pass" if write_err is None else "fail",
                    detail=f"Write probe {'succeeded' if write_err is None else 'failed'} for {repo}",
                    hint=write_err or "",
                )
            )
        else:
            skipped_detail = (
                "Skipped because the repository path was not found."
                if not path_exists
                else "Skipped because the repository path is not a directory."
            )
            checks.append(
                PreflightCheck(
                    category="repository",
                    key="git_repo",
                    label="Git repository detected",
                    status="warn",
                    detail=skipped_detail,
                    hint="Fix the repository path first.",
                )
            )
            checks.append(
                PreflightCheck(
                    category="repository",
                    key="writable",
                    label="Repository is writable",
                    status="warn",
                    detail=skipped_detail,
                    hint="Fix the repository path first.",
                )
            )
            checks.append(
                PreflightCheck(
                    category="repository",
                    key="clean_worktree",
                    label="Repository worktree is clean",
                    status="warn",
                    detail=skipped_detail,
                    hint="Fix the repository path first.",
                )
            )

    for agent in requested_agents:
        if agent not in _SUPPORTED_AGENT_SPECS:
            checks.append(
                PreflightCheck(
                    category="agents",
                    key=f"{agent}_supported",
                    label=f"Agent '{agent}' is supported",
                    status="warn",
                    detail=f"Unknown agent: {agent}",
                    hint="Use codex, claude_code, or auto.",
                )
            )
            continue

        checks.extend(
            _build_supported_agent_checks(
                agent=agent,
                binary_name=_agent_binary_for_key(
                    agent, codex_binary=codex_binary, claude_binary=claude_binary
                ),
            )
        )

    return PreflightReport(
        requested_agents=requested_agents,
        checks=checks,
        repo_path=raw_repo,
        resolved_repo_path=resolved_repo_path,
        codex_binary=codex_binary,
        claude_binary=claude_binary,
    )


def _check_status(report: PreflightReport, category: str, key: str) -> str | None:
    """Return the status for a single check, if present."""
    for check in report.checks:
        if check.category == category and check.key == key:
            return check.status
    return None


def _command_token(value: str) -> str:
    """Return a shell-friendly token for display commands."""
    token = str(value or "").strip()
    if not token:
        return '""'
    escaped = token.replace('"', '\\"')
    if any(char.isspace() for char in escaped) or '"' in token:
        return f'"{escaped}"'
    return escaped


def _doctor_command(report: PreflightReport) -> str:
    """Build a ready-to-run doctor command for the current report context."""
    repo = report.resolved_repo_path or report.repo_path
    parts = ["warpfoundry doctor"]
    if repo:
        parts.append(f'--repo "{repo}"')
    if report.requested_agents:
        parts.append(f"--agents {','.join(report.requested_agents)}")
    codex_binary = str(getattr(report, "codex_binary", "codex") or "codex").strip() or "codex"
    claude_binary = str(getattr(report, "claude_binary", "claude") or "claude").strip() or "claude"
    parts.append(f'--codex-bin "{codex_binary}"')
    parts.append(f'--claude-bin "{claude_binary}"')
    return " ".join(parts)


def build_preflight_actions(report: PreflightReport) -> list[PreflightAction]:
    """Generate prioritized setup actions from a preflight report."""
    actions: list[PreflightAction] = []
    seen: set[str] = set()
    repo = report.resolved_repo_path or report.repo_path
    codex_binary = str(getattr(report, "codex_binary", "codex") or "codex").strip() or "codex"
    claude_binary = str(getattr(report, "claude_binary", "claude") or "claude").strip() or "claude"
    codex_cmd = _command_token(codex_binary)
    claude_cmd = _command_token(claude_binary)

    def add(
        key: str,
        title: str,
        detail: str,
        *,
        command: str = "",
        severity: str = "required",
    ) -> None:
        if key in seen:
            return
        seen.add(key)
        actions.append(
            PreflightAction(
                key=key,
                title=title,
                detail=detail,
                command=command,
                severity=severity,
            )
        )

    path_status = _check_status(report, "repository", "path")
    if path_status in {"warn", "fail"}:
        if report.repo_path:
            add(
                "fix_repo_path",
                "Fix repository path",
                "Update the repository path to an existing local directory.",
            )
        else:
            add(
                "set_repo_path",
                "Set repository path",
                "Provide a local repository path before running chain/pipeline modes.",
            )

    git_status = _check_status(report, "repository", "git_repo")
    if git_status == "fail" and repo:
        add(
            "init_git_repo",
            "Initialize git in the target folder",
            "The selected folder is missing a .git directory.",
            command=f'git -C "{repo}" init',
        )

    writable_status = _check_status(report, "repository", "writable")
    if writable_status == "fail":
        add(
            "fix_repo_permissions",
            "Restore repository write access",
            "WarpFoundry needs write access to create state/log files in .codex_manager.",
        )

    clean_worktree_status = _check_status(report, "repository", "clean_worktree")
    if clean_worktree_status == "fail":
        status_command = f'git -C "{repo}" status --short' if repo else "git status --short"
        snapshot_command = (
            f'git -C "{repo}" add -A && '
            f'git -C "{repo}" commit -m "warpfoundry: preflight snapshot"'
            if repo
            else 'git add -A && git commit -m "warpfoundry: preflight snapshot"'
        )
        add(
            "snapshot_worktree_commit",
            "Snapshot local changes into a commit",
            (
                "Creates a local checkpoint commit so your current progress is preserved "
                "and the worktree becomes clean."
            ),
            command=snapshot_command,
            severity="recommended",
        )
        add(
            "clean_worktree",
            "Clean or stash local changes",
            "Local changes can be discarded in dry-run or accidentally committed in apply mode.",
            command=status_command,
        )

    codex_binary_status = _check_status(report, "codex", "binary")
    if codex_binary_status == "fail":
        add(
            "install_codex_cli",
            "Install or configure Codex CLI",
            "Ensure the configured codex binary exists on PATH.",
            command=f"{codex_cmd} --version",
        )

    codex_auth_status = _check_status(report, "codex", "auth")
    if codex_auth_status == "fail":
        add(
            "codex_login",
            "Authenticate Codex",
            "Set CODEX_API_KEY / OPENAI_API_KEY or run Codex CLI login.",
            command=f"{codex_cmd} login",
        )

    claude_binary_status = _check_status(report, "claude_code", "binary")
    if claude_binary_status == "fail":
        add(
            "install_claude_cli",
            "Install or configure Claude CLI",
            "Ensure the configured claude binary exists on PATH.",
            command=f"{claude_cmd} --version",
        )

    claude_auth_status = _check_status(report, "claude_code", "auth")
    if claude_auth_status == "fail":
        add(
            "claude_login",
            "Authenticate Claude",
            "Set ANTHROPIC_API_KEY / CLAUDE_API_KEY or run Claude CLI login.",
            command=f"{claude_cmd} login",
        )

    unsupported_status = [
        check
        for check in report.checks
        if check.category == "agents"
        and check.key.endswith("_supported")
        and check.status == "warn"
    ]
    if unsupported_status:
        add(
            "fix_agent_selection",
            "Use supported agent keys",
            "Replace unknown agents with codex, claude_code, or auto.",
        )

    if actions:
        add(
            "rerun_doctor",
            "Re-run setup diagnostics",
            "Verify all required checks pass before you start a run.",
            command=_doctor_command(report),
            severity="recommended",
        )

    if report.ready and repo:
        add(
            "first_dry_run",
            "Run a safe first strategic loop",
            "Validate your setup in dry-run mode before apply mode.",
            command=(
                f'warpfoundry strategic --repo "{repo}" --mode dry-run --rounds 1'
            ),
            severity="recommended",
        )

    return actions
