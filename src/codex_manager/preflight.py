"""Shared preflight diagnostics for CLI, GUI, and pipeline entrypoints."""

from __future__ import annotations

import os
import re
import shutil
import uuid
from collections.abc import Iterable
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
class PreflightReport:
    """Structured diagnostics output for setup readiness."""

    requested_agents: list[str]
    checks: list[PreflightCheck]
    repo_path: str
    resolved_repo_path: str

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
            "checks": [c.to_dict() for c in self.checks],
            "summary": self.summary,
            "ready": self.ready,
            "next_actions": [a.to_dict() for a in self.next_actions],
        }


def binary_exists(binary: str) -> bool:
    """Return ``True`` when an executable exists for *binary*."""
    if not binary:
        return False
    try:
        candidate = Path(binary)
        if candidate.exists():
            return True
    except Exception:
        pass
    return shutil.which(binary) is not None


def has_codex_auth() -> bool:
    """Detect Codex/OpenAI auth in env vars or local auth files."""
    if os.getenv("CODEX_API_KEY") or os.getenv("OPENAI_API_KEY"):
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
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY"):
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


def normalize_agent(agent: str) -> str:
    """Normalize agent names and map ``auto`` to ``codex``."""
    key = (agent or "codex").strip().lower()
    return "codex" if key in {"", "auto"} else key


def parse_agents(raw_agents: str | None) -> list[str]:
    """Parse comma/whitespace-delimited agents into a normalized unique list."""
    raw = (raw_agents or "").strip()
    if not raw:
        return ["codex"]
    keys: list[str] = []
    for token in re.split(r"[\s,]+", raw):
        if not token:
            continue
        normalized = normalize_agent(token)
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
    else:
        repo = Path(raw_repo)
        exists = repo.is_dir()
        if exists:
            repo = repo.resolve()
            resolved_repo_path = str(repo)
        checks.append(
            PreflightCheck(
                category="repository",
                key="path",
                label="Repository path exists",
                status="pass" if exists else "fail",
                detail=f"Path {'found' if exists else 'not found'}: {resolved_repo_path}",
                hint="Double-check the repository path." if not exists else "",
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
            checks.append(
                PreflightCheck(
                    category="repository",
                    key="git_repo",
                    label="Git repository detected",
                    status="warn",
                    detail="Skipped because the repository path was not found.",
                    hint="Fix the repository path first.",
                )
            )
            checks.append(
                PreflightCheck(
                    category="repository",
                    key="writable",
                    label="Repository is writable",
                    status="warn",
                    detail="Skipped because the repository path was not found.",
                    hint="Fix the repository path first.",
                )
            )

    for agent in requested_agents:
        if agent not in {"codex", "claude_code"}:
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

        if agent == "codex":
            codex_binary_ok = binary_exists(codex_binary)
            codex_auth_ok = has_codex_auth()
            checks.append(
                PreflightCheck(
                    category="codex",
                    key="binary",
                    label="Codex CLI binary available",
                    status="pass" if codex_binary_ok else "fail",
                    detail=f"Configured binary: {codex_binary}",
                    hint="Install Codex CLI or update --codex-bin." if not codex_binary_ok else "",
                )
            )
            checks.append(
                PreflightCheck(
                    category="codex",
                    key="auth",
                    label="Codex authentication detected",
                    status="pass" if codex_auth_ok else "fail",
                    detail=(
                        "Detected CODEX_API_KEY / OPENAI_API_KEY or Codex auth file."
                        if codex_auth_ok
                        else "No Codex/OpenAI auth detected."
                    ),
                    hint=(
                        "Set CODEX_API_KEY or OPENAI_API_KEY, or run 'codex login'."
                        if not codex_auth_ok
                        else ""
                    ),
                )
            )
        else:
            claude_binary_ok = binary_exists(claude_binary)
            claude_auth_ok = has_claude_auth()
            checks.append(
                PreflightCheck(
                    category="claude_code",
                    key="binary",
                    label="Claude Code CLI binary available",
                    status="pass" if claude_binary_ok else "fail",
                    detail=f"Configured binary: {claude_binary}",
                    hint="Install Claude Code CLI or update --claude-bin."
                    if not claude_binary_ok
                    else "",
                )
            )
            checks.append(
                PreflightCheck(
                    category="claude_code",
                    key="auth",
                    label="Claude authentication detected",
                    status="pass" if claude_auth_ok else "fail",
                    detail=(
                        "Detected ANTHROPIC_API_KEY / CLAUDE_API_KEY or Claude auth file."
                        if claude_auth_ok
                        else "No Claude auth detected."
                    ),
                    hint=(
                        "Set ANTHROPIC_API_KEY (or CLAUDE_API_KEY), or log in via Claude CLI."
                        if not claude_auth_ok
                        else ""
                    ),
                )
            )

    return PreflightReport(
        requested_agents=requested_agents,
        checks=checks,
        repo_path=raw_repo,
        resolved_repo_path=resolved_repo_path,
    )


def _check_status(report: PreflightReport, category: str, key: str) -> str | None:
    """Return the status for a single check, if present."""
    for check in report.checks:
        if check.category == category and check.key == key:
            return check.status
    return None


def _doctor_command(report: PreflightReport) -> str:
    """Build a ready-to-run doctor command for the current report context."""
    repo = report.resolved_repo_path or report.repo_path
    parts = ["python -m codex_manager doctor"]
    if repo:
        parts.append(f'--repo "{repo}"')
    if report.requested_agents:
        parts.append(f'--agents {",".join(report.requested_agents)}')
    return " ".join(parts)


def build_preflight_actions(report: PreflightReport) -> list[PreflightAction]:
    """Generate prioritized setup actions from a preflight report."""
    actions: list[PreflightAction] = []
    seen: set[str] = set()
    repo = report.resolved_repo_path or report.repo_path

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
            "Codex Manager needs write access to create state/log files in .codex_manager.",
        )

    codex_binary_status = _check_status(report, "codex", "binary")
    if codex_binary_status == "fail":
        add(
            "install_codex_cli",
            "Install or configure Codex CLI",
            "Ensure the configured codex binary exists on PATH.",
            command="codex --version",
        )

    codex_auth_status = _check_status(report, "codex", "auth")
    if codex_auth_status == "fail":
        add(
            "codex_login",
            "Authenticate Codex",
            "Set CODEX_API_KEY / OPENAI_API_KEY or run Codex CLI login.",
            command="codex login",
        )

    claude_binary_status = _check_status(report, "claude_code", "binary")
    if claude_binary_status == "fail":
        add(
            "install_claude_cli",
            "Install or configure Claude CLI",
            "Ensure the configured claude binary exists on PATH.",
            command="claude --version",
        )

    claude_auth_status = _check_status(report, "claude_code", "auth")
    if claude_auth_status == "fail":
        add(
            "claude_login",
            "Authenticate Claude",
            "Set ANTHROPIC_API_KEY / CLAUDE_API_KEY or run Claude CLI login.",
            command="claude login",
        )

    unsupported_status = [
        check
        for check in report.checks
        if check.category == "agents" and check.key.endswith("_supported") and check.status == "warn"
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
                f'python -m codex_manager strategic --repo "{repo}" '
                "--mode dry-run --rounds 1"
            ),
            severity="recommended",
        )

    return actions
