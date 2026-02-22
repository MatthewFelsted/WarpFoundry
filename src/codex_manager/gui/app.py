"""Flask web application - serves the GUI and provides API endpoints."""

from __future__ import annotations

import atexit
import faulthandler
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import socket
import string
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
import zipfile
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Timer
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from flask import Flask, Response, jsonify, render_template, request, send_file

from codex_manager import __version__
from codex_manager.file_io import read_text_utf8_resilient
from codex_manager.gui.chain import ChainExecutor
from codex_manager.gui.models import (
    DANGER_CONFIRMATION_PHRASE,
    ChainConfig,
    PipelineGUIConfig,
    TaskStep,
)
from codex_manager.gui.presets import get_preset, list_presets
from codex_manager.gui.recipes import (
    DEFAULT_RECIPE_ID,
    custom_recipes_path,
    delete_custom_recipe,
    export_custom_recipes,
    get_recipe,
    import_custom_recipes,
    list_custom_recipes,
    list_recipe_summaries,
    recipe_steps_map,
    save_custom_recipe,
)
from codex_manager.gui.stop_guidance import get_stop_guidance
from codex_manager.monitoring import ModelCatalogWatchdog
from codex_manager.preflight import (
    agent_preflight_issues as shared_agent_preflight_issues,
)
from codex_manager.preflight import (
    binary_exists as shared_binary_exists,
)
from codex_manager.preflight import (
    build_preflight_report,
    parse_agents,
)
from codex_manager.preflight import (
    env_secret_issue as shared_env_secret_issue,
)
from codex_manager.preflight import (
    has_claude_auth as shared_has_claude_auth,
)
from codex_manager.preflight import (
    has_codex_auth as shared_has_codex_auth,
)
from codex_manager.preflight import (
    image_provider_auth_issue as shared_image_provider_auth_issue,
)
from codex_manager.preflight import (
    repo_worktree_counts as shared_repo_worktree_counts,
)
from codex_manager.preflight import (
    repo_write_error as shared_repo_write_error,
)

try:
    import keyring
    from keyring.errors import KeyringError, PasswordDeleteError
except Exception:  # pragma: no cover - optional dependency/runtime availability
    keyring = None  # type: ignore[assignment]

    class KeyringError(RuntimeError):
        """Fallback keyring error type when keyring import is unavailable."""

    class PasswordDeleteError(KeyringError):
        """Fallback keyring delete error when keyring import is unavailable."""

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = str(Path(__file__).resolve().parent / "templates")
app = Flask(__name__, template_folder=_TEMPLATE_DIR)

# Single global executor - the GUI manages one chain at a time
executor = ChainExecutor()

# Pipeline executor (separate from chain)
_pipeline_executor = None  # lazy init

# Saved chain-config directory
CONFIGS_DIR = Path.home() / ".codex_manager" / "chains"
WORKSPACE_REPOS_PATH = Path.home() / ".codex_manager" / "workspace" / "repos.json"
_CONFIG_NAME_ALLOWED_CHARS = frozenset(string.ascii_letters + string.digits + "-_ ")
_KNOWN_AGENTS = {"codex", "claude_code", "auto"}
_DOCS_CATALOG: dict[str, tuple[str, str]] = {
    "quickstart": ("Quickstart", "QUICKSTART.md"),
    "output_artifacts": ("Outputs and Artifacts", "OUTPUTS_AND_ARTIFACTS.md"),
    "tutorial": ("Tutorial", "TUTORIAL.md"),
    "cli_reference": ("CLI Reference", "CLI_REFERENCE.md"),
    "troubleshooting": ("Troubleshooting", "TROUBLESHOOTING.md"),
    "model_watchdog": ("Model Watchdog", "MODEL_WATCHDOG.md"),
    "licensing_commercial": ("Licensing and Commercial", "LICENSING_AND_COMMERCIAL.md"),
}
_PIPELINE_LOG_FILES = frozenset(
    {
        "WISHLIST.md",
        "TESTPLAN.md",
        "ERRORS.md",
        "EXPERIMENTS.md",
        "RESEARCH.md",
        "PROGRESS.md",
        "SCIENTIST_REPORT.md",
        "BRAIN.md",
        "HISTORY.md",
    }
)
_RUN_ARTIFACT_BUNDLE_INCLUDE_KEYS = ("outputs", "logs", "config", "history")
_RUN_ARTIFACT_BUNDLE_LOOKBACK_LIMIT = 250
_RUNNABLE_DIAGNOSTIC_ACTION_KEYS = frozenset(
    {
        "init_git_repo",
        "install_codex_cli",
        "install_claude_cli",
        "snapshot_worktree_commit",
        "rerun_doctor",
    }
)
_DIAGNOSTIC_ACTION_TIMEOUT_SECONDS = 20
_DIAGNOSTIC_ACTION_OUTPUT_MAX_CHARS = 4000
_ATOMIC_REPLACE_MAX_RETRIES = 8
_ATOMIC_REPLACE_RETRY_SECONDS = 0.01
_SERVER_PORT = 5088
_SERVER_OPEN_BROWSER = True
_MODEL_WATCHDOG_ROOT = Path.home() / ".codex_manager" / "watchdog"
_GOVERNANCE_POLICY_PATH = Path.home() / ".codex_manager" / "governance" / "source_policy.json"
_GITHUB_AUTH_META_PATH = Path.home() / ".codex_manager" / "github" / "auth_meta.json"
_DEFAULT_PROJECT_DISPLAY_NAME = "WarpFoundry"
_GOVERNANCE_ENV_KEYS: dict[str, str] = {
    "research_allowed_domains": "CODEX_MANAGER_RESEARCH_ALLOWED_DOMAINS",
    "research_blocked_domains": "CODEX_MANAGER_RESEARCH_BLOCKED_DOMAINS",
    "deep_research_allowed_domains": "DEEP_RESEARCH_ALLOWED_SOURCE_DOMAINS",
    "deep_research_blocked_domains": "DEEP_RESEARCH_BLOCKED_SOURCE_DOMAINS",
}
_GITHUB_SECRET_SERVICE = "warpfoundry.github_auth"
_GITHUB_SECRET_SERVICE_LEGACY = "codex_manager.github_auth"
_GITHUB_PAT_SECRET_KEY = "pat"
_GITHUB_SSH_SECRET_KEY = "ssh_private_key"
_GITHUB_AUTH_METHODS = frozenset({"https", "ssh"})
_PROJECT_AUTHOR = "Matthew Felsted"
_API_KEY_SECRET_SERVICE = "warpfoundry.api_keys"
_API_KEY_SECRET_SERVICE_LEGACY = "codex_manager.api_keys"
_API_KEY_FIELD_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        "CODEX_API_KEY",
        "Codex / OpenAI (preferred)",
        "Primary key for Codex/OpenAI-backed agent and CUA flows.",
    ),
    (
        "OPENAI_API_KEY",
        "OpenAI",
        "Alternative OpenAI key accepted by Codex and native deep-research flows.",
    ),
    (
        "ANTHROPIC_API_KEY",
        "Anthropic / Claude",
        "Primary key for Claude and Anthropic-backed CUA flows.",
    ),
    (
        "CLAUDE_API_KEY",
        "Claude alias",
        "Alias environment variable accepted for Claude/Anthropic auth checks.",
    ),
    (
        "GOOGLE_API_KEY",
        "Google Gemini",
        "Primary key for Google-backed deep-research and generation providers.",
    ),
    (
        "GEMINI_API_KEY",
        "Gemini alias",
        "Alias environment variable accepted for Google Gemini provider auth checks.",
    ),
)
_API_KEY_ALLOWED_ENV_VARS = frozenset(spec[0] for spec in _API_KEY_FIELD_SPECS)
_GITHUB_TEST_TIMEOUT_SECONDS = 20
_GITHUB_REPO_METADATA_TIMEOUT_SECONDS = 8
_GITHUB_REPO_METADATA_CACHE_TTL_SECONDS = 300
_GITHUB_REPO_METADATA_ERROR_CACHE_TTL_SECONDS = 60
_GIT_REMOTE_QUERY_TIMEOUT_SECONDS = 20
_GIT_CLONE_TIMEOUT_SECONDS = 180
_GIT_SYNC_TIMEOUT_SECONDS = 30
_GIT_SIGNING_CHECK_TIMEOUT_SECONDS = 20
_GIT_SIGNING_PUSH_GUARD_KEY = "warpfoundry.signing.requirePushGuard"
_GIT_SYNC_PATH_BATCH_LIMIT_WINDOWS = 7600
_GIT_SYNC_PATH_BATCH_LIMIT_POSIX = 30000
_DEFAULT_BRANCH_SENTINEL = "__remote_default__"
_OWNER_CONTEXT_MAX_FILES = 6
_OWNER_CONTEXT_MAX_FILE_CHARS = 6000
_OWNER_CONTEXT_MAX_TOTAL_CHARS = 24000
_OWNER_REPO_IDEAS_MAX_FILES = 5000
_OWNER_REPO_IDEAS_MAX_MANIFEST_FILES = 1500
_OWNER_REPO_IDEAS_MAX_SNIPPET_FILES = 80
_OWNER_REPO_IDEAS_MAX_FILE_CHARS = 1800
_OWNER_REPO_IDEAS_MAX_TOTAL_CHARS = 60000
_OWNER_REPO_IDEAS_MAX_FILE_BYTES = 250000
_OWNER_REPO_IDEAS_EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tox",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
        "target",
        "coverage",
    }
)
_OWNER_REPO_IDEAS_EXCLUDED_SUBPATHS = frozenset(
    {
        ".codex_manager/logs",
        ".codex_manager/outputs",
        ".codex_manager/output_history",
        ".codex_manager/state",
        ".codex_manager/memory",
        ".codex_manager/ledger",
    }
)
_OWNER_REPO_IDEAS_EXCLUDED_SUFFIXES = frozenset(
    {
        ".log",
        ".tmp",
        ".cache",
        ".pyc",
        ".pyo",
        ".class",
        ".zip",
        ".tar",
        ".gz",
        ".7z",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".mp4",
        ".mov",
        ".avi",
        ".mp3",
        ".wav",
        ".pdf",
        ".db",
        ".sqlite",
    }
)
_OWNER_REPO_IDEAS_EXCLUDED_FILENAMES = frozenset(
    {
        "thumbs.db",
        ".ds_store",
        "npm-debug.log",
        "yarn-error.log",
        "pnpm-debug.log",
    }
)
_GENERAL_REQUEST_HISTORY_MAX_ITEMS = 200
_HTTP_COMPRESSION_MIN_BYTES = 1024
_HTTP_COMPRESSION_LEVEL = 6
_HTTP_COMPRESSION_CACHE_MAX_ENTRIES = 48
_INDEX_CACHE_MAX_AGE_SECONDS = 60
_INDEX_RESPONSE_CACHE_MAX_ENTRIES = 8
_HTTP_COMPRESSIBLE_MIME_TYPES = frozenset(
    {
        "text/html",
        "text/plain",
        "text/css",
        "text/javascript",
        "application/javascript",
        "application/json",
        "image/svg+xml",
    }
)
_SSE_REPLAY_BATCH_LIMIT = 500
_JSONL_ROWS_CACHE_MAX_ENTRIES = 64
_JSONL_CACHE_HEAD_SAMPLE_BYTES = 256
_RUN_COMPARISON_CACHE_MAX_ENTRIES = 64
_GIT_SYNC_BRANCH_CHOICES_CACHE_TTL_SECONDS = 1.5

def _subprocess_isolation_kwargs() -> dict[str, object]:
    """Return kwargs that prevent child console events from reaching the GUI server."""
    if os.name != "nt":
        return {}
    new_pg = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    no_win = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    flags = new_pg | no_win
    return {"creationflags": flags} if flags else {}

_GUI_STARTUP_BIND_RETRIES = 20
_GUI_STARTUP_BIND_RETRY_SECONDS = 0.25
_GUI_RESTART_CHILD_WARMUP_SECONDS = 0.35
_GUI_RESTART_LOG_PATH = Path.home() / ".codex_manager" / "logs" / "GUI_RESTART.log"
_GUI_RUNTIME_LOG_PATH = Path.home() / ".codex_manager" / "logs" / "GUI_RUNTIME.log"
_DIAGNOSTICS_CACHE_TTL_SECONDS = 8.0

_model_watchdog: ModelCatalogWatchdog | None = None
_model_watchdog_lock = threading.Lock()
_github_repo_metadata_cache: dict[str, tuple[float, dict[str, object] | None, str]] = {}
_github_repo_metadata_cache_lock = threading.Lock()
_diagnostics_report_cache: dict[str, tuple[float, dict[str, object]]] = {}
_diagnostics_report_cache_lock = threading.Lock()
_index_response_cache: dict[str, str] = {}
_index_response_cache_lock = threading.Lock()
_compressed_response_cache: dict[str, bytes] = {}
_compressed_response_cache_lock = threading.Lock()
_jsonl_rows_cache: dict[str, _JsonlRowsCacheEntry] = {}
_jsonl_rows_cache_lock = threading.Lock()
_run_comparison_cache: dict[tuple[str, int, int, str, int], dict[str, object]] = {}
_run_comparison_cache_lock = threading.Lock()
_git_sync_branch_choices_cache: dict[str, tuple[float, dict[str, object]]] = {}
_git_sync_branch_choices_cache_lock = threading.Lock()
_gui_runtime_hooks_installed = False
_gui_expected_restart_exit = False
_gui_faulthandler_enabled = False
_gui_faulthandler_stream: object | None = None


@dataclass(slots=True)
class _JsonlRowsCacheEntry:
    """Cache one parsed JSONL file with metadata needed for append fast-paths."""

    mtime_ns: int
    size: int
    rows: list[dict[str, object]]
    trailing_line: str
    head_digest: str


def _path_stat_signature(path: Path) -> tuple[int, int] | None:
    """Return ``(mtime_ns, size)`` for *path* when available."""
    try:
        stat_result = path.stat()
    except OSError:
        return None
    return int(stat_result.st_mtime_ns), int(stat_result.st_size)


def _bounded_cache_set(cache: dict[object, object], key: object, value: object, *, max_entries: int) -> None:
    """Insert one cache entry and evict oldest entries when cache grows past cap."""
    if key in cache:
        cache.pop(key, None)
    cache[key] = value
    while len(cache) > max(1, int(max_entries)):
        oldest_key = next(iter(cache))
        cache.pop(oldest_key, None)


def _recipe_template_payload() -> dict[str, object]:
    """Return recipe data that the GUI template consumes directly."""
    return {
        "default_recipe_id": DEFAULT_RECIPE_ID,
        "recipes": recipe_steps_map(),
    }


def _step_output_filename(name: str, job_type: str) -> str:
    """Mirror chain output-file naming so collisions are caught pre-run."""
    raw = (name or job_type or "step").strip()
    slug = re.sub(r"[^\w\-]+", "-", raw).strip("-")
    return f"{slug or 'step'}.md"


def _sanitize_config_name(name: object) -> str:
    """Return the persisted config-name format used by save/load APIs."""
    raw = str(name or "").strip()
    if not raw:
        return ""
    return "".join(c for c in raw if c in _CONFIG_NAME_ALLOWED_CHARS).strip()


def _is_valid_config_name(name: object) -> bool:
    """Return True when the provided config name is already canonical/safe."""
    raw = str(name or "").strip()
    if not raw:
        return False
    return raw == _sanitize_config_name(raw)


def _is_within_directory(path: Path, root: Path) -> bool:
    """Return True when *path* resolves inside *root*."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _parse_since_results_arg(raw: str | None) -> int | None:
    """Parse polling delta offset from query string."""
    if raw is None or not str(raw).strip():
        return None
    try:
        value = int(str(raw).strip())
    except ValueError:
        return None
    return max(0, value)


def _binary_exists(binary: str) -> bool:
    """Return True when a configured CLI binary is available."""
    return shared_binary_exists(binary)


def _has_codex_auth() -> bool:
    """Detect whether Codex/OpenAI credentials are present."""
    return shared_has_codex_auth()


def _has_claude_auth() -> bool:
    """Detect whether Claude credentials are present."""
    return shared_has_claude_auth()


def _repo_write_error(repo: Path) -> str | None:
    """Return a human-readable write-access error for *repo* if any."""
    return shared_repo_write_error(repo)


def _repo_worktree_counts(repo: Path) -> tuple[int, int, int] | None:
    """Return (staged, unstaged, untracked) git counts, or None if unavailable."""
    return shared_repo_worktree_counts(repo)


def _repo_dirty_issue(repo: Path) -> str | None:
    """Return a preflight issue string when the repository has local changes."""
    counts = _repo_worktree_counts(repo)
    if counts is None:
        return None
    staged, unstaged, untracked = counts
    if not (staged or unstaged or untracked):
        return None
    return (
        "Repository worktree is dirty "
        f"(staged {staged}, unstaged {unstaged}, untracked {untracked}). "
        "Clean/stash local changes first, or enable Git pre-flight with auto-stash."
    )


def _read_text_utf8_resilient(path: Path) -> str:
    """Read a text file, recovering from legacy non-UTF8 bytes when possible."""
    result = read_text_utf8_resilient(path, normalize_to_utf8=True)
    if result.used_fallback:
        logger.warning(
            "Recovered non-UTF8 text file %s using %s; rewritten as UTF-8",
            path,
            result.decoder,
        )
    return result.text


def _write_json_file_atomic(path: Path, payload: object) -> None:
    """Write JSON to disk atomically to avoid partial config files."""
    serialized = json.dumps(payload, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(serialized)
        _replace_file_with_retry(tmp_path, path)
    finally:
        with suppress(OSError):
            tmp_path.unlink(missing_ok=True)


def _replace_file_with_retry(src: Path, dst: Path) -> None:
    """Replace *dst* with *src*, retrying on transient Windows file-lock races."""
    last_error: OSError | None = None
    for attempt in range(_ATOMIC_REPLACE_MAX_RETRIES):
        try:
            src.replace(dst)
            return
        except PermissionError as exc:
            last_error = exc
        except OSError as exc:
            if exc.errno != 13:
                raise
            last_error = exc
        if attempt < _ATOMIC_REPLACE_MAX_RETRIES - 1:
            time.sleep(_ATOMIC_REPLACE_RETRY_SECONDS * (attempt + 1))
    if last_error is not None:
        raise last_error


def _normalize_agent(agent: str) -> str:
    """Normalize agent keys and map common aliases to canonical keys."""
    key = (agent or "codex").strip().lower()
    if key in {"", "auto"}:
        return "codex"
    if key in {"claude", "claude-code", "claude_code", "claudecode"}:
        return "claude_code"
    return key


def _collect_chain_agents(config: ChainConfig) -> set[str]:
    """Collect normalized agent keys used by enabled chain steps."""
    enabled = [s for s in config.steps if s.enabled]
    if not enabled:
        return {"codex"}
    return {_normalize_agent(s.agent) for s in enabled}


def _collect_pipeline_agents(config: PipelineGUIConfig) -> set[str]:
    """Collect normalized agent keys used by enabled pipeline phases."""
    default_agent = _normalize_agent(config.agent)
    if config.phases:
        enabled = [p for p in config.phases if p.enabled]
        if enabled:
            resolved: set[str] = set()
            for phase in enabled:
                phase_agent = (phase.agent or "").strip()
                if not phase_agent or phase_agent.lower() == "auto":
                    resolved.add(default_agent)
                else:
                    resolved.add(_normalize_agent(phase_agent))
            return resolved
    return {default_agent}


def _agent_preflight_issues(
    agents: set[str],
    *,
    codex_binary: str,
    claude_binary: str,
) -> list[str]:
    """Return binary/auth validation issues for the requested agents."""
    return shared_agent_preflight_issues(
        agents,
        codex_binary=codex_binary,
        claude_binary=claude_binary,
        binary_exists_detector=_binary_exists,
        codex_auth_detector=_has_codex_auth,
        claude_auth_detector=_has_claude_auth,
    )


def _image_provider_auth_issue(enabled: bool, provider: str) -> str | None:
    """Return an auth/config issue for configured image generation provider."""
    return shared_image_provider_auth_issue(
        enabled,
        provider,
        codex_auth_detector=_has_codex_auth,
    )


def _append_repository_preflight_issues(
    issues: list[str],
    *,
    repo: Path,
    git_preflight_enabled: bool,
) -> None:
    """Append repository existence/write/dirty preflight issues."""
    if not (repo / ".git").exists():
        issues.append(f"Not a git repository: {repo}")

    write_error = _repo_write_error(repo)
    if write_error:
        issues.append(write_error)

    if not git_preflight_enabled:
        dirty_issue = _repo_dirty_issue(repo)
        if dirty_issue:
            issues.append(dirty_issue)


def _append_danger_confirmation_issue(
    issues: list[str],
    *,
    bypass_approvals_and_sandbox: bool,
    danger_confirmation: str,
) -> None:
    """Append the required warning/confirmation gate for dangerous mode."""
    if (
        bypass_approvals_and_sandbox
        and danger_confirmation.strip() != DANGER_CONFIRMATION_PHRASE
    ):
        issues.append(
            "Danger confirmation missing. Set codex_danger_confirmation to "
            f"'{DANGER_CONFIRMATION_PHRASE}' to enable bypass."
        )


def _append_vector_memory_preflight_issues(
    issues: list[str],
    *,
    vector_memory_enabled: bool,
    vector_memory_backend: str,
) -> None:
    """Append vector-memory backend/import readiness issues."""
    if not vector_memory_enabled:
        return
    backend = str(vector_memory_backend or "chroma").strip().lower()
    if backend != "chroma":
        issues.append("Unsupported vector_memory_backend. Supported backend(s): chroma.")
        return
    try:
        import chromadb  # noqa: F401
    except Exception:
        issues.append(
            "Vector memory requires ChromaDB. Install with: pip install chromadb"
        )


def _normalize_requested_agents(raw_agents: object) -> list[str]:
    """Normalize requested diagnostics agents, defaulting to Codex + Claude."""
    normalized: list[str] = []
    if isinstance(raw_agents, str):
        normalized = parse_agents(raw_agents)
    elif isinstance(raw_agents, (list, tuple, set)):
        for item in raw_agents:
            for key in parse_agents(str(item)):
                if key and key not in normalized:
                    normalized.append(key)
    if not normalized:
        normalized = ["codex", "claude_code"]
    return normalized


def _build_diagnostics_report(
    *,
    repo_path: str,
    codex_binary: str,
    claude_binary: str,
    requested_agents: list[str],
) -> dict[str, object]:
    """Build diagnostics using shared preflight logic (CLI + GUI parity)."""
    report = build_preflight_report(
        repo_path=(repo_path or "").strip(),
        agents=requested_agents,
        codex_binary=codex_binary,
        claude_binary=claude_binary,
    )
    payload = report.to_dict()
    for action in payload.get("next_actions", []):
        if not isinstance(action, dict):
            continue
        key = str(action.get("key") or "").strip()
        action["can_run"] = bool(key and key in _RUNNABLE_DIAGNOSTIC_ACTION_KEYS)
    return payload


def _extract_diagnostics_request(data: dict[str, object]) -> tuple[str, str, str, list[str]]:
    """Parse common diagnostics request payload fields."""
    repo_path = str(data.get("repo_path") or data.get("path") or "").strip()
    codex_binary = str(data.get("codex_binary") or "codex").strip() or "codex"
    claude_binary = str(data.get("claude_binary") or "claude").strip() or "claude"
    requested_agents = _normalize_requested_agents(data.get("agents"))
    return repo_path, codex_binary, claude_binary, requested_agents


def _diagnostics_cache_key(
    *,
    repo_path: str,
    codex_binary: str,
    claude_binary: str,
    requested_agents: list[str],
) -> str:
    """Return a stable cache key for diagnostics requests."""
    payload = {
        "repo_path": str(repo_path or "").strip(),
        "codex_binary": str(codex_binary or "").strip() or "codex",
        "claude_binary": str(claude_binary or "").strip() or "claude",
        "requested_agents": list(requested_agents or []),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _diagnostics_action_args(action_key: str, report) -> list[str] | None:
    """Build subprocess argv for a runnable diagnostics action key."""
    repo = (str(report.resolved_repo_path or "") or str(report.repo_path or "")).strip()
    codex_binary = str(getattr(report, "codex_binary", "codex") or "codex").strip() or "codex"
    claude_binary = str(getattr(report, "claude_binary", "claude") or "claude").strip() or "claude"

    if action_key == "init_git_repo":
        if not repo:
            return None
        return ["git", "-C", repo, "init"]
    if action_key == "install_codex_cli":
        return [codex_binary, "--version"]
    if action_key == "install_claude_cli":
        return [claude_binary, "--version"]
    if action_key == "rerun_doctor":
        args = [sys.executable, "-m", "codex_manager", "doctor"]
        if repo:
            args.extend(["--repo", repo])
        requested_agents = list(getattr(report, "requested_agents", []) or [])
        if requested_agents:
            args.extend(["--agents", ",".join(requested_agents)])
        args.extend(["--codex-bin", codex_binary, "--claude-bin", claude_binary])
        return args
    return None


def _diagnostics_snapshot_worktree_commit(report) -> dict[str, object]:
    """Create a local snapshot commit for dirty worktrees without discarding changes."""
    repo_raw = (str(report.resolved_repo_path or "") or str(report.repo_path or "")).strip()
    repo, error, _status = _resolve_git_sync_repo(repo_raw)
    if repo is None:
        return {
            "ok": False,
            "exit_code": 2,
            "timed_out": False,
            "stdout": "",
            "stderr": error,
        }

    status_before = _git_sync_status_payload(repo)
    if not bool(status_before.get("dirty")):
        return {
            "ok": True,
            "exit_code": 0,
            "timed_out": False,
            "stdout": "",
            "stderr": "",
            "message": "Worktree is already clean. No snapshot commit needed.",
            "sync": status_before,
            "commit": _git_last_commit_summary(repo),
        }

    # Stage everything first so the commit captures all progress.
    stage_result = _run_git_sync_command(repo, "add", "-A")
    if stage_result.returncode != 0:
        detail = _extract_git_process_error(stage_result, "git add -A failed")
        return {
            "ok": False,
            "exit_code": stage_result.returncode,
            "timed_out": False,
            "stdout": _truncate_command_output(stage_result.stdout),
            "stderr": _truncate_command_output(detail),
            "sync": _git_sync_status_payload(repo),
        }

    # Ensure commit identity exists so auto-commit can proceed in fresh repos.
    for key, fallback in (
        ("user.name", _project_display_name()),
        ("user.email", "warpfoundry@localhost"),
    ):
        probe = _run_git_sync_command(repo, "config", key)
        if probe.returncode != 0 or not str(probe.stdout or "").strip():
            _run_git_sync_command(repo, "config", key, fallback)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    commit_message = f"warpfoundry: preflight snapshot ({timestamp})"
    commit_result = _run_git_sync_command(repo, "commit", "-m", commit_message)
    detail = _extract_git_process_error(commit_result, "git commit failed")
    if commit_result.returncode != 0:
        if "nothing to commit" in detail.lower():
            return {
                "ok": True,
                "exit_code": 0,
                "timed_out": False,
                "stdout": _truncate_command_output(commit_result.stdout),
                "stderr": _truncate_command_output(commit_result.stderr),
                "message": "No commit was created because there are no staged changes.",
                "sync": _git_sync_status_payload(repo),
                "commit": _git_last_commit_summary(repo),
            }
        return {
            "ok": False,
            "exit_code": commit_result.returncode,
            "timed_out": False,
            "stdout": _truncate_command_output(commit_result.stdout),
            "stderr": _truncate_command_output(detail),
            "sync": _git_sync_status_payload(repo),
        }

    return {
        "ok": True,
        "exit_code": 0,
        "timed_out": False,
        "stdout": _truncate_command_output(commit_result.stdout),
        "stderr": _truncate_command_output(commit_result.stderr),
        "message": "Snapshot commit created.",
        "commit_message": commit_message,
        "sync": _git_sync_status_payload(repo),
        "commit": _git_last_commit_summary(repo),
    }


def _truncate_command_output(value: str | bytes | None) -> str:
    """Trim and cap command output for safe API responses."""
    if value is None:
        return ""
    text = value.decode(errors="replace") if isinstance(value, bytes) else str(value)
    text = text.strip()
    if len(text) <= _DIAGNOSTIC_ACTION_OUTPUT_MAX_CHARS:
        return text
    keep = max(0, _DIAGNOSTIC_ACTION_OUTPUT_MAX_CHARS - len("\n...[truncated]"))
    return text[:keep] + "\n...[truncated]"


def _run_diagnostics_action(args: list[str], *, cwd: str) -> dict[str, object]:
    """Run one diagnostics command and return a structured result payload."""
    run_cwd = cwd if cwd and Path(cwd).is_dir() else None
    try:
        proc = subprocess.run(
            args,
            cwd=run_cwd,
            capture_output=True,
            text=True,
            timeout=_DIAGNOSTIC_ACTION_TIMEOUT_SECONDS,
            check=False,
            **_subprocess_isolation_kwargs(),
        )
        return {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "timed_out": False,
            "stdout": _truncate_command_output(proc.stdout),
            "stderr": _truncate_command_output(proc.stderr),
        }
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "exit_code": 127,
            "timed_out": False,
            "stdout": "",
            "stderr": _truncate_command_output(str(exc)),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "exit_code": 124,
            "timed_out": True,
            "stdout": _truncate_command_output(exc.stdout),
            "stderr": _truncate_command_output(exc.stderr),
        }


def _chain_preflight_issues(config: ChainConfig) -> list[str]:
    """Return chain-level preflight issues before execution starts."""
    repo = Path(config.repo_path).resolve()
    issues: list[str] = []

    _append_repository_preflight_issues(
        issues,
        repo=repo,
        git_preflight_enabled=bool(config.git_preflight_enabled),
    )

    # Detect output-file collisions up front (same filename from multiple steps).
    by_file: dict[str, list[str]] = {}
    display_name_by_file: dict[str, str] = {}
    for step in config.steps:
        if not step.enabled:
            continue
        out_file = _step_output_filename(step.name, step.job_type)
        key = out_file.casefold()
        display_name_by_file.setdefault(key, out_file)
        by_file.setdefault(key, []).append(step.name or step.job_type)
    for out_file_key, names in sorted(by_file.items()):
        if len(names) > 1:
            out_file = display_name_by_file.get(out_file_key, out_file_key)
            issues.append(f"Multiple enabled steps write to '{out_file}': {', '.join(names)}")

    _append_danger_confirmation_issue(
        issues,
        bypass_approvals_and_sandbox=bool(config.codex_bypass_approvals_and_sandbox),
        danger_confirmation=config.codex_danger_confirmation,
    )

    issues.extend(
        _agent_preflight_issues(
            _collect_chain_agents(config),
            codex_binary=config.codex_binary,
            claude_binary=config.claude_binary,
        )
    )

    image_issue = _image_provider_auth_issue(
        bool(config.image_generation_enabled),
        config.image_provider,
    )
    if image_issue:
        issues.append(image_issue)

    _append_vector_memory_preflight_issues(
        issues,
        vector_memory_enabled=bool(config.vector_memory_enabled),
        vector_memory_backend=config.vector_memory_backend,
    )

    return issues


def _pipeline_preflight_issues(config: PipelineGUIConfig) -> list[str]:
    """Return pipeline-level preflight issues before execution starts."""
    repo = Path(config.repo_path).resolve()
    issues: list[str] = []

    _append_repository_preflight_issues(
        issues,
        repo=repo,
        git_preflight_enabled=bool(config.git_preflight_enabled),
    )

    _append_danger_confirmation_issue(
        issues,
        bypass_approvals_and_sandbox=bool(config.codex_bypass_approvals_and_sandbox),
        danger_confirmation=config.codex_danger_confirmation,
    )

    issues.extend(
        _agent_preflight_issues(
            _collect_pipeline_agents(config),
            codex_binary=config.codex_binary,
            claude_binary=config.claude_binary,
        )
    )

    image_issue = _image_provider_auth_issue(
        bool(config.image_generation_enabled),
        config.image_provider,
    )
    if image_issue:
        issues.append(image_issue)

    _append_vector_memory_preflight_issues(
        issues,
        vector_memory_enabled=bool(config.vector_memory_enabled),
        vector_memory_backend=config.vector_memory_backend,
    )

    if config.self_improvement_auto_restart and not config.self_improvement_enabled:
        issues.append("self_improvement_auto_restart requires self_improvement_enabled.")

    if config.pr_aware_enabled and config.mode != "apply":
        issues.append("pr_aware_enabled requires mode='apply'.")

    if config.deep_research_native_enabled and config.deep_research_enabled:
        providers = str(config.deep_research_providers or "both").strip().lower()
        if providers in {"both", "openai"}:
            issue = shared_env_secret_issue(
                ("OPENAI_API_KEY", "CODEX_API_KEY"),
                "Native deep research (OpenAI) requires OPENAI_API_KEY or CODEX_API_KEY.",
            )
            if issue:
                issues.append(issue)
        if providers in {"both", "google"}:
            issue = shared_env_secret_issue(
                ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
                "Native deep research (Google) requires GOOGLE_API_KEY or GEMINI_API_KEY.",
            )
            if issue:
                issues.append(issue)

    return issues


def _resolve_chain_output_repo(repo_hint: str = "") -> Path | None:
    """Resolve a repo path for chain-output APIs from query or executor config."""
    raw = (repo_hint or "").strip()
    if raw:
        p = Path(raw)
        return p.resolve() if p.is_dir() else None
    cfg = executor.config
    if cfg and cfg.repo_path:
        p = Path(cfg.repo_path)
        return p.resolve() if p.is_dir() else None
    return None


def _chain_output_dir(repo: Path) -> Path:
    """Return the chain output directory under ``.codex_manager``."""
    return repo / ".codex_manager" / "outputs"


def _resolve_pipeline_logs_repo(repo_hint: str = "") -> Path | None:
    """Resolve a repo path for pipeline-log APIs from query or executor state."""
    raw = (repo_hint or "").strip()
    if raw:
        p = Path(raw)
        return p.resolve() if p.is_dir() else None

    global _pipeline_executor
    if _pipeline_executor is not None:
        executor_repo_raw = str(getattr(_pipeline_executor, "repo_path", "")).strip()
        if executor_repo_raw:
            p = Path(executor_repo_raw)
            if p.is_dir():
                return p.resolve()
    return None


def _pipeline_logs_dir(repo: Path) -> Path:
    """Return the pipeline logs directory under ``.codex_manager``."""
    return repo / ".codex_manager" / "logs"


def _history_jsonl_path(repo: Path) -> Path:
    """Return the run-history JSONL path for *repo*."""
    return _pipeline_logs_dir(repo) / "HISTORY.jsonl"


def _run_comparison_cache_key(
    history_path: Path,
    *,
    scope: str,
    limit: int,
) -> tuple[str, int, int, str, int] | None:
    """Build a cache key for run-comparison payloads from history file metadata."""
    signature = _path_stat_signature(history_path)
    if signature is None:
        return None
    return (
        str(history_path.resolve()),
        signature[0],
        signature[1],
        str(scope or "all"),
        int(limit),
    )


def _run_comparison_cache_get(
    cache_key: tuple[str, int, int, str, int] | None,
) -> dict[str, object] | None:
    """Return cached run-comparison payload when present."""
    if cache_key is None:
        return None
    with _run_comparison_cache_lock:
        return _run_comparison_cache.get(cache_key)


def _run_comparison_cache_set(
    cache_key: tuple[str, int, int, str, int] | None,
    payload: dict[str, object],
) -> None:
    """Persist one run-comparison payload and evict stale entries for same file."""
    if cache_key is None:
        return
    history_key = cache_key[0]
    history_signature = (cache_key[1], cache_key[2])
    with _run_comparison_cache_lock:
        stale_keys = [
            key
            for key in _run_comparison_cache
            if key[0] == history_key and (key[1], key[2]) != history_signature
        ]
        for stale_key in stale_keys:
            _run_comparison_cache.pop(stale_key, None)
        _bounded_cache_set(
            _run_comparison_cache, cache_key, payload, max_entries=_RUN_COMPARISON_CACHE_MAX_ENTRIES
        )


def _jsonl_head_digest(path: Path) -> str:
    """Return a short digest of the first bytes of a JSONL file."""
    sample_bytes = max(1, int(_JSONL_CACHE_HEAD_SAMPLE_BYTES))
    try:
        with path.open("rb") as handle:
            sample = handle.read(sample_bytes)
    except OSError:
        return ""
    if not sample:
        return "empty"
    return hashlib.sha256(sample).hexdigest()[:16]


def _parse_jsonl_text_rows(
    raw_text: str,
    *,
    base_rows: list[dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], int, str]:
    """Parse JSON-object rows from text and preserve one trailing partial line."""
    rows = list(base_rows) if base_rows is not None else []
    text = str(raw_text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not text:
        return rows, 0, ""

    trailing_line = ""
    if not text.endswith("\n"):
        split_index = text.rfind("\n")
        if split_index < 0:
            return rows, 0, text
        trailing_line = text[split_index + 1 :]
        text = text[: split_index + 1]

    invalid_rows = 0
    for line in text.split("\n"):
        item = line.strip()
        if not item:
            continue
        try:
            payload = json.loads(item)
        except json.JSONDecodeError:
            invalid_rows += 1
            continue
        if not isinstance(payload, dict):
            invalid_rows += 1
            continue
        rows.append(payload)
    return rows, invalid_rows, trailing_line


def _try_append_jsonl_cache_rows(
    *,
    path: Path,
    signature: tuple[int, int],
    cached: _JsonlRowsCacheEntry,
) -> tuple[_JsonlRowsCacheEntry | None, int]:
    """Return an incrementally-updated cache entry when the JSONL file was append-only."""
    mtime_ns, size = signature
    if size < cached.size:
        return None, 0
    try:
        with path.open("rb") as handle:
            sample = handle.read(max(1, int(_JSONL_CACHE_HEAD_SAMPLE_BYTES)))
            current_head = (
                hashlib.sha256(sample).hexdigest()[:16] if sample else "empty"
            )
            if cached.head_digest and current_head != cached.head_digest:
                return None, 0
            handle.seek(cached.size)
            delta_bytes = handle.read()
    except OSError:
        return None, 0

    delta_text = delta_bytes.decode("utf-8", errors="replace") if delta_bytes else ""
    rows, invalid_rows, trailing_line = _parse_jsonl_text_rows(
        cached.trailing_line + delta_text,
        base_rows=cached.rows,
    )
    return (
        _JsonlRowsCacheEntry(
            mtime_ns=mtime_ns,
            size=size,
            rows=rows,
            trailing_line=trailing_line,
            head_digest=current_head,
        ),
        invalid_rows,
    )


def _read_jsonl_dict_rows(path: Path, *, warn_context: str = "") -> list[dict[str, object]]:
    """Return parsed JSON-object rows from a JSONL file."""
    if not path.is_file():
        return []
    cache_key = str(path.resolve())
    signature = _path_stat_signature(path)
    cached_entry: _JsonlRowsCacheEntry | None = None
    if signature is not None:
        with _jsonl_rows_cache_lock:
            cached_entry = _jsonl_rows_cache.get(cache_key)
            if cached_entry is not None and (cached_entry.mtime_ns, cached_entry.size) == signature:
                return cached_entry.rows
        if (
            cached_entry is not None
            and signature[1] >= cached_entry.size
            and signature[0] >= cached_entry.mtime_ns
        ):
            incremental_entry, invalid_rows = _try_append_jsonl_cache_rows(
                path=path,
                signature=signature,
                cached=cached_entry,
            )
            if incremental_entry is not None:
                if invalid_rows and warn_context:
                    logger.warning(
                        "Ignored %s malformed JSONL row(s) while reading %s (%s).",
                        invalid_rows,
                        warn_context,
                        path,
                    )
                with _jsonl_rows_cache_lock:
                    _bounded_cache_set(
                        _jsonl_rows_cache,
                        cache_key,
                        incremental_entry,
                        max_entries=_JSONL_ROWS_CACHE_MAX_ENTRIES,
                    )
                return incremental_entry.rows
    try:
        raw = _read_text_utf8_resilient(path)
    except Exception as exc:
        if warn_context:
            logger.warning("Could not read %s (%s): %s", warn_context, path, exc)
        return []
    rows, invalid_rows, trailing_line = _parse_jsonl_text_rows(raw)
    if invalid_rows and warn_context:
        logger.warning(
            "Ignored %s malformed JSONL row(s) while reading %s (%s).",
            invalid_rows,
            warn_context,
            path,
        )
    if signature is not None:
        with _jsonl_rows_cache_lock:
            _bounded_cache_set(
                _jsonl_rows_cache,
                cache_key,
                _JsonlRowsCacheEntry(
                    mtime_ns=signature[0],
                    size=signature[1],
                    rows=rows,
                    trailing_line=trailing_line,
                    head_digest=_jsonl_head_digest(path),
                ),
                max_entries=_JSONL_ROWS_CACHE_MAX_ENTRIES,
            )
    return rows


def _history_context_payload(event: dict[str, object]) -> dict[str, object]:
    """Return a normalized context payload from a history event row."""
    context_obj = event.get("context")
    return context_obj if isinstance(context_obj, dict) else {}


def _scope_run_id_stack(
    open_run_ids_by_scope: dict[str, list[str]],
    scope: str,
) -> list[str]:
    """Return mutable run-id stack for one scope."""
    stack = open_run_ids_by_scope.get(scope)
    if stack is None:
        stack = []
        open_run_ids_by_scope[scope] = stack
    return stack


def _context_run_id(context: dict[str, object]) -> str:
    """Return explicit context run id when present."""
    return str(context.get("run_id") or "").strip()


def _register_scope_run_start(
    open_run_ids_by_scope: dict[str, list[str]],
    *,
    scope: str,
    context: dict[str, object],
    fallback_event_id: str,
) -> str:
    """Register a run start and return the resolved run id."""
    run_id = _context_run_id(context) or str(fallback_event_id or "").strip()
    if not run_id:
        return ""
    _scope_run_id_stack(open_run_ids_by_scope, scope).append(run_id)
    return run_id


def _resolve_scope_run_id(
    open_run_ids_by_scope: dict[str, list[str]],
    *,
    scope: str,
    context: dict[str, object],
) -> str:
    """Resolve run id for a non-start history event."""
    context_id = _context_run_id(context)
    if context_id:
        return context_id
    stack = open_run_ids_by_scope.get(scope)
    if stack:
        return stack[-1]
    return ""


def _pop_scope_run_id(
    open_run_ids_by_scope: dict[str, list[str]],
    *,
    scope: str,
    context: dict[str, object],
) -> str:
    """Resolve and remove one scope run id for a finished event."""
    stack = open_run_ids_by_scope.get(scope)
    context_id = _context_run_id(context)
    if not stack:
        return context_id
    if context_id:
        for idx in range(len(stack) - 1, -1, -1):
            if stack[idx] == context_id:
                removed = stack.pop(idx)
                if not stack:
                    open_run_ids_by_scope.pop(scope, None)
                return removed
        return context_id
    removed = stack.pop()
    if not stack:
        open_run_ids_by_scope.pop(scope, None)
    return removed


def _scope_open_run_stack(
    open_runs_by_scope: dict[str, list[dict[str, object]]],
    scope: str,
) -> list[dict[str, object]]:
    """Return mutable open-run aggregate stack for one scope."""
    stack = open_runs_by_scope.get(scope)
    if stack is None:
        stack = []
        open_runs_by_scope[scope] = stack
    return stack


def _find_scope_open_run_by_id(
    scope_runs: list[dict[str, object]],
    run_id: str,
) -> dict[str, object] | None:
    """Return a matching open run aggregate by run id from newest to oldest."""
    target = str(run_id or "").strip()
    if not target:
        return None
    for candidate in reversed(scope_runs):
        if str(candidate.get("run_id") or "").strip() == target:
            return candidate
    return None


def _resolve_scope_open_run(
    open_runs_by_scope: dict[str, list[dict[str, object]]],
    *,
    scope: str,
    context: dict[str, object],
) -> dict[str, object] | None:
    """Return the best matching open aggregate for a scope event."""
    scope_runs = open_runs_by_scope.get(scope) or []
    if not scope_runs:
        return None
    match = _find_scope_open_run_by_id(scope_runs, _context_run_id(context))
    if match is not None:
        return match
    return scope_runs[-1]


def _pop_scope_open_run(
    open_runs_by_scope: dict[str, list[dict[str, object]]],
    *,
    scope: str,
    context: dict[str, object],
) -> dict[str, object] | None:
    """Remove and return the best matching open aggregate for a finished event."""
    scope_runs = open_runs_by_scope.get(scope)
    if not scope_runs:
        return None
    context_id = _context_run_id(context)
    if context_id:
        for idx in range(len(scope_runs) - 1, -1, -1):
            candidate = scope_runs[idx]
            if str(candidate.get("run_id") or "").strip() == context_id:
                removed = scope_runs.pop(idx)
                if not scope_runs:
                    open_runs_by_scope.pop(scope, None)
                return removed
    removed = scope_runs.pop()
    if not scope_runs:
        open_runs_by_scope.pop(scope, None)
    return removed


def _pipeline_artifact_bundle_dir(repo: Path) -> Path:
    """Return the run-artifact bundle directory under ``.codex_manager/output_history``."""
    return repo / ".codex_manager" / "output_history" / "artifact_bundles"


def _normalize_run_artifact_bundle_includes(payload: object) -> dict[str, bool]:
    """Normalize include toggles for run-artifact bundle export APIs."""
    data = payload if isinstance(payload, dict) else {}
    return {
        key: _safe_bool(data.get(f"include_{key}"), True)
        for key in _RUN_ARTIFACT_BUNDLE_INCLUDE_KEYS
    }


def _find_run_comparison_run(repo: Path, run_id: str) -> dict[str, object] | None:
    """Return one run-comparison row by run id when available."""
    target = str(run_id or "").strip()
    if not target:
        return None
    payload = _pipeline_run_comparison(
        repo,
        scope="all",
        limit=_RUN_ARTIFACT_BUNDLE_LOOKBACK_LIMIT,
    )
    rows_obj = payload.get("runs") if isinstance(payload, dict) else []
    rows = rows_obj if isinstance(rows_obj, list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("run_id") or "").strip() == target:
            return dict(row)
    return None


def _history_events_for_run(repo: Path, run_id: str) -> list[dict[str, object]]:
    """Return HISTORY.jsonl events associated with a run id."""
    target = str(run_id or "").strip()
    if not target:
        return []
    history_path = _history_jsonl_path(repo)
    rows = _read_jsonl_dict_rows(history_path)
    if not rows:
        return []

    events: list[dict[str, object]] = []
    open_run_ids_by_scope: dict[str, list[str]] = {}
    for event in rows:
        scope = str(event.get("scope") or "").strip().lower()
        if scope not in {"chain", "pipeline"}:
            continue
        event_name = str(event.get("event") or "").strip().lower()
        event_id = str(event.get("id") or "").strip()
        context = _history_context_payload(event)

        if event_name == "run_started":
            derived_run_id = _register_scope_run_start(
                open_run_ids_by_scope,
                scope=scope,
                context=context,
                fallback_event_id=event_id,
            )
        elif event_name == "run_finished":
            derived_run_id = _pop_scope_run_id(
                open_run_ids_by_scope,
                scope=scope,
                context=context,
            )
        else:
            derived_run_id = _resolve_scope_run_id(
                open_run_ids_by_scope,
                scope=scope,
                context=context,
            )

        if derived_run_id == target:
            enriched = dict(event)
            patched_context = dict(context)
            if target and not _context_run_id(patched_context):
                patched_context["run_id"] = target
            enriched["context"] = patched_context
            events.append(enriched)
    return events


def _zip_add_text(archive: zipfile.ZipFile, *, arcname: str, content: str) -> None:
    """Write one UTF-8 text entry into a zip archive."""
    normalized_arcname = str(arcname or "").replace("\\", "/").strip("/") or "entry.txt"
    archive.writestr(normalized_arcname, str(content or ""))


def _zip_add_directory(
    archive: zipfile.ZipFile,
    *,
    source_dir: Path,
    archive_prefix: str,
    exclude_top_level: set[str] | None = None,
    trusted_root: Path | None = None,
) -> int:
    """Recursively add files from *source_dir* under *archive_prefix*."""
    if not source_dir.is_dir():
        return 0
    try:
        source_root = source_dir.resolve()
    except OSError:
        return 0
    trusted_root_resolved = source_root
    if trusted_root is not None:
        try:
            trusted_root_resolved = trusted_root.resolve()
        except OSError:
            return 0
        if not _is_within_directory(source_root, trusted_root_resolved):
            logger.warning("Skipping archive source outside trusted root: %s", source_dir)
            return 0
    excluded = {str(name or "").strip().lower() for name in (exclude_top_level or set())}
    added = 0
    prefix = str(archive_prefix or "").replace("\\", "/").strip("/")
    for path in sorted(source_dir.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            resolved_file = path.resolve()
        except OSError:
            continue
        if not _is_within_directory(resolved_file, source_root):
            logger.warning("Skipping archive entry outside source root: %s", path)
            continue
        if trusted_root is not None and not _is_within_directory(
            resolved_file,
            trusted_root_resolved,
        ):
            logger.warning("Skipping archive entry outside trusted root: %s", path)
            continue
        rel = resolved_file.relative_to(source_root)
        if rel.parts and rel.parts[0].strip().lower() in excluded:
            continue
        rel_posix = rel.as_posix()
        arcname = f"{prefix}/{rel_posix}" if prefix else rel_posix
        archive.write(resolved_file, arcname)
        added += 1
    return added


def _zip_add_file(
    archive: zipfile.ZipFile,
    *,
    source_path: Path,
    arcname: str,
    trusted_root: Path,
) -> int:
    """Add one file when it is a non-symlink path inside *trusted_root*."""
    if source_path.is_symlink() or not source_path.is_file():
        return 0
    try:
        resolved_file = source_path.resolve()
        trusted_root_resolved = trusted_root.resolve()
    except OSError:
        return 0
    if not _is_within_directory(resolved_file, trusted_root_resolved):
        logger.warning("Skipping archive file outside trusted root: %s", source_path)
        return 0
    normalized_arcname = str(arcname or "").replace("\\", "/").strip("/") or source_path.name
    archive.write(resolved_file, normalized_arcname)
    return 1


def _create_run_artifact_bundle(
    repo: Path,
    *,
    run_id: str,
    includes: dict[str, bool],
) -> dict[str, object]:
    """Create a zip bundle with selected run artifacts and return bundle metadata."""
    run_key = str(run_id or "").strip()
    if not run_key:
        raise ValueError("run_id is required.")

    normalized_includes = {
        key: bool(includes.get(key))
        for key in _RUN_ARTIFACT_BUNDLE_INCLUDE_KEYS
    }
    if not any(normalized_includes.values()):
        raise ValueError("Select at least one artifact category to export.")

    run = _find_run_comparison_run(repo, run_key)
    run_events = _history_events_for_run(repo, run_key)
    if run is None and not run_events:
        raise FileNotFoundError(f"Run id not found in history: {run_key}")
    if run is None:
        first_scope = str(run_events[0].get("scope") or "").strip() if run_events else ""
        run = {
            "run_id": run_key,
            "scope": first_scope,
            "event_count": len(run_events),
        }

    start_context_obj = run.get("start_context")
    start_context = start_context_obj if isinstance(start_context_obj, dict) else {}
    config_snapshot_obj = start_context.get("config_snapshot")
    config_snapshot = config_snapshot_obj if isinstance(config_snapshot_obj, dict) else {}
    try:
        repo_root = repo.resolve()
    except OSError:
        repo_root = repo

    export_dir = _pipeline_artifact_bundle_dir(repo)
    export_dir.mkdir(parents=True, exist_ok=True)
    export_dir_resolved = export_dir.resolve()

    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", run_key).strip("-") or "run"
    if len(slug) > 56:
        slug = slug[:56].rstrip("-") or "run"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle_name = f"run-artifacts-{timestamp}-{slug}.zip"
    bundle_path = (export_dir_resolved / bundle_name).resolve()
    if bundle_path.parent != export_dir_resolved:
        raise RuntimeError("Refusing to write artifact bundle outside export directory.")

    created_at = _utc_now_iso_z()
    entry_count = 0
    with zipfile.ZipFile(bundle_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        if normalized_includes["outputs"]:
            entry_count += _zip_add_directory(
                archive,
                source_dir=_chain_output_dir(repo),
                archive_prefix="outputs/current",
                trusted_root=repo_root,
            )
            entry_count += _zip_add_directory(
                archive,
                source_dir=repo / ".codex_manager" / "output_history",
                archive_prefix="outputs/history",
                exclude_top_level={"artifact_bundles"},
                trusted_root=repo_root,
            )

        if normalized_includes["logs"]:
            entry_count += _zip_add_directory(
                archive,
                source_dir=_pipeline_logs_dir(repo),
                archive_prefix="logs",
                trusted_root=repo_root,
            )

        if normalized_includes["history"]:
            history_path = _history_jsonl_path(repo)
            entry_count += _zip_add_file(
                archive,
                source_path=history_path,
                arcname="history/HISTORY.jsonl",
                trusted_root=repo_root,
            )
            if run_events:
                run_events_jsonl = "\n".join(
                    json.dumps(event, ensure_ascii=False) for event in run_events
                )
                _zip_add_text(
                    archive,
                    arcname="history/run-events.jsonl",
                    content=(run_events_jsonl + "\n") if run_events_jsonl else "",
                )
                _zip_add_text(
                    archive,
                    arcname="history/run-events.json",
                    content=json.dumps(run_events, indent=2, ensure_ascii=False),
                )
                entry_count += 2

        if normalized_includes["config"]:
            state_path = repo / ".codex_manager" / "state.json"
            entry_count += _zip_add_file(
                archive,
                source_path=state_path,
                arcname="config/state.json",
                trusted_root=repo_root,
            )
            _zip_add_text(
                archive,
                arcname="config/run-summary.json",
                content=json.dumps(run, indent=2, ensure_ascii=False),
            )
            entry_count += 1
            if start_context:
                _zip_add_text(
                    archive,
                    arcname="config/run-start-context.json",
                    content=json.dumps(start_context, indent=2, ensure_ascii=False),
                )
                entry_count += 1
            if config_snapshot:
                _zip_add_text(
                    archive,
                    arcname="config/run-config-snapshot.json",
                    content=json.dumps(config_snapshot, indent=2, ensure_ascii=False),
                )
                entry_count += 1

        manifest = {
            "created_at": created_at,
            "repo_path": str(repo),
            "run_id": run_key,
            "includes": normalized_includes,
            "history_event_count": len(run_events),
            "entry_count": entry_count,
        }
        _zip_add_text(
            archive,
            arcname="manifest.json",
            content=json.dumps(manifest, indent=2, ensure_ascii=False),
        )
        entry_count += 1

    size_bytes = bundle_path.stat().st_size if bundle_path.is_file() else 0
    repo_q = quote(str(repo), safe="")
    return {
        "status": "created",
        "repo_path": str(repo),
        "run_id": run_key,
        "run": run,
        "includes": normalized_includes,
        "history_event_count": len(run_events),
        "bundle_name": bundle_name,
        "bundle_path": str(bundle_path),
        "bundle_size_bytes": int(size_bytes),
        "entry_count": entry_count,
        "created_at": created_at,
        "download_url": (
            f"/api/pipeline/run-comparison/export/{quote(bundle_name)}"
            f"?repo_path={repo_q}"
        ),
    }


def _resolve_pipeline_logs_repo_for_api(repo_hint: str = "") -> tuple[Path | None, str, int]:
    """Resolve a safe repository path for pipeline-log API endpoints."""
    raw = (repo_hint or "").strip()
    if raw:
        repo = Path(raw).expanduser().resolve()
        if not repo.is_dir():
            return None, f"Repo path not found: {raw}", 400
        return repo, "", 200

    repo = _resolve_pipeline_logs_repo("")
    if repo is not None:
        return repo, "", 200
    return None, "repo_path is required when no active pipeline run is available.", 400


def _parse_sse_resume_id() -> int:
    """Parse SSE replay cursor from query args or Last-Event-ID header."""
    candidates = (
        request.args.get("after_id"),
        request.args.get("last_event_id"),
        request.headers.get("Last-Event-ID"),
    )
    for raw in candidates:
        if raw is None:
            continue
        try:
            return max(0, int(str(raw).strip()))
        except ValueError:
            continue
    return 0


def _sse_frame(payload: dict[str, object], *, event_id: int | None = None) -> str:
    """Build one SSE frame."""
    lines: list[str] = []
    if event_id is not None and event_id > 0:
        lines.append(f"id: {event_id}")
    lines.append(f"data: {json.dumps(payload)}")
    return "\n".join(lines) + "\n\n"


def _replay_log_events(
    source: object,
    *,
    after_id: int,
    limit: int = _SSE_REPLAY_BATCH_LIMIT,
) -> tuple[list[dict[str, object]], bool]:
    """Fetch non-destructive log events from a chain/pipeline executor."""
    getter = getattr(source, "get_log_events_since", None)
    if callable(getter):
        events, replay_gap = getter(after_id, limit=limit)
        return [dict(event) for event in events], bool(replay_gap)

    queue_obj = getattr(source, "log_queue", None)
    if queue_obj is None or not hasattr(queue_obj, "mutex"):
        return [], False

    with queue_obj.mutex:
        snapshot = [dict(item) for item in list(queue_obj.queue)]
    if not snapshot:
        return [], False
    oldest_id = _safe_int(snapshot[0].get("id"), 0)
    replay_gap = bool(after_id > 0 and oldest_id > after_id + 1)
    events = [entry for entry in snapshot if _safe_int(entry.get("id"), 0) > after_id]
    if limit > 0 and len(events) > limit:
        events = events[-limit:]
    return events, replay_gap


def _parse_iso_epoch_ms(value: object) -> int:
    """Parse ISO-8601 timestamps into epoch milliseconds."""
    raw = str(value or "").strip()
    if not raw:
        return 0
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        stamp = datetime.fromisoformat(raw)
    except ValueError:
        return 0
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return int(stamp.timestamp() * 1000)


def _run_comparison_scope(value: object) -> str:
    """Normalize run-comparison scope filter."""
    scope = str(value or "all").strip().lower()
    if scope in {"chain", "pipeline"}:
        return scope
    return "all"


def _run_comparison_limit(value: object) -> int:
    """Normalize run-comparison list limit."""
    parsed = _safe_int(value, 12)
    return max(1, min(50, parsed))


def _safe_int(value: object, default: int = 0) -> int:
    """Coerce *value* to int where possible."""
    try:
        if isinstance(value, bool):
            return int(value)
        return int(str(value).strip())
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    """Coerce *value* to float where possible."""
    try:
        if isinstance(value, bool):
            return float(int(value))
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _safe_bool(value: object, default: bool = False) -> bool:
    """Coerce *value* to bool where possible."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


_RUN_COST_PROVIDER_DEFAULT_RATES: dict[str, tuple[float, float]] = {
    # input_rate, output_rate (USD per 1k tokens)
    "openai": (0.01, 0.03),
    "anthropic": (0.015, 0.075),
    "google": (0.004, 0.012),
    "mixed": (0.01, 0.03),
    "unknown": (0.01, 0.03),
}
_RUN_COST_OUTLIER_MULTIPLIER = 2.5
_RUN_COST_EFFICIENCY_DENOMINATOR = 0.001


def _run_cost_model_key(value: object) -> str:
    """Return an env-safe model key used for per-model cost overrides."""
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", raw).strip("_")


def _run_cost_provider(agent_used: object, model: object) -> str:
    """Infer provider key from agent/model metadata."""
    agent = str(agent_used or "").strip().lower()
    model_name = str(model or "").strip().lower()

    if agent.startswith("deep_research:"):
        provider_hint = agent.split(":", 1)[1].strip()
        if provider_hint in {"openai", "google", "anthropic"}:
            return provider_hint
        if provider_hint in {"both", "mixed"}:
            return "mixed"

    if "claude" in agent or "anthropic" in agent:
        return "anthropic"
    if "google" in agent or "gemini" in agent:
        return "google"
    if "codex" in agent or "openai" in agent:
        return "openai"

    if "claude" in model_name:
        return "anthropic"
    if "gemini" in model_name:
        return "google"
    if "gpt" in model_name or model_name.startswith("o"):
        return "openai"
    return "unknown"


def _run_cost_rates(provider: str, model: object) -> tuple[float, float]:
    """Resolve input/output USD-per-1k rates with optional env overrides."""
    provider_key = str(provider or "unknown").strip().lower() or "unknown"
    model_key = _run_cost_model_key(model)

    default_in, default_out = _RUN_COST_PROVIDER_DEFAULT_RATES.get(
        provider_key,
        _RUN_COST_PROVIDER_DEFAULT_RATES["unknown"],
    )

    if model_key:
        default_in = _safe_float(
            os.getenv(f"CODEX_MANAGER_RUN_COST_{model_key.upper()}_USD_PER_1K_INPUT"),
            default_in,
        )
        default_out = _safe_float(
            os.getenv(f"CODEX_MANAGER_RUN_COST_{model_key.upper()}_USD_PER_1K_OUTPUT"),
            default_out,
        )

    default_in = _safe_float(
        os.getenv(f"CODEX_MANAGER_RUN_COST_{provider_key.upper()}_USD_PER_1K_INPUT"),
        default_in,
    )
    default_out = _safe_float(
        os.getenv(f"CODEX_MANAGER_RUN_COST_{provider_key.upper()}_USD_PER_1K_OUTPUT"),
        default_out,
    )
    return max(0.0, default_in), max(0.0, default_out)


def _estimate_run_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    provider: str,
    model: object,
) -> float:
    """Estimate USD cost for one token-usage sample."""
    in_tokens = max(0, int(input_tokens or 0))
    out_tokens = max(0, int(output_tokens or 0))
    if in_tokens <= 0 and out_tokens <= 0:
        return 0.0
    in_rate, out_rate = _run_cost_rates(provider, model)
    return round(((in_tokens / 1000.0) * in_rate) + ((out_tokens / 1000.0) * out_rate), 6)


def _median_float(values: list[float]) -> float:
    """Return median for a numeric list, or 0 when empty."""
    if not values:
        return 0.0
    sorted_values = sorted(float(item) for item in values)
    size = len(sorted_values)
    mid = size // 2
    if size % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def _run_cost_outlier_threshold(values: list[float]) -> float:
    """Return the budget-outlier threshold for run-cost estimates."""
    if len(values) < 3:
        return 0.0
    baseline = _median_float(values)
    if baseline <= 0:
        return 0.0
    return round(baseline * _RUN_COST_OUTLIER_MULTIPLIER, 6)


def _run_configuration_label(scope: str, context: dict[str, object]) -> str:
    """Return a compact run-configuration summary label."""
    mode = str(context.get("mode") or "unknown").strip() or "unknown"
    unlimited = bool(context.get("unlimited"))

    if scope == "chain":
        max_loops = _safe_int(context.get("max_loops"), 0)
        loops = "infinite" if unlimited else (str(max_loops) if max_loops > 0 else "?")
        steps = context.get("steps")
        step_count = len(steps) if isinstance(steps, list) else 0
        return f"mode={mode}, loops={loops}, steps={step_count}"

    max_cycles = _safe_int(context.get("max_cycles"), 0)
    cycles = "infinite" if unlimited else (str(max_cycles) if max_cycles > 0 else "?")
    phase_order = context.get("phase_order")
    phase_count = len(phase_order) if isinstance(phase_order, list) else 0
    science = "on" if bool(context.get("science_enabled")) else "off"
    brain = "on" if bool(context.get("brain_enabled")) else "off"
    return (
        f"mode={mode}, cycles={cycles}, phases={phase_count}, "
        f"science={science}, brain={brain}"
    )


def _run_tests_summary(tests: dict[str, int]) -> str:
    """Build a compact tests summary string."""
    passed = _safe_int(tests.get("passed"), 0)
    failed_total = _safe_int(tests.get("failed"), 0) + _safe_int(tests.get("error"), 0)
    skipped = _safe_int(tests.get("skipped"), 0)
    unknown = _safe_int(tests.get("unknown"), 0)
    summary = f"{passed} passed / {failed_total} failed / {skipped} skipped"
    if unknown > 0:
        summary += f" / {unknown} unknown"
    return summary


def _run_test_score(tests: dict[str, int]) -> int:
    """Return a simple quality score for test outcomes."""
    passed = _safe_int(tests.get("passed"), 0)
    failed = _safe_int(tests.get("failed"), 0)
    errored = _safe_int(tests.get("error"), 0)
    unknown = _safe_int(tests.get("unknown"), 0)
    return (passed * 3) - (failed * 4) - (errored * 5) - unknown


def _run_overall_score(run: dict[str, object]) -> float:
    """Rank a run by tests first, then efficiency."""
    tests = run.get("tests")
    tests_payload = tests if isinstance(tests, dict) else {}
    test_score = float(_run_test_score(tests_payload))
    commit_bonus = min(5.0, float(_safe_int(run.get("commit_count"), 0))) * 0.3
    duration_penalty = max(0.0, _safe_float(run.get("duration_seconds"), 0.0)) / 600.0
    token_penalty = max(0.0, float(_safe_int(run.get("token_usage"), 0))) / 1_500_000.0
    return round(test_score + commit_bonus - duration_penalty - token_penalty, 3)


def _run_has_executed_tests(run: dict[str, object]) -> bool:
    """Return True when the run contains passed/failed/errored test outcomes."""
    tests_obj = run.get("tests")
    tests = tests_obj if isinstance(tests_obj, dict) else {}
    return (
        _safe_int(tests.get("passed"), 0)
        + _safe_int(tests.get("failed"), 0)
        + _safe_int(tests.get("error"), 0)
    ) > 0


def _run_has_meaningful_activity(run: dict[str, object]) -> bool:
    """Ignore trivial runs (for example immediate stop with all tests skipped)."""
    if _run_has_executed_tests(run):
        return True
    if _safe_int(run.get("token_usage"), 0) > 0:
        return True
    if _safe_int(run.get("commit_count"), 0) > 0:
        return True
    diff_obj = run.get("diff_summary")
    diff = diff_obj if isinstance(diff_obj, dict) else {}
    if _safe_int(diff.get("files_changed_total"), 0) > 0:
        return True
    return _safe_int(diff.get("changed_paths_count"), 0) > 0


def _new_run_aggregate(
    *,
    scope: str,
    event_id: str,
    timestamp: str,
    context: dict[str, object],
) -> dict[str, object]:
    """Initialize an in-progress run aggregate from a ``run_started`` event."""
    started_epoch_ms = _parse_iso_epoch_ms(timestamp)
    context_run_id = str(context.get("run_id") or "").strip()
    run_id = context_run_id or event_id or f"{scope}_run_{started_epoch_ms}"
    return {
        "run_id": run_id,
        "scope": scope,
        "started_at": timestamp,
        "started_at_epoch_ms": started_epoch_ms,
        "mode": str(context.get("mode") or "unknown").strip() or "unknown",
        "configuration": _run_configuration_label(scope, context),
        "run_name": str(context.get("chain_name") or "").strip(),
        "tests": {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "error": 0,
            "unknown": 0,
        },
        "result_events": 0,
        "_token_usage_from_results": 0,
        "_estimated_cost_from_results": 0.0,
        "_cost_by_model": {},
        "_commit_shas": set(),
        "_files_changed_total": 0,
        "_net_lines_changed_total": 0,
        "_changed_path_set": set(),
        "_changed_path_samples": [],
        "_start_context": dict(context),
    }


def _record_run_result_event(run: dict[str, object], context: dict[str, object]) -> None:
    """Merge ``step_result``/``phase_result`` metrics into an in-progress run."""
    tests = run.get("tests")
    if not isinstance(tests, dict):
        tests = {}
        run["tests"] = tests

    raw_outcome = str(context.get("test_outcome") or "").strip().lower()
    outcome = raw_outcome if raw_outcome in {"passed", "failed", "skipped", "error"} else "unknown"
    tests[outcome] = _safe_int(tests.get(outcome), 0) + 1
    run["result_events"] = _safe_int(run.get("result_events"), 0) + 1

    input_tokens = max(0, _safe_int(context.get("input_tokens"), 0))
    output_tokens = max(0, _safe_int(context.get("output_tokens"), 0))
    token_usage = max(0, _safe_int(context.get("total_tokens"), 0))
    if token_usage <= 0:
        token_usage = input_tokens + output_tokens
    if (input_tokens + output_tokens) <= 0 and token_usage > 0:
        # Legacy events may only report ``total_tokens``; use a stable split
        # for estimation/ranking.
        input_tokens = int(token_usage * 0.6)
        output_tokens = max(0, token_usage - input_tokens)
    run["_token_usage_from_results"] = _safe_int(run.get("_token_usage_from_results"), 0) + max(
        0, token_usage
    )

    agent_used = str(context.get("agent_used") or "").strip()
    model_name = str(context.get("model") or "").strip()
    provider = _run_cost_provider(agent_used, model_name)
    event_cost = _estimate_run_cost_usd(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        provider=provider,
        model=model_name,
    )
    run["_estimated_cost_from_results"] = round(
        _safe_float(run.get("_estimated_cost_from_results"), 0.0) + event_cost,
        6,
    )
    model_breakdown = run.get("_cost_by_model")
    if not isinstance(model_breakdown, dict):
        model_breakdown = {}
        run["_cost_by_model"] = model_breakdown
    breakdown_key = f"{provider}:{model_name or 'unknown'}"
    breakdown_entry = model_breakdown.get(breakdown_key)
    if not isinstance(breakdown_entry, dict):
        breakdown_entry = {
            "provider": provider,
            "model": model_name or "unknown",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
    breakdown_entry["input_tokens"] = _safe_int(breakdown_entry.get("input_tokens"), 0) + input_tokens
    breakdown_entry["output_tokens"] = _safe_int(breakdown_entry.get("output_tokens"), 0) + output_tokens
    breakdown_entry["total_tokens"] = _safe_int(breakdown_entry.get("total_tokens"), 0) + token_usage
    breakdown_entry["estimated_cost_usd"] = round(
        _safe_float(breakdown_entry.get("estimated_cost_usd"), 0.0) + event_cost,
        6,
    )
    model_breakdown[breakdown_key] = breakdown_entry

    commit_sha = str(context.get("commit_sha") or "").strip()
    commits = run.get("_commit_shas")
    if isinstance(commits, set) and commit_sha and commit_sha.lower() != "none":
        commits.add(commit_sha)

    run["_files_changed_total"] = _safe_int(run.get("_files_changed_total"), 0) + max(
        0, _safe_int(context.get("files_changed"), 0)
    )
    run["_net_lines_changed_total"] = _safe_int(
        run.get("_net_lines_changed_total"), 0
    ) + _safe_int(context.get("net_lines_changed"), 0)

    changed_paths = run.get("_changed_path_set")
    changed_samples = run.get("_changed_path_samples")
    if not isinstance(changed_samples, list):
        changed_samples = []
        run["_changed_path_samples"] = changed_samples
    changed_payload = context.get("changed_files")
    if isinstance(changed_paths, set) and isinstance(changed_payload, list):
        for item in changed_payload:
            if not isinstance(item, dict):
                continue
            raw_path = str(item.get("path") or "").strip()
            if raw_path and raw_path not in changed_paths:
                changed_paths.add(raw_path)
                if len(changed_samples) < 120:
                    changed_samples.append(raw_path)


def _finalize_run_aggregate(
    run: dict[str, object],
    *,
    event_id: str,
    timestamp: str,
    summary: str,
    context: dict[str, object],
) -> dict[str, object]:
    """Finalize a run aggregate from a ``run_finished`` event."""
    finished_epoch_ms = _parse_iso_epoch_ms(timestamp)
    started_epoch_ms = _safe_int(run.get("started_at_epoch_ms"), 0)

    duration_seconds = _safe_float(context.get("elapsed_seconds"), 0.0)
    if duration_seconds <= 0 and finished_epoch_ms > 0 and started_epoch_ms > 0:
        duration_seconds = round(max(0, finished_epoch_ms - started_epoch_ms) / 1000.0, 1)

    token_usage = _safe_int(context.get("total_tokens"), 0)
    if token_usage <= 0:
        token_usage = _safe_int(run.get("_token_usage_from_results"), 0)
    estimated_cost_usd = _safe_float(context.get("estimated_cost_usd"), 0.0)
    if estimated_cost_usd <= 0:
        estimated_cost_usd = _safe_float(run.get("_estimated_cost_from_results"), 0.0)
    estimated_cost_usd = round(max(0.0, estimated_cost_usd), 6)

    stop_reason = str(context.get("stop_reason") or "").strip()
    if not stop_reason:
        match = re.search(r"stop_reason='([^']+)'", str(summary or ""))
        if match:
            stop_reason = str(match.group(1) or "").strip()
    if not stop_reason:
        stop_reason = "unknown"

    commits = run.get("_commit_shas")
    commit_count = len(commits) if isinstance(commits, set) else 0

    tests = run.get("tests")
    tests_payload = tests if isinstance(tests, dict) else {}
    tests_summary = _run_tests_summary(tests_payload)
    breakdown_raw = run.get("_cost_by_model")
    model_cost_breakdown: list[dict[str, object]] = []
    if isinstance(breakdown_raw, dict):
        for item in breakdown_raw.values():
            if not isinstance(item, dict):
                continue
            model_cost_breakdown.append(
                {
                    "provider": str(item.get("provider") or "unknown"),
                    "model": str(item.get("model") or "unknown"),
                    "input_tokens": max(0, _safe_int(item.get("input_tokens"), 0)),
                    "output_tokens": max(0, _safe_int(item.get("output_tokens"), 0)),
                    "total_tokens": max(0, _safe_int(item.get("total_tokens"), 0)),
                    "estimated_cost_usd": round(
                        max(0.0, _safe_float(item.get("estimated_cost_usd"), 0.0)),
                        6,
                    ),
                }
            )
    model_cost_breakdown.sort(
        key=lambda row: (
            _safe_float(row.get("estimated_cost_usd"), 0.0),
            _safe_int(row.get("total_tokens"), 0),
        ),
        reverse=True,
    )

    run_id = str(run.get("run_id") or "").strip()
    if not run_id:
        run_id = event_id or f"{run.get('scope', 'run')}_{finished_epoch_ms}"

    finalized = {
        "run_id": run_id,
        "scope": str(run.get("scope") or "unknown"),
        "mode": str(run.get("mode") or "unknown"),
        "configuration": str(run.get("configuration") or ""),
        "run_name": str(run.get("run_name") or ""),
        "started_at": str(run.get("started_at") or ""),
        "started_at_epoch_ms": started_epoch_ms,
        "finished_at": timestamp,
        "finished_at_epoch_ms": finished_epoch_ms,
        "duration_seconds": round(max(0.0, duration_seconds), 1),
        "token_usage": max(0, token_usage),
        "estimated_cost_usd": estimated_cost_usd,
        "model_cost_breakdown": model_cost_breakdown[:20],
        "tests": {
            "passed": _safe_int(tests_payload.get("passed"), 0),
            "failed": _safe_int(tests_payload.get("failed"), 0),
            "skipped": _safe_int(tests_payload.get("skipped"), 0),
            "error": _safe_int(tests_payload.get("error"), 0),
            "unknown": _safe_int(tests_payload.get("unknown"), 0),
        },
        "tests_summary": tests_summary,
        "result_events": _safe_int(run.get("result_events"), 0),
        "stop_reason": stop_reason,
        "commit_count": commit_count,
    }
    changed_paths = run.get("_changed_path_set")
    changed_path_count = len(changed_paths) if isinstance(changed_paths, set) else 0
    changed_samples = run.get("_changed_path_samples")
    changed_list: list[str] = []
    if isinstance(changed_samples, list):
        for item in changed_samples:
            path_value = str(item or "").strip()
            if path_value:
                changed_list.append(path_value)
    finalized["diff_summary"] = {
        "files_changed_total": _safe_int(run.get("_files_changed_total"), 0),
        "net_lines_changed_total": _safe_int(run.get("_net_lines_changed_total"), 0),
        "changed_paths_count": changed_path_count,
        "changed_paths": changed_list[:120],
    }
    start_context = run.get("_start_context")
    finalized["start_context"] = dict(start_context) if isinstance(start_context, dict) else {}
    finalized["overall_score"] = _run_overall_score(finalized)
    if estimated_cost_usd > 0:
        finalized["cost_efficiency_score"] = round(
            _safe_float(finalized.get("overall_score"), 0.0)
            / max(_RUN_COST_EFFICIENCY_DENOMINATOR, estimated_cost_usd),
            3,
        )
    else:
        finalized["cost_efficiency_score"] = 0.0
    finalized["budget_outlier"] = False
    return finalized


def _empty_run_comparison_payload(
    *,
    repo: Path | None,
    scope: str,
    limit: int,
    message: str,
) -> dict[str, object]:
    """Return an empty run-comparison payload in a consistent shape."""
    return {
        "available": False,
        "repo_path": str(repo) if repo is not None else "",
        "history_path": str(_history_jsonl_path(repo).resolve()) if repo is not None else "",
        "scope": scope,
        "limit": limit,
        "runs": [],
        "best_by": {
            "overall_run_id": "",
            "fastest_run_id": "",
            "lowest_token_run_id": "",
            "lowest_cost_run_id": "",
            "best_cost_efficiency_run_id": "",
            "strongest_tests_run_id": "",
        },
        "cost_outlier_threshold_usd": 0.0,
        "budget_outlier_count": 0,
        "message": message,
    }


def _pipeline_run_comparison(
    repo: Path,
    *,
    scope: str = "all",
    limit: int = 12,
) -> dict[str, object]:
    """Build recent run-comparison rows from ``HISTORY.jsonl`` events."""
    history_path = _history_jsonl_path(repo)
    if not history_path.is_file():
        return _empty_run_comparison_payload(
            repo=repo,
            scope=scope,
            limit=limit,
            message="Run history is not available yet for this repository.",
        )
    cache_key = _run_comparison_cache_key(history_path, scope=scope, limit=limit)
    cached_payload = _run_comparison_cache_get(cache_key)
    if cached_payload is not None:
        return cached_payload

    rows = _read_jsonl_dict_rows(history_path, warn_context="pipeline run history")
    open_runs: dict[str, list[dict[str, object]]] = {}
    finished_runs: list[dict[str, object]] = []
    result_event_by_scope = {
        "chain": "step_result",
        "pipeline": "phase_result",
    }

    for event in rows:
        scope_key = str(event.get("scope") or "").strip().lower()
        if scope_key not in {"chain", "pipeline"}:
            continue

        event_name = str(event.get("event") or "").strip().lower()
        timestamp = str(event.get("timestamp") or "").strip()
        event_id = str(event.get("id") or "").strip()
        summary = str(event.get("summary") or "").strip()
        context = _history_context_payload(event)

        if event_name == "run_started":
            _scope_open_run_stack(open_runs, scope_key).append(
                _new_run_aggregate(
                    scope=scope_key,
                    event_id=event_id,
                    timestamp=timestamp,
                    context=context,
                )
            )
            continue

        if event_name == result_event_by_scope[scope_key]:
            run = _resolve_scope_open_run(
                open_runs,
                scope=scope_key,
                context=context,
            )
            if run is not None:
                _record_run_result_event(run, context)
            continue

        if event_name == "run_finished":
            run = _pop_scope_open_run(
                open_runs,
                scope=scope_key,
                context=context,
            )
            if run is None:
                run = _new_run_aggregate(
                    scope=scope_key,
                    event_id=event_id,
                    timestamp=timestamp,
                    context=context,
                )
            finalized = _finalize_run_aggregate(
                run,
                event_id=event_id,
                timestamp=timestamp,
                summary=summary,
                context=context,
            )
            finished_runs.append(finalized)

    if scope in {"chain", "pipeline"}:
        finished_runs = [run for run in finished_runs if str(run.get("scope")) == scope]

    finished_runs.sort(
        key=lambda run: (
            _safe_int(run.get("finished_at_epoch_ms"), 0),
            _safe_int(run.get("started_at_epoch_ms"), 0),
        ),
        reverse=True,
    )
    runs = finished_runs[:limit]
    if not runs:
        empty_payload = _empty_run_comparison_payload(
            repo=repo,
            scope=scope,
            limit=limit,
            message="No completed runs were found in history yet.",
        )
        _run_comparison_cache_set(cache_key, empty_payload)
        return empty_payload

    meaningful_candidates = [run for run in runs if _run_has_meaningful_activity(run)]
    best_overall = (
        max(
            meaningful_candidates,
            key=lambda run: _safe_float(run.get("overall_score"), -999999.0),
        )
        if meaningful_candidates
        else None
    )

    duration_candidates = [
        run
        for run in meaningful_candidates
        if _safe_float(run.get("duration_seconds"), 0.0) > 0
    ]
    best_fastest = (
        min(duration_candidates, key=lambda run: _safe_float(run.get("duration_seconds"), 0.0))
        if duration_candidates
        else None
    )

    token_candidates = [
        run for run in meaningful_candidates if _safe_int(run.get("token_usage"), 0) > 0
    ]
    best_lowest_tokens = (
        min(token_candidates, key=lambda run: _safe_int(run.get("token_usage"), 0))
        if token_candidates
        else None
    )
    cost_candidates = [
        run for run in meaningful_candidates if _safe_float(run.get("estimated_cost_usd"), 0.0) > 0
    ]
    best_lowest_cost = (
        min(cost_candidates, key=lambda run: _safe_float(run.get("estimated_cost_usd"), 0.0))
        if cost_candidates
        else None
    )
    best_cost_efficiency = (
        max(
            cost_candidates,
            key=lambda run: (
                _safe_float(run.get("cost_efficiency_score"), -999999.0),
                _safe_float(run.get("overall_score"), -999999.0),
                -_safe_float(run.get("estimated_cost_usd"), 0.0),
            ),
        )
        if cost_candidates
        else None
    )
    cost_outlier_threshold = _run_cost_outlier_threshold(
        [_safe_float(run.get("estimated_cost_usd"), 0.0) for run in cost_candidates]
    )

    test_candidates = [run for run in meaningful_candidates if _run_has_executed_tests(run)]
    best_tests = (
        max(
            test_candidates,
            key=lambda run: (
                _run_test_score(run.get("tests") if isinstance(run.get("tests"), dict) else {}),
                -_safe_float(run.get("duration_seconds"), 0.0),
                -_safe_int(run.get("token_usage"), 0),
            ),
        )
        if test_candidates
        else None
    )

    for run in runs:
        badges: list[str] = []
        if best_overall is not None and run.get("run_id") == best_overall.get("run_id"):
            badges.append("best_overall")
        if best_fastest is not None and run.get("run_id") == best_fastest.get("run_id"):
            badges.append("fastest")
        if best_lowest_tokens is not None and run.get("run_id") == best_lowest_tokens.get("run_id"):
            badges.append("lowest_tokens")
        if best_lowest_cost is not None and run.get("run_id") == best_lowest_cost.get("run_id"):
            badges.append("lowest_cost")
        if (
            best_cost_efficiency is not None
            and run.get("run_id") == best_cost_efficiency.get("run_id")
        ):
            badges.append("best_cost_efficiency")
        if best_tests is not None and run.get("run_id") == best_tests.get("run_id"):
            badges.append("strongest_tests")
        run_cost = _safe_float(run.get("estimated_cost_usd"), 0.0)
        is_budget_outlier = bool(
            cost_outlier_threshold > 0
            and run_cost > cost_outlier_threshold
            and len(cost_candidates) >= 3
        )
        run["budget_outlier"] = is_budget_outlier
        if is_budget_outlier:
            badges.append("budget_outlier")
        run["badges"] = badges

    budget_outlier_count = sum(1 for run in runs if bool(run.get("budget_outlier")))
    payload = {
        "available": True,
        "repo_path": str(repo),
        "history_path": str(history_path.resolve()),
        "scope": scope,
        "limit": limit,
        "runs": runs,
        "best_by": {
            "overall_run_id": str(best_overall.get("run_id") or "") if best_overall else "",
            "fastest_run_id": str(best_fastest.get("run_id") or "") if best_fastest else "",
            "lowest_token_run_id": (
                str(best_lowest_tokens.get("run_id") or "") if best_lowest_tokens else ""
            ),
            "lowest_cost_run_id": (
                str(best_lowest_cost.get("run_id") or "") if best_lowest_cost else ""
            ),
            "best_cost_efficiency_run_id": (
                str(best_cost_efficiency.get("run_id") or "") if best_cost_efficiency else ""
            ),
            "strongest_tests_run_id": str(best_tests.get("run_id") or "") if best_tests else "",
        },
        "cost_outlier_threshold_usd": cost_outlier_threshold,
        "budget_outlier_count": budget_outlier_count,
        "message": "",
    }
    _run_comparison_cache_set(cache_key, payload)
    return payload


def _default_test_policy_for_phase_key(phase_key: str) -> str:
    """Return default test policy for a pipeline phase key."""
    key = str(phase_key or "").strip()
    if not key:
        return "skip"
    try:
        from codex_manager.pipeline.phases import PipelinePhase, default_test_policy_for_phase

        return str(default_test_policy_for_phase(PipelinePhase(key)))
    except Exception:
        return "skip"


def _phase_row_from_key(phase_key: str, *, default_agent: str) -> dict[str, object] | None:
    """Build a normalized GUI phase row from a phase key."""
    key = str(phase_key or "").strip()
    if not key:
        return None
    try:
        from codex_manager.pipeline.phases import PipelinePhase

        phase_value = PipelinePhase(key).value
    except Exception:
        return None
    agent = str(default_agent or "codex").strip() or "codex"
    return {
        "phase": phase_value,
        "enabled": True,
        "iterations": 1,
        "agent": agent,
        "on_failure": "skip",
        "max_retries": 1,
        "test_policy": _default_test_policy_for_phase_key(phase_value),
        "custom_prompt": "",
    }


def _normalize_pipeline_phase_rows(
    raw_phases: object,
    *,
    default_agent: str,
) -> list[dict[str, object]]:
    """Normalize raw phase rows into ``PipelineGUIConfig``-compatible dicts."""
    items = raw_phases if isinstance(raw_phases, list) else []
    normalized: list[dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        phase_key = str(item.get("phase") or "").strip()
        if not phase_key:
            continue
        row = _phase_row_from_key(phase_key, default_agent=default_agent)
        if row is None:
            continue
        row["enabled"] = _safe_bool(item.get("enabled"), True)
        row["iterations"] = max(1, _safe_int(item.get("iterations"), 1))
        row["agent"] = str(item.get("agent") or default_agent or "codex").strip() or "codex"
        on_failure = str(item.get("on_failure") or "skip").strip().lower()
        row["on_failure"] = on_failure if on_failure in {"skip", "retry", "abort"} else "skip"
        row["max_retries"] = min(10, max(0, _safe_int(item.get("max_retries"), 1)))
        test_policy = str(item.get("test_policy") or "").strip().lower()
        if test_policy not in {"skip", "smoke", "full"}:
            test_policy = _default_test_policy_for_phase_key(str(row.get("phase") or ""))
        row["test_policy"] = test_policy
        row["custom_prompt"] = str(item.get("custom_prompt") or "")
        normalized.append(row)
    return normalized


def _promoted_pipeline_config_from_history(
    repo: Path,
    *,
    start_context: dict[str, object],
) -> dict[str, object]:
    """Build a validated apply-mode config from a dry-run history entry."""
    repo_path = str(repo.resolve())
    defaults = PipelineGUIConfig().model_dump()

    snapshot = start_context.get("config_snapshot")
    if isinstance(snapshot, dict):
        candidate = dict(snapshot)
        candidate["repo_path"] = repo_path
        candidate["mode"] = "apply"
        default_agent = str(candidate.get("agent") or "codex").strip() or "codex"
        candidate["phases"] = _normalize_pipeline_phase_rows(
            candidate.get("phases"),
            default_agent=default_agent,
        )
        if not candidate["phases"]:
            phase_order = candidate.get("phase_order")
            if isinstance(phase_order, list):
                rows: list[dict[str, object]] = []
                for phase_key in phase_order:
                    row = _phase_row_from_key(str(phase_key), default_agent=default_agent)
                    if row is not None:
                        rows.append(row)
                if rows:
                    candidate["phases"] = rows
        try:
            validated = PipelineGUIConfig(**candidate)
            return validated.model_dump()
        except Exception:
            logger.exception("Could not validate config snapshot from dry-run history.")

    promoted = dict(defaults)
    promoted.update(
        {
            "repo_path": repo_path,
            "mode": "apply",
            "max_cycles": max(1, _safe_int(start_context.get("max_cycles"), defaults["max_cycles"])),
            "unlimited": _safe_bool(start_context.get("unlimited"), False),
            "science_enabled": _safe_bool(
                start_context.get("science_enabled"),
                bool(defaults.get("science_enabled")),
            ),
            "brain_enabled": _safe_bool(
                start_context.get("brain_enabled"),
                bool(defaults.get("brain_enabled")),
            ),
            "vector_memory_enabled": _safe_bool(
                start_context.get("vector_memory_enabled"),
                bool(defaults.get("vector_memory_enabled")),
            ),
            "vector_memory_backend": str(
                start_context.get("vector_memory_backend")
                or defaults.get("vector_memory_backend")
                or "chroma"
            ),
            "deep_research_enabled": _safe_bool(
                start_context.get("deep_research_enabled"),
                bool(defaults.get("deep_research_enabled")),
            ),
            "deep_research_providers": str(
                start_context.get("deep_research_providers")
                or defaults.get("deep_research_providers")
                or "both"
            ),
            "deep_research_max_age_hours": max(
                1,
                _safe_int(
                    start_context.get("deep_research_max_age_hours"),
                    int(defaults.get("deep_research_max_age_hours", 168)),
                ),
            ),
            "deep_research_dedupe": _safe_bool(
                start_context.get("deep_research_dedupe"),
                bool(defaults.get("deep_research_dedupe")),
            ),
            "deep_research_native_enabled": _safe_bool(
                start_context.get("deep_research_native_enabled"),
                bool(defaults.get("deep_research_native_enabled")),
            ),
            "deep_research_retry_attempts": max(
                1,
                _safe_int(
                    start_context.get("deep_research_retry_attempts"),
                    int(defaults.get("deep_research_retry_attempts", 2)),
                ),
            ),
            "deep_research_daily_quota": max(
                1,
                _safe_int(
                    start_context.get("deep_research_daily_quota"),
                    int(defaults.get("deep_research_daily_quota", 8)),
                ),
            ),
            "deep_research_max_provider_tokens": max(
                512,
                _safe_int(
                    start_context.get("deep_research_max_provider_tokens"),
                    int(defaults.get("deep_research_max_provider_tokens", 12000)),
                ),
            ),
            "deep_research_budget_usd": max(
                0.0,
                _safe_float(
                    start_context.get("deep_research_budget_usd"),
                    float(defaults.get("deep_research_budget_usd", 5.0)),
                ),
            ),
            "deep_research_openai_model": str(
                start_context.get("deep_research_openai_model")
                or defaults.get("deep_research_openai_model")
                or "gpt-5.2"
            ),
            "deep_research_google_model": str(
                start_context.get("deep_research_google_model")
                or defaults.get("deep_research_google_model")
                or "gemini-3-pro-preview"
            ),
            "self_improvement_enabled": _safe_bool(
                start_context.get("self_improvement_enabled"),
                bool(defaults.get("self_improvement_enabled")),
            ),
            "self_improvement_auto_restart": _safe_bool(
                start_context.get("self_improvement_auto_restart"),
                bool(defaults.get("self_improvement_auto_restart")),
            ),
            "pr_aware_enabled": _safe_bool(
                start_context.get("pr_aware_enabled"),
                bool(defaults.get("pr_aware_enabled")),
            ),
            "pr_feature_branch": str(start_context.get("pr_feature_branch") or ""),
            "pr_remote": str(start_context.get("pr_remote") or ""),
            "pr_base_branch": str(start_context.get("pr_base_branch") or ""),
            "pr_auto_push": _safe_bool(
                start_context.get("pr_auto_push"),
                bool(defaults.get("pr_auto_push", True)),
            ),
            "pr_sync_description": _safe_bool(
                start_context.get("pr_sync_description"),
                bool(defaults.get("pr_sync_description", True)),
            ),
        }
    )
    if not promoted["self_improvement_enabled"]:
        promoted["self_improvement_auto_restart"] = False
    if not promoted["pr_aware_enabled"]:
        promoted["pr_auto_push"] = False
        promoted["pr_sync_description"] = False

    default_agent = str(promoted.get("agent") or "codex").strip() or "codex"
    phase_order = start_context.get("phase_order")
    phase_rows: list[dict[str, object]] = []
    if isinstance(phase_order, list):
        for phase_key in phase_order:
            row = _phase_row_from_key(str(phase_key), default_agent=default_agent)
            if row is not None:
                phase_rows.append(row)
    promoted["phases"] = phase_rows

    try:
        return PipelineGUIConfig(**promoted).model_dump()
    except Exception:
        logger.exception("Could not validate reconstructed dry-run promote config.")
        return defaults


def _pipeline_promote_last_dry_run_payload(repo: Path) -> dict[str, object]:
    """Return preview payload for promoting the most recent dry-run to apply mode."""
    comparison = _pipeline_run_comparison(repo, scope="pipeline", limit=50)
    base = {
        "available": False,
        "repo_path": str(repo.resolve()),
        "history_path": str(_history_jsonl_path(repo).resolve()),
        "run": None,
        "promoted_config": None,
        "message": "",
    }
    runs_obj = comparison.get("runs")
    runs = runs_obj if isinstance(runs_obj, list) else []
    dry_run = next(
        (item for item in runs if str(item.get("mode") or "").strip().lower() == "dry-run"),
        None,
    )
    if dry_run is None:
        base["message"] = "No completed dry-run pipeline runs were found in history yet."
        return base

    start_context_obj = dry_run.get("start_context")
    start_context = start_context_obj if isinstance(start_context_obj, dict) else {}
    promoted = _promoted_pipeline_config_from_history(repo, start_context=start_context)
    diff_obj = dry_run.get("diff_summary")
    diff_summary = dict(diff_obj) if isinstance(diff_obj, dict) else {}
    tests_obj = dry_run.get("tests")
    tests = dict(tests_obj) if isinstance(tests_obj, dict) else {}

    base["available"] = True
    base["run"] = {
        "run_id": str(dry_run.get("run_id") or ""),
        "finished_at": str(dry_run.get("finished_at") or ""),
        "duration_seconds": _safe_float(dry_run.get("duration_seconds"), 0.0),
        "token_usage": _safe_int(dry_run.get("token_usage"), 0),
        "configuration": str(dry_run.get("configuration") or ""),
        "stop_reason": str(dry_run.get("stop_reason") or ""),
        "tests_summary": str(dry_run.get("tests_summary") or ""),
        "tests": {
            "passed": _safe_int(tests.get("passed"), 0),
            "failed": _safe_int(tests.get("failed"), 0),
            "skipped": _safe_int(tests.get("skipped"), 0),
            "error": _safe_int(tests.get("error"), 0),
            "unknown": _safe_int(tests.get("unknown"), 0),
        },
        "diff_summary": {
            "files_changed_total": _safe_int(diff_summary.get("files_changed_total"), 0),
            "net_lines_changed_total": _safe_int(diff_summary.get("net_lines_changed_total"), 0),
            "changed_paths_count": _safe_int(diff_summary.get("changed_paths_count"), 0),
            "changed_paths": (
                diff_summary.get("changed_paths")
                if isinstance(diff_summary.get("changed_paths"), list)
                else []
            ),
        },
        "commit_count": _safe_int(dry_run.get("commit_count"), 0),
    }
    base["promoted_config"] = promoted
    base["message"] = ""
    return base


def _extract_markdown_section_lines(
    markdown: str,
    *,
    heading_prefix: str,
    heading_level: int = 2,
) -> list[str]:
    """Extract lines in the section whose heading starts with *heading_prefix*."""
    if not markdown.strip():
        return []
    lines = markdown.splitlines()
    prefix = heading_prefix.strip().lower()
    in_section = False
    collected: list[str] = []
    boundary = re.compile(rf"^#{{1,{max(1, heading_level)}}}\s+")
    heading_start = "#" * max(1, heading_level) + " "

    for line in lines:
        stripped = line.strip()
        if not in_section:
            if stripped.startswith(heading_start):
                heading_text = stripped[len(heading_start) :].strip().lower()
                if heading_text.startswith(prefix):
                    in_section = True
            continue

        if boundary.match(stripped):
            break
        collected.append(line)
    return collected


def _parse_markdown_table(lines: list[str]) -> list[dict[str, str]]:
    """Parse the first markdown table found in *lines*."""
    if not lines:
        return []

    def _split_row(row: str) -> list[str]:
        return [cell.strip().replace(r"\|", "|") for cell in row.strip().strip("|").split("|")]

    for idx in range(0, len(lines) - 1):
        header_line = lines[idx].strip()
        sep_line = lines[idx + 1].strip()
        if not header_line.startswith("|") or not sep_line.startswith("|"):
            continue
        headers = _split_row(header_line)
        sep_cells = _split_row(sep_line)
        if not headers or len(sep_cells) < len(headers):
            continue
        if not all(re.match(r"^:?-{3,}:?$", cell) for cell in sep_cells[: len(headers)]):
            continue

        rows: list[dict[str, str]] = []
        for raw in lines[idx + 2 :]:
            row_line = raw.strip()
            if not row_line:
                break
            if not row_line.startswith("|"):
                break
            cells = _split_row(row_line)
            if len(cells) < len(headers):
                cells.extend([""] * (len(headers) - len(cells)))
            row = {headers[col].strip(): cells[col].strip() for col in range(len(headers))}
            rows.append(row)
        return rows
    return []


def _extract_code_fence_text(lines: list[str]) -> str:
    """Return text between the first fenced-code block in *lines*."""
    if not lines:
        return ""
    in_fence = False
    captured: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_fence:
                break
            in_fence = True
            continue
        if in_fence:
            captured.append(line.rstrip())
    if captured:
        return "\n".join(captured).strip()
    return "\n".join(line.rstrip() for line in lines).strip()


def _extract_first_code_fence(text: str) -> str:
    """Return content of the first fenced code block, or the raw text when none exists."""
    raw = str(text or "")
    match = re.search(r"```(?:[\w.+-]+)?\s*(.*?)```", raw, flags=re.DOTALL)
    if not match:
        return raw.strip()
    return str(match.group(1) or "").strip()


_RISKY_MARKETING_PHRASES = (
    "guaranteed",
    "risk-free",
    "instant results",
    "zero risk",
    "no downside",
    "always works",
)
_DARK_PATTERN_PHRASES = (
    "dark pattern",
    "trick users",
    "force users",
    "addictive loop",
    "dopamine trap",
    "manipulate users",
)
_LEGAL_CERTAINTY_PHRASES = (
    "fully compliant",
    "legal guaranteed",
    "zero legal risk",
    "regulation-proof",
)
_SOURCE_URL_RE = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)
_DEFAULT_BLOCKED_SOURCE_DOMAINS = frozenset(
    {
        "example.com",
        "localhost",
        "127.0.0.1",
        "facebook.com",
        "instagram.com",
        "tiktok.com",
        "x.com",
        "twitter.com",
        "pinterest.com",
    }
)


def _parse_domain_policy_env(name: str) -> set[str]:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return set()
    values = {chunk.strip().lower().lstrip(".") for chunk in raw.split(",") if chunk.strip()}
    return {item for item in values if item}


def _extract_links(text: str) -> list[str]:
    links = {match.group(0).rstrip(".,;:") for match in _SOURCE_URL_RE.finditer(text or "")}
    return sorted(link for link in links if link)


def _domain_matches(host: str, domain: str) -> bool:
    host_key = str(host or "").strip().lower()
    domain_key = str(domain or "").strip().lower().lstrip(".")
    if not host_key or not domain_key:
        return False
    return host_key == domain_key or host_key.endswith(f".{domain_key}")


def _audit_research_sources(links: list[str]) -> list[str]:
    if not links:
        return ["Research/citation links missing. Add credible source URLs before approving."]

    warnings: list[str] = []
    allowed_domains = _parse_domain_policy_env("CODEX_MANAGER_RESEARCH_ALLOWED_DOMAINS")
    blocked_domains = set(_DEFAULT_BLOCKED_SOURCE_DOMAINS)
    blocked_domains.update(_parse_domain_policy_env("CODEX_MANAGER_RESEARCH_BLOCKED_DOMAINS"))
    insecure_links = 0
    hostnames: set[str] = set()
    blocked_hits: set[str] = set()
    allowlist_violations: set[str] = set()

    for link in links:
        parsed = urlparse(link)
        if parsed.scheme.lower() != "https":
            insecure_links += 1
        host = str(parsed.hostname or "").strip().lower()
        if not host:
            continue
        hostnames.add(host)
        if any(_domain_matches(host, blocked) for blocked in blocked_domains):
            blocked_hits.add(host)
        if allowed_domains and not any(
            _domain_matches(host, allowed) for allowed in allowed_domains
        ):
            allowlist_violations.add(host)

    if insecure_links:
        warnings.append(
            "Some citations are not HTTPS. Prefer HTTPS sources for integrity and traceability."
        )
    if blocked_hits:
        warnings.append(
            "Potentially low-trust source domains detected: "
            + ", ".join(sorted(blocked_hits)[:8])
        )
    if allowlist_violations:
        warnings.append(
            "Sources outside configured governance allowlist policy: "
            + ", ".join(sorted(allowlist_violations)[:8])
        )
    if len(hostnames) <= 1 and len(links) >= 2:
        warnings.append(
            "Source diversity is low (single domain). Add corroborating references before approval."
        )
    return warnings


def _owner_decision_board_path(repo: Path) -> Path:
    return repo / ".codex_manager" / "owner" / "decision_board.json"


def _todo_wishlist_path(repo: Path) -> Path:
    return repo / ".codex_manager" / "owner" / "TODO_WISHLIST.md"


def _feature_dreams_path(repo: Path) -> Path:
    return repo / ".codex_manager" / "owner" / "FEATURE_DREAMS.md"


def _general_request_history_path(repo: Path) -> Path:
    return repo / ".codex_manager" / "owner" / "GENERAL_REQUEST_HISTORY.jsonl"


def _pipeline_resume_checkpoint_path(repo: Path) -> Path:
    return repo / ".codex_manager" / "state" / "pipeline_resume.json"


def _pipeline_resume_summary(repo: Path) -> dict[str, object]:
    checkpoint = _pipeline_resume_checkpoint_path(repo)
    payload: dict[str, object] = {
        "exists": checkpoint.is_file(),
        "checkpoint_path": str(checkpoint.resolve()),
        "resume_ready": False,
        "resume_cycle": 0,
        "resume_phase_index": 0,
        "repo_path": str(repo.resolve()),
        "saved_at_epoch_ms": 0,
    }
    if not checkpoint.is_file():
        return payload

    try:
        payload["saved_at_epoch_ms"] = int(checkpoint.stat().st_mtime * 1000)
    except OSError:
        payload["saved_at_epoch_ms"] = 0

    try:
        raw = json.loads(_read_text_utf8_resilient(checkpoint))
    except Exception as exc:
        payload["error"] = f"Could not parse checkpoint: {exc}"
        return payload

    if not isinstance(raw, dict):
        payload["error"] = "Checkpoint payload must be a JSON object."
        return payload

    checkpoint_repo_path = str(raw.get("repo_path") or "").strip()
    if checkpoint_repo_path:
        payload["checkpoint_repo_path"] = checkpoint_repo_path
    try:
        resume_cycle = int(raw.get("resume_cycle") or 1)
        resume_phase_index = int(raw.get("resume_phase_index") or 0)
    except (TypeError, ValueError):
        payload["error"] = "Checkpoint resume fields are invalid."
        return payload

    payload["resume_cycle"] = max(1, resume_cycle)
    payload["resume_phase_index"] = max(0, resume_phase_index)

    config_payload = raw.get("config")
    if not isinstance(config_payload, dict):
        payload["error"] = "Checkpoint missing pipeline config."
        return payload

    if checkpoint_repo_path:
        try:
            checkpoint_repo_resolved = str(Path(checkpoint_repo_path).expanduser().resolve())
        except Exception:
            checkpoint_repo_resolved = checkpoint_repo_path
        payload["checkpoint_repo_path"] = checkpoint_repo_resolved
        if checkpoint_repo_resolved != str(repo.resolve()):
            payload["error"] = (
                "Checkpoint repo does not match selected repository "
                f"({checkpoint_repo_resolved})."
            )
            return payload

    payload["resume_ready"] = True
    return payload


_CODEX_MANAGER_GITIGNORE_BEGIN = "# --- WarpFoundry managed rules ---"
_CODEX_MANAGER_GITIGNORE_END = "# --- End WarpFoundry managed rules ---"
_CODEX_MANAGER_GITIGNORE_RULES = [
    "# Keep WarpFoundry owner/planning docs versioned while ignoring runtime artifacts.",
    "!.codex_manager/",
    ".codex_manager/logs/",
    ".codex_manager/outputs/",
    ".codex_manager/output_history/",
    ".codex_manager/state/",
    ".codex_manager/memory/",
    ".codex_manager/ledger/",
]


def _codex_manager_gitignore_block() -> str:
    return "\n".join(
        [
            _CODEX_MANAGER_GITIGNORE_BEGIN,
            *_CODEX_MANAGER_GITIGNORE_RULES,
            _CODEX_MANAGER_GITIGNORE_END,
        ]
    )


def _ensure_codex_manager_gitignore_rules(repo: Path) -> None:
    """Ensure repository-level gitignore rules preserve owner docs under .codex_manager."""
    gitignore_path = repo / ".gitignore"
    block = _codex_manager_gitignore_block()
    existing = _read_text_utf8_resilient(gitignore_path) if gitignore_path.exists() else ""

    pattern = re.compile(
        rf"{re.escape(_CODEX_MANAGER_GITIGNORE_BEGIN)}.*?{re.escape(_CODEX_MANAGER_GITIGNORE_END)}\n?",
        flags=re.DOTALL,
    )
    if pattern.search(existing):
        updated = pattern.sub(block + "\n", existing, count=1)
    else:
        base = existing.rstrip("\n")
        updated = f"{base}\n\n{block}\n" if base else f"{block}\n"

    if updated != existing:
        gitignore_path.write_text(updated, encoding="utf-8")


def _default_todo_wishlist_markdown(project_name: str) -> str:
    name = str(project_name or "Project").strip() or "Project"
    return (
        f"# To-Do and Wishlist - {name}\n\n"
        "Use this list to track features, fixes, and implementation ideas.\n\n"
        "## High Priority\n\n"
        "- [ ] Define the most important user-facing improvement.\n"
        "- [ ] Add one high-impact reliability or quality upgrade.\n\n"
        "## Medium Priority\n\n"
        "- [ ] Improve onboarding/readme clarity.\n"
        "- [ ] Add one automation or developer-experience improvement.\n\n"
        "## Backlog Ideas\n\n"
        "- [ ] Add one optional enhancement to revisit later.\n\n"
        "## Notes\n\n"
        "- Mark completed items as `- [x] ...`.\n"
        "- Keep items concrete and implementation-ready.\n"
    )


def _default_feature_dreams_markdown(project_name: str) -> str:
    name = str(project_name or "Project").strip() or "Project"
    return (
        f"# {name} Feature Dreams\n\n"
        "Execution order: top to bottom. Keep this file feature-only.\n\n"
        "## P0 - Highest Value / Lowest Effort\n\n"
        "- [ ] [S] Add one high-impact feature that improves core user value.\n"
        "- [ ] [M] Add one workflow automation feature that removes repetitive steps.\n\n"
        "## P1 - Product Leverage\n\n"
        "- [ ] [M] Add one feature that improves onboarding or discoverability.\n"
        "- [ ] [M] Add one feature that improves reliability or observability for users.\n\n"
        "## P2 - Advanced Features\n\n"
        "- [ ] [L] Add one differentiated long-term feature bet to revisit later.\n\n"
        "## Notes\n\n"
        "- Mark completed items as `- [x] ...`.\n"
        "- Keep items concrete and implementation-ready.\n"
        "- Use effort tags `[S]`, `[M]`, `[L]` for prioritization.\n"
    )


def _read_todo_wishlist(repo: Path) -> str:
    path = _todo_wishlist_path(repo)
    if not path.is_file():
        return _default_todo_wishlist_markdown(repo.name)
    return _read_text_utf8_resilient(path)


def _read_feature_dreams(repo: Path) -> str:
    path = _feature_dreams_path(repo)
    if not path.is_file():
        return _default_feature_dreams_markdown(repo.name)
    return _read_text_utf8_resilient(path)


def _write_todo_wishlist(repo: Path, content: str) -> Path:
    _ensure_codex_manager_gitignore_rules(repo)
    path = _todo_wishlist_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = str(content or "").strip() or _default_todo_wishlist_markdown(repo.name)
    path.write_text(clean + "\n", encoding="utf-8")
    return path


def _write_feature_dreams(repo: Path, content: str) -> Path:
    _ensure_codex_manager_gitignore_rules(repo)
    path = _feature_dreams_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = str(content or "").strip() or _default_feature_dreams_markdown(repo.name)
    path.write_text(clean + "\n", encoding="utf-8")
    return path


def _todo_wishlist_has_open_items(text: str) -> bool:
    return bool(re.search(r"^\s*[-*]\s+\[\s\]\s+", str(text or ""), flags=re.MULTILINE))


def _feature_dreams_has_open_items(text: str) -> bool:
    return bool(re.search(r"^\s*[-*]\s+\[\s\]\s+", str(text or ""), flags=re.MULTILINE))


def _normalize_owner_context_files(
    raw_files: object,
    *,
    max_files: int = _OWNER_CONTEXT_MAX_FILES,
    max_chars_per_file: int = _OWNER_CONTEXT_MAX_FILE_CHARS,
    max_total_chars: int = _OWNER_CONTEXT_MAX_TOTAL_CHARS,
) -> list[dict[str, str]]:
    files = raw_files if isinstance(raw_files, list) else []
    normalized: list[dict[str, str]] = []
    total_chars = 0
    for idx, row in enumerate(files, start=1):
        if len(normalized) >= max_files:
            break
        if not isinstance(row, dict):
            continue
        name_raw = str(row.get("name") or "").strip()
        name_clean = re.sub(r"\s+", " ", name_raw).strip()
        if not name_clean:
            name_clean = f"context-{idx:02d}.txt"
        if len(name_clean) > 180:
            name_clean = name_clean[:180]

        content = str(row.get("content") or "").replace("\r\n", "\n").strip()
        if not content:
            continue
        if len(content) > max_chars_per_file:
            content = content[:max_chars_per_file]

        remaining = max_total_chars - total_chars
        if remaining <= 0:
            break
        if len(content) > remaining:
            content = content[:remaining]
        total_chars += len(content)
        normalized.append(
            {
                "name": name_clean,
                "content": content,
            }
        )
    return normalized


def _owner_context_files_prompt_section(context_files: list[dict[str, str]]) -> str:
    if not context_files:
        return "Uploaded context files: (none)\n\n"
    lines = ["Uploaded context files:\n"]
    for idx, row in enumerate(context_files, start=1):
        name = str(row.get("name") or "").strip() or f"context-{idx:02d}.txt"
        content = str(row.get("content") or "").strip()
        lines.append(f"[{idx}] {name}\n")
        lines.append("```text\n")
        lines.append(content[: _OWNER_CONTEXT_MAX_FILE_CHARS])
        if content and not content.endswith("\n"):
            lines.append("\n")
        lines.append("```\n")
    lines.append("\n")
    return "".join(lines)


def _owner_repo_ideas_skip_dir(rel_dir: Path) -> bool:
    parts = [part.strip().lower() for part in rel_dir.parts if str(part).strip() and part != "."]
    if not parts:
        return False
    if any(part in _OWNER_REPO_IDEAS_EXCLUDED_DIR_NAMES for part in parts):
        return True
    rel_posix = rel_dir.as_posix().replace("\\", "/").lower()
    while rel_posix.startswith("./"):
        rel_posix = rel_posix[2:]
    if not rel_posix:
        return False
    return any(
        rel_posix == blocked or rel_posix.startswith(f"{blocked}/")
        for blocked in _OWNER_REPO_IDEAS_EXCLUDED_SUBPATHS
    )


def _owner_repo_ideas_skip_file(rel_file: Path) -> bool:
    parts = [part.strip().lower() for part in rel_file.parts if str(part).strip() and part != "."]
    if not parts:
        return True
    if any(part in _OWNER_REPO_IDEAS_EXCLUDED_DIR_NAMES for part in parts[:-1]):
        return True
    rel_posix = rel_file.as_posix().replace("\\", "/").lower()
    while rel_posix.startswith("./"):
        rel_posix = rel_posix[2:]
    if any(
        rel_posix == blocked or rel_posix.startswith(f"{blocked}/")
        for blocked in _OWNER_REPO_IDEAS_EXCLUDED_SUBPATHS
    ):
        return True

    filename = parts[-1]
    if filename in _OWNER_REPO_IDEAS_EXCLUDED_FILENAMES:
        return True
    suffix = Path(filename).suffix.lower()
    if suffix in _OWNER_REPO_IDEAS_EXCLUDED_SUFFIXES:
        return True
    if filename.startswith("error") and suffix in {".log", ".txt"}:
        return True
    return False


def _looks_binary_blob(sample: bytes) -> bool:
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    control_chars = 0
    for byte in sample:
        if byte in {9, 10, 13}:  # tab/newline/carriage return
            continue
        if 32 <= byte <= 126:
            continue
        if 128 <= byte <= 255:
            continue
        control_chars += 1
    return (control_chars / max(len(sample), 1)) > 0.18


def _collect_owner_repo_ideas_context(
    repo: Path,
    *,
    max_files: int = _OWNER_REPO_IDEAS_MAX_FILES,
    max_manifest_files: int = _OWNER_REPO_IDEAS_MAX_MANIFEST_FILES,
    max_snippet_files: int = _OWNER_REPO_IDEAS_MAX_SNIPPET_FILES,
    max_chars_per_file: int = _OWNER_REPO_IDEAS_MAX_FILE_CHARS,
    max_total_chars: int = _OWNER_REPO_IDEAS_MAX_TOTAL_CHARS,
    max_file_bytes: int = _OWNER_REPO_IDEAS_MAX_FILE_BYTES,
) -> tuple[str, dict[str, object]]:
    manifest: list[str] = []
    snippets: list[dict[str, str]] = []
    extension_counts: Counter[str] = Counter()
    total_files = 0
    excluded_files = 0
    skipped_binary = 0
    skipped_large = 0
    scanned_bytes = 0
    total_snippet_chars = 0
    max_files_hit = False

    for root, dirs, filenames in os.walk(repo):
        root_path = Path(root)
        rel_root = root_path.relative_to(repo)
        if str(rel_root) != "." and _owner_repo_ideas_skip_dir(rel_root):
            dirs[:] = []
            continue

        kept_dirs: list[str] = []
        for dirname in dirs:
            rel_dir = rel_root / dirname if str(rel_root) != "." else Path(dirname)
            if _owner_repo_ideas_skip_dir(rel_dir):
                continue
            kept_dirs.append(dirname)
        dirs[:] = kept_dirs

        for filename in sorted(filenames):
            if total_files >= max_files:
                max_files_hit = True
                break

            rel_file = rel_root / filename if str(rel_root) != "." else Path(filename)
            if _owner_repo_ideas_skip_file(rel_file):
                excluded_files += 1
                continue

            file_path = root_path / filename
            if not file_path.is_file():
                continue

            total_files += 1
            rel_text = rel_file.as_posix()
            if len(manifest) < max_manifest_files:
                manifest.append(rel_text)

            ext = rel_file.suffix.lower().lstrip(".") or "(no-ext)"
            extension_counts[ext] += 1

            try:
                scanned_bytes += int(file_path.stat().st_size)
            except OSError:
                pass

            if len(snippets) >= max_snippet_files or total_snippet_chars >= max_total_chars:
                continue

            try:
                size_bytes = int(file_path.stat().st_size)
            except OSError:
                continue
            if size_bytes > max_file_bytes:
                skipped_large += 1
                continue

            try:
                with file_path.open("rb") as bf:
                    sample = bf.read(4096)
            except OSError:
                continue
            if _looks_binary_blob(sample):
                skipped_binary += 1
                continue

            try:
                raw = _read_text_utf8_resilient(file_path)
            except Exception:
                skipped_binary += 1
                continue
            content = raw.replace("\r\n", "\n").strip()
            if not content:
                continue
            if len(content) > max_chars_per_file:
                content = content[:max_chars_per_file].rstrip() + "\n... [truncated]"

            remaining = max_total_chars - total_snippet_chars
            if remaining <= 0:
                continue
            if len(content) > remaining:
                content = content[:remaining].rstrip()
                if content:
                    content += "\n... [truncated]"
            if not content:
                continue

            total_snippet_chars += len(content)
            snippets.append({"path": rel_text, "content": content})

        if max_files_hit:
            break

    ext_summary = ", ".join(
        f"{ext}: {count}" for ext, count in extension_counts.most_common(10)
    )
    context_lines: list[str] = [
        "Repository scan summary:",
        f"- Included files: {total_files}",
        f"- Excluded files/artifacts: {excluded_files}",
        f"- Binary/unreadable files skipped for snippets: {skipped_binary}",
        f"- Large files skipped for snippets: {skipped_large}",
        f"- Snippet files included: {len(snippets)}",
        f"- Total snippet characters: {total_snippet_chars}",
        f"- Approximate scanned bytes: {scanned_bytes}",
    ]
    if max_files_hit:
        context_lines.append(f"- Scan truncated at {max_files} files for performance.")
    if ext_summary:
        context_lines.append(f"- Top extensions: {ext_summary}")

    context_lines.append("")
    context_lines.append("Repository file inventory (relative paths):")
    if manifest:
        context_lines.extend(f"- {rel}" for rel in manifest)
    else:
        context_lines.append("- (No eligible files found after exclusions.)")
    if total_files > len(manifest):
        context_lines.append(
            f"- ... ({total_files - len(manifest)} additional files omitted from listing)"
        )

    context_lines.append("")
    context_lines.append("Repository file snippets:")
    if snippets:
        for row in snippets:
            context_lines.append(f"[{row['path']}]")
            context_lines.append("```text")
            context_lines.append(row["content"])
            if row["content"] and not row["content"].endswith("\n"):
                context_lines.append("")
            context_lines.append("```")
    else:
        context_lines.append("(No text snippets captured.)")

    scan: dict[str, object] = {
        "included_files": total_files,
        "excluded_files": excluded_files,
        "skipped_binary": skipped_binary,
        "skipped_large": skipped_large,
        "snippet_files": len(snippets),
        "snippet_chars": total_snippet_chars,
        "scan_truncated": max_files_hit,
    }
    return "\n".join(context_lines).strip(), scan


def _suggest_repo_ideas_markdown(
    *,
    repo: Path,
    model: str,
    owner_context: str,
    existing_markdown: str,
) -> tuple[str, str, dict[str, object]]:
    repo_context, scan = _collect_owner_repo_ideas_context(repo)
    prompt = (
        "You are helping the repository owner generate high-value ideas grounded in the real codebase.\n"
        "Free-tier mode: ideation only. Do not claim code changes were implemented.\n"
        "Return markdown only.\n\n"
        "Required output:\n"
        "1. Title: `# <repo name> Repository Ideas`\n"
        "2. Sections exactly:\n"
        "   - `## P0 - Highest Value / Lowest Effort`\n"
        "   - `## P1 - Product Leverage`\n"
        "   - `## P2 - Strategic Bets`\n"
        "3. Use checklist items only: `- [ ] [S|M|L] <idea> - <single-sentence user value>`\n"
        "4. Produce 10-20 concrete, repository-specific ideas and avoid duplicates.\n"
        "5. Include a final section: `## Optional Future Commercial Add-Ons (Not Implemented)`\n"
        "   with 2-5 optional ideas about admin controls, billing, monetization, or enterprise features.\n"
        "6. Keep execution order top-to-bottom by impact and practicality.\n\n"
        f"Repository: {repo.name}\n"
        f"Owner context: {owner_context or '(none)'}\n\n"
        "Existing ideas (dedupe context):\n"
        f"{existing_markdown[:5000]}\n\n"
        "Repository context (all files scanned except logs/runtime artifacts):\n"
        f"{repo_context}"
    )
    try:
        from codex_manager.brain.connector import connect
    except Exception as exc:
        logger.warning("Could not import AI connector for repo ideas suggestion: %s", exc)
        return (
            _default_feature_dreams_markdown(repo.name),
            "AI suggestion unavailable; used a deterministic starter template.",
            scan,
        )

    try:
        raw = connect(
            model=str(model or "gpt-5.2").strip() or "gpt-5.2",
            prompt=prompt,
            text_only=True,
            operation="owner_repo_ideas_generate",
            stage="owner:repo_ideas",
            max_output_tokens=2600,
            temperature=0.3,
        )
        suggested = _extract_first_code_fence(str(raw or "")).strip()
        if not suggested:
            suggested = str(raw or "").strip()
        if not suggested:
            raise RuntimeError("empty suggestion")
        return suggested, "", scan
    except Exception as exc:
        logger.warning("AI suggestion failed for repo ideas", exc_info=True)
        fallback = _default_feature_dreams_markdown(repo.name)
        return fallback, f"AI suggestion failed ({exc}); used a starter template.", scan


def _suggest_todo_wishlist_markdown(
    *,
    repo: Path,
    model: str,
    owner_context: str,
    existing_markdown: str,
    context_files: list[dict[str, str]] | None = None,
) -> tuple[str, str]:
    context_file_rows = context_files or []
    prompt = (
        "You are building a practical implementation backlog for a software repository.\n"
        "Return only markdown.\n\n"
        "Requirements:\n"
        "- Include sections: High Priority, Medium Priority, Backlog Ideas.\n"
        "- Use markdown checkboxes (`- [ ]`) for each item.\n"
        "- Provide 8-20 concrete, repository-relevant items.\n"
        "- Prioritize items that can be implemented incrementally.\n"
        "- Avoid duplicates against existing list items.\n"
        "- Keep each item specific and actionable.\n\n"
        f"Repository: {repo.name}\n"
        f"Owner context: {owner_context or '(none)'}\n\n"
        f"{_owner_context_files_prompt_section(context_file_rows)}"
        "Existing list (for dedupe context):\n"
        f"{existing_markdown[:4000]}"
    )
    try:
        from codex_manager.brain.connector import connect
    except Exception as exc:
        logger.warning("Could not import AI connector for wishlist suggestion: %s", exc)
        return (
            _default_todo_wishlist_markdown(repo.name),
            "AI suggestion unavailable; used a deterministic starter template.",
        )

    try:
        raw = connect(
            model=str(model or "gpt-5.2").strip() or "gpt-5.2",
            prompt=prompt,
            text_only=True,
            operation="todo_wishlist_suggest",
            stage="owner:todo_wishlist",
            max_output_tokens=1800,
            temperature=0.35,
        )
        suggested = _extract_first_code_fence(str(raw or "")).strip()
        if not suggested:
            suggested = str(raw or "").strip()
        if not suggested:
            raise RuntimeError("empty suggestion")
        return suggested, ""
    except Exception as exc:
        logger.warning("AI suggestion failed for todo/wishlist", exc_info=True)
        fallback = _default_todo_wishlist_markdown(repo.name)
        return fallback, f"AI suggestion failed ({exc}); used a starter template."


def _suggest_feature_dreams_markdown(
    *,
    repo: Path,
    model: str,
    owner_context: str,
    existing_markdown: str,
    context_files: list[dict[str, str]] | None = None,
) -> tuple[str, str]:
    context_file_rows = context_files or []
    prompt = (
        "You are helping an owner dream up high-value product features for a software repository.\n"
        "Return only markdown.\n\n"
        "Requirements:\n"
        "- Keep content feature-only (no bug-only chores unless tied to a user-visible feature).\n"
        "- Use sections: `P0 - Highest Value / Lowest Effort`, `P1 - Product Leverage`, `P2 - Advanced Features`.\n"
        "- Each item must be a markdown checkbox (`- [ ]`) and include effort tags `[S]`, `[M]`, or `[L]`.\n"
        "- Provide 8-20 concrete, implementation-ready feature items.\n"
        "- Prioritize incremental deliverables and avoid duplicates from the existing file.\n"
        "- Keep execution order top to bottom.\n\n"
        f"Repository: {repo.name}\n"
        f"Owner context: {owner_context or '(none)'}\n\n"
        f"{_owner_context_files_prompt_section(context_file_rows)}"
        "Existing feature dreams (for dedupe context):\n"
        f"{existing_markdown[:4000]}"
    )
    try:
        from codex_manager.brain.connector import connect
    except Exception as exc:
        logger.warning("Could not import AI connector for feature dreams suggestion: %s", exc)
        return (
            _default_feature_dreams_markdown(repo.name),
            "AI suggestion unavailable; used a deterministic starter template.",
        )

    try:
        raw = connect(
            model=str(model or "gpt-5.2").strip() or "gpt-5.2",
            prompt=prompt,
            text_only=True,
            operation="feature_dreams_suggest",
            stage="owner:feature_dreams",
            max_output_tokens=2200,
            temperature=0.35,
        )
        suggested = _extract_first_code_fence(str(raw or "")).strip()
        if not suggested:
            suggested = str(raw or "").strip()
        if not suggested:
            raise RuntimeError("empty suggestion")
        return suggested, ""
    except Exception as exc:
        logger.warning("AI suggestion failed for feature dreams", exc_info=True)
        fallback = _default_feature_dreams_markdown(repo.name)
        return fallback, f"AI suggestion failed ({exc}); used a starter template."


def _normalize_general_request_status(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"considered", "implemented", "refused"}:
        return raw
    if "refus" in raw or "cannot" in raw or "can't" in raw:
        return "refused"
    if "implement" in raw:
        return "implemented"
    return "considered"


def _parse_general_request_response(raw_output: str) -> tuple[str, str, str]:
    text = str(raw_output or "").strip()
    if not text:
        return "refused", "AI returned an empty response.", ""
    candidate = _extract_first_code_fence(text).strip() or text
    try:
        payload = json.loads(candidate)
    except Exception:
        status = "considered"
        notes = "AI returned an unstructured response."
        return status, notes, text
    if not isinstance(payload, dict):
        return "considered", "AI response JSON did not contain an object.", text
    status = _normalize_general_request_status(payload.get("status"))
    notes = str(payload.get("notes") or "").strip()
    response_text = str(payload.get("response") or "").strip()
    if not notes:
        notes = "No notes provided."
    if not response_text:
        response_text = text
    return status, notes, response_text


def _trim_general_request_history(path: Path, *, max_items: int = _GENERAL_REQUEST_HISTORY_MAX_ITEMS) -> None:
    if max_items <= 0 or not path.is_file():
        return
    try:
        rows = [line for line in _read_text_utf8_resilient(path).splitlines() if line.strip()]
    except Exception:
        return
    if len(rows) <= max_items:
        return
    trimmed = "\n".join(rows[-max_items:]) + "\n"
    path.write_text(trimmed, encoding="utf-8")


def _append_general_request_history(
    *,
    repo: Path,
    request_text: str,
    status: str,
    notes: str,
    output: str,
    source: str,
    model: str = "",
) -> dict[str, object]:
    _ensure_codex_manager_gitignore_rules(repo)
    now = _utc_now_iso_z()
    normalized_status = _normalize_general_request_status(status)
    entry: dict[str, object] = {
        "id": f"gr-{int(time.time() * 1000)}",
        "timestamp": now,
        "request": str(request_text or "").strip(),
        "status": normalized_status,
        "notes": str(notes or "").strip(),
        "output": str(output or ""),
        "source": str(source or "").strip() or "general_request",
        "model": str(model or "").strip(),
    }
    history_path = _general_request_history_path(repo)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _trim_general_request_history(history_path)
    return entry


def _read_general_request_history(repo: Path, *, limit: int = 25) -> list[dict[str, object]]:
    path = _general_request_history_path(repo)
    rows: list[dict[str, object]] = []
    for parsed in _read_jsonl_dict_rows(path):
        rows.append(
            {
                "id": str(parsed.get("id") or "").strip(),
                "timestamp": str(parsed.get("timestamp") or "").strip(),
                "request": str(parsed.get("request") or "").strip(),
                "status": _normalize_general_request_status(parsed.get("status")),
                "notes": str(parsed.get("notes") or "").strip(),
                "output": str(parsed.get("output") or ""),
                "source": str(parsed.get("source") or "").strip(),
                "model": str(parsed.get("model") or "").strip(),
            }
        )
    rows.sort(key=lambda row: str(row.get("timestamp") or ""), reverse=True)
    cap = max(1, min(200, int(limit)))
    return rows[:cap]


def _process_general_request(
    *,
    repo: Path,
    request_text: str,
    model: str,
    owner_context: str,
    context_files: list[dict[str, str]],
) -> tuple[str, str, str, str]:
    prompt = (
        "You are assisting the repository owner with a general project request.\n"
        "This mode can analyze and recommend, but does not directly edit repository files.\n"
        "Return STRICT JSON with keys: status, notes, response.\n\n"
        "Rules:\n"
        "- status must be one of: considered, refused.\n"
        "- Use `considered` when the request is valid and you can provide a useful response.\n"
        "- Use `refused` when the request is unsafe, unrelated, or missing critical details.\n"
        "- `notes` should be 1-3 concise sentences.\n"
        "- `response` should be useful markdown for the owner.\n"
        "- Never claim repository files were modified in this mode.\n\n"
        f"Repository: {repo.name}\n"
        f"Owner context: {owner_context or '(none)'}\n\n"
        f"{_owner_context_files_prompt_section(context_files)}"
        "Owner request:\n"
        f"{request_text[:6000]}"
    )
    try:
        from codex_manager.brain.connector import connect
    except Exception as exc:
        fallback = (
            '{'
            '"status":"refused",'
            '"notes":"AI connector unavailable in this environment.",'
            '"response":"Could not process the request because AI connectivity is unavailable."'
            "}"
        )
        return "refused", str(exc), fallback, "AI connector unavailable"

    try:
        raw = connect(
            model=str(model or "gpt-5.2").strip() or "gpt-5.2",
            prompt=prompt,
            text_only=True,
            operation="owner_general_request",
            stage="owner:general_request",
            max_output_tokens=2200,
            temperature=0.25,
        )
        raw_text = str(raw or "").strip()
        if not raw_text:
            raise RuntimeError("empty response")
        status, notes, response = _parse_general_request_response(raw_text)
        # This mode does not run repository edits.
        if status == "implemented":
            status = "considered"
            notes = f"{notes} (Implementation not executed in consider mode.)".strip()
        return status, notes, response, raw_text
    except Exception as exc:
        fallback = (
            '{'
            '"status":"refused",'
            '"notes":"AI request failed in this run.",'
            '"response":"The request could not be processed due to an AI/runtime error."'
            "}"
        )
        return "refused", str(exc), fallback, fallback


def _extract_governance_warnings(text: str) -> list[str]:
    raw = str(text or "").lower()
    warnings: list[str] = []
    risky_hits = [phrase for phrase in _RISKY_MARKETING_PHRASES if phrase in raw]
    if risky_hits:
        warnings.append(
            "Marketing/commercial claims may be overconfident. Review phrases: "
            + ", ".join(sorted(set(risky_hits)))
        )
    dark_hits = [phrase for phrase in _DARK_PATTERN_PHRASES if phrase in raw]
    if dark_hits:
        warnings.append(
            "Manipulative engagement language detected. Remove/replace phrases: "
            + ", ".join(sorted(set(dark_hits)))
        )
    legal_hits = [phrase for phrase in _LEGAL_CERTAINTY_PHRASES if phrase in raw]
    if legal_hits:
        warnings.append(
            "Legal/compliance certainty claims detected. Rephrase as conditional guidance: "
            + ", ".join(sorted(set(legal_hits)))
        )

    links = _extract_links(text)
    warnings.extend(_audit_research_sources(links))

    deduped: list[str] = []
    for warning in warnings:
        key = str(warning or "").strip()
        if key and key not in deduped:
            deduped.append(key)
    return deduped[:12]


def _parse_decision_cards(markdown: str, *, max_cards: int = 12) -> list[dict[str, str]]:
    text = str(markdown or "").strip()
    if not text:
        return []
    cards: list[dict[str, str]] = []

    # Preferred format: "### Option ..." sections.
    section_matches = list(
        re.finditer(
            r"^###\s+(.+?)\n(.*?)(?=^###\s+|\Z)",
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
    )
    for idx, match in enumerate(section_matches, start=1):
        title = re.sub(r"\s+", " ", match.group(1)).strip()
        body = re.sub(r"\s+", " ", match.group(2)).strip()
        if not title:
            continue
        cards.append(
            {
                "id": f"card-{idx:03d}",
                "title": title[:120],
                "summary": body[:800] if body else "(no details provided)",
                "decision": "pending",
                "owner_prompt": "",
                "source": "section",
            }
        )
        if len(cards) >= max_cards:
            return cards

    if cards:
        return cards

    # Fallback: bullet list items.
    bullet_lines = re.findall(r"^\s*(?:[-*]|\d+[.)])\s+(.+)$", text, flags=re.MULTILINE)
    for idx, line in enumerate(bullet_lines, start=1):
        content = re.sub(r"\s+", " ", line).strip()
        if not content:
            continue
        cards.append(
            {
                "id": f"card-{idx:03d}",
                "title": content[:120],
                "summary": content[:800],
                "decision": "pending",
                "owner_prompt": "",
                "source": "bullet",
            }
        )
        if len(cards) >= max_cards:
            return cards

    if cards:
        return cards

    # Last resort: one catch-all card.
    return [
        {
            "id": "card-001",
            "title": "Monetization Plan Review",
            "summary": re.sub(r"\s+", " ", text)[:1000],
            "decision": "pending",
            "owner_prompt": "",
            "source": "fallback",
        }
    ]


def _load_decision_board(repo: Path) -> dict[str, object]:
    path = _owner_decision_board_path(repo)
    if not path.is_file():
        return {
            "version": 1,
            "updated_at": "",
            "repo_path": str(repo),
            "cards": [],
            "governance_warnings": [],
            "source": "",
        }
    payload = _read_json_file(path)
    if not payload:
        return {
            "version": 1,
            "updated_at": "",
            "repo_path": str(repo),
            "cards": [],
            "governance_warnings": [],
            "source": "",
        }
    payload.setdefault("repo_path", str(repo))
    payload.setdefault("cards", [])
    payload.setdefault("governance_warnings", [])
    return payload


def _save_decision_board(repo: Path, payload: dict[str, object]) -> dict[str, object]:
    _ensure_codex_manager_gitignore_rules(repo)
    out = dict(payload)
    out["version"] = 1
    out["updated_at"] = _utc_now_iso_z()
    out["repo_path"] = str(repo)
    cards = out.get("cards")
    if not isinstance(cards, list):
        out["cards"] = []
    path = _owner_decision_board_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _safe_str(value: object, default: str = "") -> str:
    """Coerce *value* to a trimmed string, with fallback for empty values."""
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _parse_domain_list(value: object) -> list[str]:
    raw = str(value or "").strip().lower()
    if not raw:
        return []
    tokens = re.split(r"[\s,;]+", raw)
    domains: list[str] = []
    for token in tokens:
        cleaned = token.strip().lstrip(".")
        if not cleaned:
            continue
        if cleaned not in domains:
            domains.append(cleaned)
    return domains


def _normalize_domain_csv(value: object) -> str:
    return ",".join(_parse_domain_list(value))


def _governance_policy_defaults() -> dict[str, str]:
    payload: dict[str, str] = {}
    for key, env_name in _GOVERNANCE_ENV_KEYS.items():
        payload[key] = _normalize_domain_csv(os.getenv(env_name, ""))
    return payload


def _read_governance_policy_file() -> dict[str, str]:
    if not _GOVERNANCE_POLICY_PATH.is_file():
        return {}
    try:
        raw = json.loads(_read_text_utf8_resilient(_GOVERNANCE_POLICY_PATH))
    except Exception:
        logger.warning("Could not parse governance policy file: %s", _GOVERNANCE_POLICY_PATH)
        return {}
    if not isinstance(raw, dict):
        return {}
    payload: dict[str, str] = {}
    for key in _GOVERNANCE_ENV_KEYS:
        payload[key] = _normalize_domain_csv(raw.get(key, ""))
    return payload


def _apply_governance_policy_env(
    payload: dict[str, object],
    *,
    only_if_unset: bool = False,
) -> None:
    for key, env_name in _GOVERNANCE_ENV_KEYS.items():
        if only_if_unset and str(os.getenv(env_name, "")).strip():
            continue
        value = _normalize_domain_csv(payload.get(key, ""))
        if value:
            os.environ[env_name] = value
        else:
            os.environ.pop(env_name, None)


def _load_governance_policy() -> dict[str, str]:
    defaults = _governance_policy_defaults()
    from_file = _read_governance_policy_file()
    payload: dict[str, str] = {}
    for key in _GOVERNANCE_ENV_KEYS:
        payload[key] = defaults.get(key, "") or from_file.get(key, "")
    return payload


def _save_governance_policy(data: dict[str, object]) -> dict[str, str]:
    current = _load_governance_policy()
    for key in _GOVERNANCE_ENV_KEYS:
        if key in data:
            current[key] = _normalize_domain_csv(data.get(key, ""))
    payload: dict[str, object] = {
        "version": 1,
        "updated_at": _utc_now_iso_z(),
    }
    payload.update({key: current.get(key, "") for key in _GOVERNANCE_ENV_KEYS})
    _write_json_file_atomic(_GOVERNANCE_POLICY_PATH, payload)
    _apply_governance_policy_env(payload, only_if_unset=False)
    return {key: str(payload.get(key, "") or "") for key in _GOVERNANCE_ENV_KEYS}


def _initialize_governance_policy() -> None:
    payload = _read_governance_policy_file()
    if payload:
        _apply_governance_policy_env(payload, only_if_unset=True)


with suppress(Exception):
    _initialize_governance_policy()


def _utc_now_iso_z() -> str:
    """Return current UTC timestamp in compact ISO-8601 (with trailing Z)."""
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_github_auth_method(value: object) -> str:
    method = str(value or "https").strip().lower()
    return method if method in _GITHUB_AUTH_METHODS else "https"


def _api_keyring_status() -> tuple[bool, str, str]:
    """Return whether secure keyring storage is available for API keys."""
    return _github_keyring_status()


def _normalize_api_key_env_var(value: object) -> str:
    """Normalize and validate one API-key env var name."""
    env_var = str(value or "").strip().upper()
    return env_var if env_var in _API_KEY_ALLOWED_ENV_VARS else ""


def _api_key_catalog() -> list[dict[str, str]]:
    """Return API-key metadata for GUI forms and status payloads."""
    return [
        {
            "env_var": env_var,
            "provider": provider,
            "description": description,
        }
        for env_var, provider, description in _API_KEY_FIELD_SPECS
    ]


def _github_keyring_status() -> tuple[bool, str, str]:
    """Return whether secure keyring storage is usable and backend details."""
    if keyring is None:
        return False, "", "Secure storage unavailable: install the 'keyring' package."
    try:
        backend = keyring.get_keyring()
    except Exception as exc:
        return False, "", f"Secure storage unavailable: could not load keyring backend ({exc})."

    backend_name = f"{backend.__class__.__module__}.{backend.__class__.__name__}"
    module_name = str(backend.__class__.__module__ or "").lower()
    class_name = str(backend.__class__.__name__ or "").lower()
    if "fail" in module_name or "fail" in class_name:
        return (
            False,
            backend_name,
            "Secure storage unavailable: no OS keyring backend is configured.",
        )
    return True, backend_name, ""


def _github_secret_get(secret_key: str) -> str:
    """Read one GitHub credential from secure storage."""
    ok, _backend, error = _github_keyring_status()
    if not ok:
        raise RuntimeError(error or "Secure storage unavailable.")
    assert keyring is not None  # for type-checkers
    try:
        value = str(keyring.get_password(_GITHUB_SECRET_SERVICE, secret_key) or "")
        if value:
            return value
        return str(keyring.get_password(_GITHUB_SECRET_SERVICE_LEGACY, secret_key) or "")
    except KeyringError as exc:
        raise RuntimeError(f"Could not read secure credential '{secret_key}': {exc}") from exc


def _github_secret_set(secret_key: str, value: str) -> None:
    """Persist one GitHub credential in secure storage."""
    ok, _backend, error = _github_keyring_status()
    if not ok:
        raise RuntimeError(error or "Secure storage unavailable.")
    assert keyring is not None  # for type-checkers
    try:
        keyring.set_password(_GITHUB_SECRET_SERVICE, secret_key, value)
    except KeyringError as exc:
        raise RuntimeError(f"Could not store secure credential '{secret_key}': {exc}") from exc


def _github_secret_delete(secret_key: str) -> None:
    """Delete one GitHub credential from secure storage if present."""
    ok, _backend, error = _github_keyring_status()
    if not ok:
        raise RuntimeError(error or "Secure storage unavailable.")
    assert keyring is not None  # for type-checkers
    try:
        keyring.delete_password(_GITHUB_SECRET_SERVICE, secret_key)
    except PasswordDeleteError:
        pass
    except KeyringError as exc:
        raise RuntimeError(f"Could not delete secure credential '{secret_key}': {exc}") from exc
    with suppress(PasswordDeleteError, KeyringError):
        keyring.delete_password(_GITHUB_SECRET_SERVICE_LEGACY, secret_key)


def _api_key_secret_get(env_var: str) -> str:
    """Read one API key credential from secure storage."""
    ok, _backend, error = _api_keyring_status()
    if not ok:
        raise RuntimeError(error or "Secure storage unavailable.")
    assert keyring is not None  # for type-checkers
    try:
        value = str(keyring.get_password(_API_KEY_SECRET_SERVICE, env_var) or "")
        if value:
            return value
        return str(keyring.get_password(_API_KEY_SECRET_SERVICE_LEGACY, env_var) or "")
    except KeyringError as exc:
        raise RuntimeError(f"Could not read secure credential '{env_var}': {exc}") from exc


def _api_key_secret_set(env_var: str, value: str) -> None:
    """Persist one API key credential in secure storage."""
    ok, _backend, error = _api_keyring_status()
    if not ok:
        raise RuntimeError(error or "Secure storage unavailable.")
    assert keyring is not None  # for type-checkers
    try:
        keyring.set_password(_API_KEY_SECRET_SERVICE, env_var, value)
    except KeyringError as exc:
        raise RuntimeError(f"Could not store secure credential '{env_var}': {exc}") from exc


def _api_key_secret_delete(env_var: str) -> None:
    """Delete one API key credential from secure storage if present."""
    ok, _backend, error = _api_keyring_status()
    if not ok:
        raise RuntimeError(error or "Secure storage unavailable.")
    assert keyring is not None  # for type-checkers
    try:
        keyring.delete_password(_API_KEY_SECRET_SERVICE, env_var)
    except PasswordDeleteError:
        pass
    except KeyringError as exc:
        raise RuntimeError(f"Could not delete secure credential '{env_var}': {exc}") from exc
    with suppress(PasswordDeleteError, KeyringError):
        keyring.delete_password(_API_KEY_SECRET_SERVICE_LEGACY, env_var)


def _initialize_api_keys_from_secure_storage() -> None:
    """Populate process env from secure API-key storage when env vars are unset."""
    storage_ok, _storage_backend, _storage_error = _api_keyring_status()
    if not storage_ok:
        return
    for env_var in _API_KEY_ALLOWED_ENV_VARS:
        if str(os.getenv(env_var, "")).strip():
            continue
        with suppress(RuntimeError):
            value = _api_key_secret_get(env_var).strip()
            if value:
                os.environ[env_var] = value


def _api_keys_state() -> dict[str, object]:
    """Return API-key availability metadata without exposing secret values."""
    storage_ok, storage_backend, storage_error = _api_keyring_status()
    fields: list[dict[str, object]] = []

    for item in _api_key_catalog():
        env_var = str(item["env_var"])
        saved = False
        if storage_ok:
            try:
                value = _api_key_secret_get(env_var).strip()
                saved = bool(value)
                if value and not str(os.getenv(env_var, "")).strip():
                    os.environ[env_var] = value
            except RuntimeError as exc:
                storage_ok = False
                storage_error = str(exc)
        fields.append(
            {
                "env_var": env_var,
                "provider": item["provider"],
                "description": item["description"],
                "saved": saved,
                "in_environment": bool(str(os.getenv(env_var, "")).strip()),
            }
        )

    return {
        "secure_storage_available": storage_ok,
        "storage_backend": storage_backend,
        "storage_error": storage_error,
        "supported_keys": fields,
    }


def _save_api_keys_settings(
    data: dict[str, object],
) -> tuple[dict[str, object] | None, str | None, int]:
    """Persist API keys in secure storage and mirror them into this process env."""
    raw_keys = data.get("keys")
    if raw_keys is None:
        keys_payload: dict[object, object] = {}
    elif isinstance(raw_keys, dict):
        keys_payload = raw_keys
    else:
        return None, "'keys' must be a JSON object.", 400

    raw_clear = data.get("clear_keys", [])
    if raw_clear is None:
        clear_payload: list[object] = []
    elif isinstance(raw_clear, str):
        clear_payload = [raw_clear]
    elif isinstance(raw_clear, (list, tuple, set)):
        clear_payload = list(raw_clear)
    else:
        return None, "'clear_keys' must be a list of env-var names.", 400

    invalid_keys: list[str] = []
    to_set: dict[str, str] = {}
    for raw_name, raw_value in keys_payload.items():
        env_var = _normalize_api_key_env_var(raw_name)
        if not env_var:
            invalid_keys.append(str(raw_name))
            continue
        value = str(raw_value or "").strip()
        if value:
            to_set[env_var] = value

    to_clear: set[str] = set()
    for raw_name in clear_payload:
        env_var = _normalize_api_key_env_var(raw_name)
        if not env_var:
            invalid_keys.append(str(raw_name))
            continue
        to_clear.add(env_var)

    if invalid_keys:
        unique_invalid = sorted({key for key in invalid_keys if key})
        detail = ", ".join(unique_invalid) if unique_invalid else "(empty)"
        return (
            None,
            f"Unsupported API key environment variable(s): {detail}.",
            400,
        )

    clear_only = to_clear.difference(to_set.keys())
    requested_secret_mutation = bool(to_set or clear_only)
    storage_ok, _storage_backend, storage_error = _api_keyring_status()
    if requested_secret_mutation and not storage_ok:
        return None, storage_error or "Secure storage is unavailable.", 503

    try:
        for env_var, value in to_set.items():
            _api_key_secret_set(env_var, value)
            os.environ[env_var] = value
        for env_var in clear_only:
            _api_key_secret_delete(env_var)
            os.environ.pop(env_var, None)
    except RuntimeError as exc:
        return None, str(exc), 500

    return _api_keys_state(), None, 200


with suppress(Exception):
    _initialize_api_keys_from_secure_storage()


def _github_auth_meta_defaults() -> dict[str, object]:
    return {
        "version": 1,
        "preferred_auth": "https",
        "updated_at": "",
        "last_test_at": "",
        "last_test_ok": None,
    }


def _read_github_auth_meta() -> dict[str, object]:
    defaults = _github_auth_meta_defaults()
    if not _GITHUB_AUTH_META_PATH.is_file():
        return defaults
    try:
        raw = json.loads(_read_text_utf8_resilient(_GITHUB_AUTH_META_PATH))
    except Exception:
        logger.warning("Could not parse GitHub auth metadata file: %s", _GITHUB_AUTH_META_PATH)
        return defaults
    if not isinstance(raw, dict):
        return defaults
    defaults["preferred_auth"] = _normalize_github_auth_method(raw.get("preferred_auth"))
    defaults["updated_at"] = str(raw.get("updated_at") or "")
    defaults["last_test_at"] = str(raw.get("last_test_at") or "")
    last_test_ok = raw.get("last_test_ok")
    defaults["last_test_ok"] = last_test_ok if isinstance(last_test_ok, bool) else None
    return defaults


def _write_github_auth_meta(payload: dict[str, object]) -> None:
    out = _github_auth_meta_defaults()
    out["preferred_auth"] = _normalize_github_auth_method(payload.get("preferred_auth"))
    out["updated_at"] = str(payload.get("updated_at") or "")
    out["last_test_at"] = str(payload.get("last_test_at") or "")
    last_test_ok = payload.get("last_test_ok")
    out["last_test_ok"] = last_test_ok if isinstance(last_test_ok, bool) else None
    _write_json_file_atomic(_GITHUB_AUTH_META_PATH, out)


def _github_auth_state() -> dict[str, object]:
    """Return non-secret GitHub auth settings and secure-storage availability."""
    meta = _read_github_auth_meta()
    storage_ok, storage_backend, storage_error = _github_keyring_status()

    has_pat = False
    has_ssh_key = False
    if storage_ok:
        try:
            has_pat = bool(_github_secret_get(_GITHUB_PAT_SECRET_KEY).strip())
            has_ssh_key = bool(_github_secret_get(_GITHUB_SSH_SECRET_KEY).strip())
        except RuntimeError as exc:
            storage_ok = False
            storage_error = str(exc)

    return {
        "preferred_auth": _normalize_github_auth_method(meta.get("preferred_auth")),
        "has_pat": has_pat,
        "has_ssh_key": has_ssh_key,
        "secure_storage_available": storage_ok,
        "storage_backend": storage_backend,
        "storage_error": storage_error,
        "updated_at": str(meta.get("updated_at") or ""),
        "last_test_at": str(meta.get("last_test_at") or ""),
        "last_test_ok": meta.get("last_test_ok"),
    }


def _save_github_auth_settings(
    data: dict[str, object],
) -> tuple[dict[str, object] | None, str | None, int]:
    """Persist non-secret GitHub auth settings and optional secure credentials."""
    meta = _read_github_auth_meta()
    preferred_auth = _normalize_github_auth_method(data.get("preferred_auth") or meta["preferred_auth"])

    clear_pat = _safe_bool(data.get("clear_pat"), default=False)
    clear_ssh_key = _safe_bool(data.get("clear_ssh_key"), default=False)

    pat_raw = str(data.get("pat") or "")
    pat = pat_raw.strip()
    pat_provided = bool(pat)

    ssh_raw = str(data.get("ssh_private_key") or "").replace("\r\n", "\n")
    ssh_private_key = ssh_raw.strip()
    ssh_provided = bool(ssh_private_key)

    requested_secret_mutation = pat_provided or ssh_provided or clear_pat or clear_ssh_key
    storage_ok, _storage_backend, storage_error = _github_keyring_status()
    if requested_secret_mutation and not storage_ok:
        return None, storage_error or "Secure storage is unavailable.", 503

    try:
        if pat_provided:
            _github_secret_set(_GITHUB_PAT_SECRET_KEY, pat)
        elif clear_pat:
            _github_secret_delete(_GITHUB_PAT_SECRET_KEY)

        if ssh_provided:
            _github_secret_set(_GITHUB_SSH_SECRET_KEY, ssh_private_key)
        elif clear_ssh_key:
            _github_secret_delete(_GITHUB_SSH_SECRET_KEY)
    except RuntimeError as exc:
        return None, str(exc), 500

    if requested_secret_mutation or preferred_auth != meta.get("preferred_auth"):
        meta["updated_at"] = _utc_now_iso_z()
    meta["preferred_auth"] = preferred_auth
    _write_github_auth_meta(meta)
    return _github_auth_state(), None, 200


def _github_test_pat(token: str) -> dict[str, object]:
    """Validate a GitHub PAT by querying the authenticated user endpoint."""
    request_obj = Request(
        "https://api.github.com/user",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "warpfoundry-github-auth-test",
        },
        method="GET",
    )
    try:
        with urlopen(request_obj, timeout=_GITHUB_TEST_TIMEOUT_SECONDS) as response:
            payload = response.read().decode("utf-8", errors="replace")
        parsed = json.loads(payload) if payload else {}
        login = str(parsed.get("login") or "").strip() if isinstance(parsed, dict) else ""
        message = "GitHub PAT authenticated successfully."
        if login:
            message += f" Signed in as '{login}'."
        return {"ok": True, "message": message, "login": login}
    except HTTPError as exc:
        detail = ""
        with suppress(Exception):
            body = exc.read().decode("utf-8", errors="replace")
            parsed = json.loads(body) if body else {}
            if isinstance(parsed, dict):
                detail = str(parsed.get("message") or "").strip()
        if exc.code == 401:
            message = "GitHub rejected this PAT (401 Unauthorized)."
        elif exc.code == 403:
            message = "GitHub denied this PAT (403 Forbidden)."
        else:
            message = f"GitHub returned HTTP {exc.code}."
        if detail:
            message += f" {detail[:200]}"
        return {"ok": False, "message": message}
    except URLError as exc:
        reason = str(getattr(exc, "reason", exc) or "").strip()
        suffix = f": {reason}" if reason else "."
        return {"ok": False, "message": f"Could not reach api.github.com{suffix}"}
    except Exception as exc:
        return {"ok": False, "message": f"PAT test failed: {exc}"}


def _github_test_ssh_key(private_key: str) -> dict[str, object]:
    """Validate an SSH private key by attempting non-interactive GitHub auth."""
    key_text = str(private_key or "").replace("\r\n", "\n").strip()
    if not key_text:
        return {"ok": False, "message": "SSH private key is empty."}

    key_fd, key_tmp_name = tempfile.mkstemp(prefix="cm-gh-key-", suffix=".pem")
    known_fd, known_tmp_name = tempfile.mkstemp(prefix="cm-gh-known-hosts-", suffix=".txt")
    key_path = Path(key_tmp_name)
    known_hosts_path = Path(known_tmp_name)
    os.close(known_fd)
    try:
        with os.fdopen(key_fd, "w", encoding="utf-8") as f:
            f.write(key_text)
            if not key_text.endswith("\n"):
                f.write("\n")
        with suppress(OSError):
            os.chmod(key_path, 0o600)

        args = [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={known_hosts_path}",
            "-i",
            str(key_path),
            "git@github.com",
        ]
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=_GITHUB_TEST_TIMEOUT_SECONDS,
            check=False,
            **_subprocess_isolation_kwargs(),
        )
        combined = _truncate_command_output(
            "\n".join(part for part in [proc.stdout, proc.stderr] if part).strip()
        )
        lowered = combined.lower()
        if "successfully authenticated" in lowered or "does not provide shell access" in lowered:
            return {
                "ok": True,
                "message": "GitHub SSH authentication succeeded.",
                "output": combined,
            }
        if "permission denied" in lowered:
            return {
                "ok": False,
                "message": "GitHub rejected this SSH key (permission denied).",
                "output": combined,
                "exit_code": proc.returncode,
            }
        return {
            "ok": proc.returncode == 0,
            "message": (
                "GitHub SSH authentication succeeded."
                if proc.returncode == 0
                else "GitHub SSH authentication failed."
            ),
            "output": combined,
            "exit_code": proc.returncode,
        }
    except FileNotFoundError:
        return {"ok": False, "message": "SSH client not found on PATH."}
    except subprocess.TimeoutExpired:
        return {"ok": False, "message": "SSH test timed out while contacting GitHub."}
    except Exception as exc:
        return {"ok": False, "message": f"SSH test failed: {exc}"}
    finally:
        with suppress(OSError):
            key_path.unlink(missing_ok=True)
        with suppress(OSError):
            known_hosts_path.unlink(missing_ok=True)


def _github_remote_transport(remote_url: str) -> str:
    """Return ``https``/``ssh`` for GitHub remotes, else empty string."""
    raw = str(remote_url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    if parsed.scheme and parsed.netloc:
        host = str(parsed.hostname or parsed.netloc).strip().lower()
        if host != "github.com":
            return ""
        scheme = str(parsed.scheme or "").strip().lower()
        if scheme in {"http", "https"}:
            return "https"
        if scheme == "ssh":
            return "ssh"
        return ""
    if "://" not in raw and ":" in raw:
        left, right = raw.split(":", 1)
        if "@" in left and right.strip():
            host = left.split("@", 1)[1].strip().lower()
            if host == "github.com":
                return "ssh"
    return ""


def _github_auth_troubleshooting_assistant(
    *,
    auth_method: str,
    ok: bool,
    message: str,
    output: str = "",
    context: str = "",
    remote_url: str = "",
) -> dict[str, object]:
    """Return structured troubleshooting guidance for GitHub credential failures."""

    method = _normalize_github_auth_method(auth_method)
    if not method:
        method = _github_remote_transport(remote_url)
    combined = f"{message}\n{output}".lower()
    if not method:
        if "git@github.com" in combined or "publickey" in combined or "ssh" in combined:
            method = "ssh"
        elif "https://github.com" in combined or "token" in combined:
            method = "https"
    if method not in _GITHUB_AUTH_METHODS:
        method = "https"

    pat_scope_issue = any(
        token in combined
        for token in (
            "403",
            "forbidden",
            "insufficient",
            "scope",
            "resource not accessible",
            "not authorized",
            "permission to",
            "repository not found",
        )
    )
    known_hosts_issue = any(
        token in combined
        for token in (
            "host key verification failed",
            "remote host identification has changed",
            "strict host key checking",
            "known_hosts",
            "offending",
            "no matching host key",
        )
    )
    key_permission_issue = any(
        token in combined
        for token in (
            "permissions are too open",
            "bad permissions",
            "unprotected private key file",
            "load key",
            "permission denied (publickey)",
            "invalid format",
        )
    )

    checks: list[dict[str, object]] = []
    next_steps: list[str] = []
    summary = "Last GitHub credential check succeeded."
    if method == "https":
        status = "ok" if ok else ("action_required" if pat_scope_issue else "review")
        detail = (
            "Use a PAT that can write to the repository. Classic PATs need 'repo'; "
            "fine-grained PATs need repository access with Contents: Read and write "
            "and Pull requests: Read and write."
        )
        if not ok and pat_scope_issue:
            detail += " Current failure suggests missing scope/repository access."
        checks.append(
            {
                "key": "pat_scopes",
                "label": "PAT scopes",
                "status": status,
                "detail": detail,
                "commands": [
                    "Update token scopes in GitHub Settings > Developer settings > Personal access tokens.",
                    "Save the updated PAT in GitHub Auth Setup, then rerun Test Connection.",
                ],
            }
        )
        if not ok:
            summary = (
                "GitHub rejected PAT authentication. Verify PAT scopes/repository access and retest."
            )
            next_steps.extend(
                [
                    "Ensure the PAT can write to this repository (classic: repo; fine-grained: Contents + Pull requests write).",
                    "Save the PAT in GitHub Auth Setup and rerun Test Connection.",
                ]
            )
    else:
        known_hosts_status = "ok" if ok else ("action_required" if known_hosts_issue else "review")
        key_perm_status = "ok" if ok else ("action_required" if key_permission_issue else "review")
        checks.extend(
            [
                {
                    "key": "ssh_known_hosts",
                    "label": "SSH known_hosts",
                    "status": known_hosts_status,
                    "detail": (
                        "Trust github.com in your known_hosts file. If host keys changed, remove the "
                        "stale entry and reconnect to refresh the host key."
                    ),
                    "commands": [
                        "ssh-keygen -R github.com",
                        "ssh -T git@github.com",
                    ],
                },
                {
                    "key": "ssh_key_permissions",
                    "label": "SSH key permissions",
                    "status": key_perm_status,
                    "detail": (
                        "Private key files must be readable only by your account. OpenSSH rejects keys "
                        "with overly broad file permissions."
                    ),
                    "commands": [
                        "chmod 600 ~/.ssh/id_ed25519 (or equivalent ACL restrictions on Windows).",
                        "ssh-add ~/.ssh/id_ed25519",
                    ],
                },
            ]
        )
        if not ok:
            summary = (
                "GitHub rejected SSH authentication. Verify known_hosts trust and key permissions, then retest."
            )
            next_steps.extend(
                [
                    "Refresh github.com host trust (ssh-keygen -R github.com, then ssh -T git@github.com).",
                    "Ensure your private key permissions are restricted and the key is loaded in your SSH agent.",
                ]
            )

    if not ok and context == "git_push":
        next_steps.append("Retry push after fixes from the Git sync header controls.")
    elif not ok and context == "github_auth_test":
        next_steps.append("After changes, run Test Connection again from GitHub Auth Setup.")

    return {
        "available": True,
        "ok": bool(ok),
        "auth_method": method,
        "context": str(context or "").strip() or "github_auth",
        "title": "Credential Troubleshooting Assistant",
        "summary": summary,
        "checks": checks,
        "next_steps": next_steps,
    }


def _run_github_auth_test(data: dict[str, object]) -> tuple[dict[str, object], int]:
    """Resolve credentials (inline or saved) and run GitHub connectivity test."""
    meta = _read_github_auth_meta()
    auth_method = _normalize_github_auth_method(
        data.get("auth_method") or data.get("preferred_auth") or meta.get("preferred_auth")
    )
    use_saved = _safe_bool(data.get("use_saved"), default=True)

    if auth_method == "https":
        token = str(data.get("pat") or "").strip()
        if not token and use_saved:
            try:
                token = _github_secret_get(_GITHUB_PAT_SECRET_KEY).strip()
            except RuntimeError as exc:
                return {"ok": False, "auth_method": auth_method, "message": str(exc)}, 503
        if not token:
            return (
                {
                    "ok": False,
                    "auth_method": auth_method,
                    "message": "Provide a GitHub PAT or save one first.",
                    "troubleshooting": _github_auth_troubleshooting_assistant(
                        auth_method=auth_method,
                        ok=False,
                        message="Provide a GitHub PAT or save one first.",
                        context="github_auth_test",
                    ),
                },
                400,
            )
        result = _github_test_pat(token)
    else:
        ssh_key = str(data.get("ssh_private_key") or "").replace("\r\n", "\n").strip()
        if not ssh_key and use_saved:
            try:
                ssh_key = _github_secret_get(_GITHUB_SSH_SECRET_KEY).strip()
            except RuntimeError as exc:
                return {"ok": False, "auth_method": auth_method, "message": str(exc)}, 503
        if not ssh_key:
            return (
                {
                    "ok": False,
                    "auth_method": auth_method,
                    "message": "Provide an SSH private key or save one first.",
                    "troubleshooting": _github_auth_troubleshooting_assistant(
                        auth_method=auth_method,
                        ok=False,
                        message="Provide an SSH private key or save one first.",
                        context="github_auth_test",
                    ),
                },
                400,
            )
        result = _github_test_ssh_key(ssh_key)

    meta["last_test_at"] = _utc_now_iso_z()
    meta["last_test_ok"] = bool(result.get("ok"))
    _write_github_auth_meta(meta)

    payload = {
        "auth_method": auth_method,
        "troubleshooting": _github_auth_troubleshooting_assistant(
            auth_method=auth_method,
            ok=bool(result.get("ok")),
            message=str(result.get("message") or ""),
            output=str(result.get("output") or ""),
            context="github_auth_test",
        ),
    }
    payload.update(result)
    return payload, 200


_FOUNDATION_ASSISTANT_MODEL: dict[str, str] = {
    "codex": "gpt-5.2",
    "openai": "gpt-5.2",
    "google": "gemini-3-pro-preview",
    "claude": "claude-opus-4-6",
}


def _normalize_foundation_assistants(raw: object) -> list[str]:
    """Normalize assistant keys from API payload."""
    values: list[str] = []
    if isinstance(raw, str):
        values = [part.strip().lower() for part in raw.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        values = [str(part).strip().lower() for part in raw]
    normalized: list[str] = []
    for value in values:
        if value in _FOUNDATION_ASSISTANT_MODEL and value not in normalized:
            normalized.append(value)
    return normalized


def _fallback_foundation_prompt(prompt: str, project_name: str) -> str:
    """Return a deterministic improved foundational prompt when APIs fail."""
    core = str(prompt or "").strip()
    return (
        f"You are the principal engineer for project '{project_name}'.\n\n"
        "Goal:\n"
        "- Build the project from this foundational brief with production quality.\n\n"
        "Functional Scope:\n"
        f"{core}\n\n"
        "Required Output Contract:\n"
        "- Architecture summary (components + responsibilities)\n"
        "- Phase-by-phase execution plan with acceptance criteria\n"
        "- Initial implementation order with measurable milestones\n"
        "- Test strategy and rollback/safety notes\n"
        "- Documentation files to generate before coding\n"
    )


def _improve_foundational_prompt(
    prompt: str,
    *,
    project_name: str,
    assistants: list[str],
) -> dict[str, object]:
    """Improve a foundational prompt using one or more model assistants."""
    clean_prompt = str(prompt or "").strip()
    if not clean_prompt:
        return {
            "recommended_prompt": "",
            "variants": [],
            "warning": "Prompt is empty.",
        }

    assistant_order = assistants or ["codex"]
    instruction = (
        "You are refining a software project foundational prompt.\n\n"
        f"Project name: {project_name}\n\n"
        "Improve this prompt so it is execution-ready for autonomous coding agents.\n"
        "Return only the improved prompt text. Keep it concise but specific.\n\n"
        "Original prompt:\n"
        f"{clean_prompt}"
    )
    variants: list[dict[str, str]] = []

    try:
        from codex_manager.brain.connector import connect
    except Exception as exc:
        fallback = _fallback_foundation_prompt(clean_prompt, project_name)
        return {
            "recommended_prompt": fallback,
            "variants": [],
            "warning": f"AI prompt improver unavailable ({exc}). Used fallback template.",
        }

    for assistant in assistant_order:
        model = _FOUNDATION_ASSISTANT_MODEL.get(assistant)
        if not model:
            continue
        try:
            text = connect(
                model=model,
                prompt=instruction,
                text_only=True,
                operation="foundation_prompt_improve",
                stage=f"foundation:{assistant}",
                max_output_tokens=1200,
            )
            candidate = str(text or "").strip()
            if candidate:
                variants.append(
                    {
                        "assistant": assistant,
                        "model": model,
                        "prompt": candidate,
                    }
                )
        except Exception as exc:
            variants.append(
                {
                    "assistant": assistant,
                    "model": model,
                    "error": str(exc),
                    "prompt": "",
                }
            )

    successful = [item for item in variants if item.get("prompt")]
    if successful:
        return {
            "recommended_prompt": str(successful[0]["prompt"]),
            "variants": variants,
            "warning": "",
        }

    fallback = _fallback_foundation_prompt(clean_prompt, project_name)
    return {
        "recommended_prompt": fallback,
        "variants": variants,
        "warning": "No assistant returned a usable prompt. Used fallback template.",
    }


def _write_foundation_artifacts(
    *,
    project_path: Path,
    project_name: str,
    description: str,
    foundational_prompt: str,
    assistants: list[str],
    generate_docs: bool,
    bootstrap_once: bool,
) -> list[str]:
    """Write foundational planning artifacts into the new repository."""
    _ensure_codex_manager_gitignore_rules(project_path)
    root = project_path / ".codex_manager" / "foundation"
    root.mkdir(parents=True, exist_ok=True)
    now = _utc_now_iso_z()

    prompt_path = root / "FOUNDATIONAL_PROMPT.md"
    prompt_path.write_text(
        (
            "# Foundational Prompt\n\n"
            f"- Project: {project_name}\n"
            f"- Created: {now}\n"
            f"- Assistants requested: {', '.join(assistants) if assistants else 'codex'}\n\n"
            "## Prompt\n\n"
            f"{foundational_prompt.strip()}\n"
        ),
        encoding="utf-8",
    )
    written = [str(prompt_path.relative_to(project_path))]

    if generate_docs:
        plan_path = root / "FOUNDATION_PLAN.md"
        plan_path.write_text(
            (
                "# Foundation Plan\n\n"
                f"- Project: {project_name}\n"
                f"- Description: {description or '(none provided)'}\n"
                f"- Generated: {now}\n\n"
                "## Objectives\n"
                "- Define architecture and execution phases\n"
                "- Identify MVP scope and success metrics\n"
                "- Capture implementation milestones and validation checks\n\n"
                "## One-Time Bootstrap Checklist\n"
                "1. Validate foundational prompt and constraints.\n"
                "2. Generate architecture notes and technical design docs.\n"
                "3. Produce implementation plan with ordered milestones.\n"
                "4. Execute the first implementation pass.\n"
                "5. Run tests and update docs before handoff.\n\n"
                "## Owner Decision Log\n"
                "- Approved ideas:\n"
                "- On hold:\n"
                "- Rejected:\n"
            ),
            encoding="utf-8",
        )
        written.append(str(plan_path.relative_to(project_path)))

    if bootstrap_once:
        bootstrap_path = root / "BOOTSTRAP_REQUEST.json"
        payload = {
            "version": 1,
            "created_at": now,
            "project_name": project_name,
            "description": description,
            "foundational_prompt_path": "FOUNDATIONAL_PROMPT.md",
            "status": "pending",
            "one_time_bootstrap": True,
        }
        bootstrap_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written.append(str(bootstrap_path.relative_to(project_path)))

    return written


_LICENSING_STRATEGIES: dict[str, str] = {
    "oss_only": "Open-source only",
    "open_core": "Open core (OSS + paid add-ons)",
    "dual_license": "Dual license (OSS + commercial)",
    "hosted_service": "Hosted service (SaaS/API)",
}


def _normalize_licensing_strategy(value: object) -> str:
    key = str(value or "oss_only").strip().lower().replace("-", "_")
    if key not in _LICENSING_STRATEGIES:
        return "oss_only"
    return key


def _legal_review_state_path(project_path: Path) -> Path:
    return project_path / ".codex_manager" / "business" / "legal_review.json"


def _legal_review_status(*, required: bool, approved: bool) -> tuple[str, bool]:
    if not required:
        return "not_required", True
    if approved:
        return "approved", True
    return "pending", False


def _load_legal_review_state(project_path: Path) -> dict[str, object]:
    path = _legal_review_state_path(project_path)
    if not path.is_file():
        return {}
    payload = _read_json_file(path)
    return payload if isinstance(payload, dict) else {}


def _save_legal_review_state(project_path: Path, payload: dict[str, object]) -> dict[str, object]:
    _ensure_codex_manager_gitignore_rules(project_path)
    state = dict(payload)
    state["version"] = 1
    state["updated_at"] = _utc_now_iso_z()
    path = _legal_review_state_path(project_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def _upsert_legal_review_state(
    *,
    project_path: Path,
    project_name: str,
    required: bool,
    approved: bool,
    reviewer: str = "",
    notes: str = "",
    files: list[str] | None = None,
    source: str = "",
) -> dict[str, object]:
    existing = _load_legal_review_state(project_path)
    reviewer_value = str(reviewer or existing.get("reviewer") or "").strip()
    notes_value = str(notes or existing.get("notes") or "").strip()
    status, publish_ready = _legal_review_status(required=required, approved=approved)
    approved_at = str(existing.get("approved_at") or "").strip()
    if approved and not approved_at:
        approved_at = _utc_now_iso_z()
    if not approved:
        approved_at = ""
    out: dict[str, object] = {
        "project_name": project_name,
        "required": bool(required),
        "approved": bool(approved),
        "status": status,
        "publish_ready": bool(publish_ready),
        "reviewer": reviewer_value,
        "notes": notes_value,
        "approved_at": approved_at,
        "source": str(source or existing.get("source") or "").strip(),
        "files": files if isinstance(files, list) else list(existing.get("files", []) or []),
    }
    return _save_legal_review_state(project_path, out)


def _legal_review_markdown_notice(state: dict[str, object]) -> str:
    required = bool(state.get("required", False))
    approved = bool(state.get("approved", False))
    status = str(state.get("status") or "").strip().lower()
    reviewer = str(state.get("reviewer") or "").strip()
    approved_at = str(state.get("approved_at") or "").strip()

    if not required:
        return ""
    if approved or status == "approved":
        reviewer_label = reviewer or "owner"
        approved_label = approved_at or "recorded in legal review state"
        return (
            "> [!NOTE]\n"
            f"> Legal review checkpoint approved by `{reviewer_label}` ({approved_label}).\n\n"
        )
    return (
        "> [!WARNING]\n"
        "> Legal review checkpoint is **pending**. Do not publish licensing/pricing docs until sign-off is recorded.\n\n"
    )


def _write_licensing_packaging_artifacts(
    *,
    project_path: Path,
    project_name: str,
    description: str,
    strategy: str,
    include_commercial_tiers: bool,
    owner_contact_email: str,
    legal_review_required: bool = True,
    legal_signoff_approved: bool = False,
    legal_reviewer: str = "",
    legal_notes: str = "",
) -> list[str]:
    """Write licensing/commercial planning artifacts for a new project."""
    strategy_key = _normalize_licensing_strategy(strategy)
    strategy_label = _LICENSING_STRATEGIES[strategy_key]
    now = _utc_now_iso_z()
    docs_dir = project_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    business_dir = project_path / ".codex_manager" / "business"
    business_dir.mkdir(parents=True, exist_ok=True)
    contact = owner_contact_email.strip() or "owner@example.com"
    legal_state = _upsert_legal_review_state(
        project_path=project_path,
        project_name=project_name,
        required=bool(legal_review_required),
        approved=bool(legal_signoff_approved),
        reviewer=legal_reviewer,
        notes=legal_notes,
        files=[],
        source="new_project",
    )
    legal_notice = _legal_review_markdown_notice(legal_state)

    profile = {
        "version": 1,
        "project_name": project_name,
        "description": description,
        "strategy_key": strategy_key,
        "strategy_label": strategy_label,
        "include_commercial_tiers": bool(include_commercial_tiers),
        "owner_contact_email": contact,
        "created_at": now,
    }
    profile_path = business_dir / "licensing_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    strategy_path = docs_dir / "LICENSING_STRATEGY.md"
    strategy_path.write_text(
        (
            f"# Licensing Strategy - {project_name}\n\n"
            f"_Generated: {now}_\n\n"
            f"{legal_notice}"
            "## Current Direction\n\n"
            f"- Strategy: **{strategy_label}** (`{strategy_key}`)\n"
            f"- Contact: `{contact}`\n"
            f"- Project description: {description or '(none provided)'}\n\n"
            "## Operating Principles\n\n"
            "1. Keep open-source obligations clear and discoverable.\n"
            "2. Separate community and commercial promises explicitly.\n"
            "3. Avoid legal/compliance claims without counsel review.\n"
            "4. Keep pricing and support terms versioned in-repo.\n\n"
            "## Required Follow-up\n\n"
            "1. Choose and publish the exact OSS license text.\n"
            "2. Define commercial terms in `docs/COMMERCIAL_OFFERING.md`.\n"
            "3. Add contribution policy and trademark guidelines (if applicable).\n"
            "4. Have legal counsel review before public launch.\n"
        ),
        encoding="utf-8",
    )

    offering_path = docs_dir / "COMMERCIAL_OFFERING.md"
    offering_path.write_text(
        (
            f"# Commercial Offering - {project_name}\n\n"
            f"_Generated: {now}_\n\n"
            f"{legal_notice}"
            "## Packaging Tracks\n\n"
            "- Community track: source code, docs, and standard issue support.\n"
            "- Commercial track: hosted offering, premium support, and enterprise controls.\n\n"
            "## Offer Checklist\n\n"
            "- SLA and support hours\n"
            "- Data retention and security posture\n"
            "- Upgrade/migration path between tiers\n"
            "- Refund and billing dispute policy\n\n"
            "## Governance\n\n"
            "- Do not publish guarantees such as 'risk-free' or 'always compliant'.\n"
            "- Require source-backed claims for benchmark/performance statements.\n"
            "- Track approvals in `.codex_manager/owner/decision_board.json`.\n"
        ),
        encoding="utf-8",
    )

    written = [
        str(profile_path.relative_to(project_path)),
        str(strategy_path.relative_to(project_path)),
        str(offering_path.relative_to(project_path)),
    ]

    if include_commercial_tiers:
        pricing_path = docs_dir / "PRICING_TIERS.md"
        pricing_path.write_text(
            (
                f"# Pricing Tiers - {project_name}\n\n"
                f"_Generated: {now}_\n\n"
                f"{legal_notice}"
                "## Suggested Tiers (Draft)\n\n"
                "| Tier | Target | Price Anchor | Included |\n"
                "|------|--------|--------------|----------|\n"
                "| Community | Individual builders | $0 | OSS self-hosted usage |\n"
                "| Pro | Small teams | $29-$99 / month | Hosted convenience, faster support |\n"
                "| Business | Growing orgs | $299-$999 / month | SSO, admin controls, audit logs |\n"
                "| Enterprise | Regulated/large orgs | Custom annual | SLA, dedicated support, compliance docs |\n\n"
                "## Notes\n\n"
                "- Replace draft anchors with validated willingness-to-pay research.\n"
                "- Tie each tier to measurable cost-to-serve assumptions.\n"
                "- Keep this file synchronized with customer-facing pricing pages.\n"
            ),
            encoding="utf-8",
        )
        written.append(str(pricing_path.relative_to(project_path)))

    written.append(str(_legal_review_state_path(project_path).relative_to(project_path)))
    legal_state = _upsert_legal_review_state(
        project_path=project_path,
        project_name=project_name,
        required=bool(legal_review_required),
        approved=bool(legal_signoff_approved),
        reviewer=legal_reviewer,
        notes=legal_notes,
        files=written,
        source="new_project",
    )
    if legal_state.get("status") == "pending":
        logger.info(
            "Legal review checkpoint pending for %s; licensing files are marked draft.",
            project_path,
        )

    return written


def _foundation_root(project_path: Path) -> Path:
    return project_path / ".codex_manager" / "foundation"


def _foundation_bootstrap_request_path(project_path: Path) -> Path:
    return _foundation_root(project_path) / "BOOTSTRAP_REQUEST.json"


def _foundation_bootstrap_status_path(project_path: Path) -> Path:
    return _foundation_root(project_path) / "BOOTSTRAP_STATUS.json"


def _read_json_file(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(_read_text_utf8_resilient(path))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_foundation_bootstrap_status(
    *,
    project_path: Path,
    status: str,
    detail: str,
    project_name: str = "",
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": 1,
        "updated_at": _utc_now_iso_z(),
        "status": status,
        "detail": detail,
        "project_path": str(project_path),
    }
    if project_name:
        payload["project_name"] = project_name
    if extra:
        payload.update(extra)
    path = _foundation_bootstrap_status_path(project_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _update_bootstrap_request(
    *,
    project_path: Path,
    status: str,
    detail: str = "",
    extra: dict[str, object] | None = None,
) -> None:
    req_path = _foundation_bootstrap_request_path(project_path)
    if not req_path.is_file():
        return
    payload = _read_json_file(req_path)
    payload["status"] = status
    payload["updated_at"] = _utc_now_iso_z()
    if detail:
        payload["detail"] = detail
    if extra:
        payload.update(extra)
    req_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _start_foundation_bootstrap_chain(
    *,
    project_path: Path,
    project_name: str,
    description: str,
    source: str,
) -> dict[str, object]:
    """Start the one-time bootstrap chain for a newly created foundation project."""
    global executor

    foundation_root = _foundation_root(project_path)
    prompt_path = foundation_root / "FOUNDATIONAL_PROMPT.md"
    if not prompt_path.is_file():
        status = _write_foundation_bootstrap_status(
            project_path=project_path,
            project_name=project_name,
            status="failed",
            detail="Missing FOUNDATIONAL_PROMPT.md.",
            extra={"source": source},
        )
        _update_bootstrap_request(
            project_path=project_path,
            status="failed",
            detail="Missing FOUNDATIONAL_PROMPT.md.",
        )
        return status

    if executor.is_running:
        status = _write_foundation_bootstrap_status(
            project_path=project_path,
            project_name=project_name,
            status="queued",
            detail="Chain executor is busy. Run bootstrap manually when current run finishes.",
            extra={"source": source},
        )
        _update_bootstrap_request(
            project_path=project_path,
            status="queued",
            detail="Executor busy; bootstrap queued.",
        )
        return status

    plan_path = foundation_root / "FOUNDATION_PLAN.md"
    prompt_hint = (
        "Read and follow `.codex_manager/foundation/FOUNDATIONAL_PROMPT.md` exactly. "
        "If `.codex_manager/foundation/FOUNDATION_PLAN.md` exists, use it as the execution plan. "
        "Implement a production-ready MVP scaffold now, create required files, wire core flows, "
        "and run project-appropriate validation. Update README and docs with setup and usage."
    )
    validation_hint = (
        "Run and/or create practical validation checks for this new project. "
        "If tests do not exist, create minimal baseline tests or smoke checks. "
        "Document current status and next highest-priority actions."
    )
    if plan_path.is_file():
        prompt_hint += " Foundation plan available at `.codex_manager/foundation/FOUNDATION_PLAN.md`."

    chain = ChainConfig(
        name=f"{project_name or project_path.name} Foundation Bootstrap",
        repo_path=str(project_path),
        mode="apply",
        steps=[
            TaskStep(
                name="Foundation Bootstrap Implementation",
                job_type="implementation",
                prompt_mode="custom",
                custom_prompt=prompt_hint,
                enabled=True,
                agent="codex",
                loop_count=1,
            ),
            TaskStep(
                name="Foundation Bootstrap Validation",
                job_type="testing",
                prompt_mode="custom",
                custom_prompt=validation_hint,
                enabled=True,
                agent="codex",
                loop_count=1,
            ),
        ],
        max_loops=1,
        unlimited=False,
        improvement_threshold=0.1,
        max_time_minutes=90,
        max_total_tokens=1_500_000,
        strict_token_budget=False,
        stop_on_convergence=False,
        test_cmd="",
    )

    executor.start(chain)
    detail = (
        "One-time bootstrap chain started. Track progress in the main Chain Execution panel."
    )
    status = _write_foundation_bootstrap_status(
        project_path=project_path,
        project_name=project_name,
        status="running",
        detail=detail,
        extra={
            "source": source,
            "chain_name": chain.name,
            "bootstrap_mode": chain.mode,
            "steps": [step.name for step in chain.steps if step.enabled],
            "foundation_prompt_path": str(prompt_path),
            "foundation_plan_path": str(plan_path) if plan_path.is_file() else "",
        },
    )
    _update_bootstrap_request(
        project_path=project_path,
        status="running",
        detail=detail,
        extra={"chain_name": chain.name},
    )
    return status


def _foundation_bootstrap_status(repo_path: Path) -> dict[str, object]:
    """Return bootstrap status and reconcile stale 'running' states."""
    status_path = _foundation_bootstrap_status_path(repo_path)
    payload = _read_json_file(status_path) if status_path.is_file() else {}
    status = str(payload.get("status") or "").strip().lower()
    if not status:
        return {
            "status": "not_requested",
            "detail": "No bootstrap status found.",
            "project_path": str(repo_path),
        }

    if status == "running":
        cfg = executor.config
        running_same_repo = bool(
            executor.is_running
            and cfg
            and str(Path(cfg.repo_path).resolve()) == str(repo_path.resolve())
        )
        if not running_same_repo:
            payload = _write_foundation_bootstrap_status(
                project_path=repo_path,
                project_name=str(payload.get("project_name") or ""),
                status="completed",
                detail="Bootstrap chain is no longer running.",
                extra={k: v for k, v in payload.items() if k not in {"status", "detail", "updated_at"}},
            )
            _update_bootstrap_request(
                project_path=repo_path,
                status="completed",
                detail="Bootstrap chain finished.",
            )
    return payload


def _project_root_dir() -> Path:
    """Return repository root resolved from this module location."""
    return Path(__file__).resolve().parents[3]


def _readme_path() -> Path:
    """Return the repository README path."""
    return _project_root_dir() / "README.md"


def _docs_dir() -> Path | None:
    """Resolve the local docs directory when available."""
    candidates = [_project_root_dir() / "docs"]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_dir() and any(
            (resolved / filename).is_file() for _, filename in _DOCS_CATALOG.values()
        ):
            return resolved
    return None


def _docs_manifest() -> list[dict[str, object]]:
    """Return doc metadata for the in-app header links."""
    docs_dir = _docs_dir()
    items: list[dict[str, object]] = []
    for key, (title, filename) in _DOCS_CATALOG.items():
        doc_path = (docs_dir / filename) if docs_dir else None
        items.append(
            {
                "key": key,
                "title": title,
                "filename": filename,
                "available": bool(doc_path and doc_path.is_file()),
            }
        )
    return items


def _attach_stop_guidance(
    state_payload: dict[str, object],
    *,
    mode: str,
) -> dict[str, object]:
    """Attach user-facing stop guidance to a status payload."""
    payload = dict(state_payload)
    raw_reason = payload.get("stop_reason")
    reason = raw_reason if isinstance(raw_reason, str) else None
    payload["stop_guidance"] = get_stop_guidance(reason, mode=mode)
    return payload


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "off", "no"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _get_model_watchdog() -> ModelCatalogWatchdog:
    global _model_watchdog
    with _model_watchdog_lock:
        if _model_watchdog is None:
            _model_watchdog = ModelCatalogWatchdog(
                root_dir=_MODEL_WATCHDOG_ROOT,
                default_enabled=_env_bool("CODEX_MANAGER_MODEL_WATCHDOG_ENABLED", True),
                default_interval_hours=_env_int(
                    "CODEX_MANAGER_MODEL_WATCHDOG_INTERVAL_HOURS",
                    24,
                    1,
                    24 * 30,
                ),
            )
        return _model_watchdog


def _model_watchdog_health() -> dict[str, object]:
    try:
        status = _get_model_watchdog().status()
    except Exception:
        logger.exception("Could not read model watchdog status")
        return {
            "model_watchdog_enabled": False,
            "model_watchdog_running": False,
            "model_watchdog_next_due_at": "",
            "model_watchdog_last_status": "error",
        }
    cfg = status.get("config", {})
    state = status.get("state", {})
    return {
        "model_watchdog_enabled": bool(cfg.get("enabled", False)),
        "model_watchdog_running": bool(status.get("running", False)),
        "model_watchdog_next_due_at": str(status.get("next_due_at", "") or ""),
        "model_watchdog_last_status": str(state.get("last_status", "") or ""),
    }


def _project_display_name() -> str:
    """Return the user-facing program name shown in the GUI."""
    name = str(
        os.getenv("WARPFOUNDRY_PROJECT_NAME")
        or os.getenv("CODEX_MANAGER_PROJECT_NAME")
        or os.getenv("AI_MANAGER_PROJECT_NAME")
        or _DEFAULT_PROJECT_DISPLAY_NAME
    ).strip()
    return name or _DEFAULT_PROJECT_DISPLAY_NAME


# â”€â”€ Page â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _index_response_etag(
    *,
    project_display_name: str,
    recipes_payload: dict[str, object],
) -> str:
    """Return a weak-etag token for index response variants."""
    template_path = Path(_TEMPLATE_DIR) / "index.html"
    template_signature = _path_stat_signature(template_path) or (0, 0)
    stable_payload = json.dumps(recipes_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(
        (
            f"{__version__}|{template_signature[0]}:{template_signature[1]}|"
            f"{project_display_name}|{stable_payload}"
        ).encode("utf-8")
    ).hexdigest()
    return digest[:24]


def _request_accepts_gzip() -> bool:
    """Return True when the incoming request supports gzip."""
    return "gzip" in str(request.headers.get("Accept-Encoding") or "").lower()


def _set_response_vary_accept_encoding(response: Response) -> None:
    """Ensure ``Vary`` contains ``Accept-Encoding`` exactly once."""
    vary_raw = str(response.headers.get("Vary") or "")
    vary_tokens = {token.strip() for token in vary_raw.split(",") if token.strip()}
    vary_tokens.add("Accept-Encoding")
    response.headers["Vary"] = ", ".join(sorted(vary_tokens, key=str.lower))


@app.after_request
def _maybe_compress_response(response: Response) -> Response:
    """Gzip large text responses to reduce transfer size and initial-load latency."""
    if response.direct_passthrough:
        return response
    if request.method == "HEAD":
        return response
    if response.status_code < 200 or response.status_code >= 300:
        return response
    if response.mimetype == "text/event-stream":
        return response
    if response.mimetype not in _HTTP_COMPRESSIBLE_MIME_TYPES:
        return response
    if response.headers.get("Content-Encoding"):
        return response
    if not _request_accepts_gzip():
        return response

    payload = response.get_data()
    if len(payload) < _HTTP_COMPRESSION_MIN_BYTES:
        return response

    compressed_cache_key = ""
    etag = str(response.headers.get("ETag") or "").strip()
    if etag:
        compressed_cache_key = f"{response.mimetype}|{etag}|{len(payload)}"
        with _compressed_response_cache_lock:
            cached_compressed = _compressed_response_cache.get(compressed_cache_key)
        if isinstance(cached_compressed, bytes) and cached_compressed:
            response.set_data(cached_compressed)
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Content-Length"] = str(len(cached_compressed))
            _set_response_vary_accept_encoding(response)
            return response

    compressed = gzip.compress(payload, compresslevel=_HTTP_COMPRESSION_LEVEL)
    if len(compressed) >= len(payload):
        return response

    if compressed_cache_key:
        with _compressed_response_cache_lock:
            _bounded_cache_set(
                _compressed_response_cache,
                compressed_cache_key,
                compressed,
                max_entries=_HTTP_COMPRESSION_CACHE_MAX_ENTRIES,
            )

    response.set_data(compressed)
    response.headers["Content-Encoding"] = "gzip"
    response.headers["Content-Length"] = str(len(compressed))
    _set_response_vary_accept_encoding(response)
    return response


@app.route("/")
def index():
    project_display_name = _project_display_name()
    recipes_payload = _recipe_template_payload()
    etag = _index_response_etag(
        project_display_name=project_display_name,
        recipes_payload=recipes_payload,
    )
    if request.if_none_match.contains_weak(etag):
        response = Response(status=304)
    else:
        rendered_index = ""
        with _index_response_cache_lock:
            rendered_index = _index_response_cache.get(etag, "")
        if not rendered_index:
            rendered_index = render_template(
                "index.html",
                recipes_payload=recipes_payload,
                project_display_name=project_display_name,
            )
            with _index_response_cache_lock:
                _bounded_cache_set(
                    _index_response_cache,
                    etag,
                    rendered_index,
                    max_entries=_INDEX_RESPONSE_CACHE_MAX_ENTRIES,
                )
        response = Response(
            rendered_index,
            mimetype="text/html",
        )
    response.set_etag(etag, weak=True)
    response.headers["Cache-Control"] = (
        f"private, max-age={_INDEX_CACHE_MAX_AGE_SECONDS}, must-revalidate"
    )
    _set_response_vary_accept_encoding(response)
    return response


def _runtime_bool(value: object) -> bool:
    """Return a bool for values that may be callables on test doubles."""
    if callable(value):
        with suppress(Exception):
            return bool(value())
        return False
    return bool(value)


def _runtime_state_value(state: object, key: str, default: object = None) -> object:
    """Read either mapping-style or attribute-style runtime state values."""
    if isinstance(state, dict):
        return state.get(key, default)
    if state is None:
        return default
    return getattr(state, key, default)


def _chain_runtime_snapshot() -> dict[str, object]:
    """Return chain runtime context for refresh/reconnect flows."""
    cfg = getattr(executor, "config", None)
    state = getattr(executor, "state", None)
    running = _runtime_bool(getattr(executor, "is_running", False))
    paused = bool(_runtime_state_value(state, "paused", False))
    repo_path = str(getattr(cfg, "repo_path", "") or "").strip() if cfg is not None else ""
    mode = str(getattr(cfg, "mode", "") or "").strip() if cfg is not None else ""
    run_max_loops = _safe_int(getattr(cfg, "max_loops", 0), 0) if cfg is not None else 0
    run_unlimited = bool(getattr(cfg, "unlimited", False)) if cfg is not None else False

    raw_steps = getattr(cfg, "steps", []) if cfg is not None else []
    steps = raw_steps if isinstance(raw_steps, list) else []
    enabled_steps = 0
    for step in steps:
        if isinstance(step, dict):
            if bool(step.get("enabled", True)):
                enabled_steps += 1
        elif bool(getattr(step, "enabled", True)):
            enabled_steps += 1

    return {
        "running": running,
        "paused": paused,
        "active": bool(running or paused),
        "repo_path": repo_path,
        "name": str(getattr(cfg, "name", "") or "").strip() if cfg is not None else "",
        "mode": mode,
        "run_max_loops": max(0, run_max_loops),
        "run_unlimited": run_unlimited,
        "configured_steps_count": len(steps),
        "enabled_steps_count": max(0, enabled_steps),
        "current_loop": max(0, _safe_int(_runtime_state_value(state, "current_loop", 0), 0)),
        "current_step": max(0, _safe_int(_runtime_state_value(state, "current_step", 0), 0)),
        "current_step_name": str(_runtime_state_value(state, "current_step_name", "") or ""),
    }


def _pipeline_runtime_snapshot() -> dict[str, object]:
    """Return pipeline runtime context for refresh/reconnect flows."""
    global _pipeline_executor
    if _pipeline_executor is None:
        return {
            "running": False,
            "paused": False,
            "active": False,
            "repo_path": "",
            "mode": "",
            "run_max_cycles": 0,
            "run_unlimited": False,
            "current_cycle": 0,
            "current_phase": "",
            "current_iteration": 0,
        }

    state = getattr(_pipeline_executor, "state", None)
    cfg = getattr(_pipeline_executor, "config", None)
    running = _runtime_bool(getattr(_pipeline_executor, "is_running", False))
    paused = bool(_runtime_state_value(state, "paused", False))
    repo_path = str(getattr(_pipeline_executor, "repo_path", "") or "").strip()
    mode = str(getattr(cfg, "mode", "") or "").strip() if cfg is not None else ""
    run_max_cycles = _safe_int(getattr(cfg, "max_cycles", 0), 0) if cfg is not None else 0
    run_unlimited = bool(getattr(cfg, "unlimited", False)) if cfg is not None else False

    return {
        "running": running,
        "paused": paused,
        "active": bool(running or paused),
        "repo_path": repo_path,
        "mode": mode,
        "run_max_cycles": max(0, run_max_cycles),
        "run_unlimited": run_unlimited,
        "current_cycle": max(0, _safe_int(_runtime_state_value(state, "current_cycle", 0), 0)),
        "current_phase": str(_runtime_state_value(state, "current_phase", "") or ""),
        "current_iteration": max(
            0,
            _safe_int(_runtime_state_value(state, "current_iteration", 0), 0),
        ),
    }


def _attach_chain_runtime_fields(payload: dict[str, object]) -> dict[str, object]:
    """Attach chain runtime context to status payloads."""
    runtime = _chain_runtime_snapshot()
    payload.setdefault("repo_path", runtime.get("repo_path", ""))
    payload.setdefault("mode", runtime.get("mode", ""))
    payload.setdefault("run_max_loops", runtime.get("run_max_loops", 0))
    payload.setdefault("run_unlimited", runtime.get("run_unlimited", False))
    payload.setdefault("configured_steps_count", runtime.get("configured_steps_count", 0))
    payload.setdefault("enabled_steps_count", runtime.get("enabled_steps_count", 0))
    return payload


def _attach_pipeline_runtime_fields(payload: dict[str, object]) -> dict[str, object]:
    """Attach pipeline runtime context to status payloads."""
    runtime = _pipeline_runtime_snapshot()
    payload.setdefault("repo_path", runtime.get("repo_path", ""))
    payload.setdefault("mode", runtime.get("mode", ""))
    payload.setdefault("run_max_cycles", runtime.get("run_max_cycles", 0))
    payload.setdefault("run_unlimited", runtime.get("run_unlimited", False))
    return payload

@app.route("/api/health")
def api_health():
    """Lightweight liveness endpoint for frontend reconnect handling."""
    global _pipeline_executor
    pipeline_running = bool(_pipeline_executor is not None and _pipeline_executor.is_running)
    payload = {
        "ok": True,
        "time_epoch_ms": int(time.time() * 1000),
        "chain_running": bool(executor.is_running),
        "pipeline_running": pipeline_running,
    }
    payload.update(_model_watchdog_health())
    return jsonify(payload)


@app.route("/api/runtime/session")
def api_runtime_session():
    """Return active runtime context for UI refresh/reconnect reattach."""
    chain = _chain_runtime_snapshot()
    pipeline = _pipeline_runtime_snapshot()

    active_repo_path = ""
    if bool(pipeline.get("active")) and str(pipeline.get("repo_path") or "").strip():
        active_repo_path = str(pipeline.get("repo_path") or "").strip()
    elif bool(chain.get("active")) and str(chain.get("repo_path") or "").strip():
        active_repo_path = str(chain.get("repo_path") or "").strip()
    elif str(pipeline.get("repo_path") or "").strip():
        active_repo_path = str(pipeline.get("repo_path") or "").strip()
    elif str(chain.get("repo_path") or "").strip():
        active_repo_path = str(chain.get("repo_path") or "").strip()

    return jsonify(
        {
            "ok": True,
            "time_epoch_ms": int(time.time() * 1000),
            "active_repo_path": active_repo_path,
            "chain": chain,
            "pipeline": pipeline,
        }
    )


@app.route("/api/system/model-watchdog/status")
def api_model_watchdog_status():
    """Return model-watchdog scheduler status and latest persisted state."""
    return jsonify(_get_model_watchdog().status())


@app.route("/api/system/model-watchdog/alerts")
def api_model_watchdog_alerts():
    """Return latest alert-oriented watchdog summary for frontend surfacing."""
    return jsonify(_get_model_watchdog().latest_alerts())


@app.route("/api/system/model-watchdog/run", methods=["POST"])
def api_model_watchdog_run():
    """Trigger an immediate model-watchdog run."""
    data = request.get_json(silent=True) or {}
    force = bool(data.get("force", True))
    result = _get_model_watchdog().run_once(force=force, reason="manual")
    return jsonify(result)


@app.route("/api/system/model-watchdog/config", methods=["POST"])
def api_model_watchdog_config():
    """Update model-watchdog schedule settings."""
    data = request.get_json(silent=True) or {}
    updates: dict[str, object] = {}
    if "enabled" in data:
        updates["enabled"] = bool(data.get("enabled"))
    if "interval_hours" in data:
        updates["interval_hours"] = data.get("interval_hours")
    if "providers" in data:
        updates["providers"] = data.get("providers")
    if "request_timeout_seconds" in data:
        updates["request_timeout_seconds"] = data.get("request_timeout_seconds")
    if "auto_run_on_start" in data:
        updates["auto_run_on_start"] = bool(data.get("auto_run_on_start"))
    if "history_limit" in data:
        updates["history_limit"] = data.get("history_limit")

    watchdog = _get_model_watchdog()
    config = watchdog.update_config(updates)
    return jsonify({"status": "saved", "config": config, "watchdog": watchdog.status()})


@app.route("/api/governance/source-policy")
def api_governance_source_policy():
    """Return GUI-managed source-domain allow/deny policy settings."""
    return jsonify(_load_governance_policy())


@app.route("/api/governance/source-policy", methods=["POST"])
def api_governance_source_policy_save():
    """Persist GUI-managed source-domain allow/deny policy settings."""
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "JSON object body is required."}), 400
    policy = _save_governance_policy(data)
    return jsonify({"status": "saved", "policy": policy})


@app.route("/api/github/auth")
def api_github_auth():
    """Return GitHub auth settings metadata (never includes secret values)."""
    return jsonify(_github_auth_state())


@app.route("/api/github/auth", methods=["POST"])
def api_github_auth_save():
    """Persist GitHub auth settings and optional secure credentials."""
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "JSON object body is required."}), 400
    settings, error, status = _save_github_auth_settings(data)
    if error:
        return jsonify({"error": error}), status
    return jsonify({"status": "saved", "settings": settings})


@app.route("/api/github/auth/test", methods=["POST"])
def api_github_auth_test():
    """Test GitHub connectivity using saved or inline PAT/SSH credentials."""
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "JSON object body is required."}), 400
    payload, status = _run_github_auth_test(data)
    return jsonify(payload), status


# â”€â”€ Presets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@app.route("/api/api-keys")
def api_api_keys():
    """Return API-key availability and secure-storage status."""
    return jsonify(_api_keys_state())


@app.route("/api/api-keys", methods=["POST"])
def api_api_keys_save():
    """Persist API keys in secure storage and update process env."""
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "JSON object body is required."}), 400
    settings, error, status = _save_api_keys_settings(data)
    if error:
        return jsonify({"error": error}), status
    return jsonify({"status": "saved", "settings": settings})


@app.route("/api/about")
def api_about():
    """Return README-backed About payload for the GUI modal."""
    readme_path = _readme_path().resolve()
    project_root = _project_root_dir().resolve()
    if not readme_path.is_file() or readme_path.parent != project_root:
        return jsonify({"error": "README.md not found."}), 404

    try:
        readme_content = _read_text_utf8_resilient(readme_path)
    except Exception as exc:
        return jsonify({"error": f"Could not read README.md: {exc}"}), 500

    return jsonify(
        {
            "project_display_name": _project_display_name(),
            "version": str(__version__ or ""),
            "author": _PROJECT_AUTHOR,
            "readme_filename": readme_path.name,
            "readme_content": readme_content,
            "docs": _docs_manifest(),
        }
    )


@app.route("/api/docs")
def api_docs():
    return jsonify({"docs": _docs_manifest()})


@app.route("/api/docs/<key>")
def api_docs_detail(key: str):
    detail = _DOCS_CATALOG.get(key)
    if not detail:
        return jsonify({"error": f"Unknown doc key: {key}"}), 404

    docs_dir = _docs_dir()
    if docs_dir is None:
        return jsonify({"error": "Local docs directory not found"}), 404

    title, filename = detail
    doc_path = (docs_dir / filename).resolve()
    if not doc_path.is_file() or doc_path.parent != docs_dir.resolve():
        return jsonify({"error": f"Doc file not found: {filename}"}), 404

    try:
        content = _read_text_utf8_resilient(doc_path)
    except Exception as exc:
        return jsonify({"error": f"Could not read doc file: {exc}"}), 500

    return jsonify(
        {
            "key": key,
            "title": title,
            "filename": filename,
            "content": content,
        }
    )


@app.route("/api/presets")
def api_presets():
    return jsonify(list_presets())


@app.route("/api/presets/<key>")
def api_preset_detail(key: str):
    preset = get_preset(key)
    if not preset:
        return jsonify({"error": "not found"}), 404
    return jsonify(preset)


@app.route("/api/recipes")
def api_recipes():
    """List recipe summaries (built-in + optional per-repo custom) and recipe steps."""
    repo_raw = str(request.args.get("repo_path") or "").strip()
    repo: Path | None = None
    if repo_raw:
        repo = Path(repo_raw).expanduser().resolve()
        if not repo.is_dir():
            return jsonify({"error": f"Repo path not found: {repo_raw}"}), 400

    summaries = list_recipe_summaries(repo=repo)
    custom_count = sum(1 for entry in summaries if str(entry.get("source") or "") == "custom")
    return jsonify(
        {
            "default_recipe_id": DEFAULT_RECIPE_ID,
            "repo_path": str(repo) if repo is not None else "",
            "custom_store_path": str(custom_recipes_path(repo)) if repo is not None else "",
            "custom_recipe_count": custom_count,
            "recipes": summaries,
            "recipe_steps": recipe_steps_map(repo=repo),
        }
    )


@app.route("/api/recipes/<recipe_id>")
def api_recipe_detail(recipe_id: str):
    """Return one recipe with full step definitions."""
    repo_raw = str(request.args.get("repo_path") or "").strip()
    repo: Path | None = None
    if repo_raw:
        repo = Path(repo_raw).expanduser().resolve()
        if not repo.is_dir():
            return jsonify({"error": f"Repo path not found: {repo_raw}"}), 400

    recipe = get_recipe(recipe_id, repo=repo)
    if recipe is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(recipe)


@app.route("/api/recipes/custom")
def api_custom_recipes():
    """Return custom recipes for a repository."""
    repo_raw = str(request.args.get("repo_path") or "").strip()
    if not repo_raw:
        return jsonify({"error": "repo_path is required."}), 400
    repo = Path(repo_raw).expanduser().resolve()
    if not repo.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_raw}"}), 400

    recipes = list_custom_recipes(repo)
    return jsonify(
        {
            "repo_path": str(repo),
            "path": str(custom_recipes_path(repo)),
            "recipes": recipes,
        }
    )


@app.route("/api/recipes/custom/save", methods=["POST"])
def api_custom_recipes_save():
    """Create/update one custom recipe for a repository."""
    data = request.get_json(silent=True) or {}
    repo_raw = str(data.get("repo_path") or "").strip()
    if not repo_raw:
        return jsonify({"error": "repo_path is required."}), 400
    repo = Path(repo_raw).expanduser().resolve()
    if not repo.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_raw}"}), 400

    recipe_payload = data.get("recipe")
    if not isinstance(recipe_payload, dict):
        return jsonify({"error": "recipe object is required."}), 400

    try:
        _ensure_codex_manager_gitignore_rules(repo)
        recipe, created, path = save_custom_recipe(repo, recipe_payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "status": "saved",
            "created": created,
            "repo_path": str(repo),
            "path": str(path),
            "recipe": recipe,
            "custom_recipes": list_custom_recipes(repo),
        }
    )


@app.route("/api/recipes/custom/delete", methods=["POST"])
def api_custom_recipes_delete():
    """Delete one custom recipe for a repository."""
    data = request.get_json(silent=True) or {}
    repo_raw = str(data.get("repo_path") or "").strip()
    recipe_id = str(data.get("recipe_id") or "").strip()
    if not repo_raw:
        return jsonify({"error": "repo_path is required."}), 400
    if not recipe_id:
        return jsonify({"error": "recipe_id is required."}), 400
    repo = Path(repo_raw).expanduser().resolve()
    if not repo.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_raw}"}), 400

    try:
        deleted, path = delete_custom_recipe(repo, recipe_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not deleted:
        return jsonify({"error": "not found"}), 404

    return jsonify(
        {
            "status": "deleted",
            "repo_path": str(repo),
            "path": str(path),
            "recipe_id": recipe_id,
            "custom_recipes": list_custom_recipes(repo),
        }
    )


@app.route("/api/recipes/custom/import", methods=["POST"])
def api_custom_recipes_import():
    """Import custom recipe JSON into a repository."""
    data = request.get_json(silent=True) or {}
    repo_raw = str(data.get("repo_path") or "").strip()
    if not repo_raw:
        return jsonify({"error": "repo_path is required."}), 400
    repo = Path(repo_raw).expanduser().resolve()
    if not repo.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_raw}"}), 400

    payload = data.get("payload")
    if payload is None:
        return jsonify({"error": "payload is required."}), 400
    replace = bool(data.get("replace", False))

    try:
        _ensure_codex_manager_gitignore_rules(repo)
        summary, path = import_custom_recipes(repo, payload, replace=replace)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "status": "imported",
            "repo_path": str(repo),
            "path": str(path),
            **summary,
            "custom_recipes": list_custom_recipes(repo),
        }
    )


@app.route("/api/recipes/custom/export")
def api_custom_recipes_export():
    """Export custom recipe JSON for a repository."""
    repo_raw = str(request.args.get("repo_path") or "").strip()
    if not repo_raw:
        return jsonify({"error": "repo_path is required."}), 400
    repo = Path(repo_raw).expanduser().resolve()
    if not repo.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_raw}"}), 400

    recipe_id = str(request.args.get("recipe_id") or "").strip()
    try:
        payload = export_custom_recipes(repo, recipe_id=recipe_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404

    return jsonify(
        {
            "repo_path": str(repo),
            "path": str(custom_recipes_path(repo)),
            "recipe_id": recipe_id,
            "payload": payload,
        }
    )


# â”€â”€ Chain control â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@app.route("/api/chain/start", methods=["POST"])
def api_start():
    if executor.is_running:
        return jsonify({"error": "Chain is already running"}), 409

    data = request.get_json(silent=True) or {}
    try:
        config = ChainConfig(**data)
    except Exception as exc:
        return jsonify({"error": f"Invalid config: {exc}"}), 400

    if not Path(config.repo_path).is_dir():
        return jsonify({"error": f"Repo path not found: {config.repo_path}"}), 400

    issues = _chain_preflight_issues(config)
    if issues:
        msg = "Preflight checks failed:\n" + "\n".join(f"- {i}" for i in issues)
        return jsonify({"error": msg, "issues": issues}), 400

    git_preflight: dict[str, object] | None = None
    if config.git_preflight_enabled:
        try:
            git_preflight = _git_preflight_before_run(
                Path(config.repo_path).resolve(),
                auto_stash=bool(config.git_preflight_auto_stash),
                auto_pull=bool(config.git_preflight_auto_pull),
            )
        except subprocess.TimeoutExpired:
            return jsonify({"error": "Git pre-flight checks timed out."}), 504
        except RuntimeError as exc:
            return jsonify({"error": f"Git pre-flight checks failed: {exc}"}), 502
        except Exception as exc:
            return jsonify({"error": f"Git pre-flight checks failed: {exc}"}), 500

        git_issues = [str(item) for item in git_preflight.get("issues", []) if str(item).strip()]
        if git_issues:
            msg = "Git pre-flight checks failed:\n" + "\n".join(f"- {i}" for i in git_issues)
            return jsonify({"error": msg, "issues": git_issues, "git_preflight": git_preflight}), 400

    executor.start(config)
    payload: dict[str, object] = {"status": "started"}
    if git_preflight is not None:
        payload["git_preflight"] = git_preflight
    return jsonify(payload)


@app.route("/api/chain/stop", methods=["POST"])
def api_stop():
    executor.stop()
    return jsonify({"status": "stopping"})


@app.route("/api/chain/pause", methods=["POST"])
def api_pause():
    if executor.state.paused:
        executor.resume()
        return jsonify({"status": "resumed"})
    executor.pause()
    return jsonify({"status": "paused"})


@app.route("/api/chain/stop-after-step", methods=["POST"])
def api_stop_after_step():
    if not executor.is_running:
        return jsonify({"error": "No chain running"}), 400
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get("enabled", True))
    executor.set_stop_after_current_step(enabled)
    return jsonify(
        {
            "status": "armed" if enabled else "cleared",
            "stop_after_current_step": bool(executor.state.stop_after_current_step),
        }
    )


@app.route("/api/chain/status")
def api_status():
    since_results = _parse_since_results_arg(request.args.get("since_results"))
    if since_results is None:
        payload = executor.get_state()
    else:
        summary_fn = getattr(executor, "get_state_summary", None)
        if callable(summary_fn):
            payload = summary_fn(since_results=since_results)
        else:
            payload = executor.get_state()
    if isinstance(payload, dict):
        payload = _attach_chain_runtime_fields(payload)
    return jsonify(_attach_stop_guidance(payload, mode="chain"))


@app.route("/api/chain/outputs")
def api_chain_outputs():
    repo = _resolve_chain_output_repo(request.args.get("repo_path", ""))
    if repo is None:
        return jsonify({"files": [], "output_dir": "", "repo_path": ""})

    out_dir = _chain_output_dir(repo)
    files: list[dict[str, object]] = []
    if out_dir.is_dir():
        for p in sorted(out_dir.glob("*.md"), key=lambda x: x.name.lower()):
            try:
                st = p.stat()
                files.append(
                    {
                        "name": p.name,
                        "size_bytes": st.st_size,
                        "modified_epoch": int(st.st_mtime),
                    }
                )
            except Exception:
                continue
    return jsonify(
        {
            "repo_path": str(repo),
            "output_dir": str(out_dir),
            "files": files,
        }
    )


@app.route("/api/chain/outputs/<path:filename>")
def api_chain_output_file(filename: str):
    repo = _resolve_chain_output_repo(request.args.get("repo_path", ""))
    if repo is None:
        return jsonify({"error": "No valid repo path for chain outputs"}), 400

    # Prevent traversal outside the outputs directory.
    if Path(filename).name != filename:
        return jsonify({"error": "Invalid filename"}), 400

    out_dir = _chain_output_dir(repo).resolve()
    target = (out_dir / filename).resolve()
    if target.parent != out_dir:
        return jsonify({"error": "Invalid filename"}), 400
    if not target.is_file():
        return jsonify({"error": f"Output file not found: {filename}"}), 404

    try:
        content = _read_text_utf8_resilient(target)
    except Exception as exc:
        return jsonify({"error": f"Could not read output file: {exc}"}), 500

    return jsonify(
        {
            "repo_path": str(repo),
            "output_dir": str(out_dir),
            "name": filename,
            "content": content,
        }
    )


# â”€â”€ SSE live log stream â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@app.route("/api/stream")
def api_stream():
    """SSE stream for chain logs with non-destructive replay support."""
    resume_from = _parse_sse_resume_id()

    def generate():
        last_id = resume_from
        while True:
            events, replay_gap = _replay_log_events(executor, after_id=last_id)
            if replay_gap:
                yield _sse_frame(
                    {
                        "type": "warning",
                        "message": (
                            "Some older chain log events were dropped before replay "
                            "could resume."
                        ),
                    }
                )
            if events:
                for entry in events:
                    event_id = _safe_int(entry.get("id"), 0)
                    if event_id > last_id:
                        last_id = event_id
                    yield _sse_frame(entry, event_id=event_id if event_id > 0 else None)
                continue
            time.sleep(2.0)
            yield _sse_frame({"type": "heartbeat", "last_event_id": last_id})

    return Response(generate(), mimetype="text/event-stream")


# â”€â”€ Ollama (local models) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@app.route("/api/ollama/models")
def api_ollama_models():
    """Return installed Ollama models and server status."""
    from codex_manager.brain.connector import (
        OLLAMA_BASE_URL,
        _is_ollama_running,
        list_ollama_models,
    )

    running = _is_ollama_running()
    models = list_ollama_models() if running else []
    return jsonify(
        {
            "running": running,
            "base_url": OLLAMA_BASE_URL,
            "models": models,
        }
    )


# â”€â”€ Repo validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@app.route("/api/validate-repo", methods=["POST"])
def api_validate_repo():
    data = request.get_json(silent=True) or {}
    raw_path = str(data.get("path") or "").strip()
    def _repo_has_vector_memory(repo_path: Path) -> bool:
        memory_root = repo_path / ".codex_manager" / "memory"
        return (
            (memory_root / "vector_db").is_dir()
            or (memory_root / "vector_events.jsonl").is_file()
            or (memory_root / "deep_research_cache.jsonl").is_file()
        )

    if not raw_path:
        return jsonify(
            {
                "exists": False,
                "is_git": False,
                "path": "",
                "vector_memory_detected": False,
                "vector_memory_path": "",
            }
        )
    try:
        p = Path(raw_path)
    except Exception:
        return jsonify(
            {
                "exists": False,
                "is_git": False,
                "path": raw_path,
                "vector_memory_detected": False,
                "vector_memory_path": "",
            }
        )
    exists = p.is_dir()
    resolved = p.resolve() if exists else p
    vector_memory_path = str(resolved / ".codex_manager" / "memory" / "vector_db")
    return jsonify(
        {
            "exists": exists,
            "is_git": (resolved / ".git").is_dir() if exists else False,
            "path": str(resolved) if exists else raw_path,
            "vector_memory_detected": _repo_has_vector_memory(resolved) if exists else False,
            "vector_memory_path": vector_memory_path if exists else "",
        }
    )


@app.route("/api/diagnostics", methods=["POST"])
def api_diagnostics():
    """Return structured repository/auth diagnostics for the GUI."""
    data = request.get_json(silent=True) or {}
    repo_path, codex_binary, claude_binary, requested_agents = _extract_diagnostics_request(data)
    cache_key = _diagnostics_cache_key(
        repo_path=repo_path,
        codex_binary=codex_binary,
        claude_binary=claude_binary,
        requested_agents=requested_agents,
    )
    now = time.monotonic()

    with _diagnostics_report_cache_lock:
        cached = _diagnostics_report_cache.get(cache_key)
        if cached is not None:
            cached_at, cached_payload = cached
            if (now - cached_at) <= _DIAGNOSTICS_CACHE_TTL_SECONDS:
                return jsonify(cached_payload)

        report = _build_diagnostics_report(
            repo_path=repo_path,
            codex_binary=codex_binary,
            claude_binary=claude_binary,
            requested_agents=requested_agents,
        )
        _diagnostics_report_cache[cache_key] = (time.monotonic(), report)

        # Keep cache bounded and remove stale entries opportunistically.
        if len(_diagnostics_report_cache) > 64:
            cutoff = time.monotonic() - (_DIAGNOSTICS_CACHE_TTL_SECONDS * 3.0)
            stale_keys = [
                key
                for key, (cached_at, _payload) in _diagnostics_report_cache.items()
                if cached_at < cutoff
            ]
            for key in stale_keys[:48]:
                _diagnostics_report_cache.pop(key, None)
            if len(_diagnostics_report_cache) > 64:
                oldest_keys = sorted(
                    _diagnostics_report_cache.items(),
                    key=lambda item: item[1][0],
                )
                for key, _ in oldest_keys[: max(0, len(_diagnostics_report_cache) - 64)]:
                    _diagnostics_report_cache.pop(key, None)

    return jsonify(report)


@app.route("/api/diagnostics/actions/run", methods=["POST"])
def api_diagnostics_run_action():
    """Run a supported diagnostics action command by key."""
    data = request.get_json(silent=True) or {}
    action_key = str(data.get("action_key") or "").strip()
    if not action_key:
        return jsonify({"error": "Missing diagnostics action key."}), 400

    repo_path, codex_binary, claude_binary, requested_agents = _extract_diagnostics_request(data)
    report = build_preflight_report(
        repo_path=repo_path,
        agents=requested_agents,
        codex_binary=codex_binary,
        claude_binary=claude_binary,
    )

    action = next((item for item in report.next_actions if item.key == action_key), None)
    if action is None:
        return (
            jsonify(
                {
                    "error": "Diagnostics action unavailable for the current setup state.",
                    "action_key": action_key,
                }
            ),
            404,
        )

    if action_key not in _RUNNABLE_DIAGNOSTIC_ACTION_KEYS:
        return (
            jsonify(
                {
                    "error": "This diagnostics action cannot be auto-run. Copy the command instead.",
                    "action_key": action_key,
                    "command": action.command,
                }
            ),
            400,
        )

    if action_key == "snapshot_worktree_commit":
        result = _diagnostics_snapshot_worktree_commit(report)
        return jsonify(
            {
                "action_key": action.key,
                "title": action.title,
                "command": action.command,
                **result,
            }
        )

    args = _diagnostics_action_args(action_key, report)
    if not args:
        return (
            jsonify(
                {
                    "error": "Could not build command for this diagnostics action.",
                    "action_key": action_key,
                    "command": action.command,
                }
            ),
            400,
        )

    repo = str(report.resolved_repo_path or report.repo_path or "").strip()
    result = _run_diagnostics_action(args, cwd=repo)
    return jsonify(
        {
            "action_key": action.key,
            "title": action.title,
            "command": subprocess.list2cmdline(args),
            **result,
        }
    )


# â”€â”€ Owner decision board â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@app.route("/api/owner/decision-board")
def api_owner_decision_board():
    """Return the owner decision board for a repository."""
    repo_raw = str(request.args.get("repo_path", "") or "").strip()
    if not repo_raw:
        return jsonify({"error": "repo_path is required."}), 400
    repo = Path(repo_raw).expanduser().resolve()
    if not repo.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_raw}"}), 400
    return jsonify(_load_decision_board(repo))


@app.route("/api/owner/decision-board/generate", methods=["POST"])
def api_owner_decision_board_generate():
    """Generate/replace owner decision cards from monetization markdown."""
    data = request.get_json(silent=True) or {}
    repo_raw = str(data.get("repo_path") or "").strip()
    markdown = str(data.get("markdown") or "").strip()
    source = str(data.get("source") or "manual").strip()
    if not repo_raw:
        return jsonify({"error": "repo_path is required."}), 400
    repo = Path(repo_raw).expanduser().resolve()
    if not repo.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_raw}"}), 400
    if not markdown:
        return jsonify({"error": "markdown is required."}), 400

    cards = _parse_decision_cards(markdown)
    warnings = _extract_governance_warnings(markdown)
    payload = _save_decision_board(
        repo,
        {
            "source": source,
            "cards": cards,
            "governance_warnings": warnings,
        },
    )
    return jsonify(payload)


@app.route("/api/owner/decision-board/decision", methods=["POST"])
def api_owner_decision_board_decision():
    """Update one decision card with approve/hold/deny and optional follow-up prompt."""
    data = request.get_json(silent=True) or {}
    repo_raw = str(data.get("repo_path") or "").strip()
    card_id = str(data.get("card_id") or "").strip()
    decision = str(data.get("decision") or "").strip().lower()
    owner_prompt = str(data.get("owner_prompt") or "").strip()
    if not repo_raw:
        return jsonify({"error": "repo_path is required."}), 400
    if not card_id:
        return jsonify({"error": "card_id is required."}), 400
    if decision not in {"approve", "hold", "deny"}:
        return jsonify({"error": "decision must be one of: approve, hold, deny"}), 400
    repo = Path(repo_raw).expanduser().resolve()
    if not repo.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_raw}"}), 400

    board = _load_decision_board(repo)
    cards_raw = board.get("cards")
    cards = cards_raw if isinstance(cards_raw, list) else []
    updated = False
    for row in cards:
        if not isinstance(row, dict):
            continue
        if str(row.get("id") or "").strip() != card_id:
            continue
        row["decision"] = decision
        row["owner_prompt"] = owner_prompt
        row["decided_at"] = _utc_now_iso_z()
        updated = True
        break
    if not updated:
        return jsonify({"error": f"Card not found: {card_id}"}), 404

    board["cards"] = cards
    board = _save_decision_board(repo, board)
    return jsonify(board)


@app.route("/api/owner/repo-ideas")
def api_owner_repo_ideas():
    """Read the owner idea list (feature dreams) for a repository."""
    repo_raw = str(request.args.get("repo_path", "") or "").strip()
    if not repo_raw:
        return jsonify({"error": "repo_path is required."}), 400
    repo = Path(repo_raw).expanduser().resolve()
    if not repo.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_raw}"}), 400

    path = _feature_dreams_path(repo)
    content = _read_feature_dreams(repo)
    return jsonify(
        {
            "repo_path": str(repo),
            "path": str(path),
            "exists": path.is_file(),
            "has_open_items": _feature_dreams_has_open_items(content),
            "content": content,
        }
    )


@app.route("/api/owner/repo-ideas/generate", methods=["POST"])
def api_owner_repo_ideas_generate():
    """Generate repository-wide owner ideas and persist them to FEATURE_DREAMS.md."""
    data = request.get_json(silent=True) or {}
    repo_raw = str(data.get("repo_path") or "").strip()
    owner_context = str(data.get("owner_context") or "").strip()
    existing_markdown = str(data.get("existing_markdown") or "").strip()
    model = str(data.get("model") or "gpt-5.2").strip() or "gpt-5.2"
    if not repo_raw:
        return jsonify({"error": "repo_path is required."}), 400
    repo = Path(repo_raw).expanduser().resolve()
    if not repo.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_raw}"}), 400

    if not existing_markdown:
        existing_markdown = _read_feature_dreams(repo)
    suggested, warning, scan = _suggest_repo_ideas_markdown(
        repo=repo,
        model=model,
        owner_context=owner_context,
        existing_markdown=existing_markdown,
    )
    path = _write_feature_dreams(repo, suggested)
    saved = _read_text_utf8_resilient(path).strip()
    return jsonify(
        {
            "status": "generated",
            "repo_path": str(repo),
            "path": str(path),
            "model": model,
            "content": saved,
            "has_open_items": _feature_dreams_has_open_items(saved),
            "warning": warning,
            "scan": scan,
        }
    )


# -- Owner todo/wishlist workspace --------------------------------------------


@app.route("/api/owner/todo-wishlist")
def api_owner_todo_wishlist():
    """Read the owner todo/wishlist markdown for a repository."""
    repo_raw = str(request.args.get("repo_path", "") or "").strip()
    if not repo_raw:
        return jsonify({"error": "repo_path is required."}), 400
    repo = Path(repo_raw).expanduser().resolve()
    if not repo.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_raw}"}), 400

    path = _todo_wishlist_path(repo)
    content = _read_todo_wishlist(repo)
    return jsonify(
        {
            "repo_path": str(repo),
            "path": str(path),
            "exists": path.is_file(),
            "has_open_items": _todo_wishlist_has_open_items(content),
            "content": content,
        }
    )


@app.route("/api/owner/todo-wishlist/save", methods=["POST"])
def api_owner_todo_wishlist_save():
    """Save owner-provided todo/wishlist markdown content."""
    data = request.get_json(silent=True) or {}
    repo_raw = str(data.get("repo_path") or "").strip()
    content = str(data.get("content") or "").strip()
    if not repo_raw:
        return jsonify({"error": "repo_path is required."}), 400
    repo = Path(repo_raw).expanduser().resolve()
    if not repo.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_raw}"}), 400
    path = _write_todo_wishlist(repo, content)
    saved = _read_text_utf8_resilient(path)
    return jsonify(
        {
            "status": "saved",
            "repo_path": str(repo),
            "path": str(path),
            "has_open_items": _todo_wishlist_has_open_items(saved),
            "content": saved,
        }
    )


@app.route("/api/owner/todo-wishlist/suggest", methods=["POST"])
def api_owner_todo_wishlist_suggest():
    """Generate a suggested todo/wishlist markdown list using an AI model."""
    data = request.get_json(silent=True) or {}
    repo_raw = str(data.get("repo_path") or "").strip()
    owner_context = str(data.get("owner_context") or "").strip()
    existing_markdown = str(data.get("existing_markdown") or "").strip()
    context_files = _normalize_owner_context_files(data.get("context_files"))
    model = str(data.get("model") or "gpt-5.2").strip() or "gpt-5.2"
    if not repo_raw:
        return jsonify({"error": "repo_path is required."}), 400
    repo = Path(repo_raw).expanduser().resolve()
    if not repo.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_raw}"}), 400

    if not existing_markdown:
        existing_markdown = _read_todo_wishlist(repo)
    suggested, warning = _suggest_todo_wishlist_markdown(
        repo=repo,
        model=model,
        owner_context=owner_context,
        existing_markdown=existing_markdown,
        context_files=context_files,
    )
    return jsonify(
        {
            "repo_path": str(repo),
            "model": model,
            "content": suggested,
            "has_open_items": _todo_wishlist_has_open_items(suggested),
            "context_files_used": len(context_files),
            "warning": warning,
        }
    )


# â”€â”€ Directory browser â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


# -- Owner feature dreams workspace -------------------------------------------


@app.route("/api/owner/feature-dreams")
def api_owner_feature_dreams():
    """Read the owner feature-dreams markdown for a repository."""
    repo_raw = str(request.args.get("repo_path", "") or "").strip()
    if not repo_raw:
        return jsonify({"error": "repo_path is required."}), 400
    repo = Path(repo_raw).expanduser().resolve()
    if not repo.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_raw}"}), 400

    path = _feature_dreams_path(repo)
    content = _read_feature_dreams(repo)
    return jsonify(
        {
            "repo_path": str(repo),
            "path": str(path),
            "exists": path.is_file(),
            "has_open_items": _feature_dreams_has_open_items(content),
            "content": content,
        }
    )


@app.route("/api/owner/feature-dreams/save", methods=["POST"])
def api_owner_feature_dreams_save():
    """Save owner-provided feature-dreams markdown content."""
    data = request.get_json(silent=True) or {}
    repo_raw = str(data.get("repo_path") or "").strip()
    content = str(data.get("content") or "").strip()
    if not repo_raw:
        return jsonify({"error": "repo_path is required."}), 400
    repo = Path(repo_raw).expanduser().resolve()
    if not repo.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_raw}"}), 400
    path = _write_feature_dreams(repo, content)
    saved = _read_text_utf8_resilient(path)
    return jsonify(
        {
            "status": "saved",
            "repo_path": str(repo),
            "path": str(path),
            "has_open_items": _feature_dreams_has_open_items(saved),
            "content": saved,
        }
    )


@app.route("/api/owner/feature-dreams/suggest", methods=["POST"])
def api_owner_feature_dreams_suggest():
    """Generate a suggested feature-dreams markdown list using an AI model."""
    data = request.get_json(silent=True) or {}
    repo_raw = str(data.get("repo_path") or "").strip()
    owner_context = str(data.get("owner_context") or "").strip()
    existing_markdown = str(data.get("existing_markdown") or "").strip()
    context_files = _normalize_owner_context_files(data.get("context_files"))
    model = str(data.get("model") or "gpt-5.2").strip() or "gpt-5.2"
    if not repo_raw:
        return jsonify({"error": "repo_path is required."}), 400
    repo = Path(repo_raw).expanduser().resolve()
    if not repo.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_raw}"}), 400

    if not existing_markdown:
        existing_markdown = _read_feature_dreams(repo)
    suggested, warning = _suggest_feature_dreams_markdown(
        repo=repo,
        model=model,
        owner_context=owner_context,
        existing_markdown=existing_markdown,
        context_files=context_files,
    )
    return jsonify(
        {
            "repo_path": str(repo),
            "model": model,
            "content": suggested,
            "has_open_items": _feature_dreams_has_open_items(suggested),
            "context_files_used": len(context_files),
            "warning": warning,
        }
    )


@app.route("/api/owner/general-request/history")
def api_owner_general_request_history():
    """Return recent general-request history for a repository (latest first)."""
    repo_raw = str(request.args.get("repo_path", "") or "").strip()
    if not repo_raw:
        return jsonify({"error": "repo_path is required."}), 400
    repo = Path(repo_raw).expanduser().resolve()
    if not repo.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_raw}"}), 400
    limit_raw = request.args.get("limit", "25")
    try:
        limit = int(str(limit_raw).strip() or "25")
    except Exception:
        limit = 25
    limit = max(1, min(200, limit))
    entries = _read_general_request_history(repo, limit=limit)
    return jsonify(
        {
            "repo_path": str(repo),
            "path": str(_general_request_history_path(repo)),
            "entries": entries,
        }
    )


@app.route("/api/owner/general-request/process", methods=["POST"])
def api_owner_general_request_process():
    """Process one owner general request in consider mode and persist history."""
    data = request.get_json(silent=True) or {}
    repo_raw = str(data.get("repo_path") or "").strip()
    request_text = str(data.get("request_text") or "").strip()
    owner_context = str(data.get("owner_context") or "").strip()
    model = str(data.get("model") or "gpt-5.2").strip() or "gpt-5.2"
    context_files = _normalize_owner_context_files(data.get("context_files"))
    if not repo_raw:
        return jsonify({"error": "repo_path is required."}), 400
    if not request_text:
        return jsonify({"error": "request_text is required."}), 400
    repo = Path(repo_raw).expanduser().resolve()
    if not repo.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_raw}"}), 400

    status, notes, response, raw_output = _process_general_request(
        repo=repo,
        request_text=request_text,
        model=model,
        owner_context=owner_context,
        context_files=context_files,
    )
    entry = _append_general_request_history(
        repo=repo,
        request_text=request_text,
        status=status,
        notes=notes,
        output=raw_output,
        source="consider",
        model=model,
    )
    return jsonify(
        {
            "repo_path": str(repo),
            "model": model,
            "status": status,
            "notes": notes,
            "response": response,
            "output": raw_output,
            "history_entry": entry,
            "cleared_request": True,
            "context_files_used": len(context_files),
        }
    )


@app.route("/api/owner/general-request/history/add", methods=["POST"])
def api_owner_general_request_history_add():
    """Append one general-request history item (used by chain-driven implementations)."""
    data = request.get_json(silent=True) or {}
    repo_raw = str(data.get("repo_path") or "").strip()
    request_text = str(data.get("request_text") or "").strip()
    status = _normalize_general_request_status(data.get("status"))
    notes = str(data.get("notes") or "").strip()
    output = str(data.get("output") or "")
    source = str(data.get("source") or "").strip() or "chain"
    model = str(data.get("model") or "").strip()
    if not repo_raw:
        return jsonify({"error": "repo_path is required."}), 400
    if not request_text:
        return jsonify({"error": "request_text is required."}), 400
    repo = Path(repo_raw).expanduser().resolve()
    if not repo.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_raw}"}), 400
    entry = _append_general_request_history(
        repo=repo,
        request_text=request_text,
        status=status,
        notes=notes,
        output=output,
        source=source,
        model=model,
    )
    return jsonify(
        {
            "status": "saved",
            "repo_path": str(repo),
            "entry": entry,
        }
    )


# -- Directory browser --------------------------------------------------------


@app.route("/api/browse-dirs", methods=["POST"])
def api_browse_dirs():
    """Return child directories at a given path for the folder browser."""
    data = request.get_json(silent=True) or {}
    raw_path = str(data.get("path") or "").strip()

    # Default to user home if empty or invalid
    try:
        p = Path(raw_path) if raw_path else Path.home()
        if not p.is_dir():
            p = Path.home()
        p = p.resolve()
    except Exception:
        p = Path.home().resolve()

    # Collect child directories
    dirs = []
    try:
        for entry in sorted(p.iterdir(), key=lambda e: e.name.lower()):
            if entry.is_dir() and not entry.name.startswith("."):
                dirs.append(
                    {
                        "name": entry.name,
                        "is_git": (entry / ".git").is_dir(),
                    }
                )
    except PermissionError:
        pass

    # Determine parent
    parent = str(p.parent) if p.parent != p else ""

    # On Windows, add drive roots when at a drive root (e.g. C:\)
    drives = []
    if os.name == "nt" and p.parent == p:
        for letter in string.ascii_uppercase:
            drive = Path(f"{letter}:\\")
            if drive.is_dir():
                drives.append(letter)

    result = {
        "current": str(p),
        "parent": parent,
        "dirs": dirs,
    }
    if drives:
        result["drives"] = drives

    return jsonify(result)


# â”€â”€ Project creation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _sanitize_project_folder_name(value: object) -> str:
    """Return a filesystem-safe project folder name."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    cleaned = "".join(c for c in raw if c.isalnum() or c in "-_. ").strip(" .")
    return cleaned


def _derive_project_name_from_remote(remote_url: str) -> str:
    """Infer a local project folder name from a git remote URL/path."""
    raw = str(remote_url or "").strip()
    if not raw:
        return ""

    candidate_path = raw
    parsed = urlparse(raw)
    if parsed.scheme and parsed.path:
        candidate_path = parsed.path
    elif "://" not in raw and ":" in raw:
        left, right = raw.split(":", 1)
        # Handle scp-like remotes (git@github.com:owner/repo.git)
        if "@" in left and right.strip():
            candidate_path = right

    tail = candidate_path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if tail.lower().endswith(".git"):
        tail = tail[:-4]
    return _sanitize_project_folder_name(tail)


def _normalize_clone_branch_name(value: object) -> str:
    """Normalize a branch name supplied by the clone UI."""
    branch = str(value or "").strip()
    if not branch or branch == _DEFAULT_BRANCH_SENTINEL:
        return ""
    return branch


def _valid_clone_branch_name(branch: str) -> bool:
    """Return True when a branch name looks safe to pass to git clone."""
    if not branch:
        return True
    if branch.startswith("-") or branch.endswith("/") or branch.endswith("."):
        return False
    if branch.endswith(".lock") or ".." in branch:
        return False
    if any(ch.isspace() for ch in branch):
        return False
    return not any(ch in branch for ch in "~^:?*[\\")


def _extract_git_error_message(exc: subprocess.CalledProcessError) -> str:
    """Extract the most useful stderr/stdout detail from a git subprocess error."""
    stderr = str(exc.stderr or "").strip()
    if stderr:
        return stderr
    stdout = str(exc.stdout or "").strip()
    if stdout:
        return stdout
    return str(exc)


def _estimate_argv_length(argv: list[str]) -> int:
    """Estimate command-line length for conservative Windows/POSIX batching."""
    try:
        return len(subprocess.list2cmdline(argv))
    except Exception:
        return sum(len(str(part or "")) + 3 for part in argv)


def _git_sync_path_batch_length_limit() -> int:
    """Return argv length ceiling used for staged/unstaged path batching."""
    if os.name == "nt":
        return _GIT_SYNC_PATH_BATCH_LIMIT_WINDOWS
    return _GIT_SYNC_PATH_BATCH_LIMIT_POSIX


def _batch_git_paths_for_command(*, prefix_args: tuple[str, ...], paths: list[str]) -> list[list[str]]:
    """Split path lists so git argv stays below OS command-line limits."""
    if not paths:
        return []

    limit = _git_sync_path_batch_length_limit()
    base_argv = ["git", *prefix_args]
    batches: list[list[str]] = []
    current: list[str] = []
    for path in paths:
        candidate = [*current, path]
        estimated = _estimate_argv_length([*base_argv, *candidate])
        if current and estimated > limit:
            batches.append(current)
            current = [path]
            continue
        current = candidate
    if current:
        batches.append(current)
    return batches


def _run_git_sync_command(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run one git command for sync APIs and return the completed process."""
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=_GIT_SYNC_TIMEOUT_SECONDS,
        check=False,
        **_subprocess_isolation_kwargs(),
    )


def _extract_git_process_error(result: subprocess.CompletedProcess[str], fallback: str) -> str:
    """Extract stderr/stdout detail from a non-zero git process result."""
    stderr = str(result.stderr or "").strip()
    if stderr:
        return stderr
    stdout = str(result.stdout or "").strip()
    if stdout:
        return stdout
    return fallback


def _resolve_git_sync_repo(raw_repo_path: object) -> tuple[Path | None, str, int]:
    """Resolve and validate a repo path for git-sync API operations."""
    repo_raw = str(raw_repo_path or "").strip()
    if not repo_raw:
        return None, "repo_path is required.", 400

    repo = Path(repo_raw).expanduser().resolve()
    if not repo.is_dir():
        return None, f"Repo path not found: {repo_raw}", 400

    probe = _run_git_sync_command(repo, "rev-parse", "--is-inside-work-tree")
    if probe.returncode != 0 or str(probe.stdout or "").strip().lower() != "true":
        return None, f"Not a git repository: {repo}", 400

    return repo, "", 200


def _resolve_stash_ref(repo: Path) -> str:
    """Return refs/stash commit SHA when present, otherwise empty string."""
    stash_ref = _run_git_sync_command(repo, "rev-parse", "--verify", "refs/stash")
    if stash_ref.returncode != 0:
        return ""
    return str(stash_ref.stdout or "").strip()


def _resolve_git_fetch_head_path(repo: Path) -> Path | None:
    """Resolve FETCH_HEAD for a repository, returning None when unavailable."""
    git_dir = _resolve_git_dir_path(repo)
    result = _run_git_sync_command(repo, "rev-parse", "--git-path", "FETCH_HEAD")
    if result.returncode != 0:
        return None

    raw_path = str(result.stdout or "").strip()
    if not raw_path:
        return None

    fetch_head = Path(raw_path)
    if fetch_head.is_absolute():
        resolved_fetch_head = fetch_head.resolve()
    else:
        resolved_fetch_head = (repo / fetch_head).resolve()

    if git_dir is not None and not _is_within_directory(resolved_fetch_head, git_dir):
        return None
    return resolved_fetch_head


def _resolve_git_dir_path(repo: Path) -> Path | None:
    """Resolve repository git-dir path, returning None when unavailable."""
    for args in (
        ("rev-parse", "--absolute-git-dir"),
        ("rev-parse", "--git-dir"),
    ):
        result = _run_git_sync_command(repo, *args)
        if result.returncode != 0:
            continue
        raw_path = str(result.stdout or "").strip()
        if not raw_path:
            continue
        candidate = Path(raw_path)
        resolved = candidate.resolve() if candidate.is_absolute() else (repo / candidate).resolve()
        if resolved.is_dir():
            return resolved
    return None


def _git_last_fetch_metadata(repo: Path) -> tuple[int | None, str | None]:
    """Return ``(epoch_ms, iso_utc)`` for FETCH_HEAD mtime when available."""
    fetch_head = _resolve_git_fetch_head_path(repo)
    if fetch_head is None:
        return None, None

    try:
        stat_result = fetch_head.stat()
    except OSError:
        return None, None

    if stat_result.st_mtime_ns <= 0:
        return None, None

    epoch_ms = stat_result.st_mtime_ns // 1_000_000
    iso_utc = (
        datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    return epoch_ms, iso_utc


def _extract_tracking_remote_name(tracking_branch: str) -> str:
    """Extract remote name from ``<remote>/<branch>`` tracking refs."""
    tracking = str(tracking_branch or "").strip()
    if "/" not in tracking:
        return ""
    return tracking.split("/", 1)[0].strip()


def _normalize_git_sync_remote_name(value: object) -> str:
    """Normalize requested push remote, defaulting to origin."""
    remote = str(value or "").strip()
    return remote or "origin"


def _valid_git_sync_remote_name(remote: str) -> bool:
    """Return True when a remote name is safe to pass to git push."""
    if not remote:
        return False
    if remote.startswith("-"):
        return False
    if any(ch.isspace() for ch in remote):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9._/-]+", remote))


def _validate_git_remote_url(remote_url: str) -> tuple[bool, str, str]:
    """Validate an HTTPS/SSH git remote URL and return ``(ok, transport, message)``."""
    raw = str(remote_url or "").strip()
    if not raw:
        return False, "", "Remote URL is required."
    if raw.startswith("-"):
        return False, "", "Remote URL is invalid."
    if "\x00" in raw:
        return False, "", "Remote URL is invalid."
    if any(ch.isspace() for ch in raw):
        return False, "", "Remote URL may not include whitespace."

    parsed = urlparse(raw)
    if parsed.scheme:
        scheme = str(parsed.scheme or "").strip().lower()
        host = str(parsed.hostname or parsed.netloc or "").strip()
        path = str(parsed.path or "").strip().strip("/")
        if scheme == "https":
            if not host or not path:
                return False, "", "HTTPS remote URL must include host and repository path."
            return True, "https", "HTTPS remote URL looks valid."
        if scheme == "ssh":
            if not host or not path:
                return False, "", "SSH remote URL must include host and repository path."
            return True, "ssh", "SSH remote URL looks valid."
        return (
            False,
            "",
            f"Unsupported remote URL scheme: {scheme}. Use HTTPS or SSH.",
        )

    scp_like = re.fullmatch(
        r"(?P<user>[A-Za-z0-9._-]+)@(?P<host>[A-Za-z0-9._-]+):(?P<path>[^\s]+)",
        raw,
    )
    if scp_like:
        path = str(scp_like.group("path") or "").strip()
        if not path or path.startswith("/"):
            return False, "", "SSH remote URL path is invalid."
        return True, "ssh", "SSH remote URL looks valid."

    return (
        False,
        "",
        "Remote URL must be HTTPS (https://...) or SSH (ssh://... or git@host:path).",
    )


def _git_remote_exists(repo: Path, remote: str) -> bool:
    """Return True when a named git remote exists."""
    probe = _run_git_sync_command(repo, "remote", "get-url", remote)
    return probe.returncode == 0


def _git_configured_push_default_remote(repo: Path) -> str:
    """Return ``remote.pushDefault`` when configured and valid, else empty string."""
    result = _run_git_sync_command(repo, "config", "--get", "remote.pushDefault")
    if result.returncode != 0:
        return ""
    remote = str(result.stdout or "").strip()
    if not _valid_git_sync_remote_name(remote):
        return ""
    return remote


def _git_clear_push_default_remote(repo: Path) -> None:
    """Clear ``remote.pushDefault`` when set; ignore missing-key cases."""
    result = _run_git_sync_command(repo, "config", "--unset", "remote.pushDefault")
    if result.returncode == 0:
        return
    detail = _extract_git_process_error(result, "git config --unset remote.pushDefault failed")
    lowered = detail.lower()
    if "no such section or key" in lowered:
        return
    raise RuntimeError(detail)


def _git_sync_remotes_payload(repo: Path) -> dict[str, object]:
    """Return configured remotes plus default/tracking metadata for remote management UI."""
    status = _git_sync_status_payload(repo)
    tracking_remote = str(status.get("tracking_remote") or "").strip()
    if not tracking_remote:
        tracking_remote = _extract_tracking_remote_name(str(status.get("tracking_branch") or ""))
    configured_default = _git_configured_push_default_remote(repo)

    names_result = _run_git_sync_command(repo, "remote")
    if names_result.returncode != 0:
        raise RuntimeError(_extract_git_process_error(names_result, "git remote failed"))

    names = sorted(
        {
            raw.strip()
            for raw in str(names_result.stdout or "").splitlines()
            if raw.strip()
        },
        key=lambda item: item.casefold(),
    )
    name_set = set(names)

    default_remote = ""
    default_remote_source = "none"
    if configured_default and configured_default in name_set:
        default_remote = configured_default
        default_remote_source = "config"
    elif tracking_remote and tracking_remote in name_set:
        default_remote = tracking_remote
        default_remote_source = "tracking"
    elif "origin" in name_set:
        default_remote = "origin"
        default_remote_source = "origin"

    remotes: list[dict[str, object]] = []
    for name in names:
        fetch_url = _git_remote_url(repo, name)
        push_result = _run_git_sync_command(repo, "remote", "get-url", "--push", name)
        push_url = str(push_result.stdout or "").strip() if push_result.returncode == 0 else fetch_url
        remotes.append(
            {
                "name": name,
                "fetch_url": fetch_url,
                "push_url": push_url,
                "is_default": bool(default_remote and name == default_remote),
                "is_tracking_remote": bool(tracking_remote and name == tracking_remote),
            }
        )

    return {
        "repo_path": str(repo),
        "default_remote": default_remote,
        "default_remote_source": default_remote_source,
        "configured_default_remote": configured_default,
        "configured_default_missing": bool(
            configured_default and configured_default not in name_set
        ),
        "tracking_remote": tracking_remote,
        "remotes": remotes,
        "sync": status,
        "checked_at_epoch_ms": int(time.time() * 1000),
    }


def _classify_git_push_failure(result: subprocess.CompletedProcess[str]) -> str:
    """Classify common git push failures for explicit UX messaging."""
    text = f"{result.stderr or ''}\n{result.stdout or ''}".lower()
    auth_tokens = (
        "authentication failed",
        "invalid username or token",
        "permission denied",
        "could not read username",
        "could not read password",
        "support for password authentication was removed",
        "repository not found",
    )
    if any(token in text for token in auth_tokens):
        return "auth"
    if "non-fast-forward" in text or "fetch first" in text:
        return "non_fast_forward"
    if "failed to push some refs" in text and "rejected" in text:
        return "non_fast_forward"
    if "has no upstream branch" in text or "set-upstream" in text:
        return "upstream_missing"
    return "unknown"


def _git_push_recovery_steps(*, error_type: str, remote: str, branch: str) -> list[str]:
    """Provide concise guided recovery steps for common push failures."""
    push_cmd = f"git push {remote} {branch}"
    push_upstream_cmd = f"git push --set-upstream {remote} {branch}"
    if error_type == "auth":
        return [
            "Verify credentials in the GitHub Auth modal and rerun Test connection.",
            "HTTPS/PAT: confirm scopes/repository access (classic token: repo; fine-grained token: Contents + Pull requests write).",
            "SSH: refresh github.com known_hosts trust and ensure private key permissions are restricted before retrying.",
            f"Retry push after fixing auth: {push_cmd}",
        ]
    if error_type == "non_fast_forward":
        return [
            "Run Fetch, then Pull (or Stash + Pull if your worktree is dirty) to integrate remote changes.",
            "Resolve merge/rebase conflicts locally and commit the result.",
            f"Retry push: {push_cmd}",
        ]
    if error_type == "upstream_missing":
        return [
            "Enable set-upstream in Push and retry.",
            f"Equivalent command: {push_upstream_cmd}",
        ]
    return [
        "Inspect stderr/stdout from git push for the exact rejection reason.",
        "Run Fetch to refresh remote refs, then retry push.",
    ]


def _git_push_error_status_and_label(error_type: str) -> tuple[int, str]:
    """Map push failure classification to API status code + label."""
    if error_type == "auth":
        return 401, "authentication/authorization"
    if error_type == "non_fast_forward":
        return 409, "non-fast-forward"
    if error_type == "upstream_missing":
        return 409, "missing upstream tracking"
    return 502, "unexpected"


def _git_config_get_value(repo: Path, key: str) -> str:
    """Return repo-local git config value, or empty string when unset/missing."""
    result = _run_git_sync_command(repo, "config", "--get", key)
    if result.returncode != 0:
        return ""
    return str(result.stdout or "").strip()


def _git_config_set_value(repo: Path, key: str, value: str) -> None:
    """Set repo-local git config key/value, raising on failure."""
    result = _run_git_sync_command(repo, "config", "--local", key, value)
    if result.returncode == 0:
        return
    detail = _extract_git_process_error(result, f"git config --local {key} failed")
    raise RuntimeError(detail)


def _git_config_set_bool(repo: Path, key: str, value: bool) -> None:
    """Set repo-local git boolean config in canonical true/false string form."""
    _git_config_set_value(repo, key, "true" if value else "false")


def _git_config_unset_value(repo: Path, key: str) -> None:
    """Unset repo-local git config key when present; ignore missing-key cases."""
    result = _run_git_sync_command(repo, "config", "--unset", key)
    if result.returncode == 0:
        return
    detail = _extract_git_process_error(result, f"git config --unset {key} failed")
    if "no such section or key" in detail.lower():
        return
    raise RuntimeError(detail)


def _git_config_truthy(value: object) -> bool:
    """Return git-style boolean interpretation for config values."""
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_git_signing_mode(value: object) -> str:
    """Normalize input/config signing mode to 'gpg' or 'ssh'."""
    raw = str(value or "").strip().lower()
    if raw == "ssh":
        return "ssh"
    # Git stores GPG format as openpgp/x509/ssh. We currently expose openpgp as "gpg".
    return "gpg"


def _resolve_signing_key_path(repo: Path, signing_key: str) -> Path:
    """Resolve a signing-key path relative to repo when not absolute."""
    key_path = Path(signing_key).expanduser()
    if key_path.is_absolute():
        return key_path
    return (repo / key_path).resolve()


def _looks_like_inline_ssh_key(signing_key: str) -> bool:
    """Return True when value looks like an inline SSH public key string."""
    raw = str(signing_key or "").strip()
    if not raw:
        return False
    parts = [part for part in raw.split() if part]
    if len(parts) < 2:
        return False
    return parts[0].startswith("ssh-")


def _git_signing_settings(repo: Path) -> dict[str, object]:
    """Return Git signing settings for the selected repository."""
    gpg_format = str(_git_config_get_value(repo, "gpg.format") or "openpgp").strip().lower()
    mode = _normalize_git_signing_mode(gpg_format)
    commit_sign = _git_config_truthy(_git_config_get_value(repo, "commit.gpgsign"))
    tag_sign = _git_config_truthy(_git_config_get_value(repo, "tag.gpgSign"))
    enabled = bool(commit_sign or tag_sign)
    signing_key = _git_config_get_value(repo, "user.signingkey")
    require_push_guard_raw = _git_config_get_value(repo, _GIT_SIGNING_PUSH_GUARD_KEY)
    if require_push_guard_raw:
        require_push_guard = _git_config_truthy(require_push_guard_raw)
    else:
        require_push_guard = enabled

    return {
        "repo_path": str(repo),
        "mode": mode,
        "gpg_format": gpg_format or "openpgp",
        "commit_sign": commit_sign,
        "tag_sign": tag_sign,
        "enabled": enabled,
        "signing_key": signing_key,
        "has_signing_key": bool(signing_key),
        "require_push_guard": require_push_guard,
        "checked_at_epoch_ms": int(time.time() * 1000),
    }


def _git_signing_validation(repo: Path, settings: dict[str, object]) -> dict[str, object]:
    """Validate configured git signing settings (GPG/SSH) for this repository."""
    checks: list[dict[str, object]] = []
    issues: list[str] = []

    mode = _normalize_git_signing_mode(settings.get("mode"))
    enabled = bool(settings.get("enabled"))
    signing_key = str(settings.get("signing_key") or "").strip()

    checks.append(
        {
            "key": "signing_enabled",
            "label": "Signing enabled",
            "ok": enabled,
            "detail": (
                "Commit/tag signing is enabled for this repository."
                if enabled
                else "Commit/tag signing is disabled."
            ),
        }
    )
    if not enabled:
        return {
            "mode": mode,
            "valid": True,
            "issues": [],
            "checks": checks,
            "message": "Commit and tag signing are currently disabled.",
            "checked_at_epoch_ms": int(time.time() * 1000),
        }

    checks.append(
        {
            "key": "mode_supported",
            "label": "Signing mode",
            "ok": mode in {"gpg", "ssh"},
            "detail": f"Configured signing mode: {mode}",
        }
    )
    if mode not in {"gpg", "ssh"}:
        issues.append(f"Unsupported signing mode '{mode}'. Use gpg or ssh.")

    if mode == "gpg":
        gpg_bin = shutil.which("gpg")
        has_gpg = bool(gpg_bin)
        checks.append(
            {
                "key": "gpg_binary",
                "label": "GPG binary",
                "ok": has_gpg,
                "detail": (
                    f"gpg binary detected at {gpg_bin}"
                    if has_gpg
                    else "gpg is not installed or not on PATH."
                ),
            }
        )
        if not has_gpg:
            issues.append("gpg binary is not available on PATH.")
        else:
            list_args = ["gpg", "--list-secret-keys", "--with-colons"]
            if signing_key:
                list_args.append(signing_key)
            try:
                probe = subprocess.run(
                    list_args,
                    capture_output=True,
                    text=True,
                    timeout=_GIT_SIGNING_CHECK_TIMEOUT_SECONDS,
                    check=False,
                    **_subprocess_isolation_kwargs(),
                )
                has_secret = any(
                    line.startswith("sec:")
                    for line in str(probe.stdout or "").splitlines()
                )
                checks.append(
                    {
                        "key": "gpg_secret_key",
                        "label": "GPG secret key",
                        "ok": bool(probe.returncode == 0 and has_secret),
                        "detail": (
                            "Matching GPG secret key is available."
                            if probe.returncode == 0 and has_secret
                            else (
                                f"Could not find a matching GPG secret key for '{signing_key}'."
                                if signing_key
                                else "No default GPG secret key is available."
                            )
                        ),
                    }
                )
                if probe.returncode != 0 or not has_secret:
                    issues.append(
                        (
                            f"GPG key '{signing_key}' is not available."
                            if signing_key
                            else "No GPG secret key is available for signing."
                        )
                    )
            except subprocess.TimeoutExpired:
                checks.append(
                    {
                        "key": "gpg_secret_key",
                        "label": "GPG secret key",
                        "ok": False,
                        "detail": "Timed out while checking GPG secret keys.",
                    }
                )
                issues.append("Timed out while validating GPG signing keys.")

    if mode == "ssh":
        ssh_keygen_bin = shutil.which("ssh-keygen")
        has_ssh_keygen = bool(ssh_keygen_bin)
        checks.append(
            {
                "key": "ssh_keygen_binary",
                "label": "ssh-keygen binary",
                "ok": has_ssh_keygen,
                "detail": (
                    f"ssh-keygen detected at {ssh_keygen_bin}"
                    if has_ssh_keygen
                    else "ssh-keygen is not installed or not on PATH."
                ),
            }
        )
        if not has_ssh_keygen:
            issues.append("ssh-keygen binary is not available on PATH.")

        if not signing_key:
            checks.append(
                {
                    "key": "ssh_signing_key",
                    "label": "SSH signing key",
                    "ok": False,
                    "detail": "user.signingkey is not configured.",
                }
            )
            issues.append("user.signingkey is required for SSH commit/tag signing.")
        elif _looks_like_inline_ssh_key(signing_key):
            checks.append(
                {
                    "key": "ssh_signing_key",
                    "label": "SSH signing key",
                    "ok": True,
                    "detail": "Inline SSH public key is configured.",
                }
            )
        else:
            key_path = _resolve_signing_key_path(repo, signing_key)
            path_exists = key_path.is_file()
            checks.append(
                {
                    "key": "ssh_signing_key_path",
                    "label": "SSH signing key path",
                    "ok": path_exists,
                    "detail": (
                        f"Signing key path exists: {key_path}"
                        if path_exists
                        else f"Signing key path not found: {key_path}"
                    ),
                }
            )
            if not path_exists:
                issues.append(f"SSH signing key path does not exist: {key_path}")
            else:
                key_looks_valid = False
                with suppress(OSError):
                    preview = _read_text_utf8_resilient(key_path)[:200].strip()
                    if preview.startswith("ssh-") or "BEGIN OPENSSH PRIVATE KEY" in preview:
                        key_looks_valid = True
                checks.append(
                    {
                        "key": "ssh_signing_key_format",
                        "label": "SSH key format",
                        "ok": key_looks_valid,
                        "detail": (
                            "Signing key file looks like an SSH key."
                            if key_looks_valid
                            else "Signing key file does not look like an SSH key."
                        ),
                    }
                )
                if not key_looks_valid:
                    issues.append(
                        f"Signing key file is not recognized as an SSH key: {key_path}"
                    )

    valid = not issues
    return {
        "mode": mode,
        "valid": valid,
        "issues": issues,
        "checks": checks,
        "message": (
            "Signing configuration looks valid."
            if valid
            else "Signing configuration has issues."
        ),
        "checked_at_epoch_ms": int(time.time() * 1000),
    }


def _git_signing_payload(
    repo: Path,
    *,
    settings: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return git signing settings and validation for setup/push guardrails."""
    settings_payload = dict(settings) if isinstance(settings, dict) else _git_signing_settings(repo)
    validation = _git_signing_validation(repo, settings_payload)
    payload = dict(settings_payload)
    payload.update(
        {
            "valid": bool(validation.get("valid")),
            "issues": list(validation.get("issues", [])),
            "checks": list(validation.get("checks", [])),
            "validation": validation,
            "message": str(validation.get("message") or ""),
            "checked_at_epoch_ms": int(time.time() * 1000),
        }
    )
    return payload


def _git_remote_url(repo: Path, remote: str) -> str:
    """Return the configured URL for a git remote, or empty string when unknown."""
    result = _run_git_sync_command(repo, "remote", "get-url", remote)
    if result.returncode != 0:
        return ""
    return str(result.stdout or "").strip()


def _github_remote_owner_repo(remote_url: str) -> tuple[str, str]:
    """Return ``(owner, repo)`` for github.com remotes, else ``("", "")``."""
    raw = str(remote_url or "").strip()
    if not raw:
        return "", ""

    host = ""
    path = ""
    parsed = urlparse(raw)
    if parsed.scheme and parsed.netloc:
        host = str(parsed.hostname or parsed.netloc).strip().lower()
        path = str(parsed.path or "").strip()
    elif "://" not in raw and ":" in raw:
        left, right = raw.split(":", 1)
        # Handle scp-like SSH remotes (git@github.com:owner/repo.git)
        if "@" in left and right.strip():
            host = left.split("@", 1)[1].strip().lower()
            path = "/" + right.strip()

    if host != "github.com":
        return "", ""

    normalized = path.strip().strip("/")
    if normalized.lower().endswith(".git"):
        normalized = normalized[:-4]
    parts = [part.strip() for part in normalized.split("/") if part.strip()]
    if len(parts) < 2:
        return "", ""
    return parts[0], parts[1]


def _github_repo_web_base(remote_url: str) -> str:
    """Return ``https://github.com/<owner>/<repo>`` for GitHub remotes, else empty string."""
    owner, repo_name = _github_remote_owner_repo(remote_url)
    if not owner or not repo_name:
        return ""
    return f"https://github.com/{owner}/{repo_name}"


def _github_repo_metadata_cache_key(owner: str, repo_name: str, *, authenticated: bool) -> str:
    """Build a stable cache key for GitHub repo metadata lookups."""
    owner_key = str(owner or "").strip().casefold()
    repo_key = str(repo_name or "").strip().casefold()
    auth_key = "auth" if authenticated else "anon"
    return f"{owner_key}/{repo_key}|{auth_key}"


def _github_repo_metadata_cache_get(cache_key: str) -> tuple[dict[str, object] | None, str] | None:
    """Return cached repo metadata when fresh, else ``None``."""
    now = time.time()
    with _github_repo_metadata_cache_lock:
        entry = _github_repo_metadata_cache.get(cache_key)
        if not entry:
            return None
        expires_at, payload, error = entry
        if expires_at <= now:
            _github_repo_metadata_cache.pop(cache_key, None)
            return None
    return (dict(payload) if isinstance(payload, dict) else None, str(error or ""))


def _github_repo_metadata_cache_set(
    cache_key: str,
    *,
    payload: dict[str, object] | None,
    error: str,
) -> None:
    """Store GitHub repo metadata lookup result with success/error TTLs."""
    if str(error or "").strip():
        ttl_seconds = _GITHUB_REPO_METADATA_ERROR_CACHE_TTL_SECONDS
    else:
        ttl_seconds = _GITHUB_REPO_METADATA_CACHE_TTL_SECONDS
    expires_at = time.time() + max(1, int(ttl_seconds))
    with _github_repo_metadata_cache_lock:
        _github_repo_metadata_cache[cache_key] = (
            expires_at,
            dict(payload) if isinstance(payload, dict) else None,
            str(error or "").strip(),
        )


def _github_repo_metadata_from_api(
    owner: str,
    repo_name: str,
    *,
    token: str = "",
) -> tuple[dict[str, object] | None, str]:
    """Fetch repository metadata from the GitHub REST API."""
    owner_value = str(owner or "").strip()
    repo_value = str(repo_name or "").strip()
    if not owner_value or not repo_value:
        return None, "GitHub owner/repository is missing."

    token_value = str(token or "").strip()
    cache_key = _github_repo_metadata_cache_key(
        owner_value,
        repo_value,
        authenticated=bool(token_value),
    )
    cached = _github_repo_metadata_cache_get(cache_key)
    if cached is not None:
        return cached

    url = (
        "https://api.github.com/repos/"
        f"{quote(owner_value, safe='')}/{quote(repo_value, safe='')}"
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "warpfoundry-git-sync-metadata",
    }
    if token_value:
        headers["Authorization"] = f"Bearer {token_value}"

    request_obj = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request_obj, timeout=_GITHUB_REPO_METADATA_TIMEOUT_SECONDS) as response:
            payload_text = response.read().decode("utf-8", errors="replace")
        parsed = json.loads(payload_text) if payload_text else {}
        if not isinstance(parsed, dict):
            parsed = {}

        is_private = bool(parsed.get("private"))
        visibility = str(parsed.get("visibility") or "").strip().lower()
        if not visibility:
            visibility = "private" if is_private else "public"
        metadata = {
            "name": str(parsed.get("name") or repo_value).strip() or repo_value,
            "full_name": str(parsed.get("full_name") or f"{owner_value}/{repo_value}").strip()
            or f"{owner_value}/{repo_value}",
            "owner": owner_value,
            "repo": repo_value,
            "url": str(parsed.get("html_url") or f"https://github.com/{owner_value}/{repo_value}").strip()
            or f"https://github.com/{owner_value}/{repo_value}",
            "default_branch": str(parsed.get("default_branch") or "").strip(),
            "visibility": visibility,
            "private": is_private,
            "api_ok": True,
        }
        _github_repo_metadata_cache_set(cache_key, payload=metadata, error="")
        return metadata, ""
    except HTTPError as exc:
        detail = ""
        with suppress(Exception):
            body = exc.read().decode("utf-8", errors="replace")
            parsed = json.loads(body) if body else {}
            if isinstance(parsed, dict):
                detail = str(parsed.get("message") or "").strip()

        if exc.code == 404:
            message = (
                "GitHub metadata lookup returned 404 (repository may be private, missing, "
                "or inaccessible with current credentials)."
            )
        elif exc.code == 403 and "rate limit" in detail.lower():
            message = "GitHub metadata lookup hit the API rate limit."
        elif exc.code == 403:
            message = "GitHub metadata lookup returned 403 (forbidden)."
        else:
            message = f"GitHub metadata lookup returned HTTP {exc.code}."
        if detail:
            message += f" {detail[:200]}"
        _github_repo_metadata_cache_set(cache_key, payload=None, error=message)
        return None, message
    except URLError as exc:
        reason = str(getattr(exc, "reason", exc) or "").strip()
        message = (
            "Could not reach api.github.com for metadata."
            if not reason
            else f"Could not reach api.github.com for metadata: {reason}"
        )
        _github_repo_metadata_cache_set(cache_key, payload=None, error=message)
        return None, message
    except Exception as exc:
        message = f"GitHub metadata lookup failed: {exc}"
        _github_repo_metadata_cache_set(cache_key, payload=None, error=message)
        return None, message


def _git_remote_default_branch(repo: Path, remote: str) -> str:
    """Resolve ``<remote>/HEAD`` to the remote's default branch when available."""
    result = _run_git_sync_command(repo, "symbolic-ref", "--short", f"refs/remotes/{remote}/HEAD")
    if result.returncode != 0:
        return ""
    remote_head = str(result.stdout or "").strip()
    prefix = f"{remote}/"
    if remote_head.startswith(prefix):
        return remote_head[len(prefix) :].strip()
    return remote_head.rsplit("/", 1)[-1].strip()


def _git_push_pull_request_payload(*, repo: Path, remote: str, head_branch: str) -> dict[str, object]:
    """Build pull-request helper payload for GitHub remotes after push."""
    head = str(head_branch or "").strip()
    remote_url = _git_remote_url(repo, remote)
    repo_web_base = _github_repo_web_base(remote_url)
    default_branch = _git_remote_default_branch(repo, remote) if repo_web_base else ""
    payload: dict[str, object] = {
        "provider": "github",
        "remote": remote,
        "remote_url": remote_url,
        "base_branch": default_branch or None,
        "head_branch": head or None,
        "available": False,
        "url": "",
    }

    if not repo_web_base:
        payload["reason"] = "Remote is not a github.com repository URL."
        return payload
    if not head:
        payload["reason"] = "Current branch is unavailable."
        return payload

    if default_branch and default_branch != head:
        compare_ref = f"{quote(default_branch, safe='')}...{quote(head, safe='')}"
    else:
        compare_ref = quote(head, safe="")
    payload["url"] = f"{repo_web_base}/compare/{compare_ref}?expand=1"
    payload["available"] = True
    return payload


def _git_name_only_paths(repo: Path, *args: str) -> set[str]:
    """Return unique file paths from a git command that outputs one path per line."""
    result = _run_git_sync_command(repo, *args)
    if result.returncode != 0:
        detail = _extract_git_process_error(result, f"git {' '.join(args)} failed")
        raise RuntimeError(detail)

    paths: set[str] = set()
    for raw_line in str(result.stdout or "").splitlines():
        path = raw_line.strip()
        if path:
            paths.add(path)
    return paths


def _git_last_commit_summary(repo: Path) -> dict[str, object]:
    """Return last-commit metadata for commit workflow UIs."""
    log_format = "%H%x1f%h%x1f%an%x1f%ae%x1f%aI%x1f%s"
    result = _run_git_sync_command(repo, "log", "-1", f"--pretty=format:{log_format}")
    if result.returncode != 0:
        detail = _extract_git_process_error(result, "git log -1 failed")
        lowered = detail.lower()
        no_commit_tokens = (
            "does not have any commits yet",
            "unknown revision or path not in the working tree",
            "ambiguous argument 'head'",
        )
        if any(token in lowered for token in no_commit_tokens):
            return {
                "available": False,
                "hash": "",
                "short_hash": "",
                "author_name": "",
                "author_email": "",
                "authored_at": "",
                "authored_at_epoch_ms": None,
                "subject": "",
            }
        raise RuntimeError(detail)

    parts = str(result.stdout or "").split("\x1f")
    while len(parts) < 6:
        parts.append("")
    commit_hash, short_hash, author_name, author_email, authored_at, subject = [
        str(part or "").strip() for part in parts[:6]
    ]

    authored_at_epoch_ms: int | None = None
    if authored_at:
        normalized = authored_at.replace("Z", "+00:00")
        with suppress(ValueError):
            authored_at_epoch_ms = int(datetime.fromisoformat(normalized).timestamp() * 1000)

    return {
        "available": bool(commit_hash),
        "hash": commit_hash,
        "short_hash": short_hash,
        "author_name": author_name,
        "author_email": author_email,
        "authored_at": authored_at,
        "authored_at_epoch_ms": authored_at_epoch_ms,
        "subject": subject,
    }


def _git_commit_workflow_payload(repo: Path) -> dict[str, object]:
    """Return changed-file stage state plus last commit summary for commit workflows."""
    staged_paths = _git_name_only_paths(repo, "diff", "--name-only", "--cached")
    unstaged_paths = _git_name_only_paths(repo, "diff", "--name-only")
    untracked_paths = _git_name_only_paths(repo, "ls-files", "--others", "--exclude-standard")
    all_paths = sorted(staged_paths | unstaged_paths | untracked_paths, key=lambda item: item.casefold())

    files: list[dict[str, object]] = []
    for path in all_paths:
        staged = path in staged_paths
        unstaged = path in unstaged_paths
        untracked = path in untracked_paths
        files.append(
            {
                "path": path,
                "staged": staged,
                "unstaged": unstaged,
                "untracked": untracked,
                "can_stage": untracked or unstaged,
                "can_unstage": staged,
            }
        )

    stageable_paths = [str(item["path"]) for item in files if bool(item.get("can_stage"))]
    unstageable_paths = [str(item["path"]) for item in files if bool(item.get("can_unstage"))]
    counts = {
        "staged": len(staged_paths),
        "unstaged": len(unstaged_paths),
        "untracked": len(untracked_paths),
        "total_changed": len(all_paths),
    }

    return {
        "repo_path": str(repo),
        "files": files,
        "counts": counts,
        "stageable_paths": stageable_paths,
        "unstageable_paths": unstageable_paths,
        "has_changes": bool(all_paths),
        "has_staged_changes": bool(staged_paths),
        "has_stageable_changes": bool(stageable_paths),
        "last_commit": _git_last_commit_summary(repo),
        "checked_at_epoch_ms": int(time.time() * 1000),
    }


def _normalize_git_commit_paths(raw_paths: object) -> list[str]:
    """Normalize and validate commit-workflow file path selections."""
    if raw_paths is None:
        return []
    if isinstance(raw_paths, str):
        values = [raw_paths]
    elif isinstance(raw_paths, (list, tuple, set)):
        values = list(raw_paths)
    else:
        raise ValueError("paths must be a string or a list of strings.")

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        path = str(raw_value or "").strip()
        if not path:
            continue
        if "\x00" in path:
            raise ValueError("paths entries may not include NUL bytes.")
        if Path(path).is_absolute():
            raise ValueError(f"Absolute paths are not allowed: {path}")
        if path.startswith("-"):
            raise ValueError(f"Invalid path (cannot start with '-'): {path}")
        if path in seen:
            continue
        seen.add(path)
        normalized.append(path)
    return normalized


def _resolve_git_commit_target_paths(
    *,
    available_paths: set[str],
    requested_paths: list[str],
    include_all: bool,
    action_label: str,
) -> list[str]:
    """Resolve selected paths for stage/unstage actions from request payload."""
    if include_all:
        targets = sorted(available_paths, key=lambda item: item.casefold())
        if not targets:
            raise ValueError(f"No files available to {action_label}.")
        return targets

    if not requested_paths:
        raise ValueError("paths is required unless all=true.")

    invalid = [path for path in requested_paths if path not in available_paths]
    if invalid:
        preview = ", ".join(invalid[:3])
        suffix = " ..." if len(invalid) > 3 else ""
        raise ValueError(
            f"Cannot {action_label} paths outside the current change set: {preview}{suffix}"
        )
    return requested_paths


def _git_commit_user_error_status(message: str) -> int:
    """Map commit-workflow user input errors to API status codes."""
    lowered = str(message or "").lower()
    if "current change set" in lowered or "no files available" in lowered:
        return 409
    return 400


def _git_stage_paths(repo: Path, paths: list[str]) -> subprocess.CompletedProcess[str]:
    """Stage selected paths using command-length-safe batching."""
    batches = _batch_git_paths_for_command(prefix_args=("add", "--"), paths=paths)
    last_result: subprocess.CompletedProcess[str] | None = None
    for batch in batches:
        stage_result = _run_git_sync_command(repo, "add", "--", *batch)
        if stage_result.returncode != 0:
            return stage_result
        last_result = stage_result
    if last_result is not None:
        return last_result
    return _run_git_sync_command(repo, "add", "--")


def _git_unstage_paths(repo: Path, paths: list[str]) -> subprocess.CompletedProcess[str]:
    """Unstage paths (with batching) and fallback for unborn HEAD repositories."""
    batches = _batch_git_paths_for_command(prefix_args=("restore", "--staged", "--"), paths=paths)
    last_result: subprocess.CompletedProcess[str] | None = None
    for batch in batches:
        restore_result = _run_git_sync_command(repo, "restore", "--staged", "--", *batch)
        if restore_result.returncode == 0:
            last_result = restore_result
            continue

        fallback_result = _run_git_sync_command(repo, "rm", "--cached", "--quiet", "--", *batch)
        if fallback_result.returncode == 0:
            last_result = fallback_result
            continue
        return restore_result

    if last_result is not None:
        return last_result
    return _run_git_sync_command(repo, "restore", "--staged", "--")


def _git_remote_names(repo: Path) -> list[str]:
    """Return configured git remote names (best effort)."""
    result = _run_git_sync_command(repo, "remote")
    if result.returncode != 0:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for raw_line in str(result.stdout or "").splitlines():
        name = raw_line.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _git_sync_preferred_remote_name(repo: Path, status_payload: dict[str, object]) -> str:
    """Pick a default remote for repo metadata and link helpers."""
    remote_names = _git_remote_names(repo)
    if not remote_names:
        return ""

    remote_set = set(remote_names)
    tracking_remote = str(status_payload.get("tracking_remote") or "").strip()
    configured_default = _git_configured_push_default_remote(repo)

    for candidate in (tracking_remote, configured_default, "origin"):
        if candidate and candidate in remote_set:
            return candidate
    return remote_names[0]


def _git_sync_github_repo_payload(repo: Path, status_payload: dict[str, object]) -> dict[str, object]:
    """Resolve GitHub repository metadata for sync-status UI presentation."""
    remote = _git_sync_preferred_remote_name(repo, status_payload)
    payload: dict[str, object] = {
        "provider": "github",
        "remote": remote,
        "remote_url": "",
        "detected": False,
        "available": False,
        "owner": "",
        "repo": "",
        "name": "",
        "full_name": "",
        "visibility": "",
        "default_branch": "",
        "url": "",
        "source": "none",
        "api_ok": False,
    }
    if not remote:
        payload["reason"] = "No git remote is configured."
        return payload

    remote_url = _git_remote_url(repo, remote)
    payload["remote_url"] = remote_url
    owner, repo_name = _github_remote_owner_repo(remote_url)
    if not owner or not repo_name:
        payload["reason"] = "Selected remote is not a github.com repository URL."
        return payload

    repo_web_url = f"https://github.com/{owner}/{repo_name}"
    payload.update(
        {
            "detected": True,
            "available": True,
            "owner": owner,
            "repo": repo_name,
            "name": repo_name,
            "full_name": f"{owner}/{repo_name}",
            "visibility": "unknown",
            "url": repo_web_url,
            "source": "remote",
        }
    )

    default_branch = _git_remote_default_branch(repo, remote)
    if default_branch:
        payload["default_branch"] = default_branch

    token = ""
    with suppress(RuntimeError):
        token = _github_secret_get(_GITHUB_PAT_SECRET_KEY).strip()

    api_payload, api_error = _github_repo_metadata_from_api(owner, repo_name, token=token)
    if api_payload is not None:
        payload.update(api_payload)
        payload["source"] = "api"
    elif api_error:
        payload["reason"] = api_error

    return payload


def _git_sync_status_core_payload(repo: Path) -> dict[str, object]:
    """Return git sync metadata without optional GitHub API enrichment."""
    branch_result = _run_git_sync_command(repo, "rev-parse", "--abbrev-ref", "HEAD")
    if branch_result.returncode != 0:
        raise RuntimeError(
            _extract_git_process_error(branch_result, "git rev-parse --abbrev-ref HEAD failed")
        )
    branch = str(branch_result.stdout or "").strip() or "HEAD"

    tracking_result = _run_git_sync_command(
        repo,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
    )
    has_tracking_branch = tracking_result.returncode == 0
    tracking_branch = str(tracking_result.stdout or "").strip() if has_tracking_branch else ""
    tracking_remote = _extract_tracking_remote_name(tracking_branch)

    ahead: int | None = None
    behind: int | None = None
    if has_tracking_branch:
        counts_result = _run_git_sync_command(repo, "rev-list", "--left-right", "--count", "@{upstream}...HEAD")
        if counts_result.returncode == 0:
            parts = str(counts_result.stdout or "").strip().split()
            if len(parts) >= 2:
                with suppress(ValueError):
                    behind = int(parts[0])
                    ahead = int(parts[1])

    status_result = _run_git_sync_command(repo, "status", "--porcelain")
    if status_result.returncode != 0:
        raise RuntimeError(_extract_git_process_error(status_result, "git status --porcelain failed"))

    staged_changes = 0
    unstaged_changes = 0
    untracked_changes = 0
    for line in str(status_result.stdout or "").splitlines():
        if not line:
            continue
        if line.startswith("??"):
            untracked_changes += 1
            continue
        if line.startswith("!!"):
            continue
        x = line[0] if len(line) >= 1 else " "
        y = line[1] if len(line) >= 2 else " "
        if x not in {" ", "?"}:
            staged_changes += 1
        if y != " ":
            unstaged_changes += 1

    last_fetch_epoch_ms, last_fetch_at = _git_last_fetch_metadata(repo)
    dirty = bool(staged_changes or unstaged_changes or untracked_changes)
    return {
        "repo_path": str(repo),
        "branch": branch,
        "tracking_branch": tracking_branch,
        "tracking_remote": tracking_remote,
        "has_tracking_branch": has_tracking_branch,
        "ahead": ahead,
        "behind": behind,
        "dirty": dirty,
        "clean": not dirty,
        "staged_changes": staged_changes,
        "unstaged_changes": unstaged_changes,
        "untracked_changes": untracked_changes,
        "last_fetch_epoch_ms": last_fetch_epoch_ms,
        "last_fetch_at": last_fetch_at,
        "checked_at_epoch_ms": int(time.time() * 1000),
    }


def _git_sync_status_payload(repo: Path, *, force_refresh: bool = False) -> dict[str, object]:
    """Return branch/tracking/ahead-behind/dirty metadata for a repository."""
    _ = force_refresh
    payload = _git_sync_status_core_payload(repo)
    payload["github_repo"] = _git_sync_github_repo_payload(repo, payload)
    return payload


def _git_ref_exists(repo: Path, ref_name: str) -> bool:
    """Return True when a git ref exists in the repository."""
    probe = _run_git_sync_command(repo, "show-ref", "--verify", "--quiet", ref_name)
    return probe.returncode == 0


def _git_sync_branch_choices_payload(
    repo: Path,
    *,
    force_refresh: bool = False,
) -> dict[str, object]:
    """Return local/remote branch choices plus current branch metadata."""
    cache_key = str(repo.resolve())
    if not force_refresh:
        with _git_sync_branch_choices_cache_lock:
            cached = _git_sync_branch_choices_cache.get(cache_key)
        if cached is not None:
            cached_at, cached_payload = cached
            if (time.monotonic() - cached_at) <= _GIT_SYNC_BRANCH_CHOICES_CACHE_TTL_SECONDS:
                return cached_payload

    status = _git_sync_status_payload(repo, force_refresh=force_refresh)
    current_branch = str(status.get("branch") or "").strip() or "HEAD"

    local_result = _run_git_sync_command(
        repo,
        "for-each-ref",
        "--format=%(refname:short)",
        "refs/heads",
    )
    if local_result.returncode != 0:
        raise RuntimeError(_extract_git_process_error(local_result, "git for-each-ref refs/heads failed"))

    local_seen: set[str] = set()
    for raw_line in str(local_result.stdout or "").splitlines():
        branch = raw_line.strip()
        if branch:
            local_seen.add(branch)
    local_branches = sorted(local_seen, key=lambda item: item.casefold())

    remote_result = _run_git_sync_command(
        repo,
        "for-each-ref",
        "--format=%(refname:short)",
        "refs/remotes",
    )
    if remote_result.returncode != 0:
        raise RuntimeError(
            _extract_git_process_error(remote_result, "git for-each-ref refs/remotes failed")
        )

    remote_seen: set[str] = set()
    for raw_line in str(remote_result.stdout or "").splitlines():
        branch = raw_line.strip()
        if not branch:
            continue
        if branch == "HEAD" or branch.endswith("/HEAD"):
            continue
        remote_seen.add(branch)
    remote_branches = sorted(remote_seen, key=lambda item: item.casefold())

    payload = {
        "repo_path": str(repo),
        "current_branch": current_branch,
        "detached_head": current_branch == "HEAD",
        "local_branches": local_branches,
        "remote_branches": remote_branches,
        "sync": status,
        "checked_at_epoch_ms": int(time.time() * 1000),
    }
    with _git_sync_branch_choices_cache_lock:
        _bounded_cache_set(
            _git_sync_branch_choices_cache,
            cache_key,
            (time.monotonic(), payload),
            max_entries=64,
        )
    return payload


def _git_sync_dirty_recovery_steps(*, action: str) -> list[str]:
    """Return dirty-worktree recovery guidance for branch operations."""
    return [
        "Commit or stash your local changes before switching branch context.",
        "Use Stash + Pull in the header (or run `git stash push --include-untracked`) and retry.",
        f"If you intentionally need to {action} with local edits, enable 'allow dirty switch' and retry.",
    ]


def _git_sync_dirty_guardrail_response(
    *,
    repo: Path,
    status_payload: dict[str, object],
    action: str,
) -> tuple[Response, int]:
    """Build a consistent dirty-worktree guardrail response payload."""
    staged = int(status_payload.get("staged_changes") or 0)
    unstaged = int(status_payload.get("unstaged_changes") or 0)
    untracked = int(status_payload.get("untracked_changes") or 0)
    detail = f"staged {staged}, unstaged {unstaged}, untracked {untracked}"
    return (
        jsonify(
            {
                "error": (
                    "Branch operation blocked because the worktree is dirty "
                    f"({detail}). Commit/stash/discard changes first, or enable allow_dirty."
                ),
                "error_type": "dirty_worktree",
                "repo_path": str(repo),
                "action": action,
                "recovery_steps": _git_sync_dirty_recovery_steps(action=action),
                "sync": status_payload,
            }
        ),
        409,
    )


def _git_preflight_before_run(
    repo: Path,
    *,
    auto_stash: bool,
    auto_pull: bool,
) -> dict[str, object]:
    """Run optional git pre-flight checks/actions before starting a run."""
    checks: list[dict[str, str]] = []
    issues: list[str] = []
    warnings: list[str] = []
    actions: list[str] = []

    def add_check(*, key: str, label: str, status: str, detail: str) -> None:
        checks.append(
            {
                "key": key,
                "label": label,
                "status": status,
                "detail": detail,
            }
        )

    status_before = _git_sync_status_core_payload(repo)
    status_after = dict(status_before)
    branch = str(status_before.get("branch") or "").strip() or "HEAD"
    tracking_branch = str(status_before.get("tracking_branch") or "").strip()
    tracking_remote = str(status_before.get("tracking_remote") or "").strip()

    staged_changes = int(status_before.get("staged_changes") or 0)
    unstaged_changes = int(status_before.get("unstaged_changes") or 0)
    untracked_changes = int(status_before.get("untracked_changes") or 0)
    dirty = bool(status_before.get("dirty"))

    stash_created = False
    stash_ref = ""
    stash_before = ""

    if dirty:
        detail = (
            "Worktree is dirty: staged "
            f"{staged_changes}, unstaged {unstaged_changes}, untracked {untracked_changes}."
        )
        if auto_stash:
            stash_before = _resolve_stash_ref(repo)
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            stash_message = f"warpfoundry:preflight-auto-stash {stamp}"
            stash_result = _run_git_sync_command(
                repo,
                "stash",
                "push",
                "--include-untracked",
                "--message",
                stash_message,
            )
            if stash_result.returncode != 0:
                stash_error = _extract_git_process_error(
                    stash_result,
                    "git stash push --include-untracked failed",
                )
                add_check(
                    key="worktree_state",
                    label="Clean or stashed worktree",
                    status="fail",
                    detail=f"{detail} Auto-stash failed: {stash_error}",
                )
                issues.append(
                    "Git pre-flight auto-stash failed. Commit, stash, or discard local changes and retry."
                )
            else:
                stash_after = _resolve_stash_ref(repo)
                stash_created = bool(stash_after and stash_after != stash_before)
                stash_ref = stash_after if stash_created else ""
                add_check(
                    key="worktree_state",
                    label="Clean or stashed worktree",
                    status="pass",
                    detail=(
                        "Worktree was dirty and was auto-stashed."
                        if stash_created
                        else "Worktree was dirty; auto-stash ran but did not create a new stash ref."
                    ),
                )
                if stash_created:
                    actions.append("Auto-stashed local changes before run.")
        else:
            add_check(
                key="worktree_state",
                label="Clean or stashed worktree",
                status="fail",
                detail=detail,
            )
            issues.append(
                "Git pre-flight blocked the run because the worktree is dirty. "
                "Enable auto-stash or clean the worktree first."
            )
    else:
        add_check(
            key="worktree_state",
            label="Clean or stashed worktree",
            status="pass",
            detail="Worktree is clean.",
        )

    if branch == "HEAD":
        add_check(
            key="branch_validation",
            label="Branch validation",
            status="fail",
            detail="Repository is in detached HEAD state.",
        )
        issues.append(
            "Git pre-flight requires an active branch (not detached HEAD). Check out a branch first."
        )
    else:
        add_check(
            key="branch_validation",
            label="Branch validation",
            status="pass",
            detail=f"Active branch: {branch}",
        )

    if not tracking_branch:
        add_check(
            key="tracking_branch",
            label="Tracking branch configured",
            status="fail",
            detail=f"Branch '{branch}' has no upstream tracking branch.",
        )
        issues.append(
            "Git pre-flight requires an upstream tracking branch. "
            "Set it with 'git push --set-upstream <remote> <branch>'."
        )
    else:
        add_check(
            key="tracking_branch",
            label="Tracking branch configured",
            status="pass",
            detail=f"Tracking branch: {tracking_branch}",
        )

    if tracking_remote:
        remote_probe = _run_git_sync_command(repo, "remote", "get-url", tracking_remote)
        if remote_probe.returncode != 0:
            remote_error = _extract_git_process_error(
                remote_probe,
                f"git remote get-url {tracking_remote} failed",
            )
            add_check(
                key="remote_reachability",
                label="Remote reachability",
                status="fail",
                detail=f"Remote '{tracking_remote}' is not configured: {remote_error}",
            )
            issues.append(
                f"Git pre-flight could not resolve remote '{tracking_remote}'. "
                "Fix git remote settings and retry."
            )
        else:
            reachability_probe = _run_git_sync_command(
                repo,
                "ls-remote",
                "--exit-code",
                tracking_remote,
                "HEAD",
            )
            if reachability_probe.returncode != 0:
                reachability_error = _extract_git_process_error(
                    reachability_probe,
                    f"git ls-remote {tracking_remote} HEAD failed",
                )
                add_check(
                    key="remote_reachability",
                    label="Remote reachability",
                    status="fail",
                    detail=f"Remote '{tracking_remote}' is unreachable: {reachability_error}",
                )
                issues.append(
                    f"Git pre-flight could not reach remote '{tracking_remote}'. "
                    "Check network/auth settings and retry."
                )
            else:
                add_check(
                    key="remote_reachability",
                    label="Remote reachability",
                    status="pass",
                    detail=f"Remote '{tracking_remote}' is reachable.",
                )
    else:
        add_check(
            key="remote_reachability",
            label="Remote reachability",
            status="fail",
            detail="Could not determine tracking remote name.",
        )
        issues.append(
            "Git pre-flight could not determine the tracking remote. "
            "Configure upstream tracking and retry."
        )

    behind_raw = status_before.get("behind")
    behind = int(behind_raw) if isinstance(behind_raw, int) else None
    if behind is not None and behind > 0 and not auto_pull:
        warn_text = f"Local branch is behind upstream by {behind} commit(s); auto-pull is disabled."
        add_check(
            key="upstream_freshness",
            label="Upstream freshness",
            status="warn",
            detail=warn_text,
        )
        warnings.append(warn_text)
    elif behind is not None:
        add_check(
            key="upstream_freshness",
            label="Upstream freshness",
            status="pass",
            detail=f"Behind upstream by {behind} commit(s).",
        )
    else:
        add_check(
            key="upstream_freshness",
            label="Upstream freshness",
            status="warn",
            detail="Ahead/behind counts are unavailable for this branch.",
        )

    if auto_pull:
        pull_result = _run_git_sync_command(repo, "pull", "--ff-only")
        if pull_result.returncode != 0:
            pull_error = _extract_git_process_error(pull_result, "git pull --ff-only failed")
            detail = f"Auto-pull failed: {pull_error}"
            if stash_created:
                detail += " Local changes were stashed before pull."
            add_check(
                key="auto_pull",
                label="Auto-pull latest commits",
                status="fail",
                detail=detail,
            )
            issue = "Git pre-flight auto-pull failed."
            if stash_created:
                issue += " Local changes were stashed; use 'git stash list' to review."
            issues.append(issue)
        else:
            add_check(
                key="auto_pull",
                label="Auto-pull latest commits",
                status="pass",
                detail="Auto-pull succeeded with fast-forward-only mode.",
            )
            actions.append("Pulled latest commits before run.")
    else:
        add_check(
            key="auto_pull",
            label="Auto-pull latest commits",
            status="skip",
            detail="Auto-pull is disabled.",
        )

    if not issues:
        status_after = _git_sync_status_core_payload(repo)

    return {
        "ok": len(issues) == 0,
        "repo_path": str(repo),
        "auto_stash": auto_stash,
        "auto_pull": auto_pull,
        "branch": branch,
        "tracking_branch": tracking_branch,
        "tracking_remote": tracking_remote,
        "stash_created": stash_created,
        "stash_ref": stash_ref,
        "stash_ref_before": stash_before,
        "checks": checks,
        "issues": issues,
        "warnings": warnings,
        "actions": actions,
        "status_before": status_before,
        "status_after": status_after,
        "checked_at_epoch_ms": int(time.time() * 1000),
    }


def _list_git_remote_branches(remote_url: str) -> tuple[list[str], str]:
    """Return (branches, default_branch) for a git remote."""
    symref = subprocess.run(
        ["git", "ls-remote", "--symref", remote_url, "HEAD"],
        capture_output=True,
        text=True,
        timeout=_GIT_REMOTE_QUERY_TIMEOUT_SECONDS,
        check=False,
        **_subprocess_isolation_kwargs(),
    )
    if symref.returncode != 0:
        detail = str(symref.stderr or symref.stdout or "").strip()
        raise RuntimeError(detail or "git ls-remote --symref failed")

    default_branch = ""
    for raw_line in symref.stdout.splitlines():
        line = raw_line.replace("\t", " ").strip()
        match = re.match(r"^ref:\s+refs/heads/(?P<branch>\S+)\s+HEAD$", line)
        if match:
            default_branch = str(match.group("branch") or "").strip()
            break

    heads = subprocess.run(
        ["git", "ls-remote", "--heads", remote_url],
        capture_output=True,
        text=True,
        timeout=_GIT_REMOTE_QUERY_TIMEOUT_SECONDS,
        check=False,
        **_subprocess_isolation_kwargs(),
    )
    if heads.returncode != 0:
        detail = str(heads.stderr or heads.stdout or "").strip()
        raise RuntimeError(detail or "git ls-remote --heads failed")

    seen: set[str] = set()
    for raw_line in heads.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        ref = parts[1] if len(parts) == 2 else ""
        if not ref:
            tokens = line.split()
            if len(tokens) >= 2:
                ref = tokens[1]
        if not ref.startswith("refs/heads/"):
            continue
        branch = ref[len("refs/heads/") :].strip()
        if branch:
            seen.add(branch)

    branches = sorted(seen, key=lambda item: item.casefold())
    if default_branch and default_branch not in seen:
        branches.insert(0, default_branch)

    if not default_branch:
        if "main" in seen:
            default_branch = "main"
        elif "master" in seen:
            default_branch = "master"
        elif branches:
            default_branch = branches[0]

    return branches, default_branch


def _initialize_cloned_repo_codex_manager(repo_path: Path) -> list[str]:
    """Create baseline .codex_manager artifacts for a freshly-cloned repository."""
    created: list[str] = []
    root = repo_path / ".codex_manager"

    for subdir in ("outputs", "state"):
        path = root / subdir
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append((Path(".codex_manager") / subdir).as_posix())

    logs_path = root / "logs"
    protocol_path = root / "AGENT_PROTOCOL.md"
    logs_missing = not logs_path.exists()
    protocol_missing = not protocol_path.exists()

    from codex_manager.pipeline.tracker import LogTracker

    LogTracker(repo_path).initialize()
    if logs_missing:
        created.append(".codex_manager/logs")
    if protocol_missing:
        created.append(".codex_manager/AGENT_PROTOCOL.md")

    todo_path = _todo_wishlist_path(repo_path)
    if not todo_path.is_file():
        _write_todo_wishlist(repo_path, "")
        created.append(todo_path.relative_to(repo_path).as_posix())

    feature_path = _feature_dreams_path(repo_path)
    if not feature_path.is_file():
        _write_feature_dreams(repo_path, "")
        created.append(feature_path.relative_to(repo_path).as_posix())

    board_path = _owner_decision_board_path(repo_path)
    if not board_path.is_file():
        _save_decision_board(repo_path, _load_decision_board(repo_path))
        created.append(board_path.relative_to(repo_path).as_posix())

    return created


@app.route("/api/project/clone/branches", methods=["POST"])
def api_project_clone_branches():
    """Return branch choices for a remote git repository."""
    data = request.get_json(silent=True) or {}
    remote_url = str(data.get("remote_url") or "").strip()
    if not remote_url:
        return jsonify({"error": "Remote URL is required."}), 400
    if remote_url.startswith("-"):
        return jsonify({"error": "Remote URL is invalid."}), 400

    try:
        branches, default_branch = _list_git_remote_branches(remote_url)
        return jsonify(
            {
                "remote_url": remote_url,
                "branches": branches,
                "default_branch": default_branch,
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Timed out while querying remote branches."}), 504
    except RuntimeError as exc:
        return jsonify({"error": f"Could not query remote branches: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/project/clone", methods=["POST"])
def api_project_clone():
    """Clone a remote repository into a selected destination and initialize .codex_manager."""
    data = request.get_json(silent=True) or {}
    remote_url = str(data.get("remote_url") or "").strip()
    destination_dir = str(data.get("destination_dir") or "").strip()
    requested_project_name = str(data.get("project_name") or "").strip()
    branch_raw = str(data.get("default_branch") or "").strip()
    requested_branch = _normalize_clone_branch_name(branch_raw)

    if not remote_url:
        return jsonify({"error": "Remote URL is required."}), 400
    if remote_url.startswith("-"):
        return jsonify({"error": "Remote URL is invalid."}), 400
    if not destination_dir:
        return jsonify({"error": "Destination directory is required."}), 400
    if branch_raw and not _valid_clone_branch_name(requested_branch):
        return jsonify({"error": f"Invalid branch name: {branch_raw}"}), 400

    destination = Path(destination_dir).expanduser()
    if not destination.is_dir():
        return jsonify({"error": f"Destination directory does not exist: {destination_dir}"}), 400

    inferred_name = requested_project_name or _derive_project_name_from_remote(remote_url)
    safe_name = _sanitize_project_folder_name(inferred_name)
    if not safe_name:
        return jsonify(
            {
                "error": (
                    "Project folder name is required. Provide project_name or use a remote URL "
                    "that includes a repository name."
                )
            }
        ), 400

    project_path = destination / safe_name
    if project_path.exists():
        return jsonify({"error": f"Path already exists: {project_path}"}), 409

    clone_args = ["git", "clone"]
    if requested_branch:
        clone_args.extend(["--branch", requested_branch])
    clone_args.extend([remote_url, safe_name])

    try:
        subprocess.run(
            clone_args,
            cwd=str(destination),
            capture_output=True,
            text=True,
            timeout=_GIT_CLONE_TIMEOUT_SECONDS,
            check=True,
            **_subprocess_isolation_kwargs(),
        )
        if not (project_path / ".git").is_dir():
            return jsonify({"error": f"Clone failed: git metadata missing at {project_path}"}), 500

        resolved_project_path = project_path.resolve()
        created_artifacts = _initialize_cloned_repo_codex_manager(resolved_project_path)

        checkout_branch = ""
        with suppress(Exception):
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(resolved_project_path),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                **_subprocess_isolation_kwargs(),
            )
            if branch_result.returncode == 0:
                checkout_branch = str(branch_result.stdout or "").strip()

        return jsonify(
            {
                "status": "cloned",
                "path": str(resolved_project_path),
                "remote_url": remote_url,
                "project_name": safe_name,
                "checked_out_branch": checkout_branch,
                "requested_branch": requested_branch,
                "codex_manager_initialized": True,
                "codex_manager_created": created_artifacts,
            }
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(project_path, ignore_errors=True)
        return jsonify({"error": "Git clone timed out."}), 504
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(project_path, ignore_errors=True)
        return jsonify({"error": f"Git clone failed: {_extract_git_error_message(exc)}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/git/sync/status")
def api_git_sync_status():
    """Return repository sync status (branch/tracking/ahead-behind/dirty)."""
    repo, error, status = _resolve_git_sync_repo(request.args.get("repo_path"))
    if repo is None:
        return jsonify({"error": error}), status
    force_refresh = _safe_bool(request.args.get("force"), default=False)

    try:
        return jsonify(_git_sync_status_payload(repo, force_refresh=force_refresh))
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Git sync status timed out."}), 504
    except RuntimeError as exc:
        return jsonify({"error": f"Could not read git sync status: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/git/sync/branches")
def api_git_sync_branches():
    """Return local + remote branch choices for branch switching UI."""
    repo, error, status = _resolve_git_sync_repo(request.args.get("repo_path"))
    if repo is None:
        return jsonify({"error": error}), status
    force_refresh = _safe_bool(request.args.get("force"), default=False)

    try:
        return jsonify(_git_sync_branch_choices_payload(repo, force_refresh=force_refresh))
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Git branch query timed out."}), 504
    except RuntimeError as exc:
        return jsonify({"error": f"Could not read branch list: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/git/sync/remotes")
def api_git_sync_remotes():
    """Return configured git remotes plus default-remote metadata."""
    repo, error, status = _resolve_git_sync_repo(request.args.get("repo_path"))
    if repo is None:
        return jsonify({"error": error}), status

    try:
        return jsonify(_git_sync_remotes_payload(repo))
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Git remote query timed out."}), 504
    except RuntimeError as exc:
        return jsonify({"error": f"Could not read remotes: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/git/sync/remotes/validate", methods=["POST"])
def api_git_sync_remote_validate():
    """Validate a candidate git remote URL (HTTPS/SSH only)."""
    data = request.get_json(silent=True) or {}
    remote_url = str(data.get("remote_url") or data.get("url") or "").strip()
    ok, transport, message = _validate_git_remote_url(remote_url)
    if not ok:
        return jsonify({"error": message, "valid": False, "remote_url": remote_url}), 400
    return jsonify(
        {
            "status": "validated",
            "valid": True,
            "transport": transport,
            "remote_url": remote_url,
            "message": message,
        }
    )


@app.route("/api/git/sync/remotes/add", methods=["POST"])
def api_git_sync_remote_add():
    """Add one git remote and optionally set it as the default push remote."""
    data = request.get_json(silent=True) or {}
    repo, error, status = _resolve_git_sync_repo(data.get("repo_path"))
    if repo is None:
        return jsonify({"error": error}), status

    remote_name = str(data.get("remote") or data.get("name") or "").strip()
    remote_url = str(data.get("remote_url") or data.get("url") or "").strip()
    set_default = _safe_bool(data.get("set_default"), default=False)

    if not _valid_git_sync_remote_name(remote_name):
        return jsonify({"error": f"Invalid remote name: {remote_name}"}), 400
    ok, transport, validation_message = _validate_git_remote_url(remote_url)
    if not ok:
        return jsonify({"error": validation_message}), 400

    try:
        if _git_remote_exists(repo, remote_name):
            return jsonify({"error": f"Remote already exists: {remote_name}"}), 409

        add_result = _run_git_sync_command(repo, "remote", "add", remote_name, remote_url)
        if add_result.returncode != 0:
            detail = _extract_git_process_error(add_result, "git remote add failed")
            return jsonify({"error": f"Git remote add failed: {detail}"}), 502

        if set_default:
            set_default_result = _run_git_sync_command(
                repo,
                "config",
                "remote.pushDefault",
                remote_name,
            )
            if set_default_result.returncode != 0:
                rollback_result = _run_git_sync_command(repo, "remote", "remove", remote_name)
                rollback_note = ""
                if rollback_result.returncode != 0:
                    rollback_detail = _extract_git_process_error(
                        rollback_result,
                        "git remote remove failed during rollback",
                    )
                    rollback_note = f" Rollback failed: {rollback_detail}"
                detail = _extract_git_process_error(
                    set_default_result,
                    "git config remote.pushDefault failed",
                )
                return (
                    jsonify(
                        {
                            "error": (
                                f"Remote added but default-remote update failed: {detail}.{rollback_note}"
                            ).strip()
                        }
                    ),
                    502,
                )

        remotes_payload = _git_sync_remotes_payload(repo)
        return jsonify(
            {
                "status": "remote_added",
                "repo_path": str(repo),
                "remote": remote_name,
                "remote_url": remote_url,
                "transport": transport,
                "set_default": set_default,
                "message": (
                    f"Added remote {remote_name} and set it as default."
                    if set_default
                    else f"Added remote {remote_name}."
                ),
                "validation_message": validation_message,
                "stdout": _truncate_command_output(add_result.stdout),
                "stderr": _truncate_command_output(add_result.stderr),
                "remotes": remotes_payload,
                "sync": remotes_payload.get("sync"),
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Git remote add timed out."}), 504
    except RuntimeError as exc:
        return jsonify({"error": f"Could not load remotes: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/git/sync/remotes/remove", methods=["POST"])
def api_git_sync_remote_remove():
    """Remove one git remote and clear default if that remote was selected."""
    data = request.get_json(silent=True) or {}
    repo, error, status = _resolve_git_sync_repo(data.get("repo_path"))
    if repo is None:
        return jsonify({"error": error}), status

    remote_name = str(data.get("remote") or data.get("name") or "").strip()
    if not _valid_git_sync_remote_name(remote_name):
        return jsonify({"error": f"Invalid remote name: {remote_name}"}), 400

    try:
        if not _git_remote_exists(repo, remote_name):
            return jsonify({"error": f"Remote not found: {remote_name}"}), 404

        configured_default_before = _git_configured_push_default_remote(repo)
        remove_result = _run_git_sync_command(repo, "remote", "remove", remote_name)
        if remove_result.returncode != 0:
            detail = _extract_git_process_error(remove_result, "git remote remove failed")
            return jsonify({"error": f"Git remote remove failed: {detail}"}), 502

        cleared_default = False
        if configured_default_before == remote_name:
            _git_clear_push_default_remote(repo)
            cleared_default = True

        remotes_payload = _git_sync_remotes_payload(repo)
        return jsonify(
            {
                "status": "remote_removed",
                "repo_path": str(repo),
                "remote": remote_name,
                "cleared_default": cleared_default,
                "message": (
                    f"Removed remote {remote_name} and cleared default remote."
                    if cleared_default
                    else f"Removed remote {remote_name}."
                ),
                "stdout": _truncate_command_output(remove_result.stdout),
                "stderr": _truncate_command_output(remove_result.stderr),
                "remotes": remotes_payload,
                "sync": remotes_payload.get("sync"),
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Git remote remove timed out."}), 504
    except RuntimeError as exc:
        return jsonify({"error": f"Could not update remote settings: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/git/sync/remotes/default", methods=["POST"])
def api_git_sync_remote_default():
    """Set or clear the default push remote (git config remote.pushDefault)."""
    data = request.get_json(silent=True) or {}
    repo, error, status = _resolve_git_sync_repo(data.get("repo_path"))
    if repo is None:
        return jsonify({"error": error}), status

    clear_default = _safe_bool(data.get("clear"), default=False)
    remote_name = str(data.get("remote") or data.get("name") or "").strip()

    if not clear_default and not _valid_git_sync_remote_name(remote_name):
        return jsonify({"error": f"Invalid remote name: {remote_name}"}), 400

    try:
        if clear_default:
            _git_clear_push_default_remote(repo)
            remotes_payload = _git_sync_remotes_payload(repo)
            return jsonify(
                {
                    "status": "remote_default_cleared",
                    "repo_path": str(repo),
                    "default_remote": "",
                    "message": "Cleared default remote.",
                    "remotes": remotes_payload,
                    "sync": remotes_payload.get("sync"),
                }
            )

        if not _git_remote_exists(repo, remote_name):
            return jsonify({"error": f"Remote not found: {remote_name}"}), 404

        set_result = _run_git_sync_command(repo, "config", "remote.pushDefault", remote_name)
        if set_result.returncode != 0:
            detail = _extract_git_process_error(set_result, "git config remote.pushDefault failed")
            return jsonify({"error": f"Could not set default remote: {detail}"}), 502

        remotes_payload = _git_sync_remotes_payload(repo)
        return jsonify(
            {
                "status": "remote_default_set",
                "repo_path": str(repo),
                "default_remote": remote_name,
                "message": f"Default remote set to {remote_name}.",
                "stdout": _truncate_command_output(set_result.stdout),
                "stderr": _truncate_command_output(set_result.stderr),
                "remotes": remotes_payload,
                "sync": remotes_payload.get("sync"),
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Git remote default update timed out."}), 504
    except RuntimeError as exc:
        return jsonify({"error": f"Could not update remote settings: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/git/signing")
def api_git_signing_settings():
    """Return git signing settings + validation for the selected repository."""
    repo, error, status = _resolve_git_sync_repo(request.args.get("repo_path"))
    if repo is None:
        return jsonify({"error": error}), status

    try:
        return jsonify(_git_signing_payload(repo))
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Git signing check timed out."}), 504
    except RuntimeError as exc:
        return jsonify({"error": f"Could not read git signing settings: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/git/signing", methods=["POST"])
def api_git_signing_save():
    """Persist git signing settings (gpg/ssh, key, commit/tag signing, push guard)."""
    data = request.get_json(silent=True) or {}
    repo, error, status = _resolve_git_sync_repo(data.get("repo_path"))
    if repo is None:
        return jsonify({"error": error}), status

    mode = _normalize_git_signing_mode(data.get("mode") or "gpg")
    if mode not in {"gpg", "ssh"}:
        return jsonify({"error": "mode must be either 'gpg' or 'ssh'."}), 400

    commit_sign = _safe_bool(data.get("commit_sign"), default=True)
    tag_sign = _safe_bool(data.get("tag_sign"), default=True)
    if "require_push_guard" in data:
        require_push_guard = _safe_bool(data.get("require_push_guard"), default=True)
    else:
        require_push_guard = bool(commit_sign or tag_sign)

    signing_key_raw = str(data.get("signing_key") or "").strip()
    clear_signing_key = _safe_bool(data.get("clear_signing_key"), default=False)
    if "\x00" in signing_key_raw:
        return jsonify({"error": "signing_key may not include NUL bytes."}), 400
    if "\n" in signing_key_raw or "\r" in signing_key_raw:
        return jsonify({"error": "signing_key must be a single line."}), 400

    try:
        existing_key = _git_config_get_value(repo, "user.signingkey")
        target_signing_key = ""
        if clear_signing_key:
            target_signing_key = ""
        elif signing_key_raw:
            target_signing_key = signing_key_raw
        else:
            target_signing_key = existing_key

        # gpg.format accepts openpgp|x509|ssh. We expose openpgp as "gpg" in the UI.
        _git_config_set_value(repo, "gpg.format", "ssh" if mode == "ssh" else "openpgp")
        _git_config_set_bool(repo, "commit.gpgsign", commit_sign)
        _git_config_set_bool(repo, "tag.gpgSign", tag_sign)
        _git_config_set_bool(repo, _GIT_SIGNING_PUSH_GUARD_KEY, require_push_guard)
        if target_signing_key:
            _git_config_set_value(repo, "user.signingkey", target_signing_key)
        else:
            _git_config_unset_value(repo, "user.signingkey")

        payload = _git_signing_payload(repo)
        if bool(payload.get("enabled")) and bool(payload.get("valid")):
            message = "Git signing settings saved and validated."
        elif bool(payload.get("enabled")):
            message = "Git signing settings saved, but validation found issues."
        else:
            message = "Git signing settings saved (commit/tag signing disabled)."

        return jsonify(
            {
                "status": "saved",
                "repo_path": str(repo),
                "message": message,
                "settings": payload,
                "sync": _git_sync_status_payload(repo),
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Git signing update timed out."}), 504
    except RuntimeError as exc:
        return jsonify({"error": f"Could not update git signing settings: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/git/signing/validate", methods=["POST"])
def api_git_signing_validate():
    """Validate git signing settings from saved config or optional draft fields."""
    data = request.get_json(silent=True) or {}
    repo, error, status = _resolve_git_sync_repo(data.get("repo_path"))
    if repo is None:
        return jsonify({"error": error}), status

    try:
        settings = _git_signing_settings(repo)

        if "mode" in data:
            settings["mode"] = _normalize_git_signing_mode(data.get("mode"))
        if "signing_key" in data:
            settings["signing_key"] = str(data.get("signing_key") or "").strip()
            settings["has_signing_key"] = bool(settings["signing_key"])
        if _safe_bool(data.get("clear_signing_key"), default=False):
            settings["signing_key"] = ""
            settings["has_signing_key"] = False
        if "commit_sign" in data:
            settings["commit_sign"] = _safe_bool(data.get("commit_sign"), default=False)
        if "tag_sign" in data:
            settings["tag_sign"] = _safe_bool(data.get("tag_sign"), default=False)
        settings["enabled"] = bool(settings.get("commit_sign") or settings.get("tag_sign"))
        if "require_push_guard" in data:
            settings["require_push_guard"] = _safe_bool(
                data.get("require_push_guard"),
                default=bool(settings["enabled"]),
            )

        payload = _git_signing_payload(repo, settings=settings)
        return jsonify(
            {
                "status": "validated",
                "repo_path": str(repo),
                "message": str(payload.get("message") or "Validation complete."),
                "settings": payload,
                "sync": _git_sync_status_payload(repo),
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Git signing validation timed out."}), 504
    except RuntimeError as exc:
        return jsonify({"error": f"Could not validate git signing settings: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/git/sync/checkout", methods=["POST"])
def api_git_sync_checkout():
    """Checkout a selected local branch or create tracking from a remote branch."""
    data = request.get_json(silent=True) or {}
    repo, error, status = _resolve_git_sync_repo(data.get("repo_path"))
    if repo is None:
        return jsonify({"error": error}), status

    requested_branch = str(data.get("branch") or "").strip()
    requested_branch_type = str(data.get("branch_type") or "").strip().lower()
    allow_dirty = _safe_bool(data.get("allow_dirty"), default=False)

    if not requested_branch:
        return jsonify({"error": "branch is required."}), 400
    if not _valid_clone_branch_name(requested_branch):
        return jsonify({"error": f"Invalid branch name: {requested_branch}"}), 400
    if requested_branch_type not in {"", "local", "remote"}:
        return jsonify({"error": f"Invalid branch_type: {requested_branch_type}"}), 400

    try:
        status_before = _git_sync_status_payload(repo)
        if bool(status_before.get("dirty")) and not allow_dirty:
            return _git_sync_dirty_guardrail_response(
                repo=repo,
                status_payload=status_before,
                action="switch branches",
            )

        local_exists = _git_ref_exists(repo, f"refs/heads/{requested_branch}")
        remote_exists = _git_ref_exists(repo, f"refs/remotes/{requested_branch}")
        branch_type = requested_branch_type
        if not branch_type:
            if local_exists:
                branch_type = "local"
            elif remote_exists:
                branch_type = "remote"
            else:
                return jsonify({"error": f"Branch not found: {requested_branch}"}), 404

        checkout_args: list[str]
        created_tracking_branch = False
        effective_branch = requested_branch

        if branch_type == "local":
            if not local_exists:
                return jsonify({"error": f"Local branch not found: {requested_branch}"}), 404
            checkout_args = ["checkout", requested_branch]
        else:
            if not remote_exists:
                return jsonify({"error": f"Remote branch not found: {requested_branch}"}), 404
            if "/" not in requested_branch:
                return jsonify({"error": f"Remote branch must include remote prefix: {requested_branch}"}), 400
            local_branch = requested_branch.split("/", 1)[1].strip()
            if not local_branch or not _valid_clone_branch_name(local_branch):
                return jsonify({"error": f"Invalid local branch derived from {requested_branch}"}), 400
            effective_branch = local_branch
            if _git_ref_exists(repo, f"refs/heads/{local_branch}"):
                checkout_args = ["checkout", local_branch]
            else:
                checkout_args = ["checkout", "--track", "-b", local_branch, requested_branch]
                created_tracking_branch = True

        checkout_result = _run_git_sync_command(repo, *checkout_args)
        if checkout_result.returncode != 0:
            detail = _extract_git_process_error(checkout_result, f"git {' '.join(checkout_args)} failed")
            return jsonify({"error": f"Git checkout failed: {detail}"}), 502

        status_after = _git_sync_status_payload(repo)
        branches = _git_sync_branch_choices_payload(repo, force_refresh=True)
        active_branch = str(status_after.get("branch") or "").strip() or effective_branch
        if branch_type == "remote" and created_tracking_branch:
            message = f"Checked out remote branch {requested_branch} as local {active_branch}."
        elif branch_type == "remote":
            message = f"Switched to local branch {active_branch} (tracking {requested_branch})."
        else:
            message = f"Switched to branch {active_branch}."

        return jsonify(
            {
                "status": "checked_out",
                "repo_path": str(repo),
                "requested_branch": requested_branch,
                "branch_type": branch_type,
                "branch": active_branch,
                "created_tracking_branch": created_tracking_branch,
                "allow_dirty": allow_dirty,
                "message": message,
                "stdout": _truncate_command_output(checkout_result.stdout),
                "stderr": _truncate_command_output(checkout_result.stderr),
                "sync": status_after,
                "branches": branches,
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Git checkout timed out."}), 504
    except RuntimeError as exc:
        return jsonify({"error": f"Could not compute git sync status: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/git/sync/branch/create", methods=["POST"])
def api_git_sync_branch_create():
    """Create and checkout a new branch from HEAD or an explicit start point."""
    data = request.get_json(silent=True) or {}
    repo, error, status = _resolve_git_sync_repo(data.get("repo_path"))
    if repo is None:
        return jsonify({"error": error}), status

    branch_name = str(data.get("branch_name") or data.get("branch") or "").strip()
    start_point = str(data.get("start_point") or "").strip()
    allow_dirty = _safe_bool(data.get("allow_dirty"), default=False)

    if not branch_name:
        return jsonify({"error": "branch_name is required."}), 400
    if not _valid_clone_branch_name(branch_name):
        return jsonify({"error": f"Invalid branch name: {branch_name}"}), 400
    if start_point.startswith("-") or any(ch.isspace() for ch in start_point):
        return jsonify({"error": f"Invalid start_point: {start_point}"}), 400

    try:
        status_before = _git_sync_status_payload(repo)
        if bool(status_before.get("dirty")) and not allow_dirty:
            return _git_sync_dirty_guardrail_response(
                repo=repo,
                status_payload=status_before,
                action="create a branch",
            )

        if _git_ref_exists(repo, f"refs/heads/{branch_name}"):
            return jsonify({"error": f"Local branch already exists: {branch_name}"}), 409

        create_args: list[str] = ["checkout", "-b", branch_name]
        if start_point:
            create_args.append(start_point)

        create_result = _run_git_sync_command(repo, *create_args)
        if create_result.returncode != 0:
            detail = _extract_git_process_error(create_result, f"git {' '.join(create_args)} failed")
            return jsonify({"error": f"Git branch creation failed: {detail}"}), 502

        status_after = _git_sync_status_payload(repo)
        branches = _git_sync_branch_choices_payload(repo, force_refresh=True)
        message = (
            f"Created and switched to branch {branch_name} from {start_point}."
            if start_point
            else f"Created and switched to branch {branch_name}."
        )
        return jsonify(
            {
                "status": "branch_created",
                "repo_path": str(repo),
                "branch": branch_name,
                "start_point": start_point,
                "allow_dirty": allow_dirty,
                "message": message,
                "stdout": _truncate_command_output(create_result.stdout),
                "stderr": _truncate_command_output(create_result.stderr),
                "sync": status_after,
                "branches": branches,
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Git branch creation timed out."}), 504
    except RuntimeError as exc:
        return jsonify({"error": f"Could not compute git sync status: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/git/sync/commit/workflow")
def api_git_sync_commit_workflow():
    """Return stage/unstage candidates, commit summary, and last commit metadata."""
    repo, error, status = _resolve_git_sync_repo(request.args.get("repo_path"))
    if repo is None:
        return jsonify({"error": error}), status

    try:
        workflow = _git_commit_workflow_payload(repo)
        workflow["sync"] = _git_sync_status_payload(repo)
        return jsonify(workflow)
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Git commit workflow query timed out."}), 504
    except RuntimeError as exc:
        return jsonify({"error": f"Could not load commit workflow data: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/git/sync/commit/stage", methods=["POST"])
def api_git_sync_commit_stage():
    """Stage selected file paths (or all stageable paths) for commit."""
    data = request.get_json(silent=True) or {}
    repo, error, status = _resolve_git_sync_repo(data.get("repo_path"))
    if repo is None:
        return jsonify({"error": error}), status

    include_all = _safe_bool(data.get("all"), default=False)
    try:
        requested_paths = _normalize_git_commit_paths(data.get("paths"))
        workflow_before = _git_commit_workflow_payload(repo)
        available_paths = {
            str(path)
            for path in workflow_before.get("stageable_paths", [])
            if isinstance(path, str) and path.strip()
        }
        target_paths = _resolve_git_commit_target_paths(
            available_paths=available_paths,
            requested_paths=requested_paths,
            include_all=include_all,
            action_label="stage",
        )

        if include_all:
            stage_result = _run_git_sync_command(repo, "add", "--all")
        else:
            stage_result = _git_stage_paths(repo, target_paths)
        if stage_result.returncode != 0:
            detail = _extract_git_process_error(stage_result, "git add failed")
            return jsonify({"error": f"Git stage failed: {detail}"}), 502

        workflow_after = _git_commit_workflow_payload(repo)
        sync_after = _git_sync_status_payload(repo)
        return jsonify(
            {
                "status": "staged",
                "repo_path": str(repo),
                "all": include_all,
                "paths": target_paths,
                "message": (
                    f"Staged {len(target_paths)} file(s)."
                    if not include_all
                    else "Staged all pending changes."
                ),
                "stdout": _truncate_command_output(stage_result.stdout),
                "stderr": _truncate_command_output(stage_result.stderr),
                "workflow": workflow_after,
                "sync": sync_after,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), _git_commit_user_error_status(str(exc))
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Git stage operation timed out."}), 504
    except RuntimeError as exc:
        return jsonify({"error": f"Could not load commit workflow data: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/git/sync/commit/unstage", methods=["POST"])
def api_git_sync_commit_unstage():
    """Unstage selected file paths (or all staged paths)."""
    data = request.get_json(silent=True) or {}
    repo, error, status = _resolve_git_sync_repo(data.get("repo_path"))
    if repo is None:
        return jsonify({"error": error}), status

    include_all = _safe_bool(data.get("all"), default=False)
    try:
        requested_paths = _normalize_git_commit_paths(data.get("paths"))
        workflow_before = _git_commit_workflow_payload(repo)
        available_paths = {
            str(path)
            for path in workflow_before.get("unstageable_paths", [])
            if isinstance(path, str) and path.strip()
        }
        target_paths = _resolve_git_commit_target_paths(
            available_paths=available_paths,
            requested_paths=requested_paths,
            include_all=include_all,
            action_label="unstage",
        )

        unstage_result = _git_unstage_paths(repo, target_paths)
        if unstage_result.returncode != 0:
            detail = _extract_git_process_error(unstage_result, "git unstage failed")
            return jsonify({"error": f"Git unstage failed: {detail}"}), 502

        workflow_after = _git_commit_workflow_payload(repo)
        sync_after = _git_sync_status_payload(repo)
        return jsonify(
            {
                "status": "unstaged",
                "repo_path": str(repo),
                "all": include_all,
                "paths": target_paths,
                "message": (
                    f"Unstaged {len(target_paths)} file(s)."
                    if not include_all
                    else "Unstaged all staged files."
                ),
                "stdout": _truncate_command_output(unstage_result.stdout),
                "stderr": _truncate_command_output(unstage_result.stderr),
                "workflow": workflow_after,
                "sync": sync_after,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), _git_commit_user_error_status(str(exc))
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Git unstage operation timed out."}), 504
    except RuntimeError as exc:
        return jsonify({"error": f"Could not load commit workflow data: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/git/sync/commit/create", methods=["POST"])
def api_git_sync_commit_create():
    """Create a commit using staged changes and a user-supplied commit message."""
    data = request.get_json(silent=True) or {}
    repo, error, status = _resolve_git_sync_repo(data.get("repo_path"))
    if repo is None:
        return jsonify({"error": error}), status

    commit_message_raw = str(data.get("message") or "")
    commit_message = commit_message_raw.strip()
    if not commit_message:
        return jsonify({"error": "Commit message is required."}), 400

    try:
        workflow_before = _git_commit_workflow_payload(repo)
        counts = workflow_before.get("counts")
        staged_count = _safe_int(counts.get("staged"), default=0) if isinstance(counts, dict) else 0
        if staged_count <= 0:
            return jsonify({"error": "No staged changes to commit. Stage files first."}), 409

        commit_result = _run_git_sync_command(repo, "commit", "-m", commit_message_raw)
        if commit_result.returncode != 0:
            detail = _extract_git_process_error(commit_result, "git commit failed")
            lowered = detail.lower()
            if "nothing to commit" in lowered:
                return jsonify({"error": "No staged changes to commit. Stage files first."}), 409
            if "author identity unknown" in lowered or "please tell me who you are" in lowered:
                return (
                    jsonify(
                        {
                            "error": (
                                "Git commit failed: user identity is not configured. Set git user.name "
                                "and user.email, then retry."
                            ),
                            "error_type": "identity_missing",
                            "recovery_steps": [
                                "Run `git config user.name \"Your Name\"`.",
                                "Run `git config user.email \"you@example.com\"`.",
                                "Retry the commit action.",
                            ],
                        }
                    ),
                    400,
                )
            return jsonify({"error": f"Git commit failed: {detail}"}), 502

        workflow_after = _git_commit_workflow_payload(repo)
        sync_after = _git_sync_status_payload(repo)
        return jsonify(
            {
                "status": "committed",
                "repo_path": str(repo),
                "message": "Commit created.",
                "commit_message": commit_message_raw,
                "commit": workflow_after.get("last_commit"),
                "stdout": _truncate_command_output(commit_result.stdout),
                "stderr": _truncate_command_output(commit_result.stderr),
                "workflow": workflow_after,
                "sync": sync_after,
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Git commit operation timed out."}), 504
    except RuntimeError as exc:
        return jsonify({"error": f"Could not load commit workflow data: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/git/sync/fetch", methods=["POST"])
def api_git_sync_fetch():
    """Fetch latest remote refs for the active repository."""
    data = request.get_json(silent=True) or {}
    repo, error, status = _resolve_git_sync_repo(data.get("repo_path"))
    if repo is None:
        return jsonify({"error": error}), status

    try:
        fetch_result = _run_git_sync_command(repo, "fetch", "--prune")
        if fetch_result.returncode != 0:
            detail = _extract_git_process_error(fetch_result, "git fetch --prune failed")
            return jsonify({"error": f"Git fetch failed: {detail}"}), 502

        return jsonify(
            {
                "status": "fetched",
                "repo_path": str(repo),
                "message": "Fetch completed.",
                "stdout": _truncate_command_output(fetch_result.stdout),
                "stderr": _truncate_command_output(fetch_result.stderr),
                "sync": _git_sync_status_payload(repo),
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Git fetch timed out."}), 504
    except RuntimeError as exc:
        return jsonify({"error": f"Could not compute git sync status: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/git/sync/pull", methods=["POST"])
def api_git_sync_pull():
    """Pull the active branch using fast-forward-only mode."""
    data = request.get_json(silent=True) or {}
    repo, error, status = _resolve_git_sync_repo(data.get("repo_path"))
    if repo is None:
        return jsonify({"error": error}), status

    try:
        pull_result = _run_git_sync_command(repo, "pull", "--ff-only")
        if pull_result.returncode != 0:
            detail = _extract_git_process_error(pull_result, "git pull --ff-only failed")
            return jsonify({"error": f"Git pull failed: {detail}"}), 502

        return jsonify(
            {
                "status": "pulled",
                "repo_path": str(repo),
                "message": "Pull completed.",
                "stdout": _truncate_command_output(pull_result.stdout),
                "stderr": _truncate_command_output(pull_result.stderr),
                "sync": _git_sync_status_payload(repo),
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Git pull timed out."}), 504
    except RuntimeError as exc:
        return jsonify({"error": f"Could not compute git sync status: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/git/sync/stash-pull", methods=["POST"])
def api_git_sync_stash_pull():
    """Stash local changes (including untracked files), then pull latest commits."""
    data = request.get_json(silent=True) or {}
    repo, error, status = _resolve_git_sync_repo(data.get("repo_path"))
    if repo is None:
        return jsonify({"error": error}), status

    try:
        stash_before = _resolve_stash_ref(repo)
        stash_message = (
            "warpfoundry:auto-stash-before-pull "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )
        stash_result = _run_git_sync_command(
            repo,
            "stash",
            "push",
            "--include-untracked",
            "--message",
            stash_message,
        )
        if stash_result.returncode != 0:
            detail = _extract_git_process_error(
                stash_result,
                "git stash push --include-untracked failed",
            )
            return jsonify({"error": f"Git stash failed: {detail}"}), 502

        stash_after = _resolve_stash_ref(repo)
        stash_created = bool(stash_after and stash_after != stash_before)

        pull_result = _run_git_sync_command(repo, "pull", "--ff-only")
        if pull_result.returncode != 0:
            detail = _extract_git_process_error(pull_result, "git pull --ff-only failed")
            hint = (
                " Local changes were stashed. Use 'git stash list' and 'git stash pop' after "
                "resolving the pull issue."
                if stash_created
                else ""
            )
            return jsonify({"error": f"Git pull failed: {detail}{hint}"}), 502

        return jsonify(
            {
                "status": "stashed_and_pulled",
                "repo_path": str(repo),
                "stash_created": stash_created,
                "stash_ref": stash_after if stash_created else "",
                "message": (
                    "Stashed local changes and pulled latest commits."
                    if stash_created
                    else "No local changes were stashed; pull completed."
                ),
                "stash_stdout": _truncate_command_output(stash_result.stdout),
                "stash_stderr": _truncate_command_output(stash_result.stderr),
                "pull_stdout": _truncate_command_output(pull_result.stdout),
                "pull_stderr": _truncate_command_output(pull_result.stderr),
                "sync": _git_sync_status_payload(repo),
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Git stash/pull timed out."}), 504
    except RuntimeError as exc:
        return jsonify({"error": f"Could not compute git sync status: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/git/sync/push", methods=["POST"])
def api_git_sync_push():
    """Push local commits for the active branch, optionally setting upstream."""
    data = request.get_json(silent=True) or {}
    repo, error, status = _resolve_git_sync_repo(data.get("repo_path"))
    if repo is None:
        return jsonify({"error": error}), status

    set_upstream = _safe_bool(data.get("set_upstream"), default=False)
    requested_remote = str(data.get("remote") or "").strip()
    requested_branch = str(data.get("branch") or "").strip()

    if requested_remote and not _valid_git_sync_remote_name(requested_remote):
        return jsonify({"error": f"Invalid remote name: {requested_remote}"}), 400
    if requested_branch and not _valid_clone_branch_name(requested_branch):
        return jsonify({"error": f"Invalid branch name: {requested_branch}"}), 400

    try:
        status_before = _git_sync_status_payload(repo)
        active_branch = str(status_before.get("branch") or "").strip() or "HEAD"
        branch = requested_branch or active_branch
        if branch == "HEAD":
            return jsonify(
                {
                    "error": (
                        "Cannot push from detached HEAD. Checkout a branch first or pass an explicit "
                        "branch name."
                    )
                }
            ), 400

        tracking_branch = str(status_before.get("tracking_branch") or "").strip()
        tracking_remote = _extract_tracking_remote_name(tracking_branch)
        configured_default_remote = _git_configured_push_default_remote(repo)
        fallback_remote = configured_default_remote or tracking_remote or "origin"
        if fallback_remote and not _git_remote_exists(repo, fallback_remote):
            if tracking_remote and _git_remote_exists(repo, tracking_remote):
                fallback_remote = tracking_remote
            elif _git_remote_exists(repo, "origin"):
                fallback_remote = "origin"
        remote = _normalize_git_sync_remote_name(requested_remote or fallback_remote)
        if not _valid_git_sync_remote_name(remote):
            return jsonify({"error": f"Invalid remote name: {remote}"}), 400

        signing = _git_signing_payload(repo)
        signing_enabled = bool(signing.get("enabled"))
        require_push_guard = bool(signing.get("require_push_guard"))
        signing_valid = bool(signing.get("valid"))
        if signing_enabled and require_push_guard and not signing_valid:
            issues = [str(item).strip() for item in signing.get("issues", []) if str(item).strip()]
            recovery_steps = [
                "Open Git Sync -> Signing and resolve the reported validation issues.",
                "Save settings, run Validate, then retry push.",
            ]
            for issue in issues[:3]:
                recovery_steps.insert(1, f"Fix: {issue}")
            return (
                jsonify(
                    {
                        "error": (
                            "Push blocked: commit/tag signing is enabled but the signing setup is invalid."
                        ),
                        "error_type": "signing_misconfigured",
                        "repo_path": str(repo),
                        "remote": remote,
                        "branch": branch,
                        "set_upstream": set_upstream,
                        "signing": signing,
                        "recovery_steps": recovery_steps,
                        "sync": status_before,
                    }
                ),
                412,
            )

        push_args: list[str] = ["push"]
        if set_upstream:
            push_args.extend(["--set-upstream", remote, branch])
        elif requested_remote or requested_branch or bool(configured_default_remote):
            push_args.extend([remote, branch])

        push_result = _run_git_sync_command(repo, *push_args)
        if push_result.returncode != 0:
            detail = _extract_git_process_error(push_result, f"git {' '.join(push_args)} failed")
            error_type = _classify_git_push_failure(push_result)
            status_code, label = _git_push_error_status_and_label(error_type)
            stdout_text = _truncate_command_output(push_result.stdout)
            stderr_text = _truncate_command_output(push_result.stderr)
            payload: dict[str, object] = {
                "error": f"Git push failed ({label}): {detail}",
                "error_type": error_type,
                "repo_path": str(repo),
                "remote": remote,
                "branch": branch,
                "set_upstream": set_upstream,
                "recovery_steps": _git_push_recovery_steps(
                    error_type=error_type,
                    remote=remote,
                    branch=branch,
                ),
                "stdout": stdout_text,
                "stderr": stderr_text,
            }
            if error_type == "auth":
                remote_url = _git_remote_url(repo, remote)
                payload["auth_troubleshooting"] = _github_auth_troubleshooting_assistant(
                    auth_method=_github_remote_transport(remote_url),
                    ok=False,
                    message=detail,
                    output="\n".join(part for part in [stderr_text, stdout_text] if part).strip(),
                    context="git_push",
                    remote_url=remote_url,
                )
            return (
                jsonify(payload),
                status_code,
            )

        message = (
            f"Push completed and upstream set to {remote}/{branch}."
            if set_upstream
            else "Push completed."
        )
        pull_request = _git_push_pull_request_payload(
            repo=repo,
            remote=remote,
            head_branch=branch,
        )
        return jsonify(
            {
                "status": "pushed",
                "repo_path": str(repo),
                "remote": remote,
                "branch": branch,
                "set_upstream": set_upstream,
                "message": message,
                "stdout": _truncate_command_output(push_result.stdout),
                "stderr": _truncate_command_output(push_result.stderr),
                "pull_request": pull_request,
                "pull_request_url": str(pull_request.get("url") or ""),
                "sync": _git_sync_status_payload(repo),
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Git push timed out."}), 504
    except RuntimeError as exc:
        return jsonify({"error": f"Could not compute git sync status: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/project/create", methods=["POST"])
def api_create_project():
    """Create a new project directory with git init and optional remote."""
    data = request.get_json(silent=True) or {}
    parent_dir = _safe_str(data.get("parent_dir"))
    project_name = _safe_str(data.get("project_name"))
    remote_url = _safe_str(data.get("remote_url"))
    description = _safe_str(data.get("description"))
    add_readme = data.get("add_readme", True)
    add_gitignore = data.get("add_gitignore", True)
    initial_branch = _safe_str(data.get("initial_branch"), "main")
    git_name = _safe_str(data.get("git_name") or data.get("gitName"))
    git_email = _safe_str(data.get("git_email") or data.get("gitEmail"))
    foundation_enabled = _safe_bool(data.get("foundation_enabled"), False)
    foundational_prompt = str(data.get("foundational_prompt") or "").strip()
    foundational_prompt_improved = str(data.get("foundational_prompt_improved") or "").strip()
    foundation_assistants = _normalize_foundation_assistants(data.get("foundation_assistants"))
    foundation_generate_docs = _safe_bool(data.get("foundation_generate_docs"), True)
    foundation_bootstrap_once = _safe_bool(data.get("foundation_bootstrap_once"), True)
    foundation_bootstrap_autorun = _safe_bool(data.get("foundation_bootstrap_autorun"), True)
    licensing_enabled = _safe_bool(data.get("licensing_enabled"), False)
    licensing_strategy = _normalize_licensing_strategy(data.get("licensing_strategy"))
    licensing_include_commercial_tiers = _safe_bool(
        data.get("licensing_include_commercial_tiers"),
        licensing_strategy != "oss_only",
    )
    licensing_owner_contact_email = str(data.get("licensing_owner_contact_email") or "").strip()
    licensing_legal_review_required = _safe_bool(
        data.get("licensing_legal_review_required"),
        True,
    )
    licensing_legal_signoff_approved = _safe_bool(
        data.get("licensing_legal_signoff_approved"),
        False,
    )
    licensing_legal_reviewer = str(data.get("licensing_legal_reviewer") or "").strip()
    licensing_legal_notes = str(data.get("licensing_legal_notes") or "").strip()

    if not parent_dir:
        return jsonify({"error": "Parent directory is required"}), 400
    if not project_name:
        return jsonify({"error": "Project name is required"}), 400
    if foundation_enabled and not (foundational_prompt or foundational_prompt_improved):
        return jsonify({"error": "Foundational prompt is required when foundation setup is enabled."}), 400
    if remote_url and (remote_url.startswith("-") or "\x00" in remote_url):
        return jsonify({"error": "Remote URL is invalid"}), 400
    if not _valid_clone_branch_name(initial_branch):
        return jsonify({"error": f"Invalid initial branch name: {initial_branch}"}), 400

    # Sanitize project name (allow alphanumeric, hyphens, underscores, dots)
    safe_name = "".join(c for c in project_name if c.isalnum() or c in "-_. ").strip()
    if not safe_name:
        return jsonify({"error": "Project name is invalid after sanitization"}), 400

    parent = Path(parent_dir)
    if not parent.is_dir():
        return jsonify({"error": f"Parent directory does not exist: {parent_dir}"}), 400

    project_path = parent / safe_name
    if project_path.exists():
        return jsonify({"error": f"Path already exists: {project_path}"}), 409

    try:
        # Create the project directory
        project_path.mkdir(parents=True, exist_ok=False)

        # git init
        subprocess.run(
            ["git", "init", "-b", initial_branch],
            cwd=str(project_path),
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
            **_subprocess_isolation_kwargs(),
        )

        # Configure git identity (required for commits) from modal form
        if git_name:
            subprocess.run(
                ["git", "config", "user.name", git_name],
                cwd=str(project_path),
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
                **_subprocess_isolation_kwargs(),
            )
        if git_email:
            subprocess.run(
                ["git", "config", "user.email", git_email],
                cwd=str(project_path),
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
                **_subprocess_isolation_kwargs(),
            )
        # Ensure any missing identity is set so initial commit never fails
        from codex_manager.git_tools import ensure_git_identity

        ensure_git_identity(project_path)

        # Optional README
        if add_readme:
            readme = project_path / "README.md"
            header = f"# {project_name}\n"
            if description:
                header += f"\n{description}\n"
            readme.write_text(header, encoding="utf-8")

        # Optional .gitignore
        if add_gitignore:
            gi = project_path / ".gitignore"
            gi.write_text(
                "# IDE\n.vscode/\n.idea/\n*.swp\n*.swo\n\n"
                "# Python\n__pycache__/\n*.pyc\n*.pyo\n.venv/\nvenv/\nenv/\n"
                "*.egg-info/\ndist/\nbuild/\n\n"
                "# Node\nnode_modules/\n\n"
                "# OS\n.DS_Store\nThumbs.db\n",
                encoding="utf-8",
            )

        foundation_files: list[str] = []
        foundation_prompt_used = ""
        bootstrap_status: dict[str, object] | None = None
        if foundation_enabled:
            foundation_prompt_used = (
                foundational_prompt_improved.strip() or foundational_prompt.strip()
            )
            foundation_files = _write_foundation_artifacts(
                project_path=project_path,
                project_name=project_name,
                description=description,
                foundational_prompt=foundation_prompt_used,
                assistants=foundation_assistants,
                generate_docs=foundation_generate_docs,
                bootstrap_once=foundation_bootstrap_once,
            )
            if foundation_bootstrap_once and not foundation_bootstrap_autorun:
                bootstrap_status = _write_foundation_bootstrap_status(
                    project_path=project_path,
                    project_name=project_name,
                    status="pending",
                    detail="Bootstrap request created. Run it manually when ready.",
                    extra={"source": "project_create"},
                )

        licensing_files: list[str] = []
        if licensing_enabled:
            licensing_files = _write_licensing_packaging_artifacts(
                project_path=project_path,
                project_name=project_name,
                description=description,
                strategy=licensing_strategy,
                include_commercial_tiers=licensing_include_commercial_tiers,
                owner_contact_email=licensing_owner_contact_email,
                legal_review_required=licensing_legal_review_required,
                legal_signoff_approved=licensing_legal_signoff_approved,
                legal_reviewer=licensing_legal_reviewer,
                legal_notes=licensing_legal_notes,
            )
        legal_review_state = _load_legal_review_state(project_path) if licensing_enabled else {}

        # Initial commit
        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(project_path),
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
            **_subprocess_isolation_kwargs(),
        )
        subprocess.run(
            ["git", "commit", "-m", f"Initial commit - {project_name}"],
            cwd=str(project_path),
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
            **_subprocess_isolation_kwargs(),
        )

        # Optional remote
        remote_added = False
        if remote_url:
            res = subprocess.run(
                ["git", "remote", "add", "origin", remote_url],
                cwd=str(project_path),
                capture_output=True,
                text=True,
                timeout=15,
                **_subprocess_isolation_kwargs(),
            )
            remote_added = res.returncode == 0

        if foundation_enabled and foundation_bootstrap_once and foundation_bootstrap_autorun:
            bootstrap_status = _start_foundation_bootstrap_chain(
                project_path=project_path,
                project_name=project_name,
                description=description,
                source="project_create",
            )

        return jsonify(
            {
                "status": "created",
                "path": str(project_path.resolve()),
                "git_initialized": True,
                "initial_branch": initial_branch,
                "remote_added": remote_added,
                "remote_url": remote_url if remote_added else None,
                "foundation_enabled": foundation_enabled,
                "foundation_files": foundation_files,
                "foundation_assistants": foundation_assistants,
                "foundation_bootstrap_once": foundation_bootstrap_once,
                "foundation_bootstrap_autorun": foundation_bootstrap_autorun,
                "foundation_bootstrap_status": bootstrap_status or {},
                "licensing_enabled": licensing_enabled,
                "licensing_strategy": licensing_strategy,
                "licensing_include_commercial_tiers": licensing_include_commercial_tiers,
                "licensing_files": licensing_files,
                "licensing_legal_review_required": bool(
                    legal_review_state.get("required", licensing_legal_review_required)
                ),
                "licensing_legal_review_approved": bool(legal_review_state.get("approved", False)),
                "licensing_legal_review_status": str(legal_review_state.get("status", "")),
                "licensing_legal_review": legal_review_state,
            }
        )

    except subprocess.CalledProcessError as exc:
        return jsonify({"error": f"Git command failed: {exc.stderr.strip()}"}), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/project/legal-review/status")
def api_project_legal_review_status():
    """Return legal-review checkpoint status for a project repository."""
    repo_path_raw = str(request.args.get("repo_path", "") or "").strip()
    if not repo_path_raw:
        return jsonify({"error": "repo_path is required."}), 400
    repo_path = Path(repo_path_raw).expanduser().resolve()
    if not repo_path.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_path_raw}"}), 400
    state = _load_legal_review_state(repo_path)
    if not state:
        return jsonify({"error": "Legal review state not found for this repository."}), 404
    return jsonify(state)


@app.route("/api/project/legal-review/signoff", methods=["POST"])
def api_project_legal_review_signoff():
    """Record or revoke legal-review sign-off for project licensing/pricing docs."""
    data = request.get_json(silent=True) or {}
    repo_path_raw = str(data.get("repo_path") or "").strip()
    if not repo_path_raw:
        return jsonify({"error": "repo_path is required."}), 400
    repo_path = Path(repo_path_raw).expanduser().resolve()
    if not repo_path.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_path_raw}"}), 400

    current = _load_legal_review_state(repo_path)
    if not current:
        return jsonify({"error": "Legal review state not found for this repository."}), 404

    required = (
        _safe_bool(data.get("required"), bool(current.get("required", True)))
        if "required" in data
        else bool(current.get("required", True))
    )
    approved = _safe_bool(data.get("approved"), bool(current.get("approved", False)))
    reviewer = str(data.get("reviewer") or current.get("reviewer") or "").strip()
    notes = str(data.get("notes") or current.get("notes") or "").strip()
    files_raw = current.get("files")
    files = files_raw if isinstance(files_raw, list) else []
    project_name = str(current.get("project_name") or repo_path.name).strip() or repo_path.name
    updated = _upsert_legal_review_state(
        project_path=repo_path,
        project_name=project_name,
        required=required,
        approved=approved,
        reviewer=reviewer,
        notes=notes,
        files=files,
        source="api_signoff",
    )
    return jsonify({"status": "saved", "legal_review": updated})


@app.route("/api/project/foundation/improve", methods=["POST"])
def api_improve_foundational_prompt():
    """Improve a foundational prompt using selected assistants/models."""
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt") or "").strip()
    project_name = str(data.get("project_name") or "Project").strip() or "Project"
    assistants = _normalize_foundation_assistants(data.get("assistants"))
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    payload = _improve_foundational_prompt(
        prompt,
        project_name=project_name,
        assistants=assistants,
    )
    return jsonify(payload)


@app.route("/api/project/foundation/bootstrap/run", methods=["POST"])
def api_run_foundation_bootstrap():
    """Start (or queue) the one-time foundational bootstrap chain for a repo."""
    data = request.get_json(silent=True) or {}
    repo_path_raw = str(data.get("repo_path") or "").strip()
    if not repo_path_raw:
        return jsonify({"error": "repo_path is required."}), 400
    repo_path = Path(repo_path_raw).expanduser().resolve()
    if not repo_path.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_path_raw}"}), 400
    request_path = _foundation_bootstrap_request_path(repo_path)
    if not request_path.is_file():
        return jsonify(
            {
                "error": (
                    "Bootstrap request not found. "
                    "Enable foundational bootstrap when creating the project first."
                )
            }
        ), 404
    project_name = str(data.get("project_name") or repo_path.name).strip() or repo_path.name
    description = str(data.get("description") or "").strip()
    status = _start_foundation_bootstrap_chain(
        project_path=repo_path,
        project_name=project_name,
        description=description,
        source="manual_api",
    )
    return jsonify({"status": "started", "bootstrap": status})


@app.route("/api/project/foundation/bootstrap/status")
def api_foundation_bootstrap_status():
    """Return the current one-time bootstrap status for a repository."""
    repo_path_raw = str(request.args.get("repo_path", "") or "").strip()
    if not repo_path_raw:
        return jsonify({"error": "repo_path is required."}), 400
    repo_path = Path(repo_path_raw).expanduser().resolve()
    if not repo_path.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_path_raw}"}), 400
    return jsonify(_foundation_bootstrap_status(repo_path))


# â”€â”€ Config persistence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


# -- Workspace multi-repo management ------------------------------------------


def _workspace_default_store() -> dict[str, object]:
    """Return the default persisted multi-repo workspace payload."""
    return {
        "active_repo_path": "",
        "repos": [],
        "updated_at_epoch_ms": int(time.time() * 1000),
    }


def _normalize_workspace_repo_path(raw_path: object) -> str:
    """Normalize a user-supplied workspace repo path to an absolute path string."""
    raw = str(raw_path or "").strip()
    if not raw:
        return ""
    try:
        return str(Path(raw).expanduser().resolve())
    except Exception:
        return ""


def _normalize_workspace_repo_paths(raw_paths: object) -> list[str]:
    """Normalize and deduplicate stored workspace repo paths."""
    if not isinstance(raw_paths, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_paths:
        repo_path = _normalize_workspace_repo_path(raw_value)
        if not repo_path:
            continue
        key = repo_path.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(repo_path)
    return normalized


def _load_workspace_store() -> dict[str, object]:
    """Load persisted workspace repo metadata from disk."""
    defaults = _workspace_default_store()
    path = WORKSPACE_REPOS_PATH
    if not path.is_file():
        return defaults
    try:
        payload = json.loads(_read_text_utf8_resilient(path))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load workspace repo store %s: %s", path, exc)
        return defaults

    if not isinstance(payload, dict):
        return defaults

    repos = _normalize_workspace_repo_paths(payload.get("repos"))
    active_repo = _normalize_workspace_repo_path(payload.get("active_repo_path"))
    if active_repo and active_repo.casefold() not in {item.casefold() for item in repos}:
        active_repo = ""
    if not active_repo and repos:
        active_repo = repos[0]

    return {
        "active_repo_path": active_repo,
        "repos": repos,
        "updated_at_epoch_ms": _safe_int(
            payload.get("updated_at_epoch_ms"),
            int(time.time() * 1000),
        ),
    }


def _save_workspace_store(payload: dict[str, object]) -> dict[str, object]:
    """Persist workspace repo metadata and return normalized payload."""
    repos = _normalize_workspace_repo_paths(payload.get("repos"))
    active_repo = _normalize_workspace_repo_path(payload.get("active_repo_path"))
    repo_keys = {item.casefold() for item in repos}
    if active_repo and active_repo.casefold() not in repo_keys:
        active_repo = ""
    if not active_repo and repos:
        active_repo = repos[0]

    normalized = {
        "active_repo_path": active_repo,
        "repos": repos,
        "updated_at_epoch_ms": int(time.time() * 1000),
    }
    _write_json_file_atomic(WORKSPACE_REPOS_PATH, normalized)
    return normalized


def _workspace_remote_settings_defaults() -> dict[str, object]:
    """Return default remote-settings payload for workspace rows."""
    return {
        "default_remote": "",
        "default_remote_source": "none",
        "tracking_remote": "",
        "configured_default_remote": "",
        "configured_default_missing": False,
        "remote_names": [],
    }


def _workspace_recent_runs_defaults(message: str = "No recent runs found.") -> dict[str, object]:
    """Return default recent-run summary payload for workspace rows."""
    return {
        "available": False,
        "message": message,
        "latest": None,
        "count": 0,
        "runs": [],
    }


def _git_workspace_remote_settings(repo: Path, *, tracking_remote: str) -> dict[str, object]:
    """Return lightweight per-repo remote settings for workspace summaries."""
    result = _run_git_sync_command(repo, "remote")
    if result.returncode != 0:
        raise RuntimeError(_extract_git_process_error(result, "git remote failed"))

    names = sorted(
        {raw.strip() for raw in str(result.stdout or "").splitlines() if raw.strip()},
        key=lambda item: item.casefold(),
    )
    name_set = {item.casefold() for item in names}
    configured_default_remote = _git_configured_push_default_remote(repo)
    configured_default_key = configured_default_remote.casefold()
    tracking_key = str(tracking_remote or "").strip().casefold()

    default_remote = ""
    default_source = "none"
    if configured_default_remote and configured_default_key in name_set:
        default_remote = configured_default_remote
        default_source = "config"
    elif tracking_key and tracking_key in name_set:
        default_remote = next((item for item in names if item.casefold() == tracking_key), "")
        default_source = "tracking"
    elif "origin" in name_set:
        default_remote = next((item for item in names if item.casefold() == "origin"), "origin")
        default_source = "origin"

    return {
        "default_remote": default_remote,
        "default_remote_source": default_source,
        "tracking_remote": tracking_remote,
        "configured_default_remote": configured_default_remote,
        "configured_default_missing": bool(
            configured_default_remote and configured_default_key not in name_set
        ),
        "remote_names": names,
    }


def _git_workspace_branch_choices(repo: Path) -> dict[str, object]:
    """Return local/remote branch choices for workspace quick-checkout actions."""
    branch_result = _run_git_sync_command(repo, "rev-parse", "--abbrev-ref", "HEAD")
    if branch_result.returncode != 0:
        raise RuntimeError(
            _extract_git_process_error(branch_result, "git rev-parse --abbrev-ref HEAD failed")
        )
    current_branch = str(branch_result.stdout or "").strip() or "HEAD"

    local_result = _run_git_sync_command(
        repo,
        "for-each-ref",
        "--format=%(refname:short)",
        "refs/heads",
    )
    if local_result.returncode != 0:
        raise RuntimeError(_extract_git_process_error(local_result, "git for-each-ref refs/heads failed"))
    local_branches = sorted(
        {raw.strip() for raw in str(local_result.stdout or "").splitlines() if raw.strip()},
        key=lambda item: item.casefold(),
    )

    remote_result = _run_git_sync_command(
        repo,
        "for-each-ref",
        "--format=%(refname:short)",
        "refs/remotes",
    )
    if remote_result.returncode != 0:
        raise RuntimeError(
            _extract_git_process_error(remote_result, "git for-each-ref refs/remotes failed")
        )
    remote_branches = sorted(
        {
            raw.strip()
            for raw in str(remote_result.stdout or "").splitlines()
            if raw.strip() and raw.strip() != "HEAD" and not raw.strip().endswith("/HEAD")
        },
        key=lambda item: item.casefold(),
    )

    return {
        "current_branch": current_branch,
        "local_branches": local_branches,
        "remote_branches": remote_branches,
    }


def _workspace_recent_runs_summary(repo: Path, *, limit: int) -> dict[str, object]:
    """Return recent run summaries for a workspace repo entry."""
    normalized_limit = min(12, max(1, int(limit or 3)))
    comparison = _pipeline_run_comparison(repo, scope="all", limit=normalized_limit)
    runs_raw = comparison.get("runs") if isinstance(comparison, dict) else []
    runs: list[dict[str, object]] = []
    if isinstance(runs_raw, list):
        for raw_run in runs_raw:
            if not isinstance(raw_run, dict):
                continue
            runs.append(
                {
                    "run_id": str(raw_run.get("run_id") or "").strip(),
                    "scope": str(raw_run.get("scope") or "").strip(),
                    "mode": str(raw_run.get("mode") or "").strip(),
                    "finished_at": str(raw_run.get("finished_at") or "").strip(),
                    "finished_at_epoch_ms": _safe_int(raw_run.get("finished_at_epoch_ms"), 0),
                    "duration_seconds": _safe_float(raw_run.get("duration_seconds"), 0.0),
                    "token_usage": _safe_int(raw_run.get("token_usage"), 0),
                    "estimated_cost_usd": round(
                        max(0.0, _safe_float(raw_run.get("estimated_cost_usd"), 0.0)),
                        6,
                    ),
                    "tests_summary": str(raw_run.get("tests_summary") or "").strip(),
                    "stop_reason": str(raw_run.get("stop_reason") or "").strip(),
                    "configuration": str(raw_run.get("configuration") or "").strip(),
                }
            )

    latest = runs[0] if runs else None
    return {
        "available": bool(comparison.get("available")) if isinstance(comparison, dict) else False,
        "message": str(comparison.get("message") or "").strip()
        if isinstance(comparison, dict)
        else "No recent runs found.",
        "latest": latest,
        "count": len(runs),
        "runs": runs,
    }


def _workspace_repo_entry(
    repo_path: str,
    *,
    include_branches: bool,
    recent_runs_limit: int,
) -> dict[str, object]:
    """Build one workspace repository entry with sync/remotes/runs summaries."""
    resolved = Path(repo_path).expanduser().resolve()
    payload: dict[str, object] = {
        "repo_path": str(resolved),
        "name": resolved.name or str(resolved),
        "exists": resolved.is_dir(),
        "is_git": False,
        "available": False,
        "sync": None,
        "remote_settings": _workspace_remote_settings_defaults(),
        "recent_runs": _workspace_recent_runs_defaults(),
        "errors": [],
    }
    if include_branches:
        payload["branches"] = {
            "current_branch": "",
            "local_branches": [],
            "remote_branches": [],
        }

    if not resolved.is_dir():
        payload["errors"] = [f"Repository path not found: {resolved}"]
        payload["recent_runs"] = _workspace_recent_runs_defaults(
            "Repository path is unavailable on disk.",
        )
        return payload

    probe = _run_git_sync_command(resolved, "rev-parse", "--is-inside-work-tree")
    if probe.returncode != 0 or str(probe.stdout or "").strip().lower() != "true":
        payload["errors"] = [f"Not a git repository: {resolved}"]
        payload["recent_runs"] = _workspace_recent_runs_defaults(
            "Directory exists, but is not a git repository.",
        )
        return payload

    payload["is_git"] = True
    try:
        sync_payload = _git_sync_status_core_payload(resolved)
        tracking_remote = str(sync_payload.get("tracking_remote") or "").strip()
        payload["sync"] = sync_payload
        payload["remote_settings"] = _git_workspace_remote_settings(
            resolved,
            tracking_remote=tracking_remote,
        )
        payload["recent_runs"] = _workspace_recent_runs_summary(
            resolved,
            limit=recent_runs_limit,
        )
        if include_branches:
            payload["branches"] = _git_workspace_branch_choices(resolved)
        payload["available"] = True
    except subprocess.TimeoutExpired:
        payload["errors"] = [f"Timed out while reading git metadata for {resolved}."]
    except RuntimeError as exc:
        payload["errors"] = [str(exc)]
    except Exception as exc:
        payload["errors"] = [str(exc)]
    return payload


def _workspace_payload(
    *,
    include_branches: bool,
    recent_runs_limit: int,
) -> dict[str, object]:
    """Return full workspace payload including per-repo summaries."""
    store = _load_workspace_store()
    repo_paths = _normalize_workspace_repo_paths(store.get("repos"))
    active_repo_path = _normalize_workspace_repo_path(store.get("active_repo_path"))
    repo_keys = {item.casefold() for item in repo_paths}
    if active_repo_path and active_repo_path.casefold() not in repo_keys:
        active_repo_path = ""

    repos = [
        _workspace_repo_entry(
            repo_path,
            include_branches=include_branches,
            recent_runs_limit=recent_runs_limit,
        )
        for repo_path in repo_paths
    ]

    available_count = sum(1 for entry in repos if bool(entry.get("available")))
    return {
        "active_repo_path": active_repo_path,
        "repos": repos,
        "total_repos": len(repos),
        "available_repos": available_count,
        "updated_at_epoch_ms": _safe_int(store.get("updated_at_epoch_ms"), int(time.time() * 1000)),
    }


@app.route("/api/workspace/repos")
def api_workspace_repos():
    """List workspace repositories with sync/remotes/recent-run summaries."""
    include_branches = _safe_bool(request.args.get("include_branches"), default=False)
    recent_runs_limit = _safe_int(request.args.get("recent_runs_limit"), 3)
    recent_runs_limit = min(12, max(1, recent_runs_limit))
    return jsonify(
        _workspace_payload(
            include_branches=include_branches,
            recent_runs_limit=recent_runs_limit,
        )
    )


@app.route("/api/workspace/repos/add", methods=["POST"])
def api_workspace_repo_add():
    """Add one git repository path to the persisted workspace list."""
    data = request.get_json(silent=True) or {}
    repo, error, status = _resolve_git_sync_repo(data.get("repo_path"))
    if repo is None:
        return jsonify({"error": error}), status

    include_branches = _safe_bool(data.get("include_branches"), default=True)
    recent_runs_limit = min(12, max(1, _safe_int(data.get("recent_runs_limit"), 3)))
    make_active = _safe_bool(data.get("make_active"), default=False)
    normalized_repo_path = str(repo)

    store = _load_workspace_store()
    repos = _normalize_workspace_repo_paths(store.get("repos"))
    repo_keys = {item.casefold() for item in repos}
    was_added = normalized_repo_path.casefold() not in repo_keys
    if was_added:
        repos.append(normalized_repo_path)

    active_repo_path = _normalize_workspace_repo_path(store.get("active_repo_path"))
    if not active_repo_path and repos:
        active_repo_path = repos[0]
    if make_active:
        active_repo_path = normalized_repo_path

    saved = _save_workspace_store(
        {
            "repos": repos,
            "active_repo_path": active_repo_path,
        }
    )
    workspace = _workspace_payload(
        include_branches=include_branches,
        recent_runs_limit=recent_runs_limit,
    )
    return jsonify(
        {
            "status": "added" if was_added else "unchanged",
            "repo_path": normalized_repo_path,
            "active_repo_path": str(saved.get("active_repo_path") or ""),
            "workspace": workspace,
            "message": (
                f"Added workspace repo: {normalized_repo_path}"
                if was_added
                else f"Workspace repo already exists: {normalized_repo_path}"
            ),
        }
    )


@app.route("/api/workspace/repos/remove", methods=["POST"])
def api_workspace_repo_remove():
    """Remove one repository path from the workspace list."""
    data = request.get_json(silent=True) or {}
    repo_path = _normalize_workspace_repo_path(data.get("repo_path"))
    if not repo_path:
        return jsonify({"error": "repo_path is required."}), 400

    include_branches = _safe_bool(data.get("include_branches"), default=True)
    recent_runs_limit = min(12, max(1, _safe_int(data.get("recent_runs_limit"), 3)))
    store = _load_workspace_store()
    repos = _normalize_workspace_repo_paths(store.get("repos"))

    removed = False
    filtered_repos: list[str] = []
    target_key = repo_path.casefold()
    for item in repos:
        if item.casefold() == target_key:
            removed = True
            continue
        filtered_repos.append(item)
    if not removed:
        return jsonify({"error": f"Workspace repo not found: {repo_path}"}), 404

    active_repo_path = _normalize_workspace_repo_path(store.get("active_repo_path"))
    if active_repo_path.casefold() == target_key:
        active_repo_path = filtered_repos[0] if filtered_repos else ""

    saved = _save_workspace_store(
        {
            "repos": filtered_repos,
            "active_repo_path": active_repo_path,
        }
    )
    workspace = _workspace_payload(
        include_branches=include_branches,
        recent_runs_limit=recent_runs_limit,
    )
    return jsonify(
        {
            "status": "removed",
            "repo_path": repo_path,
            "active_repo_path": str(saved.get("active_repo_path") or ""),
            "workspace": workspace,
            "message": f"Removed workspace repo: {repo_path}",
        }
    )


@app.route("/api/workspace/repos/activate", methods=["POST"])
def api_workspace_repo_activate():
    """Set the active workspace repository path."""
    data = request.get_json(silent=True) or {}
    repo_path = _normalize_workspace_repo_path(data.get("repo_path"))
    if not repo_path:
        return jsonify({"error": "repo_path is required."}), 400

    include_branches = _safe_bool(data.get("include_branches"), default=True)
    recent_runs_limit = min(12, max(1, _safe_int(data.get("recent_runs_limit"), 3)))
    add_if_missing = _safe_bool(data.get("add_if_missing"), default=False)
    store = _load_workspace_store()
    repos = _normalize_workspace_repo_paths(store.get("repos"))
    repo_keys = {item.casefold() for item in repos}
    if repo_path.casefold() not in repo_keys:
        if not add_if_missing:
            return jsonify({"error": f"Workspace repo not found: {repo_path}"}), 404
        repo, error, status = _resolve_git_sync_repo(repo_path)
        if repo is None:
            return jsonify({"error": error}), status
        repo_path = str(repo)
        repos.append(repo_path)

    saved = _save_workspace_store(
        {
            "repos": repos,
            "active_repo_path": repo_path,
        }
    )
    workspace = _workspace_payload(
        include_branches=include_branches,
        recent_runs_limit=recent_runs_limit,
    )
    return jsonify(
        {
            "status": "activated",
            "repo_path": repo_path,
            "active_repo_path": str(saved.get("active_repo_path") or ""),
            "workspace": workspace,
            "message": f"Active workspace repo set to {repo_path}",
        }
    )


@app.route("/api/configs")
def api_list_configs():
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    return jsonify([{"name": f.stem, "path": str(f)} for f in sorted(CONFIGS_DIR.glob("*.json"))])


@app.route("/api/configs/save", methods=["POST"])
def api_save_config():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "untitled")
    config = data.get("config", {})
    if not isinstance(config, dict):
        return jsonify({"error": "Config must be a JSON object"}), 400

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    raw_name = str(name or "").strip()
    safe = _sanitize_config_name(raw_name) or "untitled"
    if raw_name and safe != raw_name:
        return jsonify({"error": "Invalid config name"}), 400
    path = CONFIGS_DIR / f"{safe}.json"
    try:
        _write_json_file_atomic(path, config)
    except TypeError as exc:
        return jsonify({"error": f"Config must be JSON serializable: {exc}"}), 400
    except OSError as exc:
        return jsonify({"error": f"Could not save config: {exc}"}), 500
    return jsonify({"status": "saved", "path": str(path)})


@app.route("/api/configs/load", methods=["POST"])
def api_load_config():
    data = request.get_json(silent=True) or {}
    raw_name = str(data.get("name", "") or "").strip()
    if not _is_valid_config_name(raw_name):
        return jsonify({"error": "Invalid config name"}), 400
    safe_name = raw_name

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    root = CONFIGS_DIR.resolve()
    path = (root / f"{safe_name}.json").resolve()
    if not _is_within_directory(path, root):
        return jsonify({"error": "Invalid config name"}), 400
    if not path.is_file():
        return jsonify({"error": "Config not found"}), 404

    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return jsonify({"error": "Config file is not valid JSON"}), 400
    if not isinstance(config, dict):
        return jsonify({"error": "Config file must contain a JSON object"}), 400
    return jsonify(config)


# ----------------------------------------------------------------------
# Pipeline API
# ----------------------------------------------------------------------


def _get_pipeline():
    """Get or create the global pipeline executor."""
    global _pipeline_executor
    if _pipeline_executor is None:
        # Placeholder -- will be configured on start
        _pipeline_executor = None
    return _pipeline_executor


@app.route("/api/pipeline/phases")
def api_pipeline_phases():
    """Return available pipeline phases, defaults, and prompt info."""
    from codex_manager.pipeline.phases import (
        CUA_PHASES,
        DEEP_RESEARCH_PHASES,
        DEFAULT_ITERATIONS,
        DEFAULT_PHASE_ORDER,
        PHASE_LOG_FILES,
        SCIENCE_PHASES,
        SELF_IMPROVEMENT_PHASES,
        PipelinePhase,
        default_test_policy_for_phase,
    )

    # Load prompt catalog for phase descriptions and prompt text
    try:
        from codex_manager.prompts.catalog import get_catalog

        catalog = get_catalog()
    except Exception:
        catalog = None

    phases = []
    ordered_phases = list(DEFAULT_PHASE_ORDER)
    try:
        implementation_idx = ordered_phases.index(PipelinePhase.IMPLEMENTATION)
    except ValueError:
        implementation_idx = 0
    ordered_phases[implementation_idx:implementation_idx] = list(SCIENCE_PHASES)
    try:
        prioritization_idx = ordered_phases.index(PipelinePhase.PRIORITIZATION)
    except ValueError:
        prioritization_idx = 1 if ordered_phases else 0
    ordered_phases[prioritization_idx:prioritization_idx] = list(DEEP_RESEARCH_PHASES)
    ordered_phases.extend(CUA_PHASES)
    ordered_phases.extend(SELF_IMPROVEMENT_PHASES)

    for phase in ordered_phases:
        key = phase.value
        is_science = phase in SCIENCE_PHASES
        is_deep_research = phase in DEEP_RESEARCH_PHASES
        is_cua = phase in CUA_PHASES
        is_self_improvement = phase in SELF_IMPROVEMENT_PHASES

        # Get prompt info from catalog
        prompt_text = ""
        description = ""
        if catalog:
            if is_science:
                prompt_text = catalog.scientist(key) or ""
                meta = catalog.scientist_meta(key)
            else:
                prompt_text = catalog.pipeline(key) or ""
                meta = catalog.pipeline_meta(key)
            description = meta.get("description", "")

        phases.append(
            {
                "key": key,
                "name": key.replace("_", " ").title(),
                "default_iterations": DEFAULT_ITERATIONS.get(phase, 1),
                "default_test_policy": default_test_policy_for_phase(phase),
                "log_file": PHASE_LOG_FILES.get(phase, ""),
                "is_science": is_science,
                "is_deep_research": is_deep_research,
                "is_cua": is_cua,
                "is_self_improvement": is_self_improvement,
                "description": description,
                "prompt": prompt_text,
            }
        )
    return jsonify(phases)


@app.route("/api/pipeline/resume-state")
def api_pipeline_resume_state():
    """Return whether a repository has a resumable pipeline checkpoint."""
    repo_path_raw = str(request.args.get("repo_path") or "").strip()
    if not repo_path_raw:
        return jsonify({"error": "repo_path is required."}), 400

    repo_path = Path(repo_path_raw).expanduser().resolve()
    if not repo_path.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_path_raw}"}), 400

    return jsonify(_pipeline_resume_summary(repo_path))


@app.route("/api/pipeline/resume-state/clear", methods=["POST"])
def api_pipeline_resume_state_clear():
    """Delete a repository's pipeline resume checkpoint if present."""
    data = request.get_json(silent=True) or {}
    repo_path_raw = str(data.get("repo_path") or "").strip()
    if not repo_path_raw:
        return jsonify({"error": "repo_path is required."}), 400

    repo_path = Path(repo_path_raw).expanduser().resolve()
    if not repo_path.is_dir():
        return jsonify({"error": f"Repo path not found: {repo_path_raw}"}), 400

    checkpoint = _pipeline_resume_checkpoint_path(repo_path)
    removed = False
    try:
        if checkpoint.is_file():
            checkpoint.unlink(missing_ok=True)
            removed = True
    except OSError as exc:
        return jsonify({"error": f"Could not clear checkpoint: {exc}"}), 500

    return jsonify(
        {
            "status": "cleared",
            "removed": removed,
            "checkpoint_path": str(checkpoint.resolve()),
        }
    )


def _start_pipeline_from_gui_config(gui_config: PipelineGUIConfig) -> tuple[dict[str, object], int]:
    """Validate and start pipeline execution from GUI config payload."""
    global _pipeline_executor

    if _pipeline_executor is not None and _pipeline_executor.is_running:
        return {"error": "Pipeline is already running"}, 409

    if not Path(gui_config.repo_path).is_dir():
        return {"error": f"Repo path not found: {gui_config.repo_path}"}, 400

    issues = _pipeline_preflight_issues(gui_config)
    if issues:
        msg = "Preflight checks failed:\n" + "\n".join(f"- {i}" for i in issues)
        return {"error": msg, "issues": issues}, 400

    git_preflight: dict[str, object] | None = None
    if gui_config.git_preflight_enabled:
        try:
            git_preflight = _git_preflight_before_run(
                Path(gui_config.repo_path).resolve(),
                auto_stash=bool(gui_config.git_preflight_auto_stash),
                auto_pull=bool(gui_config.git_preflight_auto_pull),
            )
        except subprocess.TimeoutExpired:
            return {"error": "Git pre-flight checks timed out."}, 504
        except RuntimeError as exc:
            return {"error": f"Git pre-flight checks failed: {exc}"}, 502
        except Exception as exc:
            return {"error": f"Git pre-flight checks failed: {exc}"}, 500

        git_issues = [str(item) for item in git_preflight.get("issues", []) if str(item).strip()]
        if git_issues:
            msg = "Git pre-flight checks failed:\n" + "\n".join(f"- {i}" for i in git_issues)
            return {"error": msg, "issues": git_issues, "git_preflight": git_preflight}, 400

    # Convert GUI config to pipeline config
    from codex_manager.pipeline.phases import PhaseConfig, PipelineConfig, PipelinePhase

    phase_configs = []
    invalid_phases: list[str] = []
    for pg in gui_config.phases:
        try:
            phase_configs.append(
                PhaseConfig(
                    phase=PipelinePhase(pg.phase),
                    enabled=pg.enabled,
                    iterations=pg.iterations,
                    agent=pg.agent,
                    on_failure=pg.on_failure,
                    max_retries=pg.max_retries,
                    test_policy=pg.test_policy,
                    custom_prompt=pg.custom_prompt,
                )
            )
        except ValueError:
            invalid_phases.append(pg.phase)
    if invalid_phases:
        msg = ", ".join(sorted(set(invalid_phases)))
        return {"error": f"Invalid pipeline phase(s): {msg}"}, 400

    config = PipelineConfig(
        mode=gui_config.mode,
        max_cycles=gui_config.max_cycles,
        unlimited=gui_config.unlimited,
        agent=gui_config.agent,
        science_enabled=gui_config.science_enabled,
        brain_enabled=gui_config.brain_enabled,
        brain_model=gui_config.brain_model,
        local_only=gui_config.local_only,
        cua_enabled=gui_config.cua_enabled,
        cua_provider=gui_config.cua_provider,
        cua_target_url=gui_config.cua_target_url,
        cua_task=gui_config.cua_task,
        smoke_test_cmd=gui_config.smoke_test_cmd,
        test_cmd=gui_config.test_cmd,
        codex_binary=gui_config.codex_binary,
        claude_binary=gui_config.claude_binary,
        codex_sandbox_mode=gui_config.codex_sandbox_mode,
        codex_approval_policy=gui_config.codex_approval_policy,
        codex_reasoning_effort=gui_config.codex_reasoning_effort,
        codex_bypass_approvals_and_sandbox=gui_config.codex_bypass_approvals_and_sandbox,
        codex_danger_confirmation=gui_config.codex_danger_confirmation,
        allow_path_creation=gui_config.allow_path_creation,
        dependency_install_policy=gui_config.dependency_install_policy,
        image_generation_enabled=gui_config.image_generation_enabled,
        image_provider=gui_config.image_provider,
        image_model=gui_config.image_model,
        vector_memory_enabled=gui_config.vector_memory_enabled,
        vector_memory_backend=gui_config.vector_memory_backend,
        vector_memory_collection=gui_config.vector_memory_collection,
        vector_memory_top_k=gui_config.vector_memory_top_k,
        deep_research_enabled=gui_config.deep_research_enabled,
        deep_research_providers=gui_config.deep_research_providers,
        deep_research_max_age_hours=gui_config.deep_research_max_age_hours,
        deep_research_dedupe=gui_config.deep_research_dedupe,
        deep_research_native_enabled=gui_config.deep_research_native_enabled,
        deep_research_retry_attempts=gui_config.deep_research_retry_attempts,
        deep_research_daily_quota=gui_config.deep_research_daily_quota,
        deep_research_max_provider_tokens=gui_config.deep_research_max_provider_tokens,
        deep_research_budget_usd=gui_config.deep_research_budget_usd,
        deep_research_openai_model=gui_config.deep_research_openai_model,
        deep_research_google_model=gui_config.deep_research_google_model,
        self_improvement_enabled=gui_config.self_improvement_enabled,
        self_improvement_auto_restart=gui_config.self_improvement_auto_restart,
        timeout_per_phase=gui_config.timeout_per_phase,
        artifact_retention_enabled=gui_config.artifact_retention_enabled,
        artifact_retention_max_age_days=gui_config.artifact_retention_max_age_days,
        artifact_retention_max_files=gui_config.artifact_retention_max_files,
        artifact_retention_max_bytes=gui_config.artifact_retention_max_bytes,
        artifact_retention_max_output_runs=gui_config.artifact_retention_max_output_runs,
        run_completion_webhooks=gui_config.run_completion_webhooks,
        run_completion_webhook_timeout_seconds=gui_config.run_completion_webhook_timeout_seconds,
        max_total_tokens=gui_config.max_total_tokens,
        strict_token_budget=gui_config.strict_token_budget,
        max_time_minutes=gui_config.max_time_minutes,
        stop_on_convergence=gui_config.stop_on_convergence,
        improvement_threshold=gui_config.improvement_threshold,
        auto_commit=gui_config.auto_commit,
        commit_frequency=gui_config.commit_frequency,
        pr_aware_enabled=gui_config.pr_aware_enabled,
        pr_feature_branch=gui_config.pr_feature_branch,
        pr_remote=gui_config.pr_remote,
        pr_base_branch=gui_config.pr_base_branch,
        pr_auto_push=gui_config.pr_auto_push,
        pr_sync_description=gui_config.pr_sync_description,
        phases=phase_configs if phase_configs else [],
    )

    from codex_manager.pipeline.orchestrator import PipelineOrchestrator

    _pipeline_executor = PipelineOrchestrator(
        repo_path=gui_config.repo_path,
        config=config,
    )
    _pipeline_executor.start()
    payload: dict[str, object] = {"status": "started"}
    if git_preflight is not None:
        payload["git_preflight"] = git_preflight
    return payload, 200


@app.route("/api/pipeline/start", methods=["POST"])
def api_pipeline_start():
    """Start the autonomous pipeline."""
    data = request.get_json(silent=True) or {}
    try:
        gui_config = PipelineGUIConfig(**data)
    except Exception as exc:
        return jsonify({"error": f"Invalid config: {exc}"}), 400

    payload, status = _start_pipeline_from_gui_config(gui_config)
    return jsonify(payload), status


@app.route("/api/pipeline/promote-last-dry-run")
def api_pipeline_promote_last_dry_run():
    """Return preview payload for promoting the latest dry-run to apply mode."""
    repo = _resolve_pipeline_logs_repo(request.args.get("repo_path", ""))
    if repo is None:
        return jsonify(
            {
                "available": False,
                "repo_path": "",
                "history_path": "",
                "run": None,
                "promoted_config": None,
                "message": "Set Repository Path in the Pipeline panel to load dry-run promotion details.",
            }
        )
    return jsonify(_pipeline_promote_last_dry_run_payload(repo))


@app.route("/api/pipeline/promote-last-dry-run/start", methods=["POST"])
def api_pipeline_promote_last_dry_run_start():
    """Promote latest dry-run pipeline config to apply mode and start the run."""
    data = request.get_json(silent=True) or {}
    repo_raw = str(data.get("repo_path") or "").strip()
    repo = _resolve_pipeline_logs_repo(repo_raw)
    if repo is None:
        return jsonify({"error": "repo_path is required and must point to a valid repository."}), 400

    preview = _pipeline_promote_last_dry_run_payload(repo)
    if not preview.get("available"):
        return jsonify({"error": preview.get("message") or "No promotable dry-run found."}), 400

    config_payload_obj = preview.get("promoted_config")
    if not isinstance(config_payload_obj, dict):
        return jsonify({"error": "Promoted config was not available from run history."}), 500

    try:
        gui_config = PipelineGUIConfig(**config_payload_obj)
    except Exception as exc:
        return jsonify({"error": f"Promoted config is invalid: {exc}"}), 500

    payload, status = _start_pipeline_from_gui_config(gui_config)
    if status != 200:
        return jsonify(payload), status

    run_obj = preview.get("run")
    run = dict(run_obj) if isinstance(run_obj, dict) else {}
    payload.update(
        {
            "promoted": True,
            "promoted_from_run_id": str(run.get("run_id") or ""),
            "promoted_from_finished_at": str(run.get("finished_at") or ""),
            "promoted_mode": "apply",
            "promoted_config": gui_config.model_dump(),
        }
    )
    return jsonify(payload), 200


@app.route("/api/pipeline/stop", methods=["POST"])
def api_pipeline_stop():
    """Stop the running pipeline."""
    global _pipeline_executor
    if _pipeline_executor is not None:
        _pipeline_executor.stop()
    return jsonify({"status": "stopping"})


@app.route("/api/pipeline/pause", methods=["POST"])
def api_pipeline_pause():
    """Pause/resume the pipeline."""
    global _pipeline_executor
    if _pipeline_executor is None:
        return jsonify({"error": "No pipeline running"}), 400
    if _pipeline_executor.state.paused:
        _pipeline_executor.resume()
        return jsonify({"status": "resumed"})
    _pipeline_executor.pause()
    return jsonify({"status": "paused"})


@app.route("/api/pipeline/status")
def api_pipeline_status():
    """Return current pipeline state."""
    global _pipeline_executor
    since_results = _parse_since_results_arg(request.args.get("since_results"))
    if _pipeline_executor is None:
        from codex_manager.pipeline.phases import PipelineState

        payload = _attach_pipeline_runtime_fields(
            PipelineState().to_summary(since_results=since_results)
        )
        return jsonify(
            _attach_stop_guidance(
                payload,
                mode="pipeline",
            )
        )

    to_summary = getattr(_pipeline_executor.state, "to_summary", None)
    if not callable(to_summary):
        payload = {}
    else:
        try:
            payload = to_summary(since_results=since_results)
        except TypeError:
            # Backward-compatible fallback for patched/mocked states in tests.
            payload = to_summary()
    if isinstance(payload, dict):
        payload = _attach_pipeline_runtime_fields(payload)
    return jsonify(_attach_stop_guidance(payload, mode="pipeline"))


@app.route("/api/pipeline/stream")
def api_pipeline_stream():
    """SSE stream for pipeline logs."""
    global _pipeline_executor
    resume_from = _parse_sse_resume_id()

    def generate():
        last_id = resume_from
        while True:
            if _pipeline_executor is not None:
                events, replay_gap = _replay_log_events(_pipeline_executor, after_id=last_id)
                if replay_gap:
                    yield _sse_frame(
                        {
                            "type": "warning",
                            "message": (
                                "Some older pipeline log events were dropped before replay "
                                "could resume."
                            ),
                        }
                    )
                if events:
                    for entry in events:
                        event_id = _safe_int(entry.get("id"), 0)
                        if event_id > last_id:
                            last_id = event_id
                        yield _sse_frame(entry, event_id=event_id if event_id > 0 else None)
                    continue
            time.sleep(2.0)
            yield _sse_frame({"type": "heartbeat", "last_event_id": last_id})

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/pipeline/logs/<filename>")
def api_pipeline_log(filename: str):
    """Read a pipeline log file (WISHLIST.md, TESTPLAN.md, etc.)."""
    global _pipeline_executor
    if filename not in _PIPELINE_LOG_FILES:
        return jsonify({"error": "Invalid log file"}), 400

    repo, resolve_error, resolve_status = _resolve_pipeline_logs_repo_for_api(
        request.args.get("repo_path", "")
    )
    if repo is None:
        if _pipeline_executor is not None and hasattr(_pipeline_executor, "tracker"):
            content = _pipeline_executor.tracker.read(filename)
            return jsonify(
                {
                    "content": content,
                    "exists": bool(content),
                    "filename": filename,
                    "repo_path": "",
                    "logs_dir": "",
                }
            )
        return (
            jsonify(
                {
                    "error": resolve_error or "Could not resolve repository for pipeline logs.",
                    "content": "",
                    "exists": False,
                    "filename": filename,
                    "repo_path": "",
                    "logs_dir": "",
                }
            ),
            resolve_status,
        )

    content = ""
    log_path = _pipeline_logs_dir(repo) / filename
    if _pipeline_executor is not None:
        executor_repo_raw = str(getattr(_pipeline_executor, "repo_path", "")).strip()
        if executor_repo_raw:
            executor_repo = Path(executor_repo_raw)
            if executor_repo.is_dir() and executor_repo.resolve() == repo:
                content = _pipeline_executor.tracker.read(filename)
                log_path = _pipeline_executor.tracker.path_for(filename)

    if not content:
        from codex_manager.pipeline.tracker import LogTracker

        tracker = LogTracker(repo)
        content = tracker.read(filename)
        log_path = tracker.path_for(filename)

    return jsonify(
        {
            "content": content,
            "exists": log_path.is_file(),
            "filename": filename,
            "repo_path": str(repo),
            "logs_dir": str(log_path.parent),
        }
    )


@app.route("/api/pipeline/run-comparison/export", methods=["POST"])
def api_pipeline_run_comparison_export():
    """Create a run-scoped artifact bundle zip for sharing/debugging."""
    data = request.get_json(silent=True) or {}
    repo, resolve_error, resolve_status = _resolve_pipeline_logs_repo_for_api(
        str(data.get("repo_path") or "")
    )
    if repo is None:
        return jsonify({"error": resolve_error or "Could not resolve repository path."}), resolve_status

    run_id = str(data.get("run_id") or "").strip()
    if not run_id:
        return jsonify({"error": "run_id is required."}), 400

    includes = _normalize_run_artifact_bundle_includes(data)
    if not any(includes.values()):
        return jsonify({"error": "Select at least one artifact category to export."}), 400

    try:
        payload = _create_run_artifact_bundle(repo, run_id=run_id, includes=includes)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive safety net
        logger.exception("Could not export run artifact bundle.")
        return jsonify({"error": f"Could not create artifact bundle: {exc}"}), 500
    return jsonify(payload)


@app.route("/api/pipeline/run-comparison/export/<path:bundle_name>")
def api_pipeline_run_comparison_export_download(bundle_name: str):
    """Download a previously-created run artifact bundle zip."""
    repo, resolve_error, resolve_status = _resolve_pipeline_logs_repo_for_api(
        request.args.get("repo_path", "")
    )
    if repo is None:
        return jsonify({"error": resolve_error or "Could not resolve repository path."}), resolve_status

    safe_name = Path(bundle_name).name
    if safe_name != bundle_name or not safe_name.lower().endswith(".zip"):
        return jsonify({"error": "Invalid bundle name."}), 400

    export_dir = _pipeline_artifact_bundle_dir(repo).resolve()
    bundle_path = (export_dir / safe_name).resolve()
    if bundle_path.parent != export_dir:
        return jsonify({"error": "Invalid bundle name."}), 400
    if not bundle_path.is_file():
        return jsonify({"error": f"Artifact bundle not found: {safe_name}"}), 404
    return send_file(
        bundle_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name=safe_name,
    )


@app.route("/api/pipeline/run-comparison")
def api_pipeline_run_comparison():
    """Return recent run-comparison metrics from HISTORY.jsonl."""
    scope = _run_comparison_scope(request.args.get("scope"))
    limit = _run_comparison_limit(request.args.get("limit"))
    repo = _resolve_pipeline_logs_repo(request.args.get("repo_path", ""))
    if repo is None:
        return jsonify(
            _empty_run_comparison_payload(
                repo=None,
                scope=scope,
                limit=limit,
                message=(
                    "Set Repository Path in the Pipeline panel to compare recent run metrics."
                ),
            )
        )
    return jsonify(_pipeline_run_comparison(repo, scope=scope, limit=limit))


@app.route("/api/pipeline/science-dashboard")
def api_pipeline_science_dashboard():
    """Return structured Scientist dashboard data for the GUI modal."""
    repo = _resolve_pipeline_logs_repo(request.args.get("repo_path", ""))
    if repo is None:
        return jsonify(
            {
                "available": False,
                "repo_path": "",
                "summary": {
                    "current_cycle": 0,
                    "science_trials": 0,
                    "hypotheses": 0,
                    "supported": 0,
                    "refuted": 0,
                    "inconclusive": 0,
                    "rollbacks": 0,
                    "trial_tokens": 0,
                    "implementation_phases": 0,
                },
                "action_items": [],
                "timeline": [],
                "phase_breakdown": [],
                "implementation": [],
                "top_files": [],
                "analysis_excerpt": "",
                "message": "Set Repository Path in the Pipeline panel to load Scientist dashboard data.",
            }
        )

    logs_dir = _pipeline_logs_dir(repo)
    science_dir = logs_dir / "scientist"
    report_path = logs_dir / "SCIENTIST_REPORT.md"
    trials_path = science_dir / "TRIALS.jsonl"

    report_text = ""
    if report_path.is_file():
        try:
            report_text = _read_text_utf8_resilient(report_path)
        except Exception as exc:
            logger.warning("Could not read Scientist report %s: %s", report_path, exc)

    timeline: list[dict[str, object]] = []
    phase_counts: dict[str, int] = {}
    verdict_counts = {"supported": 0, "refuted": 0, "inconclusive": 0}
    hypotheses: set[str] = set()
    rollback_count = 0
    trial_tokens = 0

    if trials_path.is_file():
        for payload in _read_jsonl_dict_rows(trials_path, warn_context="science trials"):
            hypothesis = payload.get("hypothesis") or {}
            baseline = payload.get("baseline") or {}
            post = payload.get("post") or {}
            usage = payload.get("usage") or {}
            hypothesis_id = str(hypothesis.get("id", "")).strip()
            verdict = str(payload.get("verdict", "")).strip().lower()
            phase = str(payload.get("phase", "")).strip().lower()
            rollback = str(payload.get("rollback_action", "")).strip().lower()

            if hypothesis_id:
                hypotheses.add(hypothesis_id)
            if verdict in verdict_counts:
                verdict_counts[verdict] += 1
            if rollback == "reverted":
                rollback_count += 1
            if phase:
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
            trial_tokens += _safe_int(usage.get("total_tokens"), 0)

            timeline.append(
                {
                    "timestamp": str(payload.get("timestamp", "")).strip(),
                    "cycle": _safe_int(payload.get("cycle"), 0),
                    "phase": phase,
                    "hypothesis_id": hypothesis_id,
                    "verdict": verdict or "n/a",
                    "confidence": str(payload.get("confidence", "")).strip().lower() or "n/a",
                    "baseline_test": str(baseline.get("test_outcome", "")).strip() or "n/a",
                    "post_test": str(post.get("test_outcome", "")).strip() or "n/a",
                    "files_changed": _safe_int(post.get("files_changed"), 0),
                    "net_lines_changed": _safe_int(post.get("net_lines_changed"), 0),
                    "rollback_action": rollback or "n/a",
                }
            )

    action_lines = _extract_markdown_section_lines(
        report_text,
        heading_prefix="Action Plan",
        heading_level=2,
    )
    action_items: list[str] = []
    for line in action_lines:
        match = re.match(r"^\s*-\s*\[[ xX]\]\s+(.+)$", line.strip())
        if not match:
            continue
        item = re.sub(r"\s+", " ", match.group(1)).strip()
        if item:
            action_items.append(item)

    impl_lines = _extract_markdown_section_lines(
        report_text,
        heading_prefix="Implementation and Code Changes",
        heading_level=2,
    )
    impl_rows = _parse_markdown_table(impl_lines)
    implementation: list[dict[str, object]] = []
    for row in impl_rows:
        implementation.append(
            {
                "cycle": _safe_int(row.get("Cycle"), 0),
                "phase": str(row.get("Phase", "")).strip().lower(),
                "iteration": _safe_int(row.get("Iter"), 0),
                "status": str(row.get("Status", "")).strip().lower() or "unknown",
                "tests": str(row.get("Tests", "")).strip() or "unknown",
                "files": _safe_int(row.get("Files"), 0),
                "net_delta": str(row.get("Net Delta", "")).strip(),
                "commit": str(row.get("Commit", "")).strip(),
            }
        )

    top_lines = _extract_markdown_section_lines(
        report_text,
        heading_prefix="Most-Touched Files",
        heading_level=3,
    )
    top_rows = _parse_markdown_table(top_lines)
    top_files: list[dict[str, object]] = []
    for row in top_rows:
        path = str(row.get("File", "")).strip()
        if not path:
            continue
        top_files.append({"path": path, "touches": _safe_int(row.get("Touches"), 0)})

    analysis_lines = _extract_markdown_section_lines(
        report_text,
        heading_prefix="Latest Analyze Output",
        heading_level=2,
    )
    analysis_excerpt = _extract_code_fence_text(analysis_lines)
    if len(analysis_excerpt) > 3000:
        analysis_excerpt = analysis_excerpt[:3000].rstrip() + "\n...[truncated]..."

    current_cycle = 0
    if timeline:
        current_cycle = max(_safe_int(item.get("cycle"), 0) for item in timeline)
    elif implementation:
        current_cycle = max(_safe_int(item.get("cycle"), 0) for item in implementation)

    available = bool(report_path.is_file() or trials_path.is_file())
    phase_breakdown = [
        {"phase": phase, "count": count}
        for phase, count in sorted(phase_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    return jsonify(
        {
            "available": available,
            "repo_path": str(repo),
            "paths": {
                "report": str(report_path),
                "trials": str(trials_path),
            },
            "summary": {
                "current_cycle": current_cycle,
                "science_trials": len(timeline),
                "hypotheses": len(hypotheses),
                "supported": verdict_counts["supported"],
                "refuted": verdict_counts["refuted"],
                "inconclusive": verdict_counts["inconclusive"],
                "rollbacks": rollback_count,
                "trial_tokens": trial_tokens,
                "implementation_phases": len(implementation),
            },
            "action_items": action_items[:20],
            "timeline": timeline[-60:],
            "phase_breakdown": phase_breakdown,
            "implementation": implementation[-60:],
            "top_files": top_files[:20],
            "analysis_excerpt": analysis_excerpt,
            "message": (
                ""
                if available
                else "Scientist artifacts are not available yet. Run a pipeline with Scientist Mode enabled."
            ),
        }
    )


# ----------------------------------------------------------------------
# CUA (Computer-Using Agent) API
# ----------------------------------------------------------------------

_cua_result = None  # latest CUA session result (including in-flight placeholder)
_cua_thread: threading.Thread | None = None
_cua_lock = threading.Lock()


@app.route("/api/cua/start", methods=["POST"])
def api_cua_start():
    """Start a standalone CUA visual testing session."""
    global _cua_result, _cua_thread
    data = request.get_json(silent=True) or {}

    try:
        from codex_manager.cua.actions import CUAProvider, CUASessionConfig, CUASessionResult
        from codex_manager.cua.session import run_cua_session_sync

        provider_str = data.get("provider", "openai").lower()
        if provider_str not in {"openai", "anthropic"}:
            return jsonify({"error": f"Unsupported CUA provider: {provider_str}"}), 400
        provider = CUAProvider.OPENAI if provider_str == "openai" else CUAProvider.ANTHROPIC

        cua_config = CUASessionConfig(
            provider=provider,
            target_url=data.get("target_url", ""),
            task=data.get("task", "Take a screenshot and describe what you see"),
            viewport_width=int(data.get("viewport_width", 1280)),
            viewport_height=int(data.get("viewport_height", 800)),
            max_steps=int(data.get("max_steps", 30)),
            timeout_seconds=int(data.get("timeout_seconds", 300)),
            headless=data.get("headless", True),
            save_screenshots=True,
        )

        with _cua_lock:
            if _cua_thread is not None and _cua_thread.is_alive():
                return jsonify({"error": "A CUA session is already running"}), 409

            # Placeholder state so /api/cua/status reports running immediately.
            _cua_result = CUASessionResult(
                task=cua_config.task,
                provider=provider.value,
                started_at=datetime.now().isoformat(),
            )

        def _run() -> None:
            global _cua_result
            try:
                session_result = run_cua_session_sync(cua_config)
            except Exception as exc:
                logger.exception("CUA session thread failed")
                session_result = CUASessionResult(
                    task=cua_config.task,
                    provider=provider.value,
                    success=False,
                    error=str(exc),
                    started_at=datetime.now().isoformat(),
                    finished_at=datetime.now().isoformat(),
                )
            with _cua_lock:
                _cua_result = session_result

        t = threading.Thread(target=_run, daemon=True)
        with _cua_lock:
            _cua_thread = t
            t.start()

        return jsonify({"status": "started", "provider": provider.value})

    except ImportError as exc:
        return jsonify(
            {
                "error": f"CUA dependencies not installed: {exc}. "
                "Install with: pip install warpfoundry[cua] then python -m playwright install"
            }
        ), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/cua/status")
def api_cua_status():
    """Return the latest CUA session result."""
    global _cua_result, _cua_thread
    with _cua_lock:
        current_result = _cua_result
        running = _cua_thread is not None and _cua_thread.is_alive()
    if current_result is None:
        return jsonify({"running": running, "result": None})

    from dataclasses import asdict

    result_dict = asdict(current_result)
    # Don't send full screenshot data over API
    for step in result_dict.get("steps", []):
        step.pop("screenshot_b64", None)
        if "action" in step and "raw" in step["action"]:
            step["action"]["raw"] = {}

    return jsonify(
        {
            "running": running,
            "result": result_dict,
        }
    )


@app.route("/api/cua/providers")
def api_cua_providers():
    """Return available CUA providers and their status."""
    providers = [
        {
            "id": "openai",
            "name": "OpenAI CUA",
            "model": "computer-use-preview",
            "description": "GPT-4o vision + reasoning for GUI interaction",
            "requires": "OPENAI_API_KEY (or CODEX_API_KEY)",
        },
        {
            "id": "anthropic",
            "name": "Anthropic Claude CUA",
            "model": "claude-opus-4-6",
            "description": "Claude computer use tool with desktop automation",
            "requires": "ANTHROPIC_API_KEY (or CLAUDE_API_KEY)",
        },
    ]
    return jsonify(providers)


# ----------------------------------------------------------------------
# Prompt Catalog API
# ----------------------------------------------------------------------


@app.route("/api/prompts")
def api_prompts_list():
    """List all prompts in the catalog."""
    try:
        from codex_manager.prompts.catalog import get_catalog

        catalog = get_catalog()
        return jsonify(catalog.all_prompts())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/prompts/pipeline-phases")
def api_prompts_pipeline():
    """List pipeline phase prompts."""
    try:
        from codex_manager.prompts.catalog import get_catalog

        catalog = get_catalog()
        phases = []
        for key in catalog.list_pipeline_phases():
            meta = catalog.pipeline_meta(key)
            phases.append(
                {
                    "key": key,
                    "name": meta["name"],
                    "description": meta["description"],
                    "prompt_preview": catalog.pipeline(key)[:200],
                }
            )
        return jsonify(phases)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# â”€â”€ Launcher â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



def _restart_creation_flags() -> int:
    """Return platform-specific subprocess creation flags for detached restart."""
    if os.name != "nt":
        return 0
    detached = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
    new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
    return int(detached | new_group)


def _build_gui_restart_command(
    *,
    port: int,
    pipeline_resume_checkpoint: str = "",
) -> list[str]:
    cmd = [sys.executable, "-m", "codex_manager", "gui", "--port", str(port), "--no-browser"]
    if pipeline_resume_checkpoint:
        cmd.extend(["--pipeline-resume-checkpoint", pipeline_resume_checkpoint])
    return cmd


def _restart_working_directory() -> Path:
    """Return the best cwd for spawning a replacement GUI process."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def _ensure_restart_log_path() -> Path:
    """Return restart log path, creating parent directories as needed."""
    _GUI_RESTART_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _GUI_RESTART_LOG_PATH


def _restart_log_field(value: object, *, limit: int = 160) -> str:
    """Return a single-line, length-limited string for restart diagnostics."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _append_restart_log(message: str) -> None:
    """Append a timestamped line to the restart diagnostics log."""
    try:
        log_path = _ensure_restart_log_path()
        with log_path.open("a", encoding="utf-8") as handle:
            stamp = datetime.now(timezone.utc).isoformat()
            handle.write(f"[{stamp}] {message}\n")
    except Exception:
        logger.debug("Could not write GUI restart diagnostic log", exc_info=True)


def _ensure_runtime_log_path() -> Path:
    """Return runtime log path, creating parent directories as needed."""
    _GUI_RUNTIME_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _GUI_RUNTIME_LOG_PATH


def _append_runtime_log(message: str) -> None:
    """Append runtime lifecycle diagnostics to GUI runtime log."""
    try:
        log_path = _ensure_runtime_log_path()
        with log_path.open("a", encoding="utf-8") as handle:
            stamp = datetime.now(timezone.utc).isoformat()
            handle.write(f"[{stamp}] {message}\n")
    except Exception:
        logger.debug("Could not write GUI runtime diagnostics log", exc_info=True)


def _enable_gui_faulthandler() -> None:
    """Route fatal interpreter tracebacks into the runtime diagnostics log."""
    global _gui_faulthandler_enabled, _gui_faulthandler_stream
    if _gui_faulthandler_enabled:
        return
    if faulthandler.is_enabled():
        _append_runtime_log("faulthandler already enabled by runtime; reusing existing sink.")
        return
    try:
        stream = _ensure_runtime_log_path().open("a", encoding="utf-8")
        faulthandler.enable(file=stream, all_threads=True)
        _gui_faulthandler_stream = stream
        _gui_faulthandler_enabled = True
        _append_runtime_log("faulthandler enabled for fatal crash tracebacks.")
    except Exception:
        _gui_faulthandler_enabled = False
        _gui_faulthandler_stream = None
        logger.debug("Could not enable faulthandler runtime diagnostics", exc_info=True)


def _disable_gui_faulthandler() -> None:
    """Tear down GUI-managed faulthandler resources during shutdown."""
    global _gui_faulthandler_enabled, _gui_faulthandler_stream
    if _gui_faulthandler_enabled:
        with suppress(Exception):
            if faulthandler.is_enabled():
                faulthandler.disable()
    stream = _gui_faulthandler_stream
    _gui_faulthandler_enabled = False
    _gui_faulthandler_stream = None
    if stream is None:
        return
    with suppress(Exception):
        stream.flush()
    with suppress(Exception):
        stream.close()


def _install_gui_runtime_hooks(*, port: int) -> None:
    """Install one-time runtime lifecycle logging hooks."""
    global _gui_runtime_hooks_installed
    if _gui_runtime_hooks_installed:
        return
    _gui_runtime_hooks_installed = True
    pid = os.getpid()
    _append_runtime_log(
        f"GUI startup: pid={pid} port={int(port)} executable={sys.executable} cwd={Path.cwd()}"
    )
    _enable_gui_faulthandler()

    def _on_exit() -> None:
        _append_runtime_log(
            f"GUI process exit: pid={pid} expected_restart={bool(_gui_expected_restart_exit)}"
        )
        _disable_gui_faulthandler()

    atexit.register(_on_exit)

    previous_excepthook = sys.excepthook

    def _main_excepthook(exc_type, exc_value, exc_traceback) -> None:
        summary = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)).strip()
        _append_runtime_log(
            "Unhandled main-thread exception: "
            + (_restart_log_field(summary, limit=8000) if summary else "<empty>")
        )
        previous_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = _main_excepthook

    previous_thread_hook = getattr(threading, "excepthook", None)
    if previous_thread_hook is not None:

        def _thread_excepthook(args) -> None:
            summary = "".join(
                traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
            ).strip()
            thread_name = getattr(getattr(args, "thread", None), "name", "") or "unknown-thread"
            _append_runtime_log(
                "Unhandled thread exception: "
                f"thread={thread_name} "
                + (_restart_log_field(summary, limit=8000) if summary else "<empty>")
            )
            previous_thread_hook(args)

        threading.excepthook = _thread_excepthook

    for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        signum = getattr(signal, signal_name, None)
        if signum is None:
            continue
        with suppress(Exception):
            previous_handler = signal.getsignal(signum)

            def _signal_handler(sig, frame, *, _name=signal_name, _previous=previous_handler):
                # On Windows, child-process console events can leak SIGINT
                # back to the parent even with CREATE_NEW_PROCESS_GROUP.
                # Suppress the signal when a chain or pipeline run is active
                # so the server stays alive.
                work_active = False
                try:
                    work_active = bool(executor.is_running) or bool(
                        _pipeline_executor is not None
                        and getattr(_pipeline_executor, "is_running", False)
                    )
                except Exception:
                    pass

                _append_runtime_log(
                    f"Signal received: {_name} ({sig}) "
                    f"expected_restart={bool(_gui_expected_restart_exit)} "
                    f"work_active={work_active}"
                )

                if work_active and sig == getattr(signal, "SIGINT", None):
                    _append_runtime_log(
                        f"Suppressed {_name} -- chain/pipeline is running "
                        "(likely leaked from a child process console event)"
                    )
                    return None

                if callable(_previous):
                    return _previous(sig, frame)
                if _previous == signal.SIG_IGN:
                    return None
                if sig == getattr(signal, "SIGINT", None):
                    return signal.default_int_handler(sig, frame)
                raise SystemExit(0)

            signal.signal(signum, _signal_handler)


def _launch_replacement_server(command: list[str]) -> subprocess.Popen:
    kwargs: dict[str, object] = {
        "cwd": str(_restart_working_directory()),
        "stdin": subprocess.DEVNULL,
    }
    _append_restart_log(
        "Launching replacement GUI process: "
        f"cwd={kwargs['cwd']} command={' '.join(command)}"
    )
    log_handle = None
    try:
        log_handle = _ensure_restart_log_path().open("ab")
    except Exception:
        logger.debug("Could not open GUI restart log for child process output", exc_info=True)
    if log_handle is not None:
        kwargs["stdout"] = log_handle
        kwargs["stderr"] = subprocess.STDOUT
    else:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    if os.name == "nt":
        kwargs["creationflags"] = _restart_creation_flags()
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(command, **kwargs)
    finally:
        if log_handle is not None:
            log_handle.close()
    return proc


def _terminate_current_process(delay_seconds: float = 0.75) -> None:
    global _gui_expected_restart_exit
    _gui_expected_restart_exit = True
    logger.warning(
        "Restart handoff accepted; terminating current GUI process in %.2fs",
        delay_seconds,
    )
    _append_restart_log(
        "Restart handoff accepted; terminating current GUI process "
        f"in {delay_seconds:.2f}s"
    )
    _append_runtime_log(
        "Restart handoff accepted; terminating current GUI process "
        f"in {delay_seconds:.2f}s"
    )

    def _exit_now() -> None:
        os._exit(0)

    Timer(delay_seconds, _exit_now).start()


def _is_address_in_use_error(exc: OSError) -> bool:
    """Return True when Flask startup failed because the port is still in use."""
    err_no = getattr(exc, "errno", None)
    win_error = getattr(exc, "winerror", None)
    text = str(exc).lower()
    return (
        err_no in {48, 98, 10048}
        or win_error in {10048}
        or "address already in use" in text
        or "only one usage of each socket address" in text
    )


def _probe_bind_error(host: str, port: int) -> OSError | None:
    """Return bind error for host/port, or ``None`` when bind appears available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, int(port)))
    except OSError as exc:
        return exc
    return None


def _run_gui_server(host: str, port: int) -> None:
    """Run Flask server with startup retries for transient bind races.

    The outer ``while True`` loop catches ``KeyboardInterrupt`` that leaks
    through when a child-process console event delivers SIGINT to the GUI
    server while a chain or pipeline run is active.  Instead of crashing,
    the server restarts its ``serve_forever`` loop.
    """
    retries = max(0, int(_GUI_STARTUP_BIND_RETRIES))
    for attempt in range(retries + 1):
        bind_probe_error = _probe_bind_error(host, port)
        if bind_probe_error is not None:
            if (not _is_address_in_use_error(bind_probe_error)) or attempt >= retries:
                raise bind_probe_error
            logger.warning(
                "GUI startup waiting for port release on %s:%s (attempt %s/%s): %s",
                host,
                port,
                attempt + 1,
                retries + 1,
                bind_probe_error,
            )
            time.sleep(_GUI_STARTUP_BIND_RETRY_SECONDS)
            continue
        try:
            while True:
                try:
                    app.run(host=host, port=port, debug=False, threaded=True)
                    return
                except KeyboardInterrupt:
                    work_active = False
                    try:
                        work_active = bool(executor.is_running) or bool(
                            _pipeline_executor is not None
                            and getattr(_pipeline_executor, "is_running", False)
                        )
                    except Exception:
                        pass
                    if work_active:
                        _append_runtime_log(
                            "KeyboardInterrupt caught in server loop while "
                            "chain/pipeline is running -- restarting serve loop "
                            "(child-process console event leak)"
                        )
                        logger.warning(
                            "Suppressed leaked KeyboardInterrupt while "
                            "chain/pipeline is active; server stays alive."
                        )
                        continue
                    raise
        except SystemExit as exc:
            bind_race_error = _probe_bind_error(host, port)
            if (
                bind_race_error is not None
                and _is_address_in_use_error(bind_race_error)
                and attempt < retries
            ):
                logger.warning(
                    "GUI startup bind race on %s:%s via SystemExit=%s (attempt %s/%s): %s",
                    host,
                    port,
                    getattr(exc, "code", exc),
                    attempt + 1,
                    retries + 1,
                    bind_race_error,
                )
                time.sleep(_GUI_STARTUP_BIND_RETRY_SECONDS)
                continue
            raise
        except OSError as exc:
            if (not _is_address_in_use_error(exc)) or attempt >= retries:
                raise
            logger.warning(
                "GUI restart bind race on %s:%s (attempt %s/%s): %s",
                host,
                port,
                attempt + 1,
                retries + 1,
                exc,
            )
            time.sleep(_GUI_STARTUP_BIND_RETRY_SECONDS)


@app.route("/api/system/restart", methods=["POST"])
def api_system_restart():
    """Spawn a replacement GUI server process, then terminate this one."""
    global _pipeline_executor

    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        data = {}
    restart_reason = _restart_log_field(data.get("restart_reason") or data.get("reason") or "")
    restart_source = _restart_log_field(data.get("restart_source") or data.get("source") or "")
    restart_auto = bool(data.get("auto"))
    restart_remote = _restart_log_field(
        request.headers.get("X-Forwarded-For") or request.remote_addr or "unknown",
        limit=100,
    )
    restart_user_agent = _restart_log_field(
        request.headers.get("User-Agent") or "",
        limit=180,
    )
    _append_restart_log(
        "Restart requested: "
        f"remote={restart_remote} "
        f"source={restart_source or '<unspecified>'} "
        f"reason={restart_reason or '<unspecified>'} "
        f"auto={restart_auto} "
        f"ua={restart_user_agent or '<unknown>'}"
    )
    logger.warning(
        "Restart request received (source=%s reason=%s auto=%s remote=%s)",
        restart_source or "<unspecified>",
        restart_reason or "<unspecified>",
        restart_auto,
        restart_remote,
    )

    if executor.is_running:
        _append_restart_log("Restart request rejected: chain run is active")
        return (
            jsonify(
                {
                    "error": (
                        "Cannot restart server while a chain run is active. "
                        "Stop the chain first."
                    )
                }
            ),
            409,
        )
    if bool(_pipeline_executor is not None and _pipeline_executor.is_running):
        _append_restart_log("Restart request rejected: pipeline run is active")
        return (
            jsonify(
                {
                    "error": (
                        "Cannot restart server while a pipeline run is active. "
                        "Stop the pipeline first."
                    )
                }
            ),
            409,
        )

    checkpoint_raw = str(
        data.get("pipeline_resume_checkpoint") or data.get("checkpoint_path") or ""
    ).strip()
    checkpoint_path = ""
    if checkpoint_raw:
        p = Path(checkpoint_raw)
        if not p.is_file():
            _append_restart_log(
                "Restart request rejected: checkpoint not found "
                f"({ _restart_log_field(checkpoint_raw, limit=240) })"
            )
            return jsonify({"error": f"Checkpoint not found: {checkpoint_raw}"}), 400
        checkpoint_path = str(p.resolve())

    command = _build_gui_restart_command(
        port=_SERVER_PORT,
        pipeline_resume_checkpoint=checkpoint_path,
    )
    try:
        child = _launch_replacement_server(command)
    except Exception as exc:
        _append_restart_log(f"Replacement launch failed: {exc}")
        return jsonify({"error": f"Could not restart server: {exc}"}), 500

    time.sleep(_GUI_RESTART_CHILD_WARMUP_SECONDS)
    child_exit = child.poll()
    if child_exit is not None:
        message = (
            "Replacement process exited before handoff "
            f"(exit={child_exit}). See {_ensure_restart_log_path()} for details."
        )
        _append_restart_log(message)
        return jsonify({"error": message}), 500

    _append_restart_log("Restart handoff confirmed: replacement process is alive after warmup")
    _terminate_current_process()
    return jsonify(
        {
            "status": "restarting",
            "port": _SERVER_PORT,
            "pipeline_resume_checkpoint": checkpoint_path,
        }
    )

def _open_browser(port: int) -> None:
    webbrowser.open(f"http://127.0.0.1:{port}")


def _resume_pipeline_from_checkpoint(checkpoint_path: str) -> tuple[bool, str]:
    """Resume a pipeline run from a restart checkpoint file."""
    global _pipeline_executor

    checkpoint_raw = str(checkpoint_path or "").strip()
    if not checkpoint_raw:
        return False, "No checkpoint provided."

    checkpoint = Path(checkpoint_raw).expanduser()
    if not checkpoint.is_file():
        return False, f"Checkpoint not found: {checkpoint}"

    try:
        payload = json.loads(_read_text_utf8_resilient(checkpoint))
    except Exception as exc:
        return False, f"Could not parse checkpoint: {exc}"

    if not isinstance(payload, dict):
        return False, "Checkpoint payload must be an object."

    repo_path = str(payload.get("repo_path") or "").strip()
    if not repo_path:
        return False, "Checkpoint missing repo_path."
    if not Path(repo_path).is_dir():
        return False, f"Checkpoint repo_path not found: {repo_path}"

    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        return False, "Checkpoint missing pipeline config."

    resume_cycle = int(payload.get("resume_cycle") or 1)
    resume_phase_index = int(payload.get("resume_phase_index") or 0)

    try:
        from codex_manager.pipeline.orchestrator import PipelineOrchestrator
        from codex_manager.pipeline.phases import PipelineConfig

        config = PipelineConfig(**config_payload)
        _pipeline_executor = PipelineOrchestrator(
            repo_path=repo_path,
            config=config,
            resume_cycle=resume_cycle,
            resume_phase_index=resume_phase_index,
        )
        _pipeline_executor.start()
        checkpoint.unlink(missing_ok=True)
        return (
            True,
            "Resumed pipeline from checkpoint "
            f"(cycle={resume_cycle}, phase_index={resume_phase_index}).",
        )
    except Exception as exc:
        return False, f"Could not resume pipeline from checkpoint: {exc}"


def run_gui(
    port: int = 5088,
    open_browser_: bool = True,
    pipeline_resume_checkpoint: str = "",
) -> None:
    """Start the Flask development server."""
    global _SERVER_PORT, _SERVER_OPEN_BROWSER, _gui_expected_restart_exit
    _SERVER_PORT = int(port)
    _SERVER_OPEN_BROWSER = bool(open_browser_)
    _gui_expected_restart_exit = False

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    _install_gui_runtime_hooks(port=_SERVER_PORT)

    try:
        watchdog = _get_model_watchdog()
        if watchdog.start():
            logger.info(
                "Model watchdog active (interval=%sh providers=%s root=%s)",
                watchdog.status().get("config", {}).get("interval_hours"),
                ",".join(watchdog.status().get("config", {}).get("providers", [])),
                _MODEL_WATCHDOG_ROOT,
            )
        else:
            logger.info("Model watchdog disabled in config (%s)", _MODEL_WATCHDOG_ROOT)
    except Exception:
        logger.exception("Could not start model watchdog")

    if pipeline_resume_checkpoint:
        resumed, detail = _resume_pipeline_from_checkpoint(pipeline_resume_checkpoint)
        if resumed:
            logger.info("Startup resume: %s", detail)
        else:
            logger.warning("Startup resume failed: %s", detail)

    if open_browser_:
        Timer(1.5, _open_browser, args=[port]).start()
    print(f"\n  {_project_display_name()} GUI -> http://127.0.0.1:{port}\n")
    _run_gui_server(host="127.0.0.1", port=port)


