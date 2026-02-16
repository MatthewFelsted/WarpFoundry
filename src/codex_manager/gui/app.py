"""Flask web application - serves the GUI and provides API endpoints."""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import shutil
import string
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from threading import Timer
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from flask import Flask, Response, jsonify, render_template, request

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
    get_recipe,
    list_recipe_summaries,
    recipe_steps_map,
)
from codex_manager.gui.stop_guidance import get_stop_guidance
from codex_manager.monitoring import ModelCatalogWatchdog
from codex_manager.preflight import (
    binary_exists as shared_binary_exists,
)
from codex_manager.preflight import (
    build_preflight_report,
    parse_agents,
)
from codex_manager.preflight import (
    has_claude_auth as shared_has_claude_auth,
)
from codex_manager.preflight import (
    has_codex_auth as shared_has_codex_auth,
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
_RUNNABLE_DIAGNOSTIC_ACTION_KEYS = frozenset(
    {
        "init_git_repo",
        "install_codex_cli",
        "install_claude_cli",
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
_GOVERNANCE_ENV_KEYS: dict[str, str] = {
    "research_allowed_domains": "CODEX_MANAGER_RESEARCH_ALLOWED_DOMAINS",
    "research_blocked_domains": "CODEX_MANAGER_RESEARCH_BLOCKED_DOMAINS",
    "deep_research_allowed_domains": "DEEP_RESEARCH_ALLOWED_SOURCE_DOMAINS",
    "deep_research_blocked_domains": "DEEP_RESEARCH_BLOCKED_SOURCE_DOMAINS",
}
_GITHUB_SECRET_SERVICE = "codex_manager.github_auth"
_GITHUB_PAT_SECRET_KEY = "pat"
_GITHUB_SSH_SECRET_KEY = "ssh_private_key"
_GITHUB_AUTH_METHODS = frozenset({"https", "ssh"})
_GITHUB_TEST_TIMEOUT_SECONDS = 20
_GITHUB_REPO_METADATA_TIMEOUT_SECONDS = 8
_GITHUB_REPO_METADATA_CACHE_TTL_SECONDS = 300
_GITHUB_REPO_METADATA_ERROR_CACHE_TTL_SECONDS = 60
_GIT_REMOTE_QUERY_TIMEOUT_SECONDS = 20
_GIT_CLONE_TIMEOUT_SECONDS = 180
_GIT_SYNC_TIMEOUT_SECONDS = 30
_DEFAULT_BRANCH_SENTINEL = "__remote_default__"
_OWNER_CONTEXT_MAX_FILES = 6
_OWNER_CONTEXT_MAX_FILE_CHARS = 6000
_OWNER_CONTEXT_MAX_TOTAL_CHARS = 24000
_GENERAL_REQUEST_HISTORY_MAX_ITEMS = 200

_model_watchdog: ModelCatalogWatchdog | None = None
_model_watchdog_lock = threading.Lock()
_github_repo_metadata_cache: dict[str, tuple[float, dict[str, object] | None, str]] = {}
_github_repo_metadata_cache_lock = threading.Lock()


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


def _read_text_utf8_resilient(path: Path) -> str:
    """Read a text file, recovering from legacy non-UTF8 bytes when possible."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raw = path.read_bytes()
        for encoding in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                text = raw.decode(encoding)
            except UnicodeDecodeError:
                continue
            logger.warning(
                "Recovered non-UTF8 text file %s using %s (%s); rewriting as UTF-8",
                path,
                encoding,
                exc,
            )
            path.write_text(text, encoding="utf-8")
            return text
        text = raw.decode("utf-8", errors="replace")
        logger.warning(
            "Recovered undecodable text file %s using replacement decode; rewriting as UTF-8",
            path,
        )
        path.write_text(text, encoding="utf-8")
        return text


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
    if config.phases:
        enabled = [p for p in config.phases if p.enabled]
        if enabled:
            return {_normalize_agent(p.agent) for p in enabled}
    return {_normalize_agent(config.agent)}


def _agent_preflight_issues(
    agents: set[str],
    *,
    codex_binary: str,
    claude_binary: str,
) -> list[str]:
    """Return binary/auth validation issues for the requested agents."""
    issues: list[str] = []
    for agent in sorted(agents):
        if agent not in {"codex", "claude_code"}:
            issues.append(f"Unknown agent '{agent}'. Supported: codex, claude_code, auto")
            continue
        if agent == "codex":
            if not _binary_exists(codex_binary):
                issues.append(f"Codex binary not found: '{codex_binary}'")
            if not _has_codex_auth():
                issues.append(
                    "Codex auth not detected. Set CODEX_API_KEY or OPENAI_API_KEY, "
                    "or run 'codex login' first."
                )
        elif agent == "claude_code":
            if not _binary_exists(claude_binary):
                issues.append(f"Claude Code binary not found: '{claude_binary}'")
            if not _has_claude_auth():
                issues.append(
                    "Claude auth not detected. Set ANTHROPIC_API_KEY (or CLAUDE_API_KEY), "
                    "or log in with the Claude CLI first."
                )
    return issues


def _image_provider_auth_issue(enabled: bool, provider: str) -> str | None:
    """Return an auth/config issue for configured image generation provider."""
    if not enabled:
        return None
    provider_key = (provider or "openai").strip().lower()
    if provider_key == "google":
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            return "Image generation (google provider) requires GOOGLE_API_KEY or GEMINI_API_KEY."
        return None
    if not os.getenv("OPENAI_API_KEY"):
        return "Image generation (openai provider) requires OPENAI_API_KEY."
    return None


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

    if not (repo / ".git").exists():
        issues.append(f"Not a git repository: {repo}")

    write_error = _repo_write_error(repo)
    if write_error:
        issues.append(write_error)

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

    if (
        config.codex_bypass_approvals_and_sandbox
        and config.codex_danger_confirmation.strip() != DANGER_CONFIRMATION_PHRASE
    ):
        issues.append(
            "Danger confirmation missing. Set codex_danger_confirmation to "
            f"'{DANGER_CONFIRMATION_PHRASE}' to enable bypass."
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

    if config.vector_memory_enabled:
        backend = str(config.vector_memory_backend or "chroma").strip().lower()
        if backend != "chroma":
            issues.append("Unsupported vector_memory_backend. Supported backend(s): chroma.")
        else:
            try:
                import chromadb  # noqa: F401
            except Exception:
                issues.append(
                    "Vector memory requires ChromaDB. Install with: pip install chromadb"
                )

    return issues


def _pipeline_preflight_issues(config: PipelineGUIConfig) -> list[str]:
    """Return pipeline-level preflight issues before execution starts."""
    repo = Path(config.repo_path).resolve()
    issues: list[str] = []

    if not (repo / ".git").exists():
        issues.append(f"Not a git repository: {repo}")

    write_error = _repo_write_error(repo)
    if write_error:
        issues.append(write_error)

    if (
        config.codex_bypass_approvals_and_sandbox
        and config.codex_danger_confirmation.strip() != DANGER_CONFIRMATION_PHRASE
    ):
        issues.append(
            "Danger confirmation missing. Set codex_danger_confirmation to "
            f"'{DANGER_CONFIRMATION_PHRASE}' to enable bypass."
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

    if config.vector_memory_enabled:
        backend = str(config.vector_memory_backend or "chroma").strip().lower()
        if backend != "chroma":
            issues.append("Unsupported vector_memory_backend. Supported backend(s): chroma.")
        else:
            try:
                import chromadb  # noqa: F401
            except Exception:
                issues.append(
                    "Vector memory requires ChromaDB. Install with: pip install chromadb"
                )

    if config.self_improvement_auto_restart and not config.self_improvement_enabled:
        issues.append("self_improvement_auto_restart requires self_improvement_enabled.")

    if config.pr_aware_enabled and config.mode != "apply":
        issues.append("pr_aware_enabled requires mode='apply'.")

    if config.deep_research_native_enabled and config.deep_research_enabled:
        providers = str(config.deep_research_providers or "both").strip().lower()
        if providers in {"both", "openai"} and not (
            os.getenv("OPENAI_API_KEY") or os.getenv("CODEX_API_KEY")
        ):
            issues.append(
                "Native deep research (OpenAI) requires OPENAI_API_KEY or CODEX_API_KEY."
            )
        if providers in {"both", "google"} and not (
            os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        ):
            issues.append(
                "Native deep research (Google) requires GOOGLE_API_KEY or GEMINI_API_KEY."
            )

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


def _safe_float(value: object, default: float = 0.0) -> float:
    """Coerce *value* to float where possible."""
    try:
        if isinstance(value, bool):
            return float(int(value))
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


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


def _new_run_aggregate(
    *,
    scope: str,
    event_id: str,
    timestamp: str,
    context: dict[str, object],
) -> dict[str, object]:
    """Initialize an in-progress run aggregate from a ``run_started`` event."""
    started_epoch_ms = _parse_iso_epoch_ms(timestamp)
    return {
        "run_id": event_id or f"{scope}_run_{started_epoch_ms}",
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
        "_commit_shas": set(),
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

    token_usage = _safe_int(context.get("total_tokens"), 0)
    if token_usage <= 0:
        token_usage = _safe_int(context.get("input_tokens"), 0) + _safe_int(
            context.get("output_tokens"), 0
        )
    run["_token_usage_from_results"] = _safe_int(run.get("_token_usage_from_results"), 0) + max(
        0, token_usage
    )

    commit_sha = str(context.get("commit_sha") or "").strip()
    commits = run.get("_commit_shas")
    if isinstance(commits, set) and commit_sha and commit_sha.lower() != "none":
        commits.add(commit_sha)


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
    finalized["overall_score"] = _run_overall_score(finalized)
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
            "strongest_tests_run_id": "",
        },
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

    try:
        raw = _read_text_utf8_resilient(history_path)
    except Exception as exc:
        return _empty_run_comparison_payload(
            repo=repo,
            scope=scope,
            limit=limit,
            message=f"Could not read run history: {exc}",
        )

    open_runs: dict[str, dict[str, object]] = {}
    finished_runs: list[dict[str, object]] = []
    result_event_by_scope = {
        "chain": "step_result",
        "pipeline": "phase_result",
    }

    for line in raw.splitlines():
        row = line.strip()
        if not row:
            continue
        try:
            event = json.loads(row)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        scope_key = str(event.get("scope") or "").strip().lower()
        if scope_key not in {"chain", "pipeline"}:
            continue

        event_name = str(event.get("event") or "").strip().lower()
        timestamp = str(event.get("timestamp") or "").strip()
        event_id = str(event.get("id") or "").strip()
        summary = str(event.get("summary") or "").strip()
        context_obj = event.get("context")
        context = context_obj if isinstance(context_obj, dict) else {}

        if event_name == "run_started":
            open_runs[scope_key] = _new_run_aggregate(
                scope=scope_key,
                event_id=event_id,
                timestamp=timestamp,
                context=context,
            )
            continue

        if event_name == result_event_by_scope[scope_key]:
            run = open_runs.get(scope_key)
            if run is not None:
                _record_run_result_event(run, context)
            continue

        if event_name == "run_finished":
            run = open_runs.pop(scope_key, None)
            if run is None:
                run = _new_run_aggregate(
                    scope=scope_key,
                    event_id=event_id,
                    timestamp=timestamp,
                    context={},
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
        return _empty_run_comparison_payload(
            repo=repo,
            scope=scope,
            limit=limit,
            message="No completed runs were found in history yet.",
        )

    best_overall = max(runs, key=lambda run: _safe_float(run.get("overall_score"), -999999.0))

    duration_candidates = [run for run in runs if _safe_float(run.get("duration_seconds"), 0.0) > 0]
    best_fastest = (
        min(duration_candidates, key=lambda run: _safe_float(run.get("duration_seconds"), 0.0))
        if duration_candidates
        else None
    )

    token_candidates = [run for run in runs if _safe_int(run.get("token_usage"), 0) > 0]
    best_lowest_tokens = (
        min(token_candidates, key=lambda run: _safe_int(run.get("token_usage"), 0))
        if token_candidates
        else None
    )

    best_tests = max(
        runs,
        key=lambda run: (
            _run_test_score(run.get("tests") if isinstance(run.get("tests"), dict) else {}),
            -_safe_float(run.get("duration_seconds"), 0.0),
            -_safe_int(run.get("token_usage"), 0),
        ),
    )

    for run in runs:
        badges: list[str] = []
        if run.get("run_id") == best_overall.get("run_id"):
            badges.append("best_overall")
        if best_fastest is not None and run.get("run_id") == best_fastest.get("run_id"):
            badges.append("fastest")
        if best_lowest_tokens is not None and run.get("run_id") == best_lowest_tokens.get("run_id"):
            badges.append("lowest_tokens")
        if run.get("run_id") == best_tests.get("run_id"):
            badges.append("strongest_tests")
        run["badges"] = badges

    return {
        "available": True,
        "repo_path": str(repo),
        "history_path": str(history_path.resolve()),
        "scope": scope,
        "limit": limit,
        "runs": runs,
        "best_by": {
            "overall_run_id": str(best_overall.get("run_id") or ""),
            "fastest_run_id": str(best_fastest.get("run_id") or "") if best_fastest else "",
            "lowest_token_run_id": (
                str(best_lowest_tokens.get("run_id") or "") if best_lowest_tokens else ""
            ),
            "strongest_tests_run_id": str(best_tests.get("run_id") or ""),
        },
        "message": "",
    }


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
    path = _todo_wishlist_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = str(content or "").strip() or _default_todo_wishlist_markdown(repo.name)
    path.write_text(clean + "\n", encoding="utf-8")
    return path


def _write_feature_dreams(repo: Path, content: str) -> Path:
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
) -> tuple[str, str]:
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
    if not path.is_file():
        return []
    try:
        raw = _read_text_utf8_resilient(path)
    except Exception:
        return []
    rows: list[dict[str, object]] = []
    for line in raw.splitlines():
        item = line.strip()
        if not item:
            continue
        try:
            parsed = json.loads(item)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
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


def _safe_int(value: object, default: int = 0) -> int:
    """Coerce *value* to int where possible."""
    try:
        if isinstance(value, bool):
            return int(value)
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _safe_str(value: object, default: str = "") -> str:
    """Coerce *value* to a trimmed string, with fallback for empty values."""
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _safe_bool(value: object, default: bool = False) -> bool:
    """Coerce common boolean-like values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


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
        return str(keyring.get_password(_GITHUB_SECRET_SERVICE, secret_key) or "")
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
        return
    except KeyringError as exc:
        raise RuntimeError(f"Could not delete secure credential '{secret_key}': {exc}") from exc


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
            "User-Agent": "codex-manager-github-auth-test",
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


def _docs_dir() -> Path | None:
    """Resolve the local docs directory when available."""
    this_file = Path(__file__).resolve()
    candidates = [this_file.parents[3] / "docs"]
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


# â”€â”€ Page â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@app.route("/")
def index():
    project_display_name = str(
        os.getenv("CODEX_MANAGER_PROJECT_NAME")
        or os.getenv("AI_MANAGER_PROJECT_NAME")
        or "Codex Manager"
    ).strip()
    if not project_display_name:
        project_display_name = "Codex Manager"
    return render_template(
        "index.html",
        recipes_payload=_recipe_template_payload(),
        project_display_name=project_display_name,
    )

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
    """List recipe summaries and default recipe id."""
    return jsonify(
        {
            "default_recipe_id": DEFAULT_RECIPE_ID,
            "recipes": list_recipe_summaries(),
        }
    )


@app.route("/api/recipes/<recipe_id>")
def api_recipe_detail(recipe_id: str):
    """Return one recipe with full step definitions."""
    recipe = get_recipe(recipe_id)
    if recipe is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(recipe)


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
    def generate():
        while True:
            try:
                entry = executor.log_queue.get(timeout=2)
                yield f"data: {json.dumps(entry)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

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

    report = _build_diagnostics_report(
        repo_path=repo_path,
        codex_binary=codex_binary,
        claude_binary=claude_binary,
        requested_agents=requested_agents,
    )
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
    )
    return jsonify(
        {
            "repo_path": str(repo),
            "model": model,
            "content": suggested,
            "has_open_items": _feature_dreams_has_open_items(suggested),
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


def _run_git_sync_command(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run one git command for sync APIs and return the completed process."""
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=_GIT_SYNC_TIMEOUT_SECONDS,
        check=False,
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
    result = _run_git_sync_command(repo, "rev-parse", "--git-path", "FETCH_HEAD")
    if result.returncode != 0:
        return None

    raw_path = str(result.stdout or "").strip()
    if not raw_path:
        return None

    fetch_head = Path(raw_path)
    if fetch_head.is_absolute():
        return fetch_head
    return (repo / fetch_head).resolve()


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
        "User-Agent": "codex-manager-git-sync-metadata",
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


def _git_unstage_paths(repo: Path, paths: list[str]) -> subprocess.CompletedProcess[str]:
    """Unstage paths, including fallback support for unborn HEAD repositories."""
    restore_result = _run_git_sync_command(repo, "restore", "--staged", "--", *paths)
    if restore_result.returncode == 0:
        return restore_result

    fallback_result = _run_git_sync_command(repo, "rm", "--cached", "--quiet", "--", *paths)
    if fallback_result.returncode == 0:
        return fallback_result
    return restore_result


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


def _git_sync_status_payload(repo: Path) -> dict[str, object]:
    """Return branch/tracking/ahead-behind/dirty metadata for a repository."""
    payload = _git_sync_status_core_payload(repo)
    payload["github_repo"] = _git_sync_github_repo_payload(repo, payload)
    return payload


def _git_ref_exists(repo: Path, ref_name: str) -> bool:
    """Return True when a git ref exists in the repository."""
    probe = _run_git_sync_command(repo, "show-ref", "--verify", "--quiet", ref_name)
    return probe.returncode == 0


def _git_sync_branch_choices_payload(repo: Path) -> dict[str, object]:
    """Return local/remote branch choices plus current branch metadata."""
    status = _git_sync_status_payload(repo)
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

    return {
        "repo_path": str(repo),
        "current_branch": current_branch,
        "detached_head": current_branch == "HEAD",
        "local_branches": local_branches,
        "remote_branches": remote_branches,
        "sync": status,
        "checked_at_epoch_ms": int(time.time() * 1000),
    }


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
            stash_message = f"codex-manager:preflight-auto-stash {stamp}"
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

    try:
        return jsonify(_git_sync_status_payload(repo))
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

    try:
        return jsonify(_git_sync_branch_choices_payload(repo))
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
        branches = _git_sync_branch_choices_payload(repo)
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
        branches = _git_sync_branch_choices_payload(repo)
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
            stage_result = _run_git_sync_command(repo, "add", "--", *target_paths)
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
            "codex-manager:auto-stash-before-pull "
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
            )
        if git_email:
            subprocess.run(
                ["git", "config", "user.email", git_email],
                cwd=str(project_path),
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
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
        )
        subprocess.run(
            ["git", "commit", "-m", f"Initial commit - {project_name}"],
            cwd=str(project_path),
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
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
    safe_name = _sanitize_config_name(raw_name)
    if not raw_name or safe_name != raw_name:
        return jsonify({"error": "Invalid config name"}), 400

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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Pipeline API
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


def _get_pipeline():
    """Get or create the global pipeline executor."""
    global _pipeline_executor
    if _pipeline_executor is None:
        # Placeholder â€” will be configured on start
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


@app.route("/api/pipeline/start", methods=["POST"])
def api_pipeline_start():
    """Start the autonomous pipeline."""
    global _pipeline_executor

    if _pipeline_executor is not None and _pipeline_executor.is_running:
        return jsonify({"error": "Pipeline is already running"}), 409

    data = request.get_json(silent=True) or {}
    try:
        gui_config = PipelineGUIConfig(**data)
    except Exception as exc:
        return jsonify({"error": f"Invalid config: {exc}"}), 400

    if not Path(gui_config.repo_path).is_dir():
        return jsonify({"error": f"Repo path not found: {gui_config.repo_path}"}), 400

    issues = _pipeline_preflight_issues(gui_config)
    if issues:
        msg = "Preflight checks failed:\n" + "\n".join(f"- {i}" for i in issues)
        return jsonify({"error": msg, "issues": issues}), 400

    git_preflight: dict[str, object] | None = None
    if gui_config.git_preflight_enabled:
        try:
            git_preflight = _git_preflight_before_run(
                Path(gui_config.repo_path).resolve(),
                auto_stash=bool(gui_config.git_preflight_auto_stash),
                auto_pull=bool(gui_config.git_preflight_auto_pull),
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
                    test_policy=pg.test_policy,
                    custom_prompt=pg.custom_prompt,
                )
            )
        except ValueError:
            invalid_phases.append(pg.phase)
    if invalid_phases:
        msg = ", ".join(sorted(set(invalid_phases)))
        return jsonify({"error": f"Invalid pipeline phase(s): {msg}"}), 400

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
    return jsonify(payload)


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

        return jsonify(
            _attach_stop_guidance(
                PipelineState().to_summary(since_results=since_results),
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
    return jsonify(_attach_stop_guidance(payload, mode="pipeline"))


@app.route("/api/pipeline/stream")
def api_pipeline_stream():
    """SSE stream for pipeline logs."""
    global _pipeline_executor

    def generate():
        while True:
            if _pipeline_executor is not None:
                try:
                    entry = _pipeline_executor.log_queue.get(timeout=2)
                    yield f"data: {json.dumps(entry)}\n\n"
                    continue
                except queue.Empty:
                    pass
            yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/pipeline/logs/<filename>")
def api_pipeline_log(filename: str):
    """Read a pipeline log file (WISHLIST.md, TESTPLAN.md, etc.)."""
    global _pipeline_executor
    if filename not in _PIPELINE_LOG_FILES:
        return jsonify({"error": "Invalid log file"}), 400

    repo = _resolve_pipeline_logs_repo(request.args.get("repo_path", ""))
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
        return jsonify(
            {
                "content": "",
                "exists": False,
                "filename": filename,
                "repo_path": "",
                "logs_dir": "",
            }
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
        try:
            raw = _read_text_utf8_resilient(trials_path)
        except Exception as exc:
            logger.warning("Could not read science trials file %s: %s", trials_path, exc)
            raw = ""

        for line in raw.splitlines():
            row = line.strip()
            if not row:
                continue
            try:
                payload = json.loads(row)
            except json.JSONDecodeError:
                continue

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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CUA (Computer-Using Agent) API
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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
                "Install with: pip install codex-manager[cua] then python -m playwright install"
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
            "requires": "OPENAI_API_KEY",
        },
        {
            "id": "anthropic",
            "name": "Anthropic Claude CUA",
            "model": "claude-opus-4-6",
            "description": "Claude computer use tool with desktop automation",
            "requires": "ANTHROPIC_API_KEY",
        },
    ]
    return jsonify(providers)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Prompt Catalog API
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


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


def _launch_replacement_server(command: list[str]) -> None:
    kwargs: dict[str, object] = {
        "cwd": str(Path.cwd()),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = _restart_creation_flags()
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(command, **kwargs)


def _terminate_current_process(delay_seconds: float = 0.75) -> None:
    def _exit_now() -> None:
        os._exit(0)

    Timer(delay_seconds, _exit_now).start()


@app.route("/api/system/restart", methods=["POST"])
def api_system_restart():
    """Spawn a replacement GUI server process, then terminate this one."""
    data = request.get_json(silent=True) or {}
    checkpoint_raw = str(
        data.get("pipeline_resume_checkpoint") or data.get("checkpoint_path") or ""
    ).strip()
    checkpoint_path = ""
    if checkpoint_raw:
        p = Path(checkpoint_raw)
        if not p.is_file():
            return jsonify({"error": f"Checkpoint not found: {checkpoint_raw}"}), 400
        checkpoint_path = str(p.resolve())

    command = _build_gui_restart_command(
        port=_SERVER_PORT,
        pipeline_resume_checkpoint=checkpoint_path,
    )
    try:
        _launch_replacement_server(command)
    except Exception as exc:
        return jsonify({"error": f"Could not restart server: {exc}"}), 500

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
    global _SERVER_PORT, _SERVER_OPEN_BROWSER
    _SERVER_PORT = int(port)
    _SERVER_OPEN_BROWSER = bool(open_browser_)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

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
    print(f"\n  Codex Manager GUI \u2192 http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


