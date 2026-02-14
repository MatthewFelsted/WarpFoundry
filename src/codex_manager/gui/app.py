"""Flask web application - serves the GUI and provides API endpoints."""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import string
import subprocess
import threading
import webbrowser
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from threading import Timer

from flask import Flask, Response, jsonify, render_template, request

from codex_manager.gui.chain import ChainExecutor
from codex_manager.gui.models import (
    DANGER_CONFIRMATION_PHRASE,
    ChainConfig,
    PipelineGUIConfig,
)
from codex_manager.gui.presets import get_preset, list_presets
from codex_manager.gui.stop_guidance import get_stop_guidance
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

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = str(Path(__file__).resolve().parent / "templates")
app = Flask(__name__, template_folder=_TEMPLATE_DIR)

# Single global executor - the GUI manages one chain at a time
executor = ChainExecutor()

# Pipeline executor (separate from chain)
_pipeline_executor = None  # lazy init

# Saved chain-config directory
CONFIGS_DIR = Path.home() / ".codex_manager" / "chains"
_CONFIG_NAME_ALLOWED_CHARS = frozenset(string.ascii_letters + string.digits + "-_ ")
_KNOWN_AGENTS = {"codex", "claude_code", "auto"}
_DOCS_CATALOG: dict[str, tuple[str, str]] = {
    "quickstart": ("Quickstart", "QUICKSTART.md"),
    "output_artifacts": ("Outputs and Artifacts", "OUTPUTS_AND_ARTIFACTS.md"),
    "tutorial": ("Tutorial", "TUTORIAL.md"),
    "cli_reference": ("CLI Reference", "CLI_REFERENCE.md"),
    "troubleshooting": ("Troubleshooting", "TROUBLESHOOTING.md"),
}
_PIPELINE_LOG_FILES = frozenset(
    {
        "WISHLIST.md",
        "TESTPLAN.md",
        "ERRORS.md",
        "EXPERIMENTS.md",
        "PROGRESS.md",
        "BRAIN.md",
        "HISTORY.md",
    }
)


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
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(serialized, encoding="utf-8")
        tmp_path.replace(path)
    finally:
        with suppress(OSError):
            tmp_path.unlink(missing_ok=True)


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
    return report.to_dict()


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


# ── Page ──────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


# ── Presets ───────────────────────────────────────────────────────────


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


# ── Chain control ─────────────────────────────────────────────────────


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

    executor.start(config)
    return jsonify({"status": "started"})


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


# ── SSE live log stream ──────────────────────────────────────────────


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


# ── Ollama (local models) ─────────────────────────────────────────────


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


# ── Repo validation ──────────────────────────────────────────────────


@app.route("/api/validate-repo", methods=["POST"])
def api_validate_repo():
    data = request.get_json(silent=True) or {}
    raw_path = str(data.get("path") or "").strip()
    if not raw_path:
        return jsonify(
            {
                "exists": False,
                "is_git": False,
                "path": "",
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
            }
        )
    return jsonify(
        {
            "exists": p.is_dir(),
            "is_git": (p / ".git").is_dir(),
            "path": str(p.resolve()) if p.is_dir() else raw_path,
        }
    )


@app.route("/api/diagnostics", methods=["POST"])
def api_diagnostics():
    """Return structured repository/auth diagnostics for the GUI."""
    data = request.get_json(silent=True) or {}
    repo_path = str(data.get("repo_path") or data.get("path") or "").strip()
    codex_binary = str(data.get("codex_binary") or "codex").strip() or "codex"
    claude_binary = str(data.get("claude_binary") or "claude").strip() or "claude"
    requested_agents = _normalize_requested_agents(data.get("agents"))

    report = _build_diagnostics_report(
        repo_path=repo_path,
        codex_binary=codex_binary,
        claude_binary=claude_binary,
        requested_agents=requested_agents,
    )
    return jsonify(report)


# ── Directory browser ────────────────────────────────────────────


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


# ── Project creation ─────────────────────────────────────────────────


@app.route("/api/project/create", methods=["POST"])
def api_create_project():
    """Create a new project directory with git init and optional remote."""
    data = request.get_json(silent=True) or {}
    parent_dir = data.get("parent_dir", "").strip()
    project_name = data.get("project_name", "").strip()
    remote_url = data.get("remote_url", "").strip()
    description = data.get("description", "").strip()
    add_readme = data.get("add_readme", True)
    add_gitignore = data.get("add_gitignore", True)
    initial_branch = data.get("initial_branch", "main").strip() or "main"
    git_name = (data.get("git_name") or data.get("gitName") or "").strip()
    git_email = (data.get("git_email") or data.get("gitEmail") or "").strip()

    if not parent_dir:
        return jsonify({"error": "Parent directory is required"}), 400
    if not project_name:
        return jsonify({"error": "Project name is required"}), 400

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

        return jsonify(
            {
                "status": "created",
                "path": str(project_path.resolve()),
                "git_initialized": True,
                "initial_branch": initial_branch,
                "remote_added": remote_added,
                "remote_url": remote_url if remote_added else None,
            }
        )

    except subprocess.CalledProcessError as exc:
        return jsonify({"error": f"Git command failed: {exc.stderr.strip()}"}), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Config persistence ───────────────────────────────────────────────


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
    raw_name = data.get("name", "")
    if not _is_valid_config_name(raw_name):
        return jsonify({"error": "Invalid config name"}), 400

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    root = CONFIGS_DIR.resolve()
    path = (root / f"{raw_name}.json").resolve()
    if path.parent != root:
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


# ══════════════════════════════════════════════════════════════════════
# Pipeline API
# ══════════════════════════════════════════════════════════════════════


def _get_pipeline():
    """Get or create the global pipeline executor."""
    global _pipeline_executor
    if _pipeline_executor is None:
        # Placeholder — will be configured on start
        _pipeline_executor = None
    return _pipeline_executor


@app.route("/api/pipeline/phases")
def api_pipeline_phases():
    """Return available pipeline phases, defaults, and prompt info."""
    from codex_manager.pipeline.phases import (
        CUA_PHASES,
        DEFAULT_ITERATIONS,
        DEFAULT_PHASE_ORDER,
        PHASE_LOG_FILES,
        SCIENCE_PHASES,
    )

    # Load prompt catalog for phase descriptions and prompt text
    try:
        from codex_manager.prompts.catalog import get_catalog

        catalog = get_catalog()
    except Exception:
        catalog = None

    phases = []
    for phase in list(DEFAULT_PHASE_ORDER) + CUA_PHASES + SCIENCE_PHASES:
        key = phase.value
        is_science = phase in SCIENCE_PHASES
        is_cua = phase in CUA_PHASES

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
                "log_file": PHASE_LOG_FILES.get(phase, ""),
                "is_science": is_science,
                "is_cua": is_cua,
                "description": description,
                "prompt": prompt_text,
            }
        )
    return jsonify(phases)


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
        test_cmd=gui_config.test_cmd,
        codex_binary=gui_config.codex_binary,
        claude_binary=gui_config.claude_binary,
        codex_sandbox_mode=gui_config.codex_sandbox_mode,
        codex_approval_policy=gui_config.codex_approval_policy,
        codex_reasoning_effort=gui_config.codex_reasoning_effort,
        codex_bypass_approvals_and_sandbox=gui_config.codex_bypass_approvals_and_sandbox,
        codex_danger_confirmation=gui_config.codex_danger_confirmation,
        timeout_per_phase=gui_config.timeout_per_phase,
        max_total_tokens=gui_config.max_total_tokens,
        strict_token_budget=gui_config.strict_token_budget,
        max_time_minutes=gui_config.max_time_minutes,
        stop_on_convergence=gui_config.stop_on_convergence,
        improvement_threshold=gui_config.improvement_threshold,
        auto_commit=gui_config.auto_commit,
        commit_frequency=gui_config.commit_frequency,
        phases=phase_configs if phase_configs else [],
    )

    from codex_manager.pipeline.orchestrator import PipelineOrchestrator

    _pipeline_executor = PipelineOrchestrator(
        repo_path=gui_config.repo_path,
        config=config,
    )
    _pipeline_executor.start()
    return jsonify({"status": "started"})


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


# ══════════════════════════════════════════════════════════════════════
# CUA (Computer-Using Agent) API
# ══════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════
# Prompt Catalog API
# ══════════════════════════════════════════════════════════════════════


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


# ── Launcher ─────────────────────────────────────────────────────────


def _open_browser(port: int) -> None:
    webbrowser.open(f"http://127.0.0.1:{port}")


def run_gui(port: int = 5088, open_browser_: bool = True) -> None:
    """Start the Flask development server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    if open_browser_:
        Timer(1.5, _open_browser, args=[port]).start()
    print(f"\n  Codex Manager GUI \u2192 http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
