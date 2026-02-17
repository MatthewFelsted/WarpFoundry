"""Helpers for generating GitHub Actions workflows for WarpFoundry."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from codex_manager.file_io import atomic_write_text

DEFAULT_WORKFLOW_FILE = "warpfoundry-pipeline.yml"
DEFAULT_WORKFLOW_NAME = "WarpFoundry Pipeline"
DEFAULT_ARTIFACT_PREFIX = "warpfoundry-pipeline"
_DEFAULT_BRANCH = "main"
_LOG_ARTIFACT_GLOBS = (".codex_manager/logs/**",)
_SUMMARY_ARTIFACT_GLOBS = (
    ".codex_manager/outputs/*.md",
    ".codex_manager/outputs/**/*.md",
    ".codex_manager/logs/PROGRESS.md",
    ".codex_manager/logs/SCIENTIST_REPORT.md",
)


@dataclass(frozen=True, slots=True)
class PipelineWorkflowConfig:
    """Configuration for the generated pipeline workflow."""

    branches: tuple[str, ...] = (_DEFAULT_BRANCH,)
    python_version: str = "3.11"
    mode: str = "dry-run"
    cycles: int = 1
    max_time_minutes: int = 120
    agent: str = "codex"
    workflow_name: str = DEFAULT_WORKFLOW_NAME
    artifact_prefix: str = DEFAULT_ARTIFACT_PREFIX


def normalize_branches(values: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Deduplicate branch names while preserving order."""
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values or []:
        branch = str(value or "").strip()
        if not branch or branch in seen:
            continue
        seen.add(branch)
        normalized.append(branch)
    return tuple(normalized)


def normalize_workflow_filename(value: str) -> str:
    """Return a validated workflow filename under ``.github/workflows``."""
    filename = str(value or "").strip()
    if not filename:
        raise ValueError("workflow filename must not be empty.")
    if "/" in filename or "\\" in filename:
        raise ValueError("workflow filename must be a file name, not a path.")
    if not filename.endswith((".yml", ".yaml")):
        raise ValueError("workflow filename must end with .yml or .yaml.")
    return filename


def _normalize_artifact_prefix(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-")
    return cleaned or DEFAULT_ARTIFACT_PREFIX


def _resolve_git_dir(repo: Path) -> Path | None:
    git_entry = repo / ".git"
    if git_entry.is_dir():
        return git_entry
    if not git_entry.is_file():
        return None

    try:
        first_line = git_entry.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip()
    except (IndexError, OSError):
        return None

    prefix = "gitdir:"
    if not first_line.lower().startswith(prefix):
        return None

    pointer = first_line[len(prefix) :].strip()
    if not pointer:
        return None

    git_dir = Path(pointer)
    if not git_dir.is_absolute():
        git_dir = (repo / git_dir).resolve()
    return git_dir


def detect_default_branch(repo: Path) -> str:
    """Infer the current branch from ``.git/HEAD`` when available."""
    git_dir = _resolve_git_dir(repo)
    if git_dir is None:
        return _DEFAULT_BRANCH

    head_path = git_dir / "HEAD"
    try:
        head_ref = head_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return _DEFAULT_BRANCH

    if not head_ref.startswith("ref:"):
        return _DEFAULT_BRANCH
    ref = head_ref.split(":", maxsplit=1)[1].strip()
    ref_prefix = "refs/heads/"
    if not ref.startswith(ref_prefix):
        return _DEFAULT_BRANCH
    branch = ref[len(ref_prefix) :].strip()
    return branch or _DEFAULT_BRANCH


def build_pipeline_workflow_yaml(config: PipelineWorkflowConfig) -> str:
    """Render GitHub Actions workflow YAML for pipeline automation."""
    branches = normalize_branches(list(config.branches)) or (_DEFAULT_BRANCH,)
    branch_lines = "\n".join(f"      - {branch}" for branch in branches)
    log_paths = "\n".join(f"            {path}" for path in _LOG_ARTIFACT_GLOBS)
    summary_paths = "\n".join(f"            {path}" for path in _SUMMARY_ARTIFACT_GLOBS)

    mode = config.mode if config.mode in {"dry-run", "apply"} else "dry-run"
    agent = config.agent if config.agent in {"codex", "claude_code"} else "codex"
    cycles = max(int(config.cycles), 1)
    max_time = max(int(config.max_time_minutes), 1)
    python_version = str(config.python_version or "3.11").strip() or "3.11"
    workflow_name = str(config.workflow_name or DEFAULT_WORKFLOW_NAME).strip() or DEFAULT_WORKFLOW_NAME
    artifact_prefix = _normalize_artifact_prefix(config.artifact_prefix)
    pipeline_command = (
        f"warpfoundry pipeline --repo . --mode {mode} --cycles {cycles} "
        f"--max-time {max_time} --agent {agent}"
    )

    return (
        f"name: {workflow_name}\n\n"
        "on:\n"
        "  push:\n"
        "    branches:\n"
        f"{branch_lines}\n"
        "  pull_request:\n"
        "    branches:\n"
        f"{branch_lines}\n"
        "  workflow_dispatch:\n\n"
        "concurrency:\n"
        "  group: warpfoundry-pipeline-${{ github.ref }}\n"
        "  cancel-in-progress: true\n\n"
        "jobs:\n"
        "  pipeline:\n"
        "    runs-on: ubuntu-latest\n"
        f"    timeout-minutes: {max_time}\n"
        "    steps:\n"
        "      - name: Checkout repository\n"
        "        uses: actions/checkout@v4\n\n"
        "      - name: Set up Python\n"
        "        uses: actions/setup-python@v5\n"
        "        with:\n"
        f"          python-version: '{python_version}'\n\n"
        "      - name: Install WarpFoundry\n"
        "        run: |\n"
        "          python -m pip install --upgrade pip\n"
        "          pip install .[dev]\n\n"
        "      - name: Run WarpFoundry pipeline\n"
        "        env:\n"
        "          CODEX_API_KEY: ${{ secrets.CODEX_API_KEY }}\n"
        "          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}\n"
        "          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}\n"
        "          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}\n"
        "          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}\n"
        "        run: |\n"
        f"          {pipeline_command}\n\n"
        "      - name: Upload .codex_manager/logs\n"
        "        if: always()\n"
        "        uses: actions/upload-artifact@v4\n"
        "        with:\n"
        f"          name: {artifact_prefix}-logs-${{{{ github.run_id }}}}\n"
        "          if-no-files-found: warn\n"
        "          path: |\n"
        f"{log_paths}\n\n"
        "      - name: Upload run summaries\n"
        "        if: always()\n"
        "        uses: actions/upload-artifact@v4\n"
        "        with:\n"
        f"          name: {artifact_prefix}-summaries-${{{{ github.run_id }}}}\n"
        "          if-no-files-found: warn\n"
        "          path: |\n"
        f"{summary_paths}\n"
    )


def generate_pipeline_workflow(
    repo: Path,
    *,
    config: PipelineWorkflowConfig,
    workflow_filename: str = DEFAULT_WORKFLOW_FILE,
    overwrite: bool = False,
) -> Path:
    """Write a repo-local GitHub Actions workflow and return its path."""
    repo_path = repo.resolve()
    if not repo_path.is_dir():
        raise ValueError(f"repo path does not exist: {repo_path}")

    filename = normalize_workflow_filename(workflow_filename)
    workflow_path = repo_path / ".github" / "workflows" / filename
    if workflow_path.exists() and not overwrite:
        raise FileExistsError(f"workflow file already exists: {workflow_path}")

    atomic_write_text(workflow_path, build_pipeline_workflow_yaml(config))
    return workflow_path
