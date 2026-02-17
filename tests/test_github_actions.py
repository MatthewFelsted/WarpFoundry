"""Tests for GitHub Actions workflow generation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from codex_manager.github_actions import (
    PipelineWorkflowConfig,
    build_pipeline_workflow_yaml,
    detect_default_branch,
    generate_pipeline_workflow,
    normalize_workflow_filename,
)


def test_detect_default_branch_from_git_head(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    git_dir = repo / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/release/2026\n", encoding="utf-8")

    assert detect_default_branch(repo) == "release/2026"


def test_detect_default_branch_from_gitdir_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    actual_git_dir = tmp_path / "repo.git"
    actual_git_dir.mkdir(parents=True)
    (actual_git_dir / "HEAD").write_text("ref: refs/heads/develop\n", encoding="utf-8")
    (repo / ".git").write_text(f"gitdir: {actual_git_dir}\n", encoding="utf-8")

    assert detect_default_branch(repo) == "develop"


def test_build_pipeline_workflow_yaml_contains_pipeline_and_artifacts() -> None:
    workflow = build_pipeline_workflow_yaml(
        PipelineWorkflowConfig(
            branches=("main", "release"),
            python_version="3.12",
            mode="dry-run",
            cycles=2,
            max_time_minutes=90,
            agent="claude_code",
            artifact_prefix="wf-ci",
        )
    )

    assert "warpfoundry pipeline --repo . --mode dry-run --cycles 2 --max-time 90" in workflow
    assert "--agent claude_code" in workflow
    assert ".codex_manager/logs/**" in workflow
    assert ".codex_manager/outputs/**/*.md" in workflow
    assert "wf-ci-logs-${{ github.run_id }}" in workflow
    assert "wf-ci-summaries-${{ github.run_id }}" in workflow
    assert "workflow_dispatch:" in workflow


def test_generate_pipeline_workflow_writes_file_and_respects_overwrite(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    first_path = generate_pipeline_workflow(
        repo,
        config=PipelineWorkflowConfig(branches=("main",)),
        workflow_filename="ci.yml",
    )
    assert first_path == repo / ".github" / "workflows" / "ci.yml"
    assert first_path.is_file()

    with pytest.raises(FileExistsError):
        generate_pipeline_workflow(
            repo,
            config=PipelineWorkflowConfig(branches=("main",)),
            workflow_filename="ci.yml",
            overwrite=False,
        )

    updated = generate_pipeline_workflow(
        repo,
        config=PipelineWorkflowConfig(branches=("release",), cycles=3),
        workflow_filename="ci.yml",
        overwrite=True,
    )
    assert updated == first_path
    content = first_path.read_text(encoding="utf-8")
    assert "--cycles 3" in content
    assert "      - release" in content


def test_normalize_workflow_filename_rejects_paths() -> None:
    with pytest.raises(ValueError):
        normalize_workflow_filename(".github/workflows/ci.yml")
