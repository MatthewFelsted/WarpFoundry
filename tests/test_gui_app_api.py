"""API tests for GUI preflight and permission validation paths."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

import codex_manager.gui.app as gui_app_module
import codex_manager.preflight as preflight_module
from codex_manager.cua.actions import CUASessionResult


@pytest.fixture()
def client():
    gui_app_module.app.config.update(TESTING=True)
    with gui_app_module.app.test_client() as test_client:
        yield test_client


def _make_repo(tmp_path: Path, *, git: bool) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    if git:
        (repo / ".git").mkdir()
    return repo


def _run_git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
    )


def _make_remote_repo(tmp_path: Path, *, default_branch: str = "main") -> Path:
    seed = tmp_path / "seed"
    seed.mkdir(parents=True, exist_ok=True)
    _run_git("init", "-b", default_branch, cwd=seed)
    _run_git("config", "user.name", "GUI API Tests", cwd=seed)
    _run_git("config", "user.email", "gui-api-tests@example.com", cwd=seed)

    (seed / "README.md").write_text("# Seed Repo\n", encoding="utf-8")
    _run_git("add", "README.md", cwd=seed)
    _run_git("commit", "-m", "initial", cwd=seed)

    _run_git("checkout", "-b", "dev", cwd=seed)
    (seed / "DEV.md").write_text("dev branch\n", encoding="utf-8")
    _run_git("add", "DEV.md", cwd=seed)
    _run_git("commit", "-m", "dev branch", cwd=seed)
    _run_git("checkout", default_branch, cwd=seed)

    bare = tmp_path / "remote.git"
    _run_git("clone", "--bare", str(seed), str(bare), cwd=tmp_path)
    return bare


def _clone_tracking_repo(tmp_path: Path, remote: Path, *, clone_name: str = "local") -> Path:
    local = tmp_path / clone_name
    _run_git("clone", str(remote), str(local), cwd=tmp_path)
    _run_git("config", "user.name", "GUI API Tests", cwd=local)
    _run_git("config", "user.email", "gui-api-tests@example.com", cwd=local)
    return local


def _push_remote_update(
    tmp_path: Path,
    remote: Path,
    *,
    clone_name: str,
    filename: str,
    content: str,
    message: str,
) -> None:
    updater = _clone_tracking_repo(tmp_path, remote, clone_name=clone_name)
    (updater / filename).write_text(content, encoding="utf-8")
    _run_git("add", filename, cwd=updater)
    _run_git("commit", "-m", message, cwd=updater)
    _run_git("push", "origin", "HEAD", cwd=updater)


def _chain_payload(repo_path: Path, **overrides):
    payload = {
        "name": "Test Chain",
        "repo_path": str(repo_path),
        "steps": [
            {
                "id": "s1",
                "name": "Implement",
                "job_type": "implementation",
                "prompt_mode": "preset",
                "custom_prompt": "",
                "on_failure": "skip",
                "max_retries": 1,
                "loop_count": 1,
                "enabled": True,
                "agent": "codex",
            }
        ],
    }
    payload.update(overrides)
    return payload


def _pipeline_payload(repo_path: Path, **overrides):
    payload = {
        "repo_path": str(repo_path),
        "phases": [
            {
                "phase": "ideation",
                "enabled": True,
                "iterations": 1,
                "agent": "codex",
                "on_failure": "skip",
                "test_policy": "skip",
                "custom_prompt": "",
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_chain_start_rejects_invalid_permission_value(client, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    resp = client.post(
        "/api/chain/start",
        json=_chain_payload(repo, codex_sandbox_mode="invalid-mode"),
    )
    data = resp.get_json()
    assert resp.status_code == 400
    assert data
    assert "Invalid config" in data["error"]


def test_chain_start_preflight_rejects_non_git_repo(client, monkeypatch, tmp_path: Path):
    repo = _make_repo(tmp_path, git=False)
    monkeypatch.setattr(gui_app_module, "_agent_preflight_issues", lambda *_a, **_k: [])

    resp = client.post("/api/chain/start", json=_chain_payload(repo))
    data = resp.get_json()
    assert resp.status_code == 400
    assert data
    assert any("Not a git repository" in issue for issue in data.get("issues", []))


def test_chain_start_preflight_rejects_output_file_collisions(client, monkeypatch, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    monkeypatch.setattr(gui_app_module, "_agent_preflight_issues", lambda *_a, **_k: [])
    payload = _chain_payload(
        repo,
        steps=[
            {
                "id": "a1",
                "name": "Same Name",
                "job_type": "implementation",
                "enabled": True,
                "agent": "codex",
            },
            {
                "id": "a2",
                "name": "Same Name",
                "job_type": "testing",
                "enabled": True,
                "agent": "codex",
            },
        ],
    )
    resp = client.post("/api/chain/start", json=payload)
    data = resp.get_json()
    assert resp.status_code == 400
    assert data
    assert any("Multiple enabled steps write to" in issue for issue in data.get("issues", []))


def test_chain_start_preflight_rejects_case_variant_output_collisions(
    client, monkeypatch, tmp_path: Path
):
    repo = _make_repo(tmp_path, git=True)
    monkeypatch.setattr(gui_app_module, "_agent_preflight_issues", lambda *_a, **_k: [])
    payload = _chain_payload(
        repo,
        steps=[
            {
                "id": "a1",
                "name": "Implement API",
                "job_type": "implementation",
                "enabled": True,
                "agent": "codex",
            },
            {
                "id": "a2",
                "name": "implement api",
                "job_type": "testing",
                "enabled": True,
                "agent": "codex",
            },
        ],
    )
    resp = client.post("/api/chain/start", json=payload)
    data = resp.get_json()
    assert resp.status_code == 400
    assert data
    assert any("Multiple enabled steps write to" in issue for issue in data.get("issues", []))


def test_chain_start_preflight_reports_missing_codex_binary(client, monkeypatch, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    monkeypatch.setattr(gui_app_module, "_binary_exists", lambda _binary: False)
    monkeypatch.setattr(gui_app_module, "_has_codex_auth", lambda: True)

    resp = client.post(
        "/api/chain/start", json=_chain_payload(repo, codex_binary="codex-not-found")
    )
    data = resp.get_json()
    assert resp.status_code == 400
    assert data
    assert any("Codex binary not found" in issue for issue in data.get("issues", []))


def test_chain_start_preflight_reports_repo_not_writable(client, monkeypatch, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    monkeypatch.setattr(gui_app_module, "_agent_preflight_issues", lambda *_a, **_k: [])
    monkeypatch.setattr(
        gui_app_module, "_repo_write_error", lambda _repo: "Repository is not writable: denied"
    )

    resp = client.post("/api/chain/start", json=_chain_payload(repo))
    data = resp.get_json()
    assert resp.status_code == 400
    assert data
    assert any("Repository is not writable" in issue for issue in data.get("issues", []))


def test_chain_start_requires_danger_confirmation_when_bypass_enabled(client, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    resp = client.post(
        "/api/chain/start",
        json=_chain_payload(repo, codex_bypass_approvals_and_sandbox=True),
    )
    data = resp.get_json()
    assert resp.status_code == 400
    assert data
    assert "Invalid config" in data["error"]
    assert "codex_danger_confirmation" in data["error"]


def test_pipeline_start_rejects_invalid_permission_value(client, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    resp = client.post(
        "/api/pipeline/start",
        json=_pipeline_payload(repo, codex_approval_policy="always"),
    )
    data = resp.get_json()
    assert resp.status_code == 400
    assert data
    assert "Invalid config" in data["error"]


def test_pipeline_start_rejects_invalid_commit_frequency(client, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    resp = client.post(
        "/api/pipeline/start",
        json=_pipeline_payload(repo, commit_frequency="every_step"),
    )
    data = resp.get_json()
    assert resp.status_code == 400
    assert data
    assert "Invalid config" in data["error"]


def test_pipeline_start_rejects_invalid_phase_key(client, monkeypatch, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    monkeypatch.setattr(gui_app_module, "_agent_preflight_issues", lambda *_a, **_k: [])
    payload = _pipeline_payload(
        repo,
        phases=[
            {
                "phase": "not_a_real_phase",
                "enabled": True,
                "iterations": 1,
                "agent": "codex",
                "on_failure": "skip",
                "test_policy": "skip",
                "custom_prompt": "",
            }
        ],
    )
    resp = client.post("/api/pipeline/start", json=payload)
    data = resp.get_json()
    assert resp.status_code == 400
    assert data
    assert "Invalid pipeline phase(s): not_a_real_phase" in data["error"]


def test_pipeline_start_rejects_invalid_phase_test_policy(client, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    resp = client.post(
        "/api/pipeline/start",
        json=_pipeline_payload(
            repo,
            phases=[
                {
                    "phase": "ideation",
                    "enabled": True,
                    "iterations": 1,
                    "agent": "codex",
                    "on_failure": "skip",
                    "test_policy": "aggressive",
                    "custom_prompt": "",
                }
            ],
        ),
    )
    data = resp.get_json()
    assert resp.status_code == 400
    assert data
    assert "Invalid config" in data["error"]
    assert "test_policy" in data["error"]


def test_pipeline_start_requires_danger_confirmation_when_bypass_enabled(client, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    resp = client.post(
        "/api/pipeline/start",
        json=_pipeline_payload(repo, codex_bypass_approvals_and_sandbox=True),
    )
    data = resp.get_json()
    assert resp.status_code == 400
    assert data
    assert "Invalid config" in data["error"]
    assert "codex_danger_confirmation" in data["error"]


def test_pipeline_phases_api_marks_self_improvement_phase(client):
    resp = client.get("/api/pipeline/phases")
    data = resp.get_json()

    assert resp.status_code == 200
    assert isinstance(data, list)
    restart_phase = next((item for item in data if item.get("key") == "apply_upgrades_and_restart"), None)
    assert restart_phase is not None
    assert restart_phase["is_self_improvement"] is True


def test_pipeline_phases_api_marks_deep_research_phase(client):
    resp = client.get("/api/pipeline/phases")
    data = resp.get_json()

    assert resp.status_code == 200
    assert isinstance(data, list)
    deep_research = next((item for item in data if item.get("key") == "deep_research"), None)
    assert deep_research is not None
    assert deep_research["is_deep_research"] is True


def test_pipeline_phases_api_places_science_before_implementation(client):
    resp = client.get("/api/pipeline/phases")
    data = resp.get_json()

    assert resp.status_code == 200
    assert isinstance(data, list)
    keys = [str(item.get("key", "")) for item in data]
    assert "theorize" in keys
    assert "implementation" in keys
    assert keys.index("theorize") < keys.index("implementation")


def test_pipeline_phases_api_returns_default_test_policies(client):
    resp = client.get("/api/pipeline/phases")
    data = resp.get_json()

    assert resp.status_code == 200
    assert isinstance(data, list)
    by_key = {str(item.get("key", "")): item for item in data}
    assert by_key["ideation"]["default_test_policy"] == "skip"
    assert by_key["testing"]["default_test_policy"] == "full"
    assert by_key["implementation"]["default_test_policy"] == "smoke"


def test_health_endpoint_reports_chain_and_pipeline_status(client, monkeypatch):
    class _ChainExec:
        is_running = True

    class _PipeExec:
        is_running = False

    monkeypatch.setattr(gui_app_module, "executor", _ChainExec())
    monkeypatch.setattr(gui_app_module, "_pipeline_executor", _PipeExec())
    monkeypatch.setattr(
        gui_app_module,
        "_model_watchdog_health",
        lambda: {
            "model_watchdog_enabled": True,
            "model_watchdog_running": True,
            "model_watchdog_next_due_at": "2026-02-16T00:00:00Z",
            "model_watchdog_last_status": "ok",
        },
    )

    resp = client.get("/api/health")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["ok"] is True
    assert isinstance(data["time_epoch_ms"], int)
    assert data["chain_running"] is True
    assert data["pipeline_running"] is False
    assert data["model_watchdog_enabled"] is True
    assert data["model_watchdog_running"] is True
    assert data["model_watchdog_last_status"] == "ok"


def test_chain_start_preflight_requires_image_provider_auth(client, monkeypatch, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    monkeypatch.setattr(gui_app_module, "_agent_preflight_issues", lambda *_a, **_k: [])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    resp = client.post(
        "/api/chain/start",
        json=_chain_payload(
            repo,
            image_generation_enabled=True,
            image_provider="openai",
            image_model="gpt-image-1",
        ),
    )
    data = resp.get_json()

    assert resp.status_code == 400
    assert data
    assert any("OPENAI_API_KEY" in issue for issue in data.get("issues", []))


def test_pipeline_start_preflight_requires_image_provider_auth(
    client, monkeypatch, tmp_path: Path
):
    repo = _make_repo(tmp_path, git=True)
    monkeypatch.setattr(gui_app_module, "_agent_preflight_issues", lambda *_a, **_k: [])
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    resp = client.post(
        "/api/pipeline/start",
        json=_pipeline_payload(
            repo,
            image_generation_enabled=True,
            image_provider="google",
            image_model="nano-banana",
        ),
    )
    data = resp.get_json()

    assert resp.status_code == 400
    assert data
    assert any(
        ("GOOGLE_API_KEY" in issue or "GEMINI_API_KEY" in issue)
        for issue in data.get("issues", [])
    )


def test_pipeline_start_rejects_auto_restart_without_self_improvement(
    client, monkeypatch, tmp_path: Path
):
    repo = _make_repo(tmp_path, git=True)
    monkeypatch.setattr(gui_app_module, "_agent_preflight_issues", lambda *_a, **_k: [])

    resp = client.post(
        "/api/pipeline/start",
        json=_pipeline_payload(
            repo,
            self_improvement_enabled=False,
            self_improvement_auto_restart=True,
        ),
    )
    data = resp.get_json()

    assert resp.status_code == 400
    assert data
    assert any("self_improvement_auto_restart" in issue for issue in data.get("issues", []))


def test_pipeline_start_maps_capability_and_self_improvement_fields(
    client, monkeypatch, tmp_path: Path
):
    repo = _make_repo(tmp_path, git=True)
    monkeypatch.setattr(gui_app_module, "_agent_preflight_issues", lambda *_a, **_k: [])
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(gui_app_module, "_pipeline_executor", None)
    monkeypatch.setitem(sys.modules, "chromadb", types.ModuleType("chromadb"))

    captured: dict[str, object] = {}

    class _StubOrchestrator:
        def __init__(self, repo_path, config):
            captured["repo_path"] = str(repo_path)
            captured["config"] = config
            self.is_running = False

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(
        "codex_manager.pipeline.orchestrator.PipelineOrchestrator",
        _StubOrchestrator,
    )

    payload = _pipeline_payload(
        repo,
        smoke_test_cmd="python -m pytest -q -m smoke",
        allow_path_creation=False,
        dependency_install_policy="allow_system",
        image_generation_enabled=True,
        image_provider="openai",
        image_model="gpt-image-1",
        vector_memory_enabled=True,
        vector_memory_backend="chroma",
        vector_memory_collection="repo-memory",
        vector_memory_top_k=12,
        deep_research_enabled=True,
        deep_research_providers="both",
        deep_research_max_age_hours=240,
        deep_research_dedupe=True,
        self_improvement_enabled=True,
        self_improvement_auto_restart=True,
    )
    resp = client.post("/api/pipeline/start", json=payload)
    data = resp.get_json()

    assert resp.status_code == 200
    assert data == {"status": "started"}
    assert captured.get("started") is True
    assert captured.get("repo_path") == str(repo)

    config = captured["config"]
    assert config.allow_path_creation is False
    assert config.smoke_test_cmd == "python -m pytest -q -m smoke"
    assert config.dependency_install_policy == "allow_system"
    assert config.image_generation_enabled is True
    assert config.image_provider == "openai"
    assert config.image_model == "gpt-image-1"
    assert config.vector_memory_enabled is True
    assert config.vector_memory_backend == "chroma"
    assert config.vector_memory_collection == "repo-memory"
    assert config.vector_memory_top_k == 12
    assert config.deep_research_enabled is True
    assert config.deep_research_providers == "both"
    assert config.deep_research_max_age_hours == 240
    assert config.deep_research_dedupe is True
    assert config.self_improvement_enabled is True
    assert config.self_improvement_auto_restart is True
    assert config.phases
    assert config.phases[0].test_policy == "skip"


def test_foundation_prompt_improve_endpoint_returns_payload(client, monkeypatch):
    monkeypatch.setattr(
        gui_app_module,
        "_improve_foundational_prompt",
        lambda prompt, project_name, assistants: {
            "recommended_prompt": "improved",
            "variants": [{"assistant": "codex", "model": "gpt-5.2", "prompt": "improved"}],
            "warning": "",
        },
    )
    resp = client.post(
        "/api/project/foundation/improve",
        json={
            "prompt": "build a todo app",
            "project_name": "Todo",
            "assistants": ["codex"],
        },
    )
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["recommended_prompt"] == "improved"
    assert data["variants"][0]["assistant"] == "codex"


def test_write_licensing_packaging_artifacts_generates_expected_files(tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    generated = gui_app_module._write_licensing_packaging_artifacts(
        project_path=repo,
        project_name="Demo Project",
        description="Demo",
        strategy="dual_license",
        include_commercial_tiers=True,
        owner_contact_email="owner@example.com",
    )
    normalized = {Path(item).as_posix() for item in generated}
    assert ".codex_manager/business/licensing_profile.json" in normalized
    assert "docs/LICENSING_STRATEGY.md" in normalized
    assert "docs/COMMERCIAL_OFFERING.md" in normalized
    assert "docs/PRICING_TIERS.md" in normalized
    assert (repo / ".codex_manager" / "business" / "licensing_profile.json").is_file()
    assert (repo / "docs" / "LICENSING_STRATEGY.md").is_file()
    assert (repo / "docs" / "COMMERCIAL_OFFERING.md").is_file()
    assert (repo / "docs" / "PRICING_TIERS.md").is_file()


def test_write_licensing_packaging_artifacts_records_pending_legal_review(tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    generated = gui_app_module._write_licensing_packaging_artifacts(
        project_path=repo,
        project_name="Demo Project",
        description="Demo",
        strategy="dual_license",
        include_commercial_tiers=True,
        owner_contact_email="owner@example.com",
        legal_review_required=True,
        legal_signoff_approved=False,
        legal_reviewer="Owner",
        legal_notes="Need counsel sign-off.",
    )
    normalized = {Path(item).as_posix() for item in generated}
    assert ".codex_manager/business/legal_review.json" in normalized

    state_path = repo / ".codex_manager" / "business" / "legal_review.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["required"] is True
    assert state["approved"] is False
    assert state["status"] == "pending"
    assert state["publish_ready"] is False

    strategy_text = (repo / "docs" / "LICENSING_STRATEGY.md").read_text(encoding="utf-8")
    assert "Legal review checkpoint is **pending**" in strategy_text


def test_extract_governance_warnings_flags_claims_and_source_quality():
    text = (
        "This launch is guaranteed and risk-free. "
        "Read details at http://example.com/a and http://example.com/b"
    )
    warnings = gui_app_module._extract_governance_warnings(text)
    assert any("overconfident" in warning.lower() for warning in warnings)
    assert any("not https" in warning.lower() for warning in warnings)
    assert any("low-trust source domains" in warning.lower() for warning in warnings)


def test_governance_source_policy_endpoint_roundtrip(client, monkeypatch, tmp_path: Path):
    policy_path = tmp_path / "source_policy.json"
    monkeypatch.setattr(gui_app_module, "_GOVERNANCE_POLICY_PATH", policy_path)
    monkeypatch.delenv("CODEX_MANAGER_RESEARCH_ALLOWED_DOMAINS", raising=False)
    monkeypatch.delenv("CODEX_MANAGER_RESEARCH_BLOCKED_DOMAINS", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_ALLOWED_SOURCE_DOMAINS", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_BLOCKED_SOURCE_DOMAINS", raising=False)

    save_resp = client.post(
        "/api/governance/source-policy",
        json={
            "research_allowed_domains": "docs.python.org, openai.com",
            "research_blocked_domains": "example.com",
            "deep_research_allowed_domains": "arxiv.org",
            "deep_research_blocked_domains": "x.com",
        },
    )
    save_data = save_resp.get_json()
    assert save_resp.status_code == 200
    assert save_data
    assert save_data["status"] == "saved"
    assert policy_path.is_file()
    assert os.getenv("CODEX_MANAGER_RESEARCH_ALLOWED_DOMAINS") == "docs.python.org,openai.com"
    assert os.getenv("DEEP_RESEARCH_BLOCKED_SOURCE_DOMAINS") == "x.com"

    load_resp = client.get("/api/governance/source-policy")
    load_data = load_resp.get_json()
    assert load_resp.status_code == 200
    assert load_data
    assert load_data["research_allowed_domains"] == "docs.python.org,openai.com"
    assert load_data["research_blocked_domains"] == "example.com"
    assert load_data["deep_research_allowed_domains"] == "arxiv.org"
    assert load_data["deep_research_blocked_domains"] == "x.com"


def test_github_auth_settings_roundtrip_with_secure_storage(client, monkeypatch, tmp_path: Path):
    meta_path = tmp_path / "github_auth.json"
    secrets: dict[str, str] = {}

    monkeypatch.setattr(gui_app_module, "_GITHUB_AUTH_META_PATH", meta_path)
    monkeypatch.setattr(gui_app_module, "_github_keyring_status", lambda: (True, "tests.fake", ""))
    monkeypatch.setattr(gui_app_module, "_github_secret_get", lambda key: secrets.get(key, ""))
    monkeypatch.setattr(gui_app_module, "_github_secret_set", lambda key, value: secrets.__setitem__(key, value))

    def _delete_secret(key: str) -> None:
        secrets.pop(key, None)

    monkeypatch.setattr(gui_app_module, "_github_secret_delete", _delete_secret)

    save_resp = client.post(
        "/api/github/auth",
        json={
            "preferred_auth": "ssh",
            "pat": "ghp_test_123",
            "ssh_private_key": "-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n-----END OPENSSH PRIVATE KEY-----",
        },
    )
    save_data = save_resp.get_json()
    assert save_resp.status_code == 200
    assert save_data
    assert save_data["status"] == "saved"
    assert meta_path.is_file()
    assert secrets[gui_app_module._GITHUB_PAT_SECRET_KEY] == "ghp_test_123"
    assert gui_app_module._GITHUB_SSH_SECRET_KEY in secrets
    assert save_data["settings"]["preferred_auth"] == "ssh"
    assert save_data["settings"]["has_pat"] is True
    assert save_data["settings"]["has_ssh_key"] is True

    load_resp = client.get("/api/github/auth")
    load_data = load_resp.get_json()
    assert load_resp.status_code == 200
    assert load_data
    assert load_data["preferred_auth"] == "ssh"
    assert load_data["has_pat"] is True
    assert load_data["has_ssh_key"] is True
    assert "ghp_test_123" not in json.dumps(load_data)

    clear_resp = client.post(
        "/api/github/auth",
        json={
            "preferred_auth": "https",
            "clear_pat": True,
        },
    )
    clear_data = clear_resp.get_json()
    assert clear_resp.status_code == 200
    assert clear_data
    assert clear_data["settings"]["preferred_auth"] == "https"
    assert clear_data["settings"]["has_pat"] is False
    assert clear_data["settings"]["has_ssh_key"] is True


def test_github_auth_save_requires_available_secure_storage(client, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(gui_app_module, "_GITHUB_AUTH_META_PATH", tmp_path / "github_auth.json")
    monkeypatch.setattr(
        gui_app_module,
        "_github_keyring_status",
        lambda: (False, "keyring.backends.fail.Keyring", "no keyring backend"),
    )

    resp = client.post(
        "/api/github/auth",
        json={
            "preferred_auth": "https",
            "pat": "ghp_fail_me",
        },
    )
    data = resp.get_json()
    assert resp.status_code == 503
    assert data
    assert "no keyring backend" in data["error"]


def test_github_auth_test_endpoint_uses_saved_pat(client, monkeypatch, tmp_path: Path):
    meta_path = tmp_path / "github_auth.json"
    monkeypatch.setattr(gui_app_module, "_GITHUB_AUTH_META_PATH", meta_path)
    monkeypatch.setattr(gui_app_module, "_github_secret_get", lambda _key: "ghp_saved_token")
    monkeypatch.setattr(
        gui_app_module,
        "_github_test_pat",
        lambda _token: {"ok": True, "message": "PAT accepted", "login": "octocat"},
    )

    resp = client.post(
        "/api/github/auth/test",
        json={
            "auth_method": "https",
            "use_saved": True,
        },
    )
    data = resp.get_json()
    assert resp.status_code == 200
    assert data
    assert data["ok"] is True
    assert data["auth_method"] == "https"
    assert data["login"] == "octocat"
    assert meta_path.is_file()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["last_test_ok"] is True
    assert meta["last_test_at"]


def test_github_auth_test_endpoint_returns_pat_scope_troubleshooting(
    client, monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(gui_app_module, "_GITHUB_AUTH_META_PATH", tmp_path / "github_auth.json")
    monkeypatch.setattr(gui_app_module, "_github_secret_get", lambda _key: "ghp_saved_token")
    monkeypatch.setattr(
        gui_app_module,
        "_github_test_pat",
        lambda _token: {
            "ok": False,
            "message": (
                "GitHub denied this PAT (403 Forbidden). "
                "Resource not accessible by personal access token"
            ),
        },
    )

    resp = client.post(
        "/api/github/auth/test",
        json={
            "auth_method": "https",
            "use_saved": True,
        },
    )
    data = resp.get_json()
    assert resp.status_code == 200
    assert data
    assert data["ok"] is False
    troubleshooting = data["troubleshooting"]
    assert troubleshooting["auth_method"] == "https"
    checks = troubleshooting["checks"]
    assert isinstance(checks, list)
    pat_scope = next(check for check in checks if check["key"] == "pat_scopes")
    assert pat_scope["status"] == "action_required"
    assert "Contents: Read and write" in pat_scope["detail"]


def test_github_auth_test_endpoint_returns_ssh_known_hosts_and_key_permissions_guidance(
    client, monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(gui_app_module, "_GITHUB_AUTH_META_PATH", tmp_path / "github_auth.json")
    monkeypatch.setattr(
        gui_app_module,
        "_github_test_ssh_key",
        lambda _key: {
            "ok": False,
            "message": "Host key verification failed.",
            "output": "Host key verification failed.",
        },
    )

    resp = client.post(
        "/api/github/auth/test",
        json={
            "auth_method": "ssh",
            "use_saved": False,
            "ssh_private_key": "-----BEGIN OPENSSH PRIVATE KEY-----\nkey\n-----END OPENSSH PRIVATE KEY-----",
        },
    )
    data = resp.get_json()
    assert resp.status_code == 200
    assert data
    assert data["ok"] is False
    troubleshooting = data["troubleshooting"]
    assert troubleshooting["auth_method"] == "ssh"
    checks = troubleshooting["checks"]
    assert isinstance(checks, list)
    check_keys = {check.get("key") for check in checks}
    assert "ssh_known_hosts" in check_keys
    assert "ssh_key_permissions" in check_keys


def test_github_auth_test_endpoint_requires_credentials(client, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(gui_app_module, "_GITHUB_AUTH_META_PATH", tmp_path / "github_auth.json")
    monkeypatch.setattr(gui_app_module, "_github_secret_get", lambda _key: "")

    resp = client.post(
        "/api/github/auth/test",
        json={
            "auth_method": "https",
            "use_saved": True,
        },
    )
    data = resp.get_json()
    assert resp.status_code == 400
    assert data
    assert data["ok"] is False
    assert "Provide a GitHub PAT" in data["message"]


def test_project_legal_review_signoff_endpoint_updates_state(client, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    gui_app_module._upsert_legal_review_state(
        project_path=repo,
        project_name="repo",
        required=True,
        approved=False,
        reviewer="",
        notes="Pending",
        files=["docs/LICENSING_STRATEGY.md"],
        source="test",
    )

    resp = client.post(
        "/api/project/legal-review/signoff",
        json={
            "repo_path": str(repo),
            "approved": True,
            "reviewer": "Counsel",
            "notes": "Reviewed for launch.",
        },
    )
    data = resp.get_json()
    assert resp.status_code == 200
    assert data
    state = data["legal_review"]
    assert state["status"] == "approved"
    assert state["approved"] is True
    assert state["publish_ready"] is True
    assert state["reviewer"] == "Counsel"
    assert state["approved_at"]


def test_system_restart_rejects_missing_checkpoint(client):
    resp = client.post(
        "/api/system/restart",
        json={"pipeline_resume_checkpoint": "C:/missing/pipeline_resume.json"},
    )
    data = resp.get_json()

    assert resp.status_code == 400
    assert data
    assert "Checkpoint not found" in data["error"]


def test_system_restart_spawns_replacement_server_with_checkpoint(
    client, monkeypatch, tmp_path: Path
):
    checkpoint = tmp_path / "pipeline_resume.json"
    checkpoint.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(gui_app_module, "_SERVER_PORT", 6111)

    observed: dict[str, object] = {}

    def _fake_launch(command):
        observed["command"] = command

    def _fake_terminate(delay_seconds: float = 0.75):
        observed["terminated"] = delay_seconds

    monkeypatch.setattr(gui_app_module, "_launch_replacement_server", _fake_launch)
    monkeypatch.setattr(gui_app_module, "_terminate_current_process", _fake_terminate)

    resp = client.post(
        "/api/system/restart",
        json={"pipeline_resume_checkpoint": str(checkpoint)},
    )
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["status"] == "restarting"
    assert data["port"] == 6111
    assert data["pipeline_resume_checkpoint"] == str(checkpoint.resolve())

    command = observed["command"]
    assert command[:5] == [gui_app_module.sys.executable, "-m", "codex_manager", "gui", "--port"]
    assert "--pipeline-resume-checkpoint" in command
    idx = command.index("--pipeline-resume-checkpoint")
    assert command[idx + 1] == str(checkpoint.resolve())
    assert "terminated" in observed


def test_pipeline_resume_state_reports_checkpoint_payload(client, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    checkpoint = repo / ".codex_manager" / "state" / "pipeline_resume.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(
        json.dumps(
            {
                "repo_path": str(repo.resolve()),
                "config": {"mode": "dry-run"},
                "resume_cycle": 3,
                "resume_phase_index": 2,
            }
        ),
        encoding="utf-8",
    )

    resp = client.get(
        "/api/pipeline/resume-state",
        query_string={"repo_path": str(repo)},
    )
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["exists"] is True
    assert data["resume_ready"] is True
    assert data["resume_cycle"] == 3
    assert data["resume_phase_index"] == 2
    assert data["saved_at_epoch_ms"] > 0
    assert Path(data["checkpoint_path"]).resolve() == checkpoint.resolve()


def test_pipeline_resume_state_clear_deletes_checkpoint(client, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    checkpoint = repo / ".codex_manager" / "state" / "pipeline_resume.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("{}", encoding="utf-8")

    resp = client.post(
        "/api/pipeline/resume-state/clear",
        json={"repo_path": str(repo)},
    )
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["status"] == "cleared"
    assert data["removed"] is True
    assert checkpoint.exists() is False


def test_model_watchdog_status_endpoint_returns_watchdog_payload(client, monkeypatch):
    class _WatchdogStub:
        def status(self):
            return {
                "running": True,
                "config": {"enabled": True, "interval_hours": 24},
                "state": {"last_status": "ok"},
                "next_due_at": "2026-02-16T00:00:00Z",
            }

    monkeypatch.setattr(gui_app_module, "_get_model_watchdog", lambda: _WatchdogStub())

    resp = client.get("/api/system/model-watchdog/status")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["running"] is True
    assert data["config"]["enabled"] is True
    assert data["state"]["last_status"] == "ok"


def test_model_watchdog_alerts_endpoint_returns_watchdog_alerts(client, monkeypatch):
    class _WatchdogStub:
        def latest_alerts(self):
            return {
                "has_alerts": True,
                "alerts": [
                    {
                        "severity": "warn",
                        "title": "openai models removed",
                        "detail": "1 model disappeared",
                        "action": "Migrate defaults",
                    }
                ],
                "last_generated_at": "2026-02-15T10:00:00Z",
                "status": {"running": True},
            }

    monkeypatch.setattr(gui_app_module, "_get_model_watchdog", lambda: _WatchdogStub())

    resp = client.get("/api/system/model-watchdog/alerts")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["has_alerts"] is True
    assert data["alerts"][0]["severity"] == "warn"


def test_model_watchdog_run_endpoint_forwards_force_flag(client, monkeypatch):
    observed: dict[str, object] = {}

    class _WatchdogStub:
        def run_once(self, *, force: bool = True, reason: str = "manual"):
            observed["force"] = force
            observed["reason"] = reason
            return {"status": "ok", "reason": reason, "catalog_changed": False}

    monkeypatch.setattr(gui_app_module, "_get_model_watchdog", lambda: _WatchdogStub())

    resp = client.post("/api/system/model-watchdog/run", json={"force": False})
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["status"] == "ok"
    assert observed["force"] is False
    assert observed["reason"] == "manual"


def test_model_watchdog_config_endpoint_updates_watchdog(client, monkeypatch):
    observed: dict[str, object] = {}

    class _WatchdogStub:
        def __init__(self):
            self._config = {
                "enabled": True,
                "interval_hours": 24,
                "providers": ["openai"],
                "request_timeout_seconds": 10,
                "auto_run_on_start": True,
                "history_limit": 100,
            }

        def update_config(self, updates):
            observed["updates"] = updates
            self._config.update(updates)
            return dict(self._config)

        def status(self):
            return {
                "running": bool(self._config["enabled"]),
                "config": dict(self._config),
                "state": {"last_status": "ok"},
                "next_due_at": "2026-02-16T00:00:00Z",
            }

    monkeypatch.setattr(gui_app_module, "_get_model_watchdog", lambda: _WatchdogStub())

    resp = client.post(
        "/api/system/model-watchdog/config",
        json={
            "enabled": False,
            "interval_hours": 72,
            "providers": ["openai", "google"],
            "request_timeout_seconds": 15,
            "auto_run_on_start": False,
            "history_limit": 200,
        },
    )
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["status"] == "saved"
    assert observed["updates"]["enabled"] is False
    assert observed["updates"]["interval_hours"] == 72
    assert observed["updates"]["providers"] == ["openai", "google"]
    assert observed["updates"]["request_timeout_seconds"] == 15
    assert observed["updates"]["auto_run_on_start"] is False
    assert observed["updates"]["history_limit"] == 200


def test_chain_outputs_list_and_read_file(client, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    out_dir = repo / ".codex_manager" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "Implement.md"
    out_file.write_text("# Output\n\nhello\n", encoding="utf-8")

    list_resp = client.get("/api/chain/outputs", query_string={"repo_path": str(repo)})
    list_data = list_resp.get_json()
    assert list_resp.status_code == 200
    assert list_data
    assert list_data["output_dir"].endswith(str(Path(".codex_manager") / "outputs"))
    assert any(f["name"] == "Implement.md" for f in list_data["files"])

    read_resp = client.get(
        "/api/chain/outputs/Implement.md",
        query_string={"repo_path": str(repo)},
    )
    read_data = read_resp.get_json()
    assert read_resp.status_code == 200
    assert read_data
    assert "hello" in read_data["content"]


def test_chain_outputs_read_recovers_non_utf8_file(client, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    out_dir = repo / ".codex_manager" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "Implement.md"
    out_file.write_bytes(b"alpha\x97omega\n")

    read_resp = client.get(
        "/api/chain/outputs/Implement.md",
        query_string={"repo_path": str(repo)},
    )
    read_data = read_resp.get_json()
    assert read_resp.status_code == 200
    assert read_data
    assert "alpha" in read_data["content"]
    assert "omega" in read_data["content"]
    assert "\u2014" in read_data["content"]
    assert b"\x97" not in out_file.read_bytes()


def test_chain_outputs_reject_path_traversal(client, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    resp = client.get(
        "/api/chain/outputs/../secrets.txt",
        query_string={"repo_path": str(repo)},
    )
    data = resp.get_json()
    assert resp.status_code == 400
    assert data
    assert "Invalid filename" in data["error"]


def test_configs_save_and_load_roundtrip(client, monkeypatch, tmp_path: Path):
    cfg_dir = tmp_path / "chains"
    monkeypatch.setattr(gui_app_module, "CONFIGS_DIR", cfg_dir)

    payload = {"repo_path": "C:/repo", "steps": [{"id": "s1", "enabled": True}]}

    save_resp = client.post(
        "/api/configs/save",
        json={"name": "My Config", "config": payload},
    )
    save_data = save_resp.get_json()

    assert save_resp.status_code == 200
    assert save_data
    assert save_data["status"] == "saved"
    assert (cfg_dir / "My Config.json").is_file()

    load_resp = client.post("/api/configs/load", json={"name": "My Config"})
    load_data = load_resp.get_json()

    assert load_resp.status_code == 200
    assert load_data == payload


@pytest.mark.parametrize("name", ["../secret", "..\\secret", "secret.json", ""])
def test_configs_load_rejects_invalid_names(client, monkeypatch, tmp_path: Path, name: str):
    cfg_dir = tmp_path / "chains"
    monkeypatch.setattr(gui_app_module, "CONFIGS_DIR", cfg_dir)

    resp = client.post("/api/configs/load", json={"name": name})
    data = resp.get_json()

    assert resp.status_code == 400
    assert data
    assert "Invalid config name" in data["error"]


@pytest.mark.parametrize("name", ["../secret", "..\\secret", "secret.json", "bad/name"])
def test_configs_save_rejects_invalid_names(client, monkeypatch, tmp_path: Path, name: str):
    cfg_dir = tmp_path / "chains"
    monkeypatch.setattr(gui_app_module, "CONFIGS_DIR", cfg_dir)

    resp = client.post(
        "/api/configs/save",
        json={"name": name, "config": {"repo_path": "C:/repo"}},
    )
    data = resp.get_json()

    assert resp.status_code == 400
    assert data
    assert "Invalid config name" in data["error"]


def test_configs_load_rejects_invalid_json_file(client, monkeypatch, tmp_path: Path):
    cfg_dir = tmp_path / "chains"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(gui_app_module, "CONFIGS_DIR", cfg_dir)
    (cfg_dir / "Broken.json").write_text("{not json", encoding="utf-8")

    resp = client.post("/api/configs/load", json={"name": "Broken"})
    data = resp.get_json()

    assert resp.status_code == 400
    assert data
    assert "not valid JSON" in data["error"]


def test_configs_load_rejects_non_object_json(client, monkeypatch, tmp_path: Path):
    cfg_dir = tmp_path / "chains"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(gui_app_module, "CONFIGS_DIR", cfg_dir)
    (cfg_dir / "Listy.json").write_text('["not", "a", "dict"]', encoding="utf-8")

    resp = client.post("/api/configs/load", json={"name": "Listy"})
    data = resp.get_json()

    assert resp.status_code == 400
    assert data
    assert "must contain a JSON object" in data["error"]


def test_configs_load_missing_valid_name_returns_404(client, monkeypatch, tmp_path: Path):
    cfg_dir = tmp_path / "chains"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(gui_app_module, "CONFIGS_DIR", cfg_dir)

    resp = client.post("/api/configs/load", json={"name": "Missing"})
    data = resp.get_json()

    assert resp.status_code == 404
    assert data
    assert "Config not found" in data["error"]


def test_configs_load_rejects_resolved_path_outside_configs_dir(client, monkeypatch, tmp_path: Path):
    cfg_dir = tmp_path / "chains"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(gui_app_module, "CONFIGS_DIR", cfg_dir)

    outside = tmp_path / "outside.json"
    outside.write_text('{"repo_path": "C:/repo"}', encoding="utf-8")
    link = cfg_dir / "Alias.json"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation not supported in this environment")

    resp = client.post("/api/configs/load", json={"name": "Alias"})
    data = resp.get_json()

    assert resp.status_code == 400
    assert data
    assert "Invalid config name" in data["error"]


def test_configs_save_rejects_non_object_json(client, monkeypatch, tmp_path: Path):
    cfg_dir = tmp_path / "chains"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(gui_app_module, "CONFIGS_DIR", cfg_dir)

    resp = client.post(
        "/api/configs/save",
        json={"name": "Listy", "config": ["not", "a", "dict"]},
    )
    data = resp.get_json()

    assert resp.status_code == 400
    assert data
    assert "JSON object" in data["error"]
    assert not (cfg_dir / "Listy.json").exists()


def test_write_json_file_atomic_handles_concurrent_writers(tmp_path: Path):
    target = tmp_path / "chains" / "Concurrent.json"
    errors: list[Exception] = []

    def _writer(worker_id: int) -> None:
        try:
            for iteration in range(20):
                gui_app_module._write_json_file_atomic(
                    target,
                    {
                        "worker_id": worker_id,
                        "iteration": iteration,
                    },
                )
        except Exception as exc:  # pragma: no cover - defensive capture
            errors.append(exc)

    threads = [threading.Thread(target=_writer, args=(idx,)) for idx in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert isinstance(payload.get("worker_id"), int)
    assert isinstance(payload.get("iteration"), int)


def test_pipeline_logs_allows_brain_log_file(client, monkeypatch):
    class _Tracker:
        def read(self, filename):
            return "brain note\n" if filename == "BRAIN.md" else ""

    class _Exec:
        tracker = _Tracker()

    monkeypatch.setattr(gui_app_module, "_pipeline_executor", _Exec())
    resp = client.get("/api/pipeline/logs/BRAIN.md")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["filename"] == "BRAIN.md"
    assert "brain note" in data["content"]


def test_pipeline_logs_allows_history_log_file(client, monkeypatch):
    class _Tracker:
        def read(self, filename):
            return "history note\n" if filename == "HISTORY.md" else ""

    class _Exec:
        tracker = _Tracker()

    monkeypatch.setattr(gui_app_module, "_pipeline_executor", _Exec())
    resp = client.get("/api/pipeline/logs/HISTORY.md")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["filename"] == "HISTORY.md"
    assert "history note" in data["content"]


def test_pipeline_logs_allows_scientist_report_file(client, monkeypatch):
    class _Tracker:
        def read(self, filename):
            return "science report\n" if filename == "SCIENTIST_REPORT.md" else ""

    class _Exec:
        tracker = _Tracker()

    monkeypatch.setattr(gui_app_module, "_pipeline_executor", _Exec())
    resp = client.get("/api/pipeline/logs/SCIENTIST_REPORT.md")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["filename"] == "SCIENTIST_REPORT.md"
    assert "science report" in data["content"]


def test_pipeline_logs_allows_research_log_file(client, monkeypatch):
    class _Tracker:
        def read(self, filename):
            return "research note\n" if filename == "RESEARCH.md" else ""

    class _Exec:
        tracker = _Tracker()

    monkeypatch.setattr(gui_app_module, "_pipeline_executor", _Exec())
    resp = client.get("/api/pipeline/logs/RESEARCH.md")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["filename"] == "RESEARCH.md"
    assert "research note" in data["content"]


def test_pipeline_logs_reads_from_repo_path_without_active_executor(
    client, monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(gui_app_module, "_pipeline_executor", None)
    repo = _make_repo(tmp_path, git=True)
    logs_dir = repo / ".codex_manager" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "WISHLIST.md").write_text("wishlist entry\n", encoding="utf-8")

    resp = client.get(
        "/api/pipeline/logs/WISHLIST.md",
        query_string={"repo_path": str(repo)},
    )
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["exists"] is True
    assert data["filename"] == "WISHLIST.md"
    assert data["repo_path"] == str(repo.resolve())
    assert "wishlist entry" in data["content"]


def test_pipeline_logs_rejects_invalid_filename_when_executor_missing(client, monkeypatch):
    monkeypatch.setattr(gui_app_module, "_pipeline_executor", None)

    resp = client.get("/api/pipeline/logs/not_allowed.md")
    data = resp.get_json()

    assert resp.status_code == 400
    assert data
    assert "Invalid log file" in data["error"]


def test_pipeline_run_comparison_returns_unavailable_without_repo_hint(client, monkeypatch):
    monkeypatch.setattr(gui_app_module, "_pipeline_executor", None)

    resp = client.get("/api/pipeline/run-comparison")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["available"] is False
    assert data["runs"] == []
    assert "Set Repository Path" in data["message"]


def test_pipeline_run_comparison_aggregates_recent_runs(client, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(gui_app_module, "_pipeline_executor", None)
    repo = _make_repo(tmp_path, git=True)
    logs_dir = repo / ".codex_manager" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    history_path = logs_dir / "HISTORY.jsonl"

    events = [
        {
            "id": "hist_chain_start",
            "timestamp": "2026-02-15T10:00:00+00:00",
            "scope": "chain",
            "event": "run_started",
            "level": "info",
            "summary": "Chain run started.",
            "context": {
                "mode": "apply",
                "max_loops": 3,
                "unlimited": False,
                "steps": ["Implement", "Test"],
            },
        },
        {
            "id": "hist_chain_step_1",
            "timestamp": "2026-02-15T10:00:20+00:00",
            "scope": "chain",
            "event": "step_result",
            "level": "info",
            "summary": "Step result",
            "context": {
                "test_outcome": "passed",
                "input_tokens": 50,
                "output_tokens": 10,
                "commit_sha": "abc111",
            },
        },
        {
            "id": "hist_chain_step_2",
            "timestamp": "2026-02-15T10:00:30+00:00",
            "scope": "chain",
            "event": "step_result",
            "level": "warn",
            "summary": "Step result",
            "context": {
                "test_outcome": "failed",
                "input_tokens": 20,
                "output_tokens": 8,
                "commit_sha": "",
            },
        },
        {
            "id": "hist_chain_finish",
            "timestamp": "2026-02-15T10:00:45+00:00",
            "scope": "chain",
            "event": "run_finished",
            "level": "info",
            "summary": "Chain finished with stop_reason='max_loops_reached'.",
            "context": {
                "stop_reason": "max_loops_reached",
                "total_tokens": 123,
                "elapsed_seconds": 45.5,
            },
        },
        {
            "id": "hist_pipe_start",
            "timestamp": "2026-02-15T11:00:00+00:00",
            "scope": "pipeline",
            "event": "run_started",
            "level": "info",
            "summary": "Pipeline run started.",
            "context": {
                "mode": "dry-run",
                "max_cycles": 2,
                "unlimited": False,
                "phase_order": ["ideation", "implementation"],
                "science_enabled": True,
                "brain_enabled": False,
            },
        },
        {
            "id": "hist_pipe_phase_1",
            "timestamp": "2026-02-15T11:00:15+00:00",
            "scope": "pipeline",
            "event": "phase_result",
            "level": "info",
            "summary": "Phase result",
            "context": {
                "test_outcome": "skipped",
                "total_tokens": 200,
                "commit_sha": "def222",
            },
        },
        {
            "id": "hist_pipe_phase_2",
            "timestamp": "2026-02-15T11:00:40+00:00",
            "scope": "pipeline",
            "event": "phase_result",
            "level": "info",
            "summary": "Phase result",
            "context": {
                "test_outcome": "passed",
                "input_tokens": 25,
                "output_tokens": 5,
                "commit_sha": "def222",
            },
        },
        {
            "id": "hist_pipe_finish",
            "timestamp": "2026-02-15T11:01:30+00:00",
            "scope": "pipeline",
            "event": "run_finished",
            "level": "warn",
            "summary": "Pipeline finished with stop_reason='budget_exhausted'.",
            "context": {
                "stop_reason": "budget_exhausted",
                "total_tokens": 230,
                "elapsed_seconds": 90.0,
            },
        },
    ]
    history_path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    resp = client.get(
        "/api/pipeline/run-comparison",
        query_string={"repo_path": str(repo), "limit": "5", "scope": "all"},
    )
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["available"] is True
    assert data["repo_path"] == str(repo.resolve())
    assert len(data["runs"]) == 2

    newest = data["runs"][0]
    oldest = data["runs"][1]

    assert newest["scope"] == "pipeline"
    assert newest["duration_seconds"] == 90.0
    assert newest["token_usage"] == 230
    assert newest["tests"]["passed"] == 1
    assert newest["tests"]["skipped"] == 1
    assert newest["commit_count"] == 1
    assert "phases=2" in newest["configuration"]

    assert oldest["scope"] == "chain"
    assert oldest["duration_seconds"] == 45.5
    assert oldest["token_usage"] == 123
    assert oldest["tests"]["passed"] == 1
    assert oldest["tests"]["failed"] == 1
    assert oldest["commit_count"] == 1
    assert "steps=2" in oldest["configuration"]

    assert data["best_by"]["fastest_run_id"] == oldest["run_id"]
    assert data["best_by"]["lowest_token_run_id"] == oldest["run_id"]

    pipeline_only = client.get(
        "/api/pipeline/run-comparison",
        query_string={"repo_path": str(repo), "scope": "pipeline"},
    ).get_json()
    assert pipeline_only
    assert pipeline_only["available"] is True
    assert len(pipeline_only["runs"]) == 1
    assert pipeline_only["runs"][0]["scope"] == "pipeline"


def test_pipeline_science_dashboard_returns_structured_payload(
    client, monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(gui_app_module, "_pipeline_executor", None)
    repo = _make_repo(tmp_path, git=True)
    logs_dir = repo / ".codex_manager" / "logs"
    science_dir = logs_dir / "scientist"
    science_dir.mkdir(parents=True, exist_ok=True)

    trials = [
        {
            "timestamp": "2026-02-14T00:00:00+00:00",
            "cycle": 1,
            "phase": "experiment",
            "verdict": "supported",
            "confidence": "high",
            "rollback_action": "kept",
            "hypothesis": {"id": "EXP-001"},
            "baseline": {"test_outcome": "skipped"},
            "post": {
                "test_outcome": "passed",
                "files_changed": 2,
                "net_lines_changed": 12,
            },
            "usage": {"total_tokens": 111},
        },
        {
            "timestamp": "2026-02-14T00:10:00+00:00",
            "cycle": 1,
            "phase": "skeptic",
            "verdict": "inconclusive",
            "confidence": "medium",
            "rollback_action": "reverted",
            "hypothesis": {"id": "EXP-001"},
            "baseline": {"test_outcome": "passed"},
            "post": {
                "test_outcome": "failed",
                "files_changed": 1,
                "net_lines_changed": -4,
            },
            "usage": {"total_tokens": 222},
        },
    ]
    (science_dir / "TRIALS.jsonl").write_text(
        "\n".join(json.dumps(item) for item in trials) + "\n",
        encoding="utf-8",
    )

    (logs_dir / "SCIENTIST_REPORT.md").write_text(
        (
            "# Scientist Mode Report\n\n"
            "## Action Plan (Implementation TODO)\n"
            "- [ ] Implement robust retries for unstable tests\n"
            "- [ ] Add regression guard for skeptic rollback scenario\n\n"
            "## Implementation and Code Changes\n"
            "| Cycle | Phase | Iter | Status | Tests | Files | Net Delta | Commit |\n"
            "|---:|---|---:|---|---|---:|---:|---|\n"
            "| 1 | implementation | 1 | ok | passed | 2 | +12 | abc123 |\n"
            "| 1 | debugging | 1 | failed | failed | 1 | -4 | - |\n\n"
            "### Most-Touched Files\n"
            "| File | Touches |\n"
            "|---|---:|\n"
            "| src/core.py | 2 |\n"
            "| tests/test_core.py | 1 |\n\n"
            "## Latest Analyze Output (Excerpt)\n"
            "```text\n"
            "Prioritize retry stabilization and rollback-safe validation.\n"
            "```\n"
        ),
        encoding="utf-8",
    )

    resp = client.get(
        "/api/pipeline/science-dashboard",
        query_string={"repo_path": str(repo)},
    )
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["available"] is True
    assert data["repo_path"] == str(repo.resolve())
    assert data["summary"]["science_trials"] == 2
    assert data["summary"]["supported"] == 1
    assert data["summary"]["inconclusive"] == 1
    assert data["summary"]["rollbacks"] == 1
    assert data["summary"]["trial_tokens"] == 333
    assert len(data["action_items"]) == 2
    assert len(data["timeline"]) == 2
    assert len(data["implementation"]) == 2
    assert len(data["top_files"]) == 2
    assert "retry stabilization" in data["analysis_excerpt"].lower()


def test_pipeline_science_dashboard_returns_unavailable_without_repo_hint(client, monkeypatch):
    monkeypatch.setattr(gui_app_module, "_pipeline_executor", None)
    resp = client.get("/api/pipeline/science-dashboard")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["available"] is False
    assert "Set Repository Path" in data["message"]


def test_recipes_api_lists_default_and_known_recipes(client):
    resp = client.get("/api/recipes")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["default_recipe_id"] == "autopilot_default"
    recipes = {entry["id"]: entry for entry in data["recipes"]}
    assert "autopilot_default" in recipes
    assert recipes["autopilot_default"]["step_count"] == 7
    assert "New Features" in recipes["autopilot_default"]["sequence"]


def test_recipes_api_detail_exposes_new_features_prompt(client):
    resp = client.get("/api/recipes/autopilot_default")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["id"] == "autopilot_default"
    steps = data["steps"]
    assert isinstance(steps, list)
    new_features = next((step for step in steps if step.get("name") == "02 New Features"), None)
    assert new_features is not None
    assert "Identify the highest-impact features" in new_features["custom_prompt"]


def test_recipes_api_detail_rejects_unknown_id(client):
    resp = client.get("/api/recipes/not-real")
    data = resp.get_json()

    assert resp.status_code == 404
    assert data
    assert data["error"] == "not found"


def test_owner_todo_wishlist_get_and_save_roundtrip(client, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)

    read_resp = client.get(
        "/api/owner/todo-wishlist",
        query_string={"repo_path": str(repo)},
    )
    read_data = read_resp.get_json()
    assert read_resp.status_code == 200
    assert read_data
    assert read_data["exists"] is False
    assert "To-Do and Wishlist" in read_data["content"]

    save_resp = client.post(
        "/api/owner/todo-wishlist/save",
        json={
            "repo_path": str(repo),
            "content": "# To-Do\n\n- [ ] Ship MVP\n- [x] Initialize repo\n",
        },
    )
    save_data = save_resp.get_json()
    assert save_resp.status_code == 200
    assert save_data
    assert save_data["status"] == "saved"
    assert save_data["has_open_items"] is True

    reread_resp = client.get(
        "/api/owner/todo-wishlist",
        query_string={"repo_path": str(repo)},
    )
    reread_data = reread_resp.get_json()
    assert reread_resp.status_code == 200
    assert reread_data
    assert reread_data["exists"] is True
    assert "Ship MVP" in reread_data["content"]


def test_owner_todo_wishlist_suggest_endpoint_uses_helper(client, monkeypatch, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    monkeypatch.setattr(
        gui_app_module,
        "_suggest_todo_wishlist_markdown",
        lambda **_kwargs: (
            "# To-Do\n\n## High Priority\n- [ ] Add feature flags\n",
            "Generated by stub",
        ),
    )

    resp = client.post(
        "/api/owner/todo-wishlist/suggest",
        json={
            "repo_path": str(repo),
            "model": "gpt-5.2",
            "owner_context": "Focus on quality and UX",
            "existing_markdown": "- [ ] Existing item",
        },
    )
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["model"] == "gpt-5.2"
    assert data["has_open_items"] is True
    assert "feature flags" in data["content"]
    assert data["warning"] == "Generated by stub"


def test_owner_feature_dreams_get_and_save_roundtrip(client, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)

    read_resp = client.get(
        "/api/owner/feature-dreams",
        query_string={"repo_path": str(repo)},
    )
    read_data = read_resp.get_json()
    assert read_resp.status_code == 200
    assert read_data
    assert read_data["exists"] is False
    assert "Feature Dreams" in read_data["content"]

    save_resp = client.post(
        "/api/owner/feature-dreams/save",
        json={
            "repo_path": str(repo),
            "content": "# Feature Dreams\n\n- [ ] Add public roadmap view\n- [x] Existing capability\n",
        },
    )
    save_data = save_resp.get_json()
    assert save_resp.status_code == 200
    assert save_data
    assert save_data["status"] == "saved"
    assert save_data["has_open_items"] is True

    reread_resp = client.get(
        "/api/owner/feature-dreams",
        query_string={"repo_path": str(repo)},
    )
    reread_data = reread_resp.get_json()
    assert reread_resp.status_code == 200
    assert reread_data
    assert reread_data["exists"] is True
    assert "public roadmap" in reread_data["content"]


def test_owner_feature_dreams_suggest_endpoint_uses_helper(client, monkeypatch, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    monkeypatch.setattr(
        gui_app_module,
        "_suggest_feature_dreams_markdown",
        lambda **_kwargs: (
            "# Feature Dreams\n\n## P0 - Highest Value / Lowest Effort\n- [ ] [S] Add inline release notes feed\n",
            "Generated by stub",
        ),
    )

    resp = client.post(
        "/api/owner/feature-dreams/suggest",
        json={
            "repo_path": str(repo),
            "model": "gpt-5.2",
            "owner_context": "Focus on discoverability and retention",
            "existing_markdown": "- [ ] Existing feature",
        },
    )
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["model"] == "gpt-5.2"
    assert data["has_open_items"] is True
    assert "release notes feed" in data["content"]
    assert data["warning"] == "Generated by stub"


def test_index_includes_feature_dreams_workspace_controls(client):
    resp = client.get("/")
    html = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert 'showFeatureDreamsModal()' in html
    assert 'id="feature-dreams-overlay"' in html
    assert 'id="feature-dreams-model"' in html
    assert "/api/owner/feature-dreams/suggest" in html
    assert 'onclick="saveAndStartFeatureDreamAutopilot()"' in html
    assert "async function startFeatureDreamAutopilot()" in html
    assert "selectRecipe('feature_dream_autopilot')" in html
    assert 'id="btn-feature-dream-next"' in html
    assert 'onclick="implementNextDreamedFeature()"' in html
    assert "async function implementNextDreamedFeature()" in html
    assert "_implementNextDreamPrompt(" in html
    assert 'id="pipe-resume-card"' in html
    assert 'onclick="resumePipelineFromCheckpoint()"' in html
    assert 'onclick="startFreshPipelineRun()"' in html
    assert "async function resumePipelineFromCheckpoint()" in html
    assert "async function startFreshPipelineRun()" in html
    assert "/api/pipeline/resume-state?repo_path=" in html
    assert "/api/pipeline/resume-state/clear" in html
    assert 'id="pipe-run-compare-body"' in html
    assert "async function refreshPipelineRunComparison" in html
    assert "/api/pipeline/run-comparison?repo_path=" in html
    assert 'id="pipe-smoke-test-cmd"' in html
    assert "default_test_policy" in html
    assert "data-phase-test-policy" in html


def test_index_includes_git_branch_switcher_controls(client):
    resp = client.get("/")
    html = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert 'id="git-sync-remotes-btn"' in html
    assert 'onclick="showGitRemoteModal()"' in html
    assert 'id="git-remote-overlay"' in html
    assert 'id="git-remote-body"' in html
    assert 'id="git-remote-name-input"' in html
    assert 'id="git-remote-url-input"' in html
    assert "async function showGitRemoteModal()" in html
    assert "async function refreshGitRemotesNow()" in html
    assert "async function gitRemoteAdd()" in html
    assert "async function gitRemoteSetDefault(name)" in html
    assert "async function gitRemoteRemove(name)" in html
    assert "async function gitRemoteValidateUrl()" in html
    assert "/api/git/sync/remotes?repo_path=" in html
    assert "/api/git/sync/remotes/add" in html
    assert "/api/git/sync/remotes/remove" in html
    assert "/api/git/sync/remotes/default" in html
    assert "/api/git/sync/remotes/validate" in html
    assert 'id="git-sync-commit-panel-btn"' in html
    assert 'onclick="showGitCommitModal()"' in html
    assert 'id="git-commit-overlay"' in html
    assert 'id="git-commit-files-body"' in html
    assert 'id="git-commit-message"' in html
    assert "async function showGitCommitModal()" in html
    assert "async function refreshGitCommitWorkflowNow()" in html
    assert "async function gitCommitCreate()" in html
    assert 'id="git-sync-open-pr-btn"' in html
    assert 'onclick="gitSyncOpenPullRequest()"' in html
    assert 'id="git-sync-copy-pr-btn"' in html
    assert 'onclick="gitSyncCopyPullRequestUrl()"' in html
    assert "function gitSyncOpenPullRequest()" in html
    assert "async function gitSyncCopyPullRequestUrl()" in html
    assert 'id="git-sync-open-repo-btn"' in html
    assert 'onclick="gitSyncOpenRepoPage()"' in html
    assert "function gitSyncOpenRepoPage()" in html
    assert 'id="git-sync-widget-github-name"' in html
    assert 'id="git-sync-widget-github-visibility"' in html
    assert 'id="git-sync-widget-github-default-branch"' in html
    assert 'id="git-sync-branch-select"' in html
    assert 'id="git-sync-branch-refresh-btn"' in html
    assert 'onclick="refreshGitSyncBranchesNow()"' in html
    assert 'onclick="gitSyncSwitchBranch()"' in html
    assert 'id="git-sync-branch-create-name"' in html
    assert 'onclick="gitSyncCreateBranch()"' in html
    assert "/api/git/sync/branches?repo_path=" in html
    assert "/api/git/sync/checkout" in html
    assert "/api/git/sync/branch/create" in html
    assert "/api/git/sync/commit/workflow?repo_path=" in html
    assert "/api/git/sync/commit/stage" in html
    assert "/api/git/sync/commit/unstage" in html
    assert "/api/git/sync/commit/create" in html
    assert "allow dirty switch" in html


def test_owner_feature_dreams_helpers_default_template_and_open_item_detection(tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    path = gui_app_module._feature_dreams_path(repo)

    content = gui_app_module._read_feature_dreams(repo)
    assert path.is_file() is False
    assert "Feature Dreams" in content
    assert gui_app_module._feature_dreams_has_open_items(content) is True


def test_owner_feature_dreams_helpers_write_roundtrip_and_fallback(tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)

    saved_path = gui_app_module._write_feature_dreams(
        repo,
        "# Feature Dreams\n\n- [x] Shipped a capability\n",
    )
    saved = gui_app_module._read_text_utf8_resilient(saved_path)
    assert saved_path.name == "FEATURE_DREAMS.md"
    assert gui_app_module._feature_dreams_has_open_items(saved) is False

    fallback_path = gui_app_module._write_feature_dreams(repo, "   ")
    fallback_content = gui_app_module._read_text_utf8_resilient(fallback_path)
    assert "Feature Dreams" in fallback_content
    assert gui_app_module._feature_dreams_has_open_items(fallback_content) is True


def test_chain_status_includes_actionable_stop_guidance(client, monkeypatch):
    class _Exec:
        @staticmethod
        def get_state():
            return {
                "running": False,
                "paused": False,
                "stop_reason": "budget_exhausted",
                "total_loops_completed": 2,
            }

    monkeypatch.setattr(gui_app_module, "executor", _Exec())
    resp = client.get("/api/chain/status")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    guidance = data["stop_guidance"]
    assert guidance
    assert guidance["code"] == "budget_exhausted"
    assert guidance["label"] == "Token budget exhausted"
    assert guidance["severity"] == "warn"
    assert any("Increase the token budget" in step for step in guidance["next_steps"])


def test_chain_status_supports_results_delta_polling(client, monkeypatch):
    observed_since: list[int | None] = []

    class _Exec:
        @staticmethod
        def get_state():
            return {"running": True, "paused": False, "stop_reason": None}

        @staticmethod
        def get_state_summary(*, since_results: int | None = None):
            observed_since.append(since_results)
            return {
                "running": True,
                "paused": False,
                "stop_reason": None,
                "total_results": 4,
                "results_delta": [
                    {
                        "loop_number": 2,
                        "step_index": 1,
                        "step_name": "Testing",
                        "test_outcome": "passed",
                    }
                ],
            }

    monkeypatch.setattr(gui_app_module, "executor", _Exec())
    resp = client.get("/api/chain/status", query_string={"since_results": "3"})
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert observed_since == [3]
    assert data["total_results"] == 4
    assert len(data["results_delta"]) == 1
    assert "results" not in data


def test_chain_stop_after_step_api_arms_and_clears_toggle(client, monkeypatch):
    observed: list[bool] = []

    class _State:
        stop_after_current_step = False

    class _Exec:
        is_running = True
        state = _State()

        def set_stop_after_current_step(self, enabled: bool):
            observed.append(bool(enabled))
            self.state.stop_after_current_step = bool(enabled)

    monkeypatch.setattr(gui_app_module, "executor", _Exec())

    arm_resp = client.post("/api/chain/stop-after-step", json={"enabled": True})
    arm_data = arm_resp.get_json()
    assert arm_resp.status_code == 200
    assert arm_data
    assert arm_data["status"] == "armed"
    assert arm_data["stop_after_current_step"] is True

    clear_resp = client.post("/api/chain/stop-after-step", json={"enabled": False})
    clear_data = clear_resp.get_json()
    assert clear_resp.status_code == 200
    assert clear_data
    assert clear_data["status"] == "cleared"
    assert clear_data["stop_after_current_step"] is False
    assert observed == [True, False]


def test_chain_stop_after_step_api_rejects_when_not_running(client, monkeypatch):
    class _Exec:
        is_running = False

    monkeypatch.setattr(gui_app_module, "executor", _Exec())
    resp = client.post("/api/chain/stop-after-step", json={"enabled": True})
    data = resp.get_json()
    assert resp.status_code == 400
    assert data
    assert "No chain running" in data["error"]


def test_pipeline_status_includes_actionable_stop_guidance(client, monkeypatch):
    class _PipelineState:
        @staticmethod
        def to_summary():
            return {
                "running": False,
                "paused": False,
                "stop_reason": "preflight_failed",
                "total_cycles": 0,
                "total_phases": 0,
                "results": [],
            }

    class _Exec:
        state = _PipelineState()

    monkeypatch.setattr(gui_app_module, "_pipeline_executor", _Exec())
    resp = client.get("/api/pipeline/status")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    guidance = data["stop_guidance"]
    assert guidance
    assert guidance["code"] == "preflight_failed"
    assert guidance["label"] == "Preflight checks failed"
    assert guidance["severity"] == "error"
    assert any("Setup Diagnostics" in step for step in guidance["next_steps"])


def test_pipeline_status_supports_results_delta_polling(client, monkeypatch):
    observed_since: list[int | None] = []

    class _PipelineState:
        @staticmethod
        def to_summary(*, since_results: int | None = None):
            observed_since.append(since_results)
            return {
                "running": True,
                "paused": False,
                "stop_reason": None,
                "total_cycles": 2,
                "total_phases": 8,
                "total_results": 8,
                "results_delta": [
                    {
                        "phase": "implementation",
                        "iteration": 2,
                        "success": True,
                        "test_outcome": "passed",
                    }
                ],
            }

    class _Exec:
        state = _PipelineState()

    monkeypatch.setattr(gui_app_module, "_pipeline_executor", _Exec())
    resp = client.get("/api/pipeline/status", query_string={"since_results": "7"})
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert observed_since == [7]
    assert data["total_results"] == 8
    assert len(data["results_delta"]) == 1
    assert "results" not in data


def _check_by_key(payload: dict, category: str, key: str) -> dict:
    checks = payload.get("checks", [])
    for check in checks:
        if check.get("category") == category and check.get("key") == key:
            return check
    raise AssertionError(f"Missing diagnostics check: {category}/{key}")


def test_validate_repo_treats_empty_or_invalid_path_as_missing(client):
    empty_resp = client.post("/api/validate-repo", json={"path": ""})
    empty_data = empty_resp.get_json()
    assert empty_resp.status_code == 200
    assert empty_data
    assert empty_data["exists"] is False
    assert empty_data["is_git"] is False
    assert empty_data["path"] == ""
    assert empty_data["vector_memory_detected"] is False
    assert empty_data["vector_memory_path"] == ""

    invalid_resp = client.post("/api/validate-repo", json={"path": None})
    invalid_data = invalid_resp.get_json()
    assert invalid_resp.status_code == 200
    assert invalid_data
    assert invalid_data["exists"] is False
    assert invalid_data["is_git"] is False
    assert invalid_data["path"] == ""
    assert invalid_data["vector_memory_detected"] is False
    assert invalid_data["vector_memory_path"] == ""


def test_validate_repo_reports_vector_memory_artifacts(client, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    vector_dir = repo / ".codex_manager" / "memory" / "vector_db"
    vector_dir.mkdir(parents=True, exist_ok=True)

    resp = client.post("/api/validate-repo", json={"path": str(repo)})
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["exists"] is True
    assert data["is_git"] is True
    assert data["vector_memory_detected"] is True
    assert Path(data["vector_memory_path"]).resolve() == vector_dir.resolve()


def test_browse_dirs_accepts_non_string_path_payload(client, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(gui_app_module.Path, "home", staticmethod(lambda: tmp_path))

    resp = client.post("/api/browse-dirs", json={"path": None})
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["current"] == str(tmp_path.resolve())
    assert isinstance(data["dirs"], list)


def test_project_clone_branches_returns_default_and_heads(client, tmp_path: Path):
    remote = _make_remote_repo(tmp_path)

    resp = client.post(
        "/api/project/clone/branches",
        json={"remote_url": str(remote)},
    )
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["remote_url"] == str(remote)
    assert data["default_branch"] == "main"
    assert "main" in data["branches"]
    assert "dev" in data["branches"]


def test_project_clone_endpoint_clones_and_initializes_codex_manager(client, tmp_path: Path):
    remote = _make_remote_repo(tmp_path)
    destination = tmp_path / "workspace"
    destination.mkdir(parents=True, exist_ok=True)

    resp = client.post(
        "/api/project/clone",
        json={
            "remote_url": str(remote),
            "destination_dir": str(destination),
            "project_name": "cloned-repo",
            "default_branch": "dev",
        },
    )
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["status"] == "cloned"
    assert data["project_name"] == "cloned-repo"
    assert data["requested_branch"] == "dev"
    assert data["checked_out_branch"] == "dev"
    assert data["codex_manager_initialized"] is True

    cloned = Path(data["path"])
    assert cloned.is_dir()
    assert (cloned / ".git").is_dir()
    assert (cloned / ".codex_manager" / "owner" / "TODO_WISHLIST.md").is_file()
    assert (cloned / ".codex_manager" / "owner" / "FEATURE_DREAMS.md").is_file()
    assert (cloned / ".codex_manager" / "owner" / "decision_board.json").is_file()
    assert (cloned / ".codex_manager" / "logs").is_dir()
    assert (cloned / ".codex_manager" / "outputs").is_dir()
    assert (cloned / ".codex_manager" / "state").is_dir()


def test_project_clone_endpoint_rejects_existing_target_path(client, tmp_path: Path):
    remote = _make_remote_repo(tmp_path)
    destination = tmp_path / "workspace"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "existing-repo").mkdir(parents=True, exist_ok=True)

    resp = client.post(
        "/api/project/clone",
        json={
            "remote_url": str(remote),
            "destination_dir": str(destination),
            "project_name": "existing-repo",
        },
    )
    data = resp.get_json()

    assert resp.status_code == 409
    assert data
    assert "Path already exists" in data["error"]


def test_project_clone_endpoint_rejects_dash_prefixed_remote_url(client, tmp_path: Path):
    destination = tmp_path / "workspace"
    destination.mkdir(parents=True, exist_ok=True)

    resp = client.post(
        "/api/project/clone",
        json={
            "remote_url": "--upload-pack=malicious",
            "destination_dir": str(destination),
            "project_name": "safe-name",
        },
    )
    data = resp.get_json()

    assert resp.status_code == 400
    assert data
    assert "Remote URL is invalid" in data["error"]


def test_git_sync_status_reports_tracking_and_dirty_state(client, tmp_path: Path):
    remote = _make_remote_repo(tmp_path)
    local = _clone_tracking_repo(tmp_path, remote, clone_name="local-sync-status")

    clean_resp = client.get("/api/git/sync/status", query_string={"repo_path": str(local)})
    clean_data = clean_resp.get_json()

    assert clean_resp.status_code == 200
    assert clean_data
    assert clean_data["branch"] == "main"
    assert clean_data["has_tracking_branch"] is True
    assert clean_data["tracking_remote"] == "origin"
    assert clean_data["ahead"] == 0
    assert clean_data["behind"] == 0
    assert clean_data["dirty"] is False
    assert clean_data["clean"] is True
    assert "last_fetch_epoch_ms" in clean_data
    assert "last_fetch_at" in clean_data
    assert isinstance(clean_data["github_repo"], dict)
    assert clean_data["github_repo"]["provider"] == "github"
    assert clean_data["github_repo"]["available"] is False
    if clean_data["last_fetch_epoch_ms"] is not None:
        assert isinstance(clean_data["last_fetch_epoch_ms"], int)
        assert clean_data["last_fetch_epoch_ms"] > 0
        assert str(clean_data["last_fetch_at"] or "").endswith("Z")

    (local / "UNTRACKED.txt").write_text("dirty file\n", encoding="utf-8")
    dirty_resp = client.get("/api/git/sync/status", query_string={"repo_path": str(local)})
    dirty_data = dirty_resp.get_json()

    assert dirty_resp.status_code == 200
    assert dirty_data
    assert dirty_data["dirty"] is True
    assert dirty_data["untracked_changes"] >= 1


def test_git_sync_status_includes_github_repo_metadata(client, monkeypatch, tmp_path: Path):
    remote = _make_remote_repo(tmp_path)
    local = _clone_tracking_repo(tmp_path, remote, clone_name="local-sync-status-github-metadata")
    _run_git(
        "remote",
        "set-url",
        "origin",
        "https://github.com/example/demo-repo.git",
        cwd=local,
    )

    def _fake_github_repo_metadata(owner: str, repo_name: str, *, token: str = ""):
        assert owner == "example"
        assert repo_name == "demo-repo"
        assert token == ""
        return (
            {
                "name": "demo-repo",
                "full_name": "example/demo-repo",
                "owner": "example",
                "repo": "demo-repo",
                "url": "https://github.com/example/demo-repo",
                "default_branch": "main",
                "visibility": "private",
                "private": True,
                "api_ok": True,
            },
            "",
        )

    monkeypatch.setattr(gui_app_module, "_github_repo_metadata_from_api", _fake_github_repo_metadata)

    resp = client.get("/api/git/sync/status", query_string={"repo_path": str(local)})
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    github_repo = data["github_repo"]
    assert github_repo["provider"] == "github"
    assert github_repo["remote"] == "origin"
    assert github_repo["detected"] is True
    assert github_repo["available"] is True
    assert github_repo["name"] == "demo-repo"
    assert github_repo["full_name"] == "example/demo-repo"
    assert github_repo["visibility"] == "private"
    assert github_repo["default_branch"] == "main"
    assert github_repo["url"] == "https://github.com/example/demo-repo"
    assert github_repo["source"] == "api"


def test_git_sync_branches_lists_local_and_remote_choices(client, tmp_path: Path):
    remote = _make_remote_repo(tmp_path)
    local = _clone_tracking_repo(tmp_path, remote, clone_name="local-sync-branches")
    _run_git("checkout", "-b", "feature/local-only", cwd=local)
    _run_git("checkout", "main", cwd=local)

    resp = client.get("/api/git/sync/branches", query_string={"repo_path": str(local)})
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["current_branch"] == "main"
    assert "main" in data["local_branches"]
    assert "feature/local-only" in data["local_branches"]
    assert "origin/main" in data["remote_branches"]
    assert "origin/dev" in data["remote_branches"]
    assert not any(branch.endswith("/HEAD") for branch in data["remote_branches"])


def test_git_sync_checkout_remote_branch_creates_tracking_branch(client, tmp_path: Path):
    remote = _make_remote_repo(tmp_path)
    local = _clone_tracking_repo(tmp_path, remote, clone_name="local-sync-checkout-remote")

    resp = client.post(
        "/api/git/sync/checkout",
        json={
            "repo_path": str(local),
            "branch": "origin/dev",
            "branch_type": "remote",
        },
    )
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["status"] == "checked_out"
    assert data["branch"] == "dev"
    assert data["created_tracking_branch"] is True
    assert data["sync"]["branch"] == "dev"
    assert data["sync"]["tracking_branch"] == "origin/dev"

    upstream = _run_git(
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
        cwd=local,
    ).stdout
    assert upstream.strip() == "origin/dev"


def test_git_sync_checkout_blocks_dirty_worktree_without_allow_dirty(client, tmp_path: Path):
    remote = _make_remote_repo(tmp_path)
    local = _clone_tracking_repo(tmp_path, remote, clone_name="local-sync-checkout-dirty")
    (local / "DIRTY_BRANCH_SWITCH.txt").write_text("dirty branch switch\n", encoding="utf-8")

    resp = client.post(
        "/api/git/sync/checkout",
        json={
            "repo_path": str(local),
            "branch": "origin/dev",
            "branch_type": "remote",
            "allow_dirty": False,
        },
    )
    data = resp.get_json()

    assert resp.status_code == 409
    assert data
    assert data["error_type"] == "dirty_worktree"
    assert isinstance(data["recovery_steps"], list)
    assert data["sync"]["dirty"] is True


def test_git_sync_branch_create_creates_and_switches_branch(client, tmp_path: Path):
    remote = _make_remote_repo(tmp_path)
    local = _clone_tracking_repo(tmp_path, remote, clone_name="local-sync-branch-create")

    resp = client.post(
        "/api/git/sync/branch/create",
        json={
            "repo_path": str(local),
            "branch_name": "feature/sync-branch-create",
        },
    )
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["status"] == "branch_created"
    assert data["branch"] == "feature/sync-branch-create"
    assert data["sync"]["branch"] == "feature/sync-branch-create"

    ref = _run_git(
        "rev-parse",
        "--verify",
        "refs/heads/feature/sync-branch-create",
        cwd=local,
    ).stdout
    assert ref.strip()


def test_git_sync_branch_create_blocks_dirty_worktree_without_allow_dirty(client, tmp_path: Path):
    remote = _make_remote_repo(tmp_path)
    local = _clone_tracking_repo(tmp_path, remote, clone_name="local-sync-branch-create-dirty")
    (local / "DIRTY_BRANCH_CREATE.txt").write_text("dirty branch create\n", encoding="utf-8")

    resp = client.post(
        "/api/git/sync/branch/create",
        json={
            "repo_path": str(local),
            "branch_name": "feature/dirty-create",
            "allow_dirty": False,
        },
    )
    data = resp.get_json()

    assert resp.status_code == 409
    assert data
    assert data["error_type"] == "dirty_worktree"
    assert isinstance(data["recovery_steps"], list)
    assert data["sync"]["dirty"] is True


def test_git_sync_remotes_list_includes_default_and_tracking_remote(client, tmp_path: Path):
    remote = _make_remote_repo(tmp_path)
    local = _clone_tracking_repo(tmp_path, remote, clone_name="local-sync-remotes-list")
    _run_git("remote", "add", "backup", "git@github.com:example/demo.git", cwd=local)

    resp = client.get("/api/git/sync/remotes", query_string={"repo_path": str(local)})
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["default_remote"] == "origin"
    assert data["default_remote_source"] in {"tracking", "origin"}
    assert data["tracking_remote"] == "origin"

    remotes = {str(item["name"]): item for item in data["remotes"]}
    assert "origin" in remotes
    assert "backup" in remotes
    assert remotes["origin"]["is_default"] is True
    assert remotes["origin"]["is_tracking_remote"] is True
    assert remotes["backup"]["fetch_url"] == "git@github.com:example/demo.git"


def test_git_sync_remote_validate_endpoint_accepts_https_and_ssh(client):
    https_resp = client.post(
        "/api/git/sync/remotes/validate",
        json={"remote_url": "https://github.com/example/demo.git"},
    )
    https_data = https_resp.get_json()
    assert https_resp.status_code == 200
    assert https_data
    assert https_data["valid"] is True
    assert https_data["transport"] == "https"

    ssh_resp = client.post(
        "/api/git/sync/remotes/validate",
        json={"remote_url": "git@github.com:example/demo.git"},
    )
    ssh_data = ssh_resp.get_json()
    assert ssh_resp.status_code == 200
    assert ssh_data
    assert ssh_data["valid"] is True
    assert ssh_data["transport"] == "ssh"

    invalid_resp = client.post(
        "/api/git/sync/remotes/validate",
        json={"remote_url": "file:///tmp/demo.git"},
    )
    invalid_data = invalid_resp.get_json()
    assert invalid_resp.status_code == 400
    assert invalid_data
    assert invalid_data["valid"] is False
    assert "HTTPS or SSH" in invalid_data["error"]


def test_git_sync_remote_add_set_default_clear_and_remove(client, tmp_path: Path):
    remote = _make_remote_repo(tmp_path)
    local = _clone_tracking_repo(tmp_path, remote, clone_name="local-sync-remotes-manage")

    add_resp = client.post(
        "/api/git/sync/remotes/add",
        json={
            "repo_path": str(local),
            "name": "backup",
            "remote_url": "git@github.com:example/demo.git",
            "set_default": True,
        },
    )
    add_data = add_resp.get_json()
    assert add_resp.status_code == 200
    assert add_data
    assert add_data["status"] == "remote_added"
    assert add_data["remote"] == "backup"
    assert add_data["set_default"] is True
    assert add_data["remotes"]["default_remote"] == "backup"

    configured_default = _run_git("config", "--get", "remote.pushDefault", cwd=local).stdout.strip()
    assert configured_default == "backup"

    set_default_resp = client.post(
        "/api/git/sync/remotes/default",
        json={"repo_path": str(local), "name": "origin"},
    )
    set_default_data = set_default_resp.get_json()
    assert set_default_resp.status_code == 200
    assert set_default_data
    assert set_default_data["status"] == "remote_default_set"
    assert set_default_data["default_remote"] == "origin"

    configured_default = _run_git("config", "--get", "remote.pushDefault", cwd=local).stdout.strip()
    assert configured_default == "origin"

    clear_default_resp = client.post(
        "/api/git/sync/remotes/default",
        json={"repo_path": str(local), "clear": True},
    )
    clear_default_data = clear_default_resp.get_json()
    assert clear_default_resp.status_code == 200
    assert clear_default_data
    assert clear_default_data["status"] == "remote_default_cleared"
    assert clear_default_data["default_remote"] == ""

    cleared_probe = subprocess.run(
        ["git", "config", "--get", "remote.pushDefault"],
        cwd=str(local),
        capture_output=True,
        text=True,
        check=False,
    )
    assert cleared_probe.returncode != 0

    remove_resp = client.post(
        "/api/git/sync/remotes/remove",
        json={"repo_path": str(local), "name": "backup"},
    )
    remove_data = remove_resp.get_json()
    assert remove_resp.status_code == 200
    assert remove_data
    assert remove_data["status"] == "remote_removed"
    assert remove_data["remote"] == "backup"

    remote_probe = subprocess.run(
        ["git", "remote", "get-url", "backup"],
        cwd=str(local),
        capture_output=True,
        text=True,
        check=False,
    )
    assert remote_probe.returncode != 0


def test_git_sync_remote_add_rejects_invalid_url_scheme(client, tmp_path: Path):
    remote = _make_remote_repo(tmp_path)
    local = _clone_tracking_repo(tmp_path, remote, clone_name="local-sync-remotes-invalid-url")

    resp = client.post(
        "/api/git/sync/remotes/add",
        json={
            "repo_path": str(local),
            "name": "badremote",
            "remote_url": "http://github.com/example/demo.git",
        },
    )
    data = resp.get_json()

    assert resp.status_code == 400
    assert data
    assert "HTTPS or SSH" in data["error"]


def test_git_commit_workflow_lists_changes_and_last_commit(client, tmp_path: Path):
    remote = _make_remote_repo(tmp_path)
    local = _clone_tracking_repo(tmp_path, remote, clone_name="local-commit-workflow")

    (local / "STAGED_FILE.txt").write_text("staged change\n", encoding="utf-8")
    _run_git("add", "STAGED_FILE.txt", cwd=local)
    (local / "README.md").write_text("# Updated README\n", encoding="utf-8")
    (local / "UNTRACKED_FILE.txt").write_text("untracked change\n", encoding="utf-8")

    resp = client.get("/api/git/sync/commit/workflow", query_string={"repo_path": str(local)})
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    files = {str(item["path"]): item for item in data["files"]}
    assert files["STAGED_FILE.txt"]["staged"] is True
    assert files["STAGED_FILE.txt"]["can_unstage"] is True
    assert files["README.md"]["unstaged"] is True
    assert files["README.md"]["can_stage"] is True
    assert files["UNTRACKED_FILE.txt"]["untracked"] is True
    assert data["counts"]["staged"] >= 1
    assert data["counts"]["unstaged"] >= 1
    assert data["counts"]["untracked"] >= 1

    last_commit = data["last_commit"]
    assert last_commit["available"] is True
    assert last_commit["hash"]
    assert last_commit["author_name"] == "GUI API Tests"
    assert last_commit["authored_at"]


def test_git_commit_stage_then_unstage_file(client, tmp_path: Path):
    remote = _make_remote_repo(tmp_path)
    local = _clone_tracking_repo(tmp_path, remote, clone_name="local-commit-stage-unstage")
    (local / "STAGE_ROUNDTRIP.txt").write_text("roundtrip\n", encoding="utf-8")

    stage_resp = client.post(
        "/api/git/sync/commit/stage",
        json={"repo_path": str(local), "paths": ["STAGE_ROUNDTRIP.txt"]},
    )
    stage_data = stage_resp.get_json()

    assert stage_resp.status_code == 200
    assert stage_data
    assert stage_data["status"] == "staged"
    assert "STAGE_ROUNDTRIP.txt" in stage_data["paths"]
    staged_entry = {
        str(item["path"]): item for item in stage_data["workflow"]["files"]
    }["STAGE_ROUNDTRIP.txt"]
    assert staged_entry["staged"] is True

    unstage_resp = client.post(
        "/api/git/sync/commit/unstage",
        json={"repo_path": str(local), "paths": ["STAGE_ROUNDTRIP.txt"]},
    )
    unstage_data = unstage_resp.get_json()

    assert unstage_resp.status_code == 200
    assert unstage_data
    assert unstage_data["status"] == "unstaged"
    unstage_entry = {
        str(item["path"]): item for item in unstage_data["workflow"]["files"]
    }["STAGE_ROUNDTRIP.txt"]
    assert unstage_entry["staged"] is False
    assert unstage_entry["untracked"] is True


def test_git_commit_create_commits_staged_changes(client, tmp_path: Path):
    remote = _make_remote_repo(tmp_path)
    local = _clone_tracking_repo(tmp_path, remote, clone_name="local-commit-create")
    (local / "COMMIT_CREATE.txt").write_text("commit create\n", encoding="utf-8")
    _run_git("add", "COMMIT_CREATE.txt", cwd=local)

    resp = client.post(
        "/api/git/sync/commit/create",
        json={"repo_path": str(local), "message": "add commit workflow coverage"},
    )
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["status"] == "committed"
    assert data["commit"]["available"] is True
    assert data["commit"]["subject"] == "add commit workflow coverage"
    assert data["workflow"]["counts"]["staged"] == 0

    subject = _run_git("log", "-1", "--pretty=%s", cwd=local).stdout.strip()
    assert subject == "add commit workflow coverage"


def test_git_commit_create_requires_staged_changes(client, tmp_path: Path):
    remote = _make_remote_repo(tmp_path)
    local = _clone_tracking_repo(tmp_path, remote, clone_name="local-commit-create-empty")

    resp = client.post(
        "/api/git/sync/commit/create",
        json={"repo_path": str(local), "message": "should fail"},
    )
    data = resp.get_json()

    assert resp.status_code == 409
    assert data
    assert "No staged changes to commit" in data["error"]


def test_git_sync_fetch_then_pull_updates_behind_count(client, tmp_path: Path):
    remote = _make_remote_repo(tmp_path)
    local = _clone_tracking_repo(tmp_path, remote, clone_name="local-sync-pull")
    _push_remote_update(
        tmp_path,
        remote,
        clone_name="updater-sync-pull",
        filename="REMOTE.md",
        content="upstream update\n",
        message="remote update",
    )

    fetch_resp = client.post("/api/git/sync/fetch", json={"repo_path": str(local)})
    fetch_data = fetch_resp.get_json()

    assert fetch_resp.status_code == 200
    assert fetch_data
    assert fetch_data["status"] == "fetched"
    assert fetch_data["sync"]["behind"] == 1
    assert isinstance(fetch_data["sync"]["last_fetch_epoch_ms"], int)
    assert fetch_data["sync"]["last_fetch_epoch_ms"] > 0
    assert str(fetch_data["sync"]["last_fetch_at"] or "").endswith("Z")

    pull_resp = client.post("/api/git/sync/pull", json={"repo_path": str(local)})
    pull_data = pull_resp.get_json()

    assert pull_resp.status_code == 200
    assert pull_data
    assert pull_data["status"] == "pulled"
    assert pull_data["sync"]["behind"] == 0
    assert pull_data["sync"]["dirty"] is False
    assert (local / "REMOTE.md").is_file()


def test_git_sync_stash_pull_stashes_local_changes_then_pulls(client, tmp_path: Path):
    remote = _make_remote_repo(tmp_path)
    local = _clone_tracking_repo(tmp_path, remote, clone_name="local-sync-stash-pull")
    _push_remote_update(
        tmp_path,
        remote,
        clone_name="updater-sync-stash",
        filename="REMOTE_STASH.md",
        content="remote stash pull update\n",
        message="remote update for stash pull",
    )

    (local / "README.md").write_text("# Local Dirty Change\n", encoding="utf-8")
    (local / "LOCAL_ONLY.txt").write_text("local worktree edits\n", encoding="utf-8")

    resp = client.post("/api/git/sync/stash-pull", json={"repo_path": str(local)})
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["status"] == "stashed_and_pulled"
    assert data["stash_created"] is True
    assert data["stash_ref"]
    assert data["sync"]["behind"] == 0
    assert data["sync"]["dirty"] is False
    assert (local / "REMOTE_STASH.md").is_file()

    stash_list = _run_git("stash", "list", cwd=local).stdout
    assert "codex-manager:auto-stash-before-pull" in stash_list


def test_git_sync_push_pushes_local_commit(client, tmp_path: Path):
    remote = _make_remote_repo(tmp_path)
    local = _clone_tracking_repo(tmp_path, remote, clone_name="local-sync-push")

    (local / "LOCAL_PUSH.md").write_text("local push update\n", encoding="utf-8")
    _run_git("add", "LOCAL_PUSH.md", cwd=local)
    _run_git("commit", "-m", "local push update", cwd=local)

    resp = client.post("/api/git/sync/push", json={"repo_path": str(local)})
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["status"] == "pushed"
    assert data["set_upstream"] is False
    assert data["pull_request"]["available"] is False
    assert data["pull_request_url"] == ""
    assert data["sync"]["ahead"] == 0
    assert data["sync"]["behind"] == 0
    assert data["sync"]["dirty"] is False

    verifier = _clone_tracking_repo(tmp_path, remote, clone_name="verify-sync-push")
    assert (verifier / "LOCAL_PUSH.md").is_file()


def test_git_sync_push_with_set_upstream_for_new_branch(client, tmp_path: Path):
    remote = _make_remote_repo(tmp_path)
    local = _clone_tracking_repo(tmp_path, remote, clone_name="local-sync-push-upstream")
    _run_git("checkout", "-b", "feature/sync-push-upstream", cwd=local)

    (local / "FEATURE_PUSH.md").write_text("feature push update\n", encoding="utf-8")
    _run_git("add", "FEATURE_PUSH.md", cwd=local)
    _run_git("commit", "-m", "feature push update", cwd=local)

    resp = client.post(
        "/api/git/sync/push",
        json={"repo_path": str(local), "set_upstream": True},
    )
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["status"] == "pushed"
    assert data["set_upstream"] is True
    assert data["remote"] == "origin"
    assert data["branch"] == "feature/sync-push-upstream"
    assert data["sync"]["tracking_branch"] == "origin/feature/sync-push-upstream"
    assert data["sync"]["has_tracking_branch"] is True

    upstream = _run_git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}", cwd=local).stdout
    assert upstream.strip() == "origin/feature/sync-push-upstream"


def test_git_sync_push_returns_pull_request_url_for_github_remote(client, monkeypatch, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    sync_payload = {
        "branch": "feature/pr-helper",
        "tracking_branch": "origin/feature/pr-helper",
        "has_tracking_branch": True,
        "ahead": 0,
        "behind": 0,
        "dirty": False,
        "staged_changes": 0,
        "unstaged_changes": 0,
        "untracked_changes": 0,
    }

    monkeypatch.setattr(gui_app_module, "_resolve_git_sync_repo", lambda _raw: (repo, "", 200))
    monkeypatch.setattr(gui_app_module, "_git_sync_status_payload", lambda _repo: dict(sync_payload))

    def _fake_run_git_sync(repo_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
        assert repo_path == repo
        if args[:1] == ("push",):
            return subprocess.CompletedProcess(["git", *args], 0, stdout="ok", stderr="")
        if args == ("remote", "get-url", "origin"):
            return subprocess.CompletedProcess(
                ["git", *args],
                0,
                stdout="https://github.com/example/demo-repo.git\n",
                stderr="",
            )
        if args == ("symbolic-ref", "--short", "refs/remotes/origin/HEAD"):
            return subprocess.CompletedProcess(["git", *args], 0, stdout="origin/main\n", stderr="")
        return subprocess.CompletedProcess(["git", *args], 1, stdout="", stderr="unexpected")

    monkeypatch.setattr(gui_app_module, "_run_git_sync_command", _fake_run_git_sync)

    resp = client.post("/api/git/sync/push", json={"repo_path": str(repo)})
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    expected_url = "https://github.com/example/demo-repo/compare/main...feature%2Fpr-helper?expand=1"
    assert data["pull_request_url"] == expected_url
    assert data["pull_request"]["available"] is True
    assert data["pull_request"]["url"] == expected_url
    assert data["pull_request"]["base_branch"] == "main"
    assert data["pull_request"]["head_branch"] == "feature/pr-helper"


def test_git_sync_push_reports_auth_failures_with_recovery_steps(client, monkeypatch, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    monkeypatch.setattr(gui_app_module, "_resolve_git_sync_repo", lambda _raw: (repo, "", 200))
    monkeypatch.setattr(
        gui_app_module,
        "_git_sync_status_payload",
        lambda _repo: {"branch": "main", "tracking_branch": "origin/main"},
    )

    def _fake_run_git_sync(repo_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
        assert repo_path == repo
        assert args
        return subprocess.CompletedProcess(
            ["git", *args],
            1,
            stdout="",
            stderr=(
                "remote: Invalid username or token.\n"
                "fatal: Authentication failed for 'https://github.com/example/repo.git/'"
            ),
        )

    monkeypatch.setattr(gui_app_module, "_run_git_sync_command", _fake_run_git_sync)

    resp = client.post("/api/git/sync/push", json={"repo_path": str(repo)})
    data = resp.get_json()

    assert resp.status_code == 401
    assert data
    assert data["error_type"] == "auth"
    assert "authentication/authorization" in data["error"]
    assert isinstance(data["recovery_steps"], list)
    assert any("GitHub Auth" in step for step in data["recovery_steps"])
    assert any("scopes" in step.lower() for step in data["recovery_steps"])
    assert any("known_hosts" in step.lower() for step in data["recovery_steps"])
    assistant = data["auth_troubleshooting"]
    assert assistant["title"] == "Credential Troubleshooting Assistant"
    assert assistant["auth_method"] == "https"
    checks = assistant["checks"]
    assert isinstance(checks, list)
    assert any(check.get("key") == "pat_scopes" for check in checks)


def test_git_sync_push_reports_non_fast_forward_with_recovery_steps(client, monkeypatch, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    monkeypatch.setattr(gui_app_module, "_resolve_git_sync_repo", lambda _raw: (repo, "", 200))
    monkeypatch.setattr(
        gui_app_module,
        "_git_sync_status_payload",
        lambda _repo: {"branch": "main", "tracking_branch": "origin/main"},
    )

    def _fake_run_git_sync(repo_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
        assert repo_path == repo
        assert args
        return subprocess.CompletedProcess(
            ["git", *args],
            1,
            stdout="",
            stderr=(
                "! [rejected]        main -> main (non-fast-forward)\n"
                "error: failed to push some refs to 'origin'"
            ),
        )

    monkeypatch.setattr(gui_app_module, "_run_git_sync_command", _fake_run_git_sync)

    resp = client.post("/api/git/sync/push", json={"repo_path": str(repo)})
    data = resp.get_json()

    assert resp.status_code == 409
    assert data
    assert data["error_type"] == "non_fast_forward"
    assert "non-fast-forward" in data["error"]
    assert isinstance(data["recovery_steps"], list)
    assert any("Fetch" in step for step in data["recovery_steps"])


def test_git_sync_status_requires_repo_path(client):
    resp = client.get("/api/git/sync/status")
    data = resp.get_json()

    assert resp.status_code == 400
    assert data
    assert "repo_path is required" in data["error"]


def test_diagnostics_reports_actionable_failures(client, monkeypatch):
    monkeypatch.setattr(preflight_module, "binary_exists", lambda _binary: False)
    monkeypatch.setattr(preflight_module, "has_codex_auth", lambda: False)
    monkeypatch.setattr(preflight_module, "has_claude_auth", lambda: False)

    resp = client.post(
        "/api/diagnostics",
        json={
            "repo_path": "",
            "codex_binary": "codex-missing",
            "claude_binary": "claude-missing",
            "agents": ["codex", "claude_code"],
        },
    )
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["summary"]["fail"] >= 4
    assert data["ready"] is False
    assert _check_by_key(data, "repository", "path")["status"] == "warn"
    assert _check_by_key(data, "codex", "binary")["status"] == "fail"
    assert _check_by_key(data, "codex", "auth")["status"] == "fail"
    assert _check_by_key(data, "claude_code", "binary")["status"] == "fail"
    assert _check_by_key(data, "claude_code", "auth")["status"] == "fail"
    actions = data.get("next_actions", [])
    assert any(a.get("key") == "install_codex_cli" for a in actions)
    assert any(a.get("key") == "codex_login" for a in actions)
    assert any(a.get("key") == "rerun_doctor" for a in actions)
    install_codex = next(a for a in actions if a.get("key") == "install_codex_cli")
    assert install_codex.get("can_run") is True
    codex_login = next(a for a in actions if a.get("key") == "codex_login")
    assert codex_login.get("can_run") is False
    rerun = next(a for a in actions if a.get("key") == "rerun_doctor")
    assert rerun.get("can_run") is True
    assert '--codex-bin "codex-missing"' in rerun.get("command", "")
    assert '--claude-bin "claude-missing"' in rerun.get("command", "")


def test_diagnostics_reports_ready_state(client, monkeypatch, tmp_path: Path):
    repo = _make_repo(tmp_path, git=True)
    monkeypatch.setattr(preflight_module, "binary_exists", lambda _binary: True)
    monkeypatch.setattr(preflight_module, "has_codex_auth", lambda: True)
    monkeypatch.setattr(preflight_module, "has_claude_auth", lambda: True)
    monkeypatch.setattr(preflight_module, "repo_write_error", lambda _repo: None)

    resp = client.post(
        "/api/diagnostics",
        json={
            "repo_path": str(repo),
            "codex_binary": "codex",
            "claude_binary": "claude",
            "agents": ["codex", "claude_code"],
        },
    )
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["summary"]["fail"] == 0
    assert data["ready"] is True
    assert _check_by_key(data, "repository", "path")["status"] == "pass"
    assert _check_by_key(data, "repository", "git_repo")["status"] == "pass"
    assert _check_by_key(data, "repository", "writable")["status"] == "pass"
    assert _check_by_key(data, "codex", "binary")["status"] == "pass"
    assert _check_by_key(data, "codex", "auth")["status"] == "pass"
    assert _check_by_key(data, "claude_code", "binary")["status"] == "pass"
    assert _check_by_key(data, "claude_code", "auth")["status"] == "pass"
    actions = data.get("next_actions", [])
    assert any(a.get("key") == "first_dry_run" for a in actions)


def test_diagnostics_run_action_executes_runnable_command(client, monkeypatch):
    monkeypatch.setattr(preflight_module, "binary_exists", lambda _binary: False)
    monkeypatch.setattr(preflight_module, "has_codex_auth", lambda: True)
    monkeypatch.setattr(preflight_module, "has_claude_auth", lambda: True)

    observed = {}

    class _Completed:
        returncode = 0
        stdout = "codex 1.2.3\n"
        stderr = ""

    def _fake_run(args, **kwargs):
        observed["args"] = args
        observed["cwd"] = kwargs.get("cwd")
        return _Completed()

    monkeypatch.setattr(gui_app_module.subprocess, "run", _fake_run)

    resp = client.post(
        "/api/diagnostics/actions/run",
        json={
            "repo_path": "",
            "codex_binary": "codex-missing",
            "claude_binary": "claude",
            "agents": ["codex"],
            "action_key": "install_codex_cli",
        },
    )
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["ok"] is True
    assert data["exit_code"] == 0
    assert observed["args"] == ["codex-missing", "--version"]
    assert observed["cwd"] is None
    assert data["command"] == gui_app_module.subprocess.list2cmdline(observed["args"])


def test_diagnostics_run_action_rejects_non_runnable_action(client, monkeypatch):
    monkeypatch.setattr(preflight_module, "binary_exists", lambda _binary: True)
    monkeypatch.setattr(preflight_module, "has_codex_auth", lambda: False)
    monkeypatch.setattr(preflight_module, "has_claude_auth", lambda: True)

    resp = client.post(
        "/api/diagnostics/actions/run",
        json={
            "repo_path": "",
            "codex_binary": "codex",
            "claude_binary": "claude",
            "agents": ["codex"],
            "action_key": "codex_login",
        },
    )
    data = resp.get_json()

    assert resp.status_code == 400
    assert data
    assert "cannot be auto-run" in data["error"]


def test_diagnostics_run_action_rejects_missing_or_unavailable_action(client, monkeypatch):
    monkeypatch.setattr(preflight_module, "binary_exists", lambda _binary: True)
    monkeypatch.setattr(preflight_module, "has_codex_auth", lambda: True)
    monkeypatch.setattr(preflight_module, "has_claude_auth", lambda: True)

    missing_resp = client.post("/api/diagnostics/actions/run", json={})
    missing_data = missing_resp.get_json()
    assert missing_resp.status_code == 400
    assert missing_data
    assert "Missing diagnostics action key" in missing_data["error"]

    unavailable_resp = client.post(
        "/api/diagnostics/actions/run",
        json={
            "repo_path": "",
            "codex_binary": "codex",
            "claude_binary": "claude",
            "agents": ["codex"],
            "action_key": "install_codex_cli",
        },
    )
    unavailable_data = unavailable_resp.get_json()
    assert unavailable_resp.status_code == 404
    assert unavailable_data
    assert "unavailable" in unavailable_data["error"]


def test_diagnostics_warns_on_unknown_agents(client):
    resp = client.post(
        "/api/diagnostics",
        json={"repo_path": "", "agents": ["mystery_agent"]},
    )
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    unknown = _check_by_key(data, "agents", "mystery_agent_supported")
    assert unknown["status"] == "warn"
    assert "Unknown agent" in unknown["detail"]


def test_diagnostics_accepts_string_agents_with_aliases(client, monkeypatch):
    monkeypatch.setattr(preflight_module, "binary_exists", lambda _binary: True)
    monkeypatch.setattr(preflight_module, "has_codex_auth", lambda: True)
    monkeypatch.setattr(preflight_module, "has_claude_auth", lambda: True)

    resp = client.post(
        "/api/diagnostics",
        json={"repo_path": "", "agents": "codex,claude"},
    )
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["requested_agents"] == ["codex", "claude_code"]


def test_normalize_requested_agents_supports_iterable_items_with_csv_tokens():
    assert gui_app_module._normalize_requested_agents(("codex,claude", "auto")) == [
        "codex",
        "claude_code",
    ]


def test_docs_api_lists_curated_docs(client):
    resp = client.get("/api/docs")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    docs = {entry["key"]: entry for entry in data["docs"]}
    assert "quickstart" in docs
    assert "output_artifacts" in docs
    assert "tutorial" in docs
    assert "cli_reference" in docs
    assert "troubleshooting" in docs
    assert "model_watchdog" in docs
    assert docs["quickstart"]["filename"] == "QUICKSTART.md"
    assert docs["output_artifacts"]["filename"] == "OUTPUTS_AND_ARTIFACTS.md"
    assert docs["model_watchdog"]["filename"] == "MODEL_WATCHDOG.md"


def test_docs_api_returns_doc_content(client):
    resp = client.get("/api/docs/quickstart")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["key"] == "quickstart"
    assert data["filename"] == "QUICKSTART.md"
    assert "Prerequisites" in data["content"]


def test_docs_api_returns_outputs_artifacts_content(client):
    resp = client.get("/api/docs/output_artifacts")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data
    assert data["key"] == "output_artifacts"
    assert data["filename"] == "OUTPUTS_AND_ARTIFACTS.md"
    assert "Chain mode artifacts" in data["content"]


def test_docs_api_rejects_unknown_doc_key(client):
    resp = client.get("/api/docs/not-real")
    data = resp.get_json()

    assert resp.status_code == 404
    assert data
    assert "Unknown doc key" in data["error"]


def test_docs_api_reports_missing_docs_directory(client, monkeypatch):
    monkeypatch.setattr(gui_app_module, "_docs_dir", lambda: None)

    resp = client.get("/api/docs/quickstart")
    data = resp.get_json()

    assert resp.status_code == 404
    assert data
    assert "Local docs directory not found" in data["error"]


def test_cua_status_reports_running_while_session_is_active(client, monkeypatch):
    start_gate = threading.Event()
    finish_gate = threading.Event()

    def _fake_run(config):
        start_gate.set()
        finish_gate.wait(timeout=5)
        return CUASessionResult(
            task=config.task,
            provider=config.provider.value,
            success=True,
            total_steps=2,
            duration_seconds=0.1,
            finished_at="done",
        )

    monkeypatch.setattr("codex_manager.cua.session.run_cua_session_sync", _fake_run)
    monkeypatch.setattr(gui_app_module, "_cua_result", None)
    monkeypatch.setattr(gui_app_module, "_cua_thread", None)

    try:
        start_resp = client.post("/api/cua/start", json={"provider": "openai", "task": "check"})
        start_data = start_resp.get_json()
        assert start_resp.status_code == 200
        assert start_data
        assert start_gate.wait(timeout=2)

        status_resp = client.get("/api/cua/status")
        status_data = status_resp.get_json()
        assert status_resp.status_code == 200
        assert status_data
        assert status_data["running"] is True
        assert status_data["result"]["provider"] == "openai"
        assert status_data["result"]["finished_at"] == ""

        finish_gate.set()
        for _ in range(30):
            status_data = client.get("/api/cua/status").get_json()
            if status_data and not status_data["running"]:
                break
            time.sleep(0.05)

        assert status_data
        assert status_data["running"] is False
        assert status_data["result"]["success"] is True
    finally:
        finish_gate.set()
        thread = getattr(gui_app_module, "_cua_thread", None)
        if thread is not None:
            thread.join(timeout=2)


def test_cua_start_rejects_overlapping_sessions(client, monkeypatch):
    start_gate = threading.Event()
    finish_gate = threading.Event()

    def _fake_run(config):
        start_gate.set()
        finish_gate.wait(timeout=5)
        return CUASessionResult(
            task=config.task,
            provider=config.provider.value,
            success=True,
            total_steps=1,
            duration_seconds=0.1,
            finished_at="done",
        )

    monkeypatch.setattr("codex_manager.cua.session.run_cua_session_sync", _fake_run)
    monkeypatch.setattr(gui_app_module, "_cua_result", None)
    monkeypatch.setattr(gui_app_module, "_cua_thread", None)

    try:
        first = client.post("/api/cua/start", json={"provider": "openai", "task": "one"})
        assert first.status_code == 200
        assert start_gate.wait(timeout=2)

        second = client.post("/api/cua/start", json={"provider": "anthropic", "task": "two"})
        second_data = second.get_json()
        assert second.status_code == 409
        assert second_data
        assert "already running" in second_data["error"]
    finally:
        finish_gate.set()
        thread = getattr(gui_app_module, "_cua_thread", None)
        if thread is not None:
            thread.join(timeout=2)
