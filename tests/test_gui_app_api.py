"""API tests for GUI preflight and permission validation paths."""

from __future__ import annotations

import json
import os
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
                "custom_prompt": "",
            }
        ],
    )
    resp = client.post("/api/pipeline/start", json=payload)
    data = resp.get_json()
    assert resp.status_code == 400
    assert data
    assert "Invalid pipeline phase(s): not_a_real_phase" in data["error"]


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
