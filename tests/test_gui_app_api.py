"""API tests for GUI preflight and permission validation paths."""

from __future__ import annotations

import threading
import time
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

    invalid_resp = client.post("/api/validate-repo", json={"path": None})
    invalid_data = invalid_resp.get_json()
    assert invalid_resp.status_code == 200
    assert invalid_data
    assert invalid_data["exists"] is False
    assert invalid_data["is_git"] is False
    assert invalid_data["path"] == ""


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
    rerun = next(a for a in actions if a.get("key") == "rerun_doctor")
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
    assert docs["quickstart"]["filename"] == "QUICKSTART.md"
    assert docs["output_artifacts"]["filename"] == "OUTPUTS_AND_ARTIFACTS.md"


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
