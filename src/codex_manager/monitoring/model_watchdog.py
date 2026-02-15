"""Scheduled provider/model catalog snapshots with change tracking."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

AVAILABLE_MODEL_PROVIDERS: tuple[str, ...] = ("openai", "anthropic", "google", "xai", "ollama")
DEFAULT_INTERVAL_HOURS = 24
MIN_INTERVAL_HOURS = 1
MAX_INTERVAL_HOURS = 24 * 30
DEFAULT_REQUEST_TIMEOUT_SECONDS = 10
DEFAULT_HISTORY_LIMIT = 100
DEFAULT_DEPENDENCY_PACKAGES: tuple[str, ...] = (
    "codex-manager",
    "flask",
    "pydantic",
    "openai",
    "anthropic",
    "google-generativeai",
    "chromadb",
)

ProviderFetcher = Callable[[int], list[str]]


def _utc_now_iso(now_epoch_s: float | None = None) -> str:
    now = datetime.fromtimestamp(now_epoch_s or time.time(), tz=timezone.utc)
    return now.isoformat().replace("+00:00", "Z")


def _parse_utc_iso(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return dict(fallback)
    except Exception:
        logger.warning("Could not parse JSON file %s; using fallback payload", path, exc_info=True)
        return dict(fallback)
    if not isinstance(data, dict):
        return dict(fallback)
    return data


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        tmp_path.replace(path)
    finally:
        with suppress(Exception):
            tmp_path.unlink(missing_ok=True)


def _normalize_models(models: list[str]) -> list[str]:
    return sorted({str(item or "").strip() for item in models if str(item or "").strip()})


def _model_family_token(model_id: str) -> str:
    """Return a coarse family token used for migration suggestions."""
    key = str(model_id or "").strip().lower()
    if not key:
        return ""
    for known in ("gpt", "claude", "gemini", "grok", "llama", "mistral", "qwen", "gemma"):
        if key.startswith(known):
            return known
    for sep in ("-", "_", ":", "/", "."):
        if sep in key:
            return key.split(sep, 1)[0]
    return key


class ModelCatalogWatchdog:
    """Periodic model/dependency catalog watcher with persisted snapshots."""

    def __init__(
        self,
        *,
        root_dir: Path | str,
        default_enabled: bool = True,
        default_interval_hours: int = DEFAULT_INTERVAL_HOURS,
        provider_fetchers: Mapping[str, ProviderFetcher] | None = None,
        dependency_packages: tuple[str, ...] = DEFAULT_DEPENDENCY_PACKAGES,
        poll_seconds: int = 60,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.config_path = self.root_dir / "config.json"
        self.state_path = self.root_dir / "state.json"
        self.latest_snapshot_path = self.root_dir / "model_catalog_latest.json"
        self.history_path = self.root_dir / "model_catalog_history.ndjson"
        self.poll_seconds = max(10, int(poll_seconds))
        self._dependency_packages = tuple(dependency_packages)

        self._lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._provider_fetchers: dict[str, ProviderFetcher] = (
            dict(provider_fetchers) if provider_fetchers is not None else self._default_fetchers()
        )
        self._config = self._load_or_init_config(
            default_enabled=bool(default_enabled),
            default_interval_hours=int(default_interval_hours),
        )
        self._state = self._load_state()

    def _default_fetchers(self) -> dict[str, ProviderFetcher]:
        return {
            "openai": self._fetch_openai_models,
            "anthropic": self._fetch_anthropic_models,
            "google": self._fetch_google_models,
            "xai": self._fetch_xai_models,
            "ollama": self._fetch_ollama_models,
        }

    def _load_or_init_config(
        self,
        *,
        default_enabled: bool,
        default_interval_hours: int,
    ) -> dict[str, Any]:
        raw = _read_json(self.config_path, {})
        if not raw:
            raw = {
                "enabled": default_enabled,
                "interval_hours": default_interval_hours,
                "providers": list(AVAILABLE_MODEL_PROVIDERS),
                "request_timeout_seconds": DEFAULT_REQUEST_TIMEOUT_SECONDS,
                "auto_run_on_start": True,
                "history_limit": DEFAULT_HISTORY_LIMIT,
            }
        normalized = self._normalize_config(raw)
        _write_json_atomic(self.config_path, normalized)
        return normalized

    def _load_state(self) -> dict[str, Any]:
        fallback = {
            "last_status": "never_ran",
            "last_started_at": "",
            "last_finished_at": "",
            "last_error": "",
            "last_reason": "",
            "catalog_changed": False,
        }
        state = _read_json(self.state_path, fallback)
        normalized = dict(fallback)
        normalized.update({k: state.get(k, fallback.get(k)) for k in fallback})
        _write_json_atomic(self.state_path, normalized)
        return normalized

    def _normalize_config(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        enabled = bool(payload.get("enabled", True))
        try:
            interval_hours = int(payload.get("interval_hours", DEFAULT_INTERVAL_HOURS))
        except Exception:
            interval_hours = DEFAULT_INTERVAL_HOURS
        interval_hours = max(MIN_INTERVAL_HOURS, min(MAX_INTERVAL_HOURS, interval_hours))

        providers_raw = payload.get("providers", list(AVAILABLE_MODEL_PROVIDERS))
        providers: list[str] = []
        if isinstance(providers_raw, str):
            tokens = [part.strip().lower() for part in providers_raw.split(",")]
            providers = [token for token in tokens if token in AVAILABLE_MODEL_PROVIDERS]
        elif isinstance(providers_raw, list):
            for item in providers_raw:
                token = str(item or "").strip().lower()
                if token in AVAILABLE_MODEL_PROVIDERS and token not in providers:
                    providers.append(token)
        if not providers:
            providers = list(AVAILABLE_MODEL_PROVIDERS)

        try:
            timeout_s = int(payload.get("request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS))
        except Exception:
            timeout_s = DEFAULT_REQUEST_TIMEOUT_SECONDS
        timeout_s = max(2, min(60, timeout_s))

        try:
            history_limit = int(payload.get("history_limit", DEFAULT_HISTORY_LIMIT))
        except Exception:
            history_limit = DEFAULT_HISTORY_LIMIT
        history_limit = max(10, min(1000, history_limit))

        return {
            "enabled": enabled,
            "interval_hours": interval_hours,
            "providers": providers,
            "request_timeout_seconds": timeout_s,
            "auto_run_on_start": bool(payload.get("auto_run_on_start", True)),
            "history_limit": history_limit,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            config = dict(self._config)
            state = dict(self._state)
            running = bool(self._thread and self._thread.is_alive())
        next_due_at = self._compute_next_due_at(config, state)
        return {
            "running": running,
            "config": config,
            "state": state,
            "next_due_at": next_due_at,
            "paths": {
                "root_dir": str(self.root_dir),
                "config": str(self.config_path),
                "state": str(self.state_path),
                "latest_snapshot": str(self.latest_snapshot_path),
                "history": str(self.history_path),
            },
        }

    def latest_alerts(self) -> dict[str, Any]:
        """Return alert-oriented summary from the newest history diff."""
        status = self.status()
        try:
            lines = self.history_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return {
                "has_alerts": False,
                "alerts": [],
                "last_generated_at": "",
                "status": status,
            }
        except Exception:
            logger.warning("Could not read model watchdog history", exc_info=True)
            return {
                "has_alerts": False,
                "alerts": [],
                "last_generated_at": "",
                "status": status,
            }

        if not lines:
            return {
                "has_alerts": False,
                "alerts": [],
                "last_generated_at": "",
                "status": status,
            }

        latest_row: dict[str, Any] = {}
        for raw in reversed(lines):
            row = str(raw or "").strip()
            if not row:
                continue
            try:
                parsed = json.loads(row)
            except Exception:
                continue
            if isinstance(parsed, dict):
                latest_row = parsed
                break
        if not latest_row:
            return {
                "has_alerts": False,
                "alerts": [],
                "last_generated_at": "",
                "status": status,
            }

        alerts: list[dict[str, Any]] = []
        diff = latest_row.get("diff")
        if not isinstance(diff, dict):
            diff = {}
        provider_diff = diff.get("providers")
        if isinstance(provider_diff, dict):
            for provider, details in provider_diff.items():
                if not isinstance(details, dict):
                    continue
                added_count = int(details.get("added_count") or 0)
                removed_count = int(details.get("removed_count") or 0)
                added_models = [
                    str(item or "").strip()
                    for item in details.get("added", [])
                    if str(item or "").strip()
                ]
                removed_models = [
                    str(item or "").strip()
                    for item in details.get("removed", [])
                    if str(item or "").strip()
                ]
                current_models = [
                    str(item or "").strip()
                    for item in dict(latest_row.get("providers", {}))
                    .get(str(provider), {})
                    .get("models", [])
                    if str(item or "").strip()
                ]
                playbook = self._build_model_migration_playbook(
                    provider=str(provider),
                    added_models=added_models,
                    removed_models=removed_models,
                    current_models=current_models,
                )
                if removed_count > 0:
                    alert: dict[str, Any] = {
                        "severity": "warn",
                        "title": f"{provider} models removed",
                        "detail": (
                            f"{removed_count} model(s) disappeared from provider catalog. "
                            "Review model defaults and fallbacks."
                        ),
                        "provider": str(provider),
                        "action": "Check configured models and migrate off removed IDs.",
                    }
                    if playbook:
                        alert["playbook"] = playbook
                        alert["migrations"] = playbook.get("migrations", [])
                        alert["action"] = (
                            "Apply the suggested migration playbook or update model IDs manually."
                        )
                    alerts.append(alert)
                if added_count > 0:
                    alerts.append(
                        {
                            "severity": "info",
                            "title": f"{provider} models added",
                            "detail": (
                                f"{added_count} new model(s) detected. Consider benchmarking for upgrades."
                            ),
                            "provider": str(provider),
                            "action": "Review new models for cost/quality tradeoffs.",
                        }
                    )

        dep_diff = diff.get("dependencies")
        if isinstance(dep_diff, dict):
            for package, details in dep_diff.items():
                if not isinstance(details, dict):
                    continue
                before = str(details.get("before") or "")
                after = str(details.get("after") or "")
                if before == after:
                    continue
                severity = "warn" if "not-installed" in {before, after} else "info"
                alerts.append(
                    {
                        "severity": severity,
                        "title": f"Dependency changed: {package}",
                        "detail": f"{before or 'unknown'} -> {after or 'unknown'}",
                        "provider": "",
                        "action": "Run compatibility checks and update pinned versions if needed.",
                    }
                )

        playbooks: list[dict[str, Any]] = []
        for alert in alerts:
            playbook = alert.get("playbook")
            if isinstance(playbook, dict):
                playbooks.append(playbook)

        return {
            "has_alerts": bool(alerts),
            "alerts": alerts,
            "playbooks": playbooks,
            "last_generated_at": str(latest_row.get("generated_at") or ""),
            "status": status,
        }

    def _suggest_replacement_model(
        self,
        *,
        provider: str,
        removed_model: str,
        added_models: list[str],
        current_models: list[str],
    ) -> str:
        """Choose a best-effort replacement model for a removed ID."""
        provider_key = str(provider or "").strip().lower()
        removed = str(removed_model or "").strip()
        if not removed:
            return ""
        pool = [m for m in added_models if m] or [m for m in current_models if m]
        if not pool:
            return ""

        removed_family = _model_family_token(removed)
        family_matches = [
            candidate
            for candidate in pool
            if _model_family_token(candidate) == removed_family
        ]
        if family_matches:
            return sorted(family_matches)[-1]

        provider_hints: dict[str, tuple[str, ...]] = {
            "openai": ("gpt-5.3", "gpt-5", "gpt-4.1"),
            "anthropic": ("claude-opus-4-6", "claude-sonnet-4"),
            "google": ("gemini-3", "gemini-2.5"),
            "xai": ("grok-4", "grok-3"),
            "ollama": ("gemma3", "llama3", "qwen", "mistral"),
        }
        hints = provider_hints.get(provider_key, ())
        for hint in hints:
            for candidate in sorted(pool, reverse=True):
                if hint in candidate.lower():
                    return candidate

        return sorted(pool)[-1]

    def _build_model_migration_playbook(
        self,
        *,
        provider: str,
        added_models: list[str],
        removed_models: list[str],
        current_models: list[str],
    ) -> dict[str, Any]:
        """Build remediation playbook suggestions for removed provider models."""
        clean_removed = [str(item or "").strip() for item in removed_models if str(item or "").strip()]
        if not clean_removed:
            return {}
        migrations: list[dict[str, str]] = []
        for removed in clean_removed[:8]:
            replacement = self._suggest_replacement_model(
                provider=provider,
                removed_model=removed,
                added_models=added_models,
                current_models=current_models,
            )
            if not replacement or replacement == removed:
                continue
            migrations.append(
                {
                    "provider": str(provider or "").strip().lower(),
                    "from_model": removed,
                    "to_model": replacement,
                }
            )
        if not migrations:
            return {}
        provider_label = str(provider or "").strip() or "provider"
        return {
            "id": f"{provider_label.lower()}_model_remediation",
            "provider": provider_label.lower(),
            "title": f"{provider_label.title()} model migration playbook",
            "summary": "Replace removed model IDs with currently available equivalents.",
            "migrations": migrations,
            "steps": [
                "Apply suggested replacements in your active pipeline settings.",
                "Re-run smoke tests and benchmark quality/cost deltas.",
                "Commit updated model IDs and rollout notes.",
            ],
        }

    def start(self) -> bool:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return True
            if not bool(self._config.get("enabled", True)):
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="codex-model-watchdog",
                daemon=True,
            )
            self._thread.start()
            return True

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            self._thread = None
            self._stop_event.set()
        if thread and thread.is_alive():
            thread.join(timeout=1.0)

    def update_config(self, updates: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            merged = dict(self._config)
            merged.update(dict(updates or {}))
            self._config = self._normalize_config(merged)
            _write_json_atomic(self.config_path, self._config)
            config = dict(self._config)

        if config["enabled"]:
            self.start()
        else:
            self.stop()
        return config

    def run_once(self, *, force: bool = True, reason: str = "manual") -> dict[str, Any]:
        if not self._run_lock.acquire(blocking=False):
            return {
                "status": "busy",
                "reason": reason,
            }

        try:
            with self._lock:
                config = dict(self._config)
                state = dict(self._state)

            if not bool(config.get("enabled", True)):
                result = {
                    "status": "disabled",
                    "reason": reason,
                    "next_due_at": self._compute_next_due_at(config, state),
                }
                return result

            now = datetime.now(tz=timezone.utc)
            if not force and not self._is_due(config, state, now):
                return {
                    "status": "skipped_not_due",
                    "reason": reason,
                    "next_due_at": self._compute_next_due_at(config, state),
                }

            started_at = _utc_now_iso()
            with self._lock:
                self._state.update(
                    {
                        "last_status": "running",
                        "last_started_at": started_at,
                        "last_reason": reason,
                        "last_error": "",
                    }
                )
                _write_json_atomic(self.state_path, self._state)

            previous = _read_json(self.latest_snapshot_path, {})
            snapshot = self._collect_snapshot(config=config, reason=reason, started_at=started_at)
            diff = self._compute_diff(previous, snapshot)

            _write_json_atomic(self.latest_snapshot_path, snapshot)
            self._append_history(snapshot, diff, config)

            finished_at = _utc_now_iso()
            success_providers = [
                p for p, details in snapshot.get("providers", {}).items() if details.get("ok")
            ]
            failed_providers = [
                p for p, details in snapshot.get("providers", {}).items() if not details.get("ok")
            ]

            with self._lock:
                self._state.update(
                    {
                        "last_status": "ok",
                        "last_started_at": started_at,
                        "last_finished_at": finished_at,
                        "last_reason": reason,
                        "last_error": "",
                        "catalog_changed": bool(diff.get("catalog_changed")),
                    }
                )
                _write_json_atomic(self.state_path, self._state)
                state_out = dict(self._state)

            return {
                "status": "ok",
                "reason": reason,
                "started_at": started_at,
                "finished_at": finished_at,
                "catalog_changed": bool(diff.get("catalog_changed")),
                "successful_providers": success_providers,
                "failed_providers": failed_providers,
                "snapshot_path": str(self.latest_snapshot_path),
                "history_path": str(self.history_path),
                "next_due_at": self._compute_next_due_at(config, state_out),
                "diff": diff,
            }
        except Exception as exc:
            logger.exception("Model watchdog run failed")
            finished_at = _utc_now_iso()
            with self._lock:
                self._state.update(
                    {
                        "last_status": "error",
                        "last_finished_at": finished_at,
                        "last_error": str(exc),
                        "last_reason": reason,
                    }
                )
                _write_json_atomic(self.state_path, self._state)
            return {
                "status": "error",
                "reason": reason,
                "error": str(exc),
                "finished_at": finished_at,
            }
        finally:
            self._run_lock.release()

    def _run_loop(self) -> None:
        with self._lock:
            should_start = bool(self._config.get("auto_run_on_start", True))
        if should_start:
            self.run_once(force=False, reason="startup")

        while not self._stop_event.wait(self.poll_seconds):
            with self._lock:
                enabled = bool(self._config.get("enabled", True))
            if not enabled:
                continue
            self.run_once(force=False, reason="scheduled")

    def _is_due(self, config: Mapping[str, Any], state: Mapping[str, Any], now: datetime) -> bool:
        last_finished = _parse_utc_iso(str(state.get("last_finished_at", "")))
        if last_finished is None:
            return True
        interval_hours = int(config.get("interval_hours", DEFAULT_INTERVAL_HOURS))
        delta = now - last_finished
        return delta >= timedelta(hours=interval_hours)

    def _compute_next_due_at(
        self,
        config: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> str:
        last_finished = _parse_utc_iso(str(state.get("last_finished_at", "")))
        if last_finished is None:
            return _utc_now_iso()
        interval_hours = int(config.get("interval_hours", DEFAULT_INTERVAL_HOURS))
        due = last_finished + timedelta(hours=interval_hours)
        return due.isoformat().replace("+00:00", "Z")

    def _collect_snapshot(
        self,
        *,
        config: Mapping[str, Any],
        reason: str,
        started_at: str,
    ) -> dict[str, Any]:
        timeout_s = int(config.get("request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS))
        providers = list(config.get("providers", []))
        payload: dict[str, Any] = {
            "version": 1,
            "generated_at": started_at,
            "run_reason": reason,
            "providers": {},
            "dependencies": self._collect_dependency_versions(),
        }

        for provider in providers:
            fetcher = self._provider_fetchers.get(provider)
            if fetcher is None:
                payload["providers"][provider] = {
                    "ok": False,
                    "error": f"No fetcher configured for provider '{provider}'",
                    "models": [],
                    "model_count": 0,
                }
                continue
            try:
                models = _normalize_models(fetcher(timeout_s))
                payload["providers"][provider] = {
                    "ok": True,
                    "error": "",
                    "models": models,
                    "model_count": len(models),
                }
            except Exception as exc:
                payload["providers"][provider] = {
                    "ok": False,
                    "error": str(exc),
                    "models": [],
                    "model_count": 0,
                }
        return payload

    def _collect_dependency_versions(self) -> dict[str, str]:
        versions: dict[str, str] = {}
        for package in self._dependency_packages:
            try:
                versions[package] = importlib_metadata.version(package)
            except importlib_metadata.PackageNotFoundError:
                versions[package] = "not-installed"
            except Exception:
                versions[package] = "unknown"
        return versions

    def _compute_diff(
        self,
        previous_snapshot: Mapping[str, Any],
        current_snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        previous_providers = dict(previous_snapshot.get("providers", {}))
        current_providers = dict(current_snapshot.get("providers", {}))
        provider_keys = sorted(set(previous_providers) | set(current_providers))

        provider_changes: dict[str, Any] = {}
        changed = False
        for provider in provider_keys:
            prev_models = set(previous_providers.get(provider, {}).get("models", []))
            cur_models = set(current_providers.get(provider, {}).get("models", []))
            added = sorted(cur_models - prev_models)
            removed = sorted(prev_models - cur_models)
            provider_changed = bool(added or removed)
            changed = changed or provider_changed
            provider_changes[provider] = {
                "added_count": len(added),
                "removed_count": len(removed),
                "added": added,
                "removed": removed,
            }

        prev_deps = {
            str(k): str(v)
            for k, v in dict(previous_snapshot.get("dependencies", {})).items()
            if str(k).strip()
        }
        cur_deps = {
            str(k): str(v)
            for k, v in dict(current_snapshot.get("dependencies", {})).items()
            if str(k).strip()
        }
        dep_keys = sorted(set(prev_deps) | set(cur_deps))
        dependency_changes: dict[str, Any] = {}
        for key in dep_keys:
            before = prev_deps.get(key, "")
            after = cur_deps.get(key, "")
            if before == after:
                continue
            dependency_changes[key] = {"before": before, "after": after}
            changed = True

        return {
            "catalog_changed": changed,
            "providers": provider_changes,
            "dependencies": dependency_changes,
        }

    def _append_history(
        self,
        snapshot: Mapping[str, Any],
        diff: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        history_item = {
            "generated_at": snapshot.get("generated_at"),
            "run_reason": snapshot.get("run_reason"),
            "catalog_changed": bool(diff.get("catalog_changed")),
            "providers": snapshot.get("providers", {}),
            "dependencies": snapshot.get("dependencies", {}),
            "diff": diff,
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(history_item, sort_keys=True))
            handle.write("\n")
        self._trim_history_file(int(config.get("history_limit", DEFAULT_HISTORY_LIMIT)))

    def _trim_history_file(self, history_limit: int) -> None:
        try:
            rows = self.history_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return
        if len(rows) <= history_limit:
            return
        kept = rows[-history_limit:]
        self.history_path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    def _http_json(self, url: str, *, headers: dict[str, str], timeout_s: int) -> dict[str, Any]:
        request = Request(url, headers=headers, method="GET")
        try:
            with urlopen(request, timeout=float(timeout_s)) as response:
                raw = response.read()
                charset = response.headers.get_content_charset() or "utf-8"
        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            msg = f"HTTP {exc.code} for {url}"
            if detail:
                msg = f"{msg}: {detail[:200]}"
            raise RuntimeError(msg) from exc
        except URLError as exc:
            raise RuntimeError(f"Network error for {url}: {exc.reason}") from exc

        try:
            payload = json.loads(raw.decode(charset, errors="replace"))
        except Exception as exc:
            raise RuntimeError(f"Non-JSON response from {url}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected payload type from {url}")
        return payload

    def _fetch_openai_models(self, timeout_s: int) -> list[str]:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CODEX_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY (or CODEX_API_KEY)")
        payload = self._http_json(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout_s=timeout_s,
        )
        data = payload.get("data", [])
        if not isinstance(data, list):
            return []
        return [str(item.get("id", "")).strip() for item in data if isinstance(item, dict)]

    def _fetch_anthropic_models(self, timeout_s: int) -> list[str]:
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY (or CLAUDE_API_KEY)")
        payload = self._http_json(
            "https://api.anthropic.com/v1/models",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout_s=timeout_s,
        )
        data = payload.get("data", [])
        if not isinstance(data, list):
            return []
        return [str(item.get("id", "")).strip() for item in data if isinstance(item, dict)]

    def _fetch_google_models(self, timeout_s: int) -> list[str]:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY)")
        query = urlencode({"key": api_key})
        payload = self._http_json(
            f"https://generativelanguage.googleapis.com/v1beta/models?{query}",
            headers={},
            timeout_s=timeout_s,
        )
        rows = payload.get("models", [])
        if not isinstance(rows, list):
            return []
        models: list[str] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if name.startswith("models/"):
                name = name.split("/", 1)[1]
            if name:
                models.append(name)
        return models

    def _fetch_xai_models(self, timeout_s: int) -> list[str]:
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing XAI_API_KEY")
        payload = self._http_json(
            "https://api.x.ai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout_s=timeout_s,
        )
        data = payload.get("data", [])
        if not isinstance(data, list):
            return []
        return [str(item.get("id", "")).strip() for item in data if isinstance(item, dict)]

    def _fetch_ollama_models(self, timeout_s: int) -> list[str]:
        from codex_manager.brain.connector import list_ollama_models

        rows = list_ollama_models(timeout_s=float(timeout_s))
        models: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            item = str(row.get("ollama_id") or row.get("name") or "").strip()
            if item:
                models.append(item)
        if not models:
            raise RuntimeError("No local Ollama models detected")
        return models
