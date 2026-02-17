"""Native provider deep-research execution with retries, quotas, and budgets."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from codex_manager.preflight import first_env_secret

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)
_DEFAULT_TIMEOUT_SECONDS = 45
_DEFAULT_OPENAI_MODEL = "gpt-5.2"
_DEFAULT_GOOGLE_MODEL = "gemini-3-pro-preview"
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
_USAGE_LOCKS: dict[str, threading.Lock] = {}
_USAGE_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class DeepResearchSettings:
    """Controls for native deep-research calls."""

    providers: str = "both"  # openai | google | both
    retry_attempts: int = 2
    daily_quota: int = 8
    max_provider_tokens: int = 12_000
    daily_budget_usd: float = 5.0
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS
    openai_model: str = _DEFAULT_OPENAI_MODEL
    google_model: str = _DEFAULT_GOOGLE_MODEL


@dataclass(frozen=True)
class DeepResearchProviderResult:
    """One provider response inside a deep-research run."""

    provider: str
    ok: bool
    summary: str
    sources: list[str]
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    error: str = ""


@dataclass(frozen=True)
class DeepResearchRunResult:
    """Combined provider-native deep-research output."""

    ok: bool
    topic: str
    providers: list[DeepResearchProviderResult]
    merged_summary: str
    merged_sources: list[str]
    total_input_tokens: int
    total_output_tokens: int
    total_estimated_cost_usd: float
    governance_warnings: list[str] = field(default_factory=list)
    filtered_source_count: int = 0
    budget_blocked: bool = False
    quota_blocked: bool = False
    provider_prompt_previews: dict[str, str] = field(default_factory=dict)
    error: str = ""


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat()


def _usage_path(repo_path: Path) -> Path:
    return repo_path / ".codex_manager" / "memory" / "deep_research_usage.json"


def _usage_lock(repo_path: Path) -> threading.Lock:
    key = str(repo_path.resolve())
    if os.name == "nt":
        key = key.casefold()
    with _USAGE_LOCKS_GUARD:
        lock = _USAGE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _USAGE_LOCKS[key] = lock
    return lock


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return dict(fallback)
    except Exception:
        logger.warning("Could not parse JSON %s; using fallback.", path, exc_info=True)
        return dict(fallback)
    if not isinstance(payload, dict):
        return dict(fallback)
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _ensure_parent(path)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        tmp_path.replace(path)
    finally:
        with suppress(Exception):
            tmp_path.unlink(missing_ok=True)


def _extract_urls(text: str) -> list[str]:
    urls = {match.group(0).rstrip(".,;:") for match in _URL_RE.finditer(text or "")}
    return sorted(urls)


def _parse_domain_policy_env(name: str) -> set[str]:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return set()
    domains = {chunk.strip().lower().lstrip(".") for chunk in raw.split(",") if chunk.strip()}
    return {domain for domain in domains if domain}


def _domain_matches(host: str, domain: str) -> bool:
    host_key = str(host or "").strip().lower()
    domain_key = str(domain or "").strip().lower().lstrip(".")
    if not host_key or not domain_key:
        return False
    return host_key == domain_key or host_key.endswith(f".{domain_key}")


def _filter_sources_by_policy(sources: list[str]) -> tuple[list[str], list[str]]:
    allowed_domains = _parse_domain_policy_env("DEEP_RESEARCH_ALLOWED_SOURCE_DOMAINS")
    blocked_domains = set(_DEFAULT_BLOCKED_SOURCE_DOMAINS)
    blocked_domains.update(_parse_domain_policy_env("DEEP_RESEARCH_BLOCKED_SOURCE_DOMAINS"))
    warnings: list[str] = []
    accepted: list[str] = []
    blocked_hits: set[str] = set()
    allowlist_violations: set[str] = set()
    insecure_count = 0

    for source in sorted({str(item or "").strip() for item in sources if str(item or "").strip()}):
        parsed = urlparse(source)
        scheme = str(parsed.scheme or "").lower()
        host = str(parsed.hostname or "").strip().lower()

        reject = False
        if scheme != "https":
            insecure_count += 1
            reject = True
        if host and any(_domain_matches(host, blocked) for blocked in blocked_domains):
            blocked_hits.add(host)
            reject = True
        if allowed_domains and host and not any(
            _domain_matches(host, allowed) for allowed in allowed_domains
        ):
            allowlist_violations.add(host)
            reject = True

        if not reject:
            accepted.append(source)

    if insecure_count:
        warnings.append(
            "Filtered non-HTTPS citations from deep research output. Prefer HTTPS-only references."
        )
    if blocked_hits:
        warnings.append(
            "Filtered low-trust source domains: " + ", ".join(sorted(blocked_hits)[:8])
        )
    if allowlist_violations:
        warnings.append(
            "Filtered citations outside DEEP_RESEARCH_ALLOWED_SOURCE_DOMAINS: "
            + ", ".join(sorted(allowlist_violations)[:8])
        )

    return accepted, warnings


def _clamp_int(value: int, *, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _estimate_cost_usd(provider: str, input_tokens: int, output_tokens: int) -> float:
    provider_key = (provider or "").strip().lower()
    in_rate = 0.0
    out_rate = 0.0
    if provider_key == "openai":
        in_rate = float(os.getenv("DEEP_RESEARCH_OPENAI_USD_PER_1K_INPUT", "0.01"))
        out_rate = float(os.getenv("DEEP_RESEARCH_OPENAI_USD_PER_1K_OUTPUT", "0.03"))
    elif provider_key == "google":
        in_rate = float(os.getenv("DEEP_RESEARCH_GOOGLE_USD_PER_1K_INPUT", "0.004"))
        out_rate = float(os.getenv("DEEP_RESEARCH_GOOGLE_USD_PER_1K_OUTPUT", "0.012"))
    total = (max(0, input_tokens) / 1000.0 * in_rate) + (max(0, output_tokens) / 1000.0 * out_rate)
    return round(max(0.0, total), 6)


def _retry_call(
    fn,
    *,
    attempts: int,
    provider: str,
    topic: str,
) -> dict[str, Any]:
    max_attempts = _clamp_int(attempts, minimum=1, maximum=6)
    delay = 1.0
    last_error: Exception | None = None
    for index in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if index >= max_attempts - 1:
                break
            logger.warning(
                "Deep research provider %s failed for topic '%s' (attempt %s/%s): %s",
                provider,
                topic,
                index + 1,
                max_attempts,
                exc,
            )
            time.sleep(delay)
            delay = min(delay * 2.0, 8.0)
    if last_error is None:
        raise RuntimeError(f"Unknown provider failure for {provider}")
    raise last_error


def _http_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    timeout_seconds: int,
) -> dict[str, Any]:
    body_bytes = b""
    req_headers = dict(headers or {})
    if payload is not None:
        body_bytes = json.dumps(payload).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")
    request = Request(url, data=body_bytes if payload is not None else None, headers=req_headers, method=method)
    try:
        with urlopen(request, timeout=float(timeout_seconds)) as resp:
            raw = resp.read()
            charset = resp.headers.get_content_charset() or "utf-8"
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        msg = f"HTTP {exc.code} {method} {url}"
        if detail:
            msg += f": {detail[:220]}"
        raise RuntimeError(msg) from exc
    except URLError as exc:
        raise RuntimeError(f"Network error {method} {url}: {exc.reason}") from exc
    try:
        parsed = json.loads(raw.decode(charset, errors="replace"))
    except Exception as exc:
        raise RuntimeError(f"Non-JSON response from {url}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Unexpected payload type from {url}")
    return parsed


def _load_daily_usage(repo_path: Path) -> dict[str, Any]:
    path = _usage_path(repo_path)
    payload = _read_json(
        path,
        {
            "date": "",
            "runs": 0,
            "estimated_cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "providers": {},
            "updated_at": "",
        },
    )
    today = _utc_now().date().isoformat()
    if str(payload.get("date") or "") != today:
        payload = {
            "date": today,
            "runs": 0,
            "estimated_cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "providers": {},
            "updated_at": _iso_now(),
        }
        _write_json_atomic(path, payload)
    return payload


def _save_daily_usage(repo_path: Path, usage: dict[str, Any]) -> None:
    usage["updated_at"] = _iso_now()
    _write_json_atomic(_usage_path(repo_path), usage)


def _usage_allows_run(
    usage: dict[str, Any],
    *,
    daily_quota: int,
    daily_budget_usd: float,
) -> tuple[bool, bool, bool]:
    # returns: allowed, quota_blocked, budget_blocked
    quota = max(1, int(daily_quota))
    budget = max(0.0, float(daily_budget_usd))
    runs = int(usage.get("runs") or 0)
    cost = float(usage.get("estimated_cost_usd") or 0.0)

    quota_blocked = runs >= quota
    budget_blocked = budget > 0 and cost >= budget
    return (not quota_blocked and not budget_blocked), quota_blocked, budget_blocked


def _record_run_usage(
    repo_path: Path,
    usage: dict[str, Any],
    provider_results: list[DeepResearchProviderResult],
) -> None:
    usage["runs"] = int(usage.get("runs") or 0) + 1
    usage["input_tokens"] = int(usage.get("input_tokens") or 0) + sum(
        max(0, int(item.input_tokens)) for item in provider_results
    )
    usage["output_tokens"] = int(usage.get("output_tokens") or 0) + sum(
        max(0, int(item.output_tokens)) for item in provider_results
    )
    usage["estimated_cost_usd"] = round(
        float(usage.get("estimated_cost_usd") or 0.0)
        + sum(float(item.estimated_cost_usd) for item in provider_results),
        6,
    )
    providers = usage.get("providers")
    if not isinstance(providers, dict):
        providers = {}
    for item in provider_results:
        stats = providers.get(item.provider)
        if not isinstance(stats, dict):
            stats = {"runs": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0}
        stats["runs"] = int(stats.get("runs") or 0) + 1
        stats["input_tokens"] = int(stats.get("input_tokens") or 0) + max(0, int(item.input_tokens))
        stats["output_tokens"] = int(stats.get("output_tokens") or 0) + max(0, int(item.output_tokens))
        stats["estimated_cost_usd"] = round(
            float(stats.get("estimated_cost_usd") or 0.0) + float(item.estimated_cost_usd),
            6,
        )
        providers[item.provider] = stats
    usage["providers"] = providers
    _save_daily_usage(repo_path, usage)


def _normalize_provider_list(providers: str) -> list[str]:
    key = (providers or "both").strip().lower()
    if key == "openai":
        return ["openai"]
    if key == "google":
        return ["google"]
    return ["openai", "google"]


def _extract_openai_text(payload: dict[str, Any]) -> str:
    output_text = str(payload.get("output_text") or "").strip()
    if output_text:
        return output_text
    output = payload.get("output")
    if not isinstance(output, list):
        return ""
    collected: list[str] = []
    for block in output:
        if not isinstance(block, dict):
            continue
        content = block.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if text:
                collected.append(text)
    return "\n\n".join(collected).strip()


def _call_openai_native(
    *,
    topic: str,
    guidance: str,
    model: str,
    max_output_tokens: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    api_key = str(first_env_secret(("OPENAI_API_KEY", "CODEX_API_KEY")) or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY (or CODEX_API_KEY) is not configured.")
    prompt_text = _openai_native_prompt_text(topic=topic, guidance=guidance)
    payload = {
        "model": model,
        "input": prompt_text,
        "max_output_tokens": max_output_tokens,
        "tools": [{"type": "web_search_preview"}],
    }
    out = _http_json(
        "https://api.openai.com/v1/responses",
        method="POST",
        headers={"Authorization": f"Bearer {api_key}"},
        payload=payload,
        timeout_seconds=timeout_seconds,
    )
    usage = out.get("usage") if isinstance(out.get("usage"), dict) else {}
    summary = _extract_openai_text(out)
    if not summary:
        raise RuntimeError("OpenAI deep research returned empty content.")
    return {
        "summary": summary,
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
    }


def _extract_google_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return ""
    parts: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        rows = content.get("parts")
        if not isinstance(rows, list):
            continue
        for item in rows:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts).strip()


def _call_google_native(
    *,
    topic: str,
    guidance: str,
    model: str,
    max_output_tokens: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    api_key = str(first_env_secret(("GOOGLE_API_KEY", "GEMINI_API_KEY")) or "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not configured.")
    prompt_text = _google_native_prompt_text(topic=topic, guidance=guidance)
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": max_output_tokens,
        },
        "tools": [{"google_search": {}}],
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    out = _http_json(
        url,
        method="POST",
        payload=payload,
        timeout_seconds=timeout_seconds,
    )
    usage = out.get("usageMetadata") if isinstance(out.get("usageMetadata"), dict) else {}
    summary = _extract_google_text(out)
    if not summary:
        raise RuntimeError("Google deep research returned empty content.")
    return {
        "summary": summary,
        "input_tokens": int(usage.get("promptTokenCount") or 0),
        "output_tokens": int(usage.get("candidatesTokenCount") or 0),
    }


def _provider_result(
    provider: str,
    *,
    ok: bool,
    summary: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    error: str = "",
) -> DeepResearchProviderResult:
    sources = _extract_urls(summary)
    return DeepResearchProviderResult(
        provider=provider,
        ok=ok,
        summary=summary.strip(),
        sources=sources,
        input_tokens=max(0, int(input_tokens)),
        output_tokens=max(0, int(output_tokens)),
        estimated_cost_usd=_estimate_cost_usd(provider, input_tokens, output_tokens),
        error=str(error or "").strip(),
    )


def run_native_deep_research(
    *,
    repo_path: Path | str,
    topic: str,
    project_context: str,
    settings: DeepResearchSettings,
) -> DeepResearchRunResult:
    """Run provider-native deep research and return merged results."""
    repo = Path(repo_path).resolve()
    clean_topic = str(topic or "").strip()
    if not clean_topic:
        return DeepResearchRunResult(
            ok=False,
            topic="",
            providers=[],
            merged_summary="",
            merged_sources=[],
            total_input_tokens=0,
            total_output_tokens=0,
            total_estimated_cost_usd=0.0,
            governance_warnings=[],
            filtered_source_count=0,
            error="Deep research topic is empty.",
        )

    provider_prompt_previews: dict[str, str] = {}

    # Serialize quota/budget checks per repository so concurrent requests
    # cannot both pass stale counters before usage is persisted.
    with _usage_lock(repo):
        usage = _load_daily_usage(repo)
        allowed, quota_blocked, budget_blocked = _usage_allows_run(
            usage,
            daily_quota=settings.daily_quota,
            daily_budget_usd=settings.daily_budget_usd,
        )
        if not allowed:
            reason = (
                "daily quota reached"
                if quota_blocked
                else "daily research budget reached"
                if budget_blocked
                else "blocked by usage policy"
            )
            return DeepResearchRunResult(
                ok=False,
                topic=clean_topic,
                providers=[],
                merged_summary="",
                merged_sources=[],
                total_input_tokens=0,
                total_output_tokens=0,
                total_estimated_cost_usd=0.0,
                governance_warnings=[],
                filtered_source_count=0,
                quota_blocked=quota_blocked,
                budget_blocked=budget_blocked,
                provider_prompt_previews=provider_prompt_previews,
                error=f"Native deep research blocked: {reason}.",
            )

        provider_list = _normalize_provider_list(settings.providers)
        retry_attempts = _clamp_int(settings.retry_attempts, minimum=1, maximum=6)
        max_tokens = _clamp_int(settings.max_provider_tokens, minimum=512, maximum=64_000)
        timeout_seconds = _clamp_int(settings.timeout_seconds, minimum=10, maximum=120)
        guidance = str(project_context or "").strip()
        if len(guidance) > 5000:
            guidance = guidance[:5000].rstrip() + "..."

        results: list[DeepResearchProviderResult] = []
        for provider in provider_list:
            try:
                if provider == "openai":
                    provider_prompt_previews["openai"] = _openai_native_prompt_text(
                        topic=clean_topic,
                        guidance=guidance,
                    )
                    payload = _retry_call(
                        lambda: _call_openai_native(
                            topic=clean_topic,
                            guidance=guidance,
                            model=str(settings.openai_model or _DEFAULT_OPENAI_MODEL).strip()
                            or _DEFAULT_OPENAI_MODEL,
                            max_output_tokens=max_tokens,
                            timeout_seconds=timeout_seconds,
                        ),
                        attempts=retry_attempts,
                        provider=provider,
                        topic=clean_topic,
                    )
                    results.append(
                        _provider_result(
                            provider,
                            ok=True,
                            summary=str(payload.get("summary") or ""),
                            input_tokens=int(payload.get("input_tokens") or 0),
                            output_tokens=int(payload.get("output_tokens") or 0),
                        )
                    )
                    continue

                if provider == "google":
                    provider_prompt_previews["google"] = _google_native_prompt_text(
                        topic=clean_topic,
                        guidance=guidance,
                    )
                    payload = _retry_call(
                        lambda: _call_google_native(
                            topic=clean_topic,
                            guidance=guidance,
                            model=str(settings.google_model or _DEFAULT_GOOGLE_MODEL).strip()
                            or _DEFAULT_GOOGLE_MODEL,
                            max_output_tokens=max_tokens,
                            timeout_seconds=timeout_seconds,
                        ),
                        attempts=retry_attempts,
                        provider=provider,
                        topic=clean_topic,
                    )
                    results.append(
                        _provider_result(
                            provider,
                            ok=True,
                            summary=str(payload.get("summary") or ""),
                            input_tokens=int(payload.get("input_tokens") or 0),
                            output_tokens=int(payload.get("output_tokens") or 0),
                        )
                    )
                    continue

                results.append(
                    _provider_result(
                        provider,
                        ok=False,
                        error=f"Unsupported deep research provider: {provider}",
                    )
                )
            except Exception as exc:
                results.append(_provider_result(provider, ok=False, error=str(exc)))

        successful = [item for item in results if item.ok and item.summary]
        merged_sources = sorted(
            {
                source
                for item in successful
                for source in item.sources
                if str(source or "").strip()
            }
        )
        merged_sections: list[str] = []
        for item in successful:
            merged_sections.append(f"## {item.provider.title()} Findings\n\n{item.summary}")
        merged_summary = "\n\n".join(merged_sections).strip()
        governed_sources, governance_warnings = _filter_sources_by_policy(merged_sources)
        total_input_tokens = sum(item.input_tokens for item in results)
        total_output_tokens = sum(item.output_tokens for item in results)
        total_cost = round(sum(item.estimated_cost_usd for item in results), 6)
        filtered_count = max(0, len(merged_sources) - len(governed_sources))

        if successful:
            _record_run_usage(repo, usage, successful)
            return DeepResearchRunResult(
                ok=True,
                topic=clean_topic,
                providers=results,
                merged_summary=merged_summary,
                merged_sources=governed_sources,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                total_estimated_cost_usd=total_cost,
                governance_warnings=governance_warnings,
                filtered_source_count=filtered_count,
                provider_prompt_previews=provider_prompt_previews,
            )

        errors = "; ".join(
            f"{item.provider}: {item.error}" for item in results if not item.ok and item.error
        )
        return DeepResearchRunResult(
            ok=False,
            topic=clean_topic,
            providers=results,
            merged_summary="",
            merged_sources=[],
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_estimated_cost_usd=total_cost,
            governance_warnings=[],
            filtered_source_count=0,
            provider_prompt_previews=provider_prompt_previews,
            error=errors or "Native deep research failed for all providers.",
        )


def _openai_native_prompt_text(*, topic: str, guidance: str) -> str:
    return (
        "You are running deep technical research for a software project.\n"
        "Focus on verifiable, implementation-relevant findings and include source links.\n\n"
        f"Topic: {topic}\n\n"
        "Project context:\n"
        f"{guidance}\n\n"
        "Output format:\n"
        "- Executive summary\n"
        "- Concrete implementation recommendations\n"
        "- Risks/caveats\n"
        "- Sources list with URLs"
    )


def _google_native_prompt_text(*, topic: str, guidance: str) -> str:
    return (
        "Run deep technical research for this software project. "
        "Focus on implementation-grade findings with citations.\n\n"
        f"Topic: {topic}\n\n"
        "Project context:\n"
        f"{guidance}\n\n"
        "Return:\n"
        "1) Executive summary\n"
        "2) Concrete implementation recommendations\n"
        "3) Caveats and risk notes\n"
        "4) Sources with URLs"
    )
