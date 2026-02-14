"""Multi-AI connector — adapted for Codex Manager.

Supports: OpenAI (GPT), Anthropic (Claude), xAI (Grok), Google (Gemini),
and local Ollama models.  Includes retry logic, persistent caching,
adaptive concurrency control, and optional guardrails.

Adapted from the user's standalone connector.py with external custom
module dependencies (red_team_ai, gemini wrapper, recovery_hints,
model_logger) made optional with built-in fallbacks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shelve
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any

# ── Optional imports (graceful fallbacks) ─────────────────────────

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env must be loaded by the caller or set in the environment

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

try:
    from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
except Exception:
    APIConnectionError = APITimeoutError = RateLimitError = APIStatusError = Exception  # type: ignore[assignment,misc]

try:
    # Suppress deprecation warning on import; we still support this SDK for compatibility.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as genai  # standard Gemini SDK
    _GEMINI_SDK = True
except ImportError:
    _GEMINI_SDK = False

# These are from the user's broader project — optional here
try:
    import recovery_hints  # type: ignore[import-untyped]
except Exception:
    recovery_hints = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def strip_json_wrappers(text: str) -> str:
    """Remove ```json ... ``` or ~~~json ... ~~~ wrappers."""
    pattern = re.compile(r"(```|~~~)json\s*(.*?)\1", re.DOTALL | re.IGNORECASE)
    stripped = re.sub(pattern, lambda m: m.group(2).strip(), text)
    return stripped.strip()


# ── Defaults / knobs ──────────────────────────────────────────────

DEFAULT_TEXT_ONLY: bool = os.getenv("AI_TEXT_ONLY", "true").lower() == "true"
DEFAULT_PER_REQUEST_TIMEOUT_S: float = float(os.getenv("AI_PER_REQUEST_TIMEOUT_S", str(60 * 10)))
DEFAULT_RECONNECT_WAIT_S: int = int(os.getenv("AI_RECONNECT_WAIT_S", "30"))
CLIENT_MAX_AGE_S: int = int(os.getenv("AI_CLIENT_MAX_AGE_S", str(45 * 60)))
_CACHE_DIR = Path.home() / ".codex_manager"
_CACHE_PATH = os.getenv("AI_RESULT_CACHE_PATH", str(_CACHE_DIR / "ai_cache.db"))

# Model tiers
DEFAULT_LEAD_MODEL: str = os.getenv("AI_LEAD_MODEL", "gpt-5.2").strip() or "gpt-5.2"
DEFAULT_CHEAP_MODEL: str = os.getenv("AI_CHEAP_MODEL", "grok-4-1-fast-reasoning").strip() or "grok-4-1-fast-reasoning"
DEFAULT_MEDIUM_MODEL: str = os.getenv("AI_MEDIUM_MODEL", "gpt-5.2").strip() or "gpt-5.2"
DEFAULT_FREE_MODEL: str = os.getenv("AI_FREE_MODEL", "ollama:gemma3:27b").strip() or "ollama:gemma3:27b"

# Retry knobs
OPENAI_MAX_ATTEMPTS: int = int(os.getenv("AI_OPENAI_MAX_ATTEMPTS", "5"))
GEMINI_MAX_ATTEMPTS: int = int(os.getenv("AI_GEMINI_MAX_ATTEMPTS", "3"))
ANTHROPIC_MAX_ATTEMPTS: int = int(os.getenv("AI_ANTHROPIC_MAX_ATTEMPTS", "3"))
XAI_MAX_ATTEMPTS: int = int(os.getenv("AI_XAI_MAX_ATTEMPTS", "3"))
OLLAMA_MAX_ATTEMPTS: int = int(os.getenv("AI_OLLAMA_MAX_ATTEMPTS", "3"))
RETRY_BACKOFF_BASE: float = float(os.getenv("AI_RETRY_BACKOFF_BASE", "2.0"))
RETRY_BACKOFF_MAX_S: float = float(os.getenv("AI_RETRY_BACKOFF_MAX_S", "60.0"))

OLLAMA_AUTO_START: bool = os.getenv("OLLAMA_AUTO_START", "false").lower() == "true"
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

NON_RETRYABLE_ERROR_PATTERNS = [
    "quota exceeded", "insufficient_quota", "model not found", "no api key", "not set",
]
RETRYABLE_ERROR_PATTERNS = [
    "rate limit", "429", "timeout", "timed out", "temporarily unavailable",
    "server error", "503", "connection", "reset by peer", "service unavailable",
]

_ollama_last_check: float = 0.0
_ollama_last_ok: bool = False


# ── Persistent cache ──────────────────────────────────────────────

class ResultCache:
    """Persistent cache for model responses keyed by provider/model/prompt variant."""

    def __init__(self, path: str = _CACHE_PATH):
        """Initialize cache storage and ensure its parent directory exists."""
        self.path = path
        self._lock = threading.Lock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _key(self, provider: str, model: str, prompt: str, variant: str = "") -> str:
        """Build a stable cache key for a request."""
        h = hashlib.sha256(f"{provider}|{model}|{variant}|{prompt}".encode()).hexdigest()
        return f"{provider}:{model}:{h}"

    def get(self, provider: str, model: str, prompt: str, variant: str = "") -> dict | None:
        """Return cached payload for a request, or ``None`` on cache miss/error."""
        try:
            with self._lock, shelve.open(self.path) as db:
                return db.get(self._key(provider, model, prompt, variant=variant))
        except Exception:
            logger.debug("Cache read failed for %s:%s", provider, model, exc_info=True)
            return None

    def set(self, provider: str, model: str, prompt: str, value: dict, variant: str = "") -> None:
        """Store a payload in the persistent cache."""
        try:
            with self._lock, shelve.open(self.path) as db:
                db[self._key(provider, model, prompt, variant=variant)] = value
        except Exception:
            logger.debug("Cache write failed for %s:%s", provider, model, exc_info=True)


_cache = ResultCache()


# ── Provider detection ────────────────────────────────────────────

def provider_from_model(model: str) -> str:
    """Determine the provider from a model name string."""
    m = (model or "").lower().strip()
    if m.startswith("ollama:") or m.startswith("ollama/"):
        return "ollama"
    if "gpt" in m or "o1" in m or "o3" in m or "o4" in m:
        return "openai"
    if "gemini" in m:
        return "gemini"
    if "grok" in m:
        return "xai"
    if any(k in m for k in ("claude", "opus", "sonnet", "haiku")):
        return "anthropic"
    return "unknown"


# ── Client registry ───────────────────────────────────────────────

_clients: dict[str, dict[str, Any]] = {
    p: {"client": None, "created_at": 0.0}
    for p in ["openai", "gemini", "xai", "anthropic"]
}
_clients_lock = threading.Lock()

def _now() -> float:
    """Return monotonic time for age/expiry checks."""
    return time.monotonic()

def _expired(ts: float) -> bool:
    """Return True when a cached client timestamp is stale."""
    return ts == 0.0 or (_now() - ts) > CLIENT_MAX_AGE_S


# ── Retry / backoff helpers ───────────────────────────────────────

def _sleep_with_backoff(attempt: int) -> None:
    """Sleep using exponential backoff for retry attempts."""
    delay = min(RETRY_BACKOFF_BASE ** (attempt - 1), RETRY_BACKOFF_MAX_S)
    time.sleep(delay)


def _is_retryable_generic_error(e: Exception) -> bool:
    """Heuristically classify whether a generic provider error is retryable."""
    if isinstance(e, (socket.timeout, OSError, ConnectionError, urllib.error.URLError)):
        return True
    msg = str(e).lower()
    for p in NON_RETRYABLE_ERROR_PATTERNS:
        if p in msg:
            return False
    return any(p in msg for p in RETRYABLE_ERROR_PATTERNS)


def _is_rate_limit_error(e: Exception) -> bool:
    """Return True when an exception indicates provider rate limiting."""
    if isinstance(e, RateLimitError):
        return True
    msg = str(e).lower()
    return any(p in msg for p in ("rate limit", "429", "too many requests", "throttl"))


def _cache_variant(max_output_tokens: int | None = None, temperature: float | None = None) -> str:
    """Build a cache-variant suffix for request-shaping parameters."""
    parts: list[str] = []
    if max_output_tokens is not None:
        parts.append(f"max={int(max_output_tokens)}")
    if temperature is not None:
        parts.append(f"temp={float(temperature):.3f}")
    return "|".join(parts)


# ── Reachability ──────────────────────────────────────────────────

def is_online(host: str = "api.openai.com", port: int = 443, timeout: float = 3.0) -> bool:
    """Return True if a TCP connection to host/port succeeds within timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ── Available models ──────────────────────────────────────────────

ALL_MODELS: list[str] = [
    "gpt-5.2",
    "gemini-3-pro-preview",
    "grok-4-1-fast-reasoning",
    "claude-opus-4-6",
]


# ── Ollama helpers ────────────────────────────────────────────────

def list_ollama_models(timeout_s: float = 3.0) -> list[dict[str, Any]]:
    """Query the Ollama server for installed models.

    Returns a list of dicts, each with at least ``name`` and ``size``.
    Falls back to an empty list if Ollama is unreachable.
    """
    try:
        url = f"{OLLAMA_BASE_URL}/api/tags"
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
        data = json.loads(body)
        models = data.get("models", [])
        result: list[dict[str, Any]] = []
        for m in models:
            name = m.get("name", m.get("model", ""))
            if not name:
                continue
            # Normalise: strip ":latest" tag for display
            display = name.replace(":latest", "")
            size_bytes = m.get("size", 0)
            size_gb = round(size_bytes / (1024 ** 3), 1) if size_bytes else 0
            result.append({
                "name": name,
                "display": display,
                "ollama_id": f"ollama:{name}",
                "size_gb": size_gb,
                "parameter_size": m.get("details", {}).get("parameter_size", ""),
                "family": m.get("details", {}).get("family", ""),
            })
        return result
    except Exception:
        logger.debug("Failed to list Ollama models", exc_info=True)
        return []


def get_default_ollama_model() -> str:
    """Return the first installed Ollama model, or the configured free model."""
    models = list_ollama_models()
    if models:
        return models[0]["ollama_id"]
    return DEFAULT_FREE_MODEL


def _is_ollama_running(timeout_s: float = 2.0) -> bool:
    global _ollama_last_check, _ollama_last_ok
    now = time.time()
    if now - _ollama_last_check < 5:
        return _ollama_last_ok
    _ollama_last_check = now
    try:
        url = f"{OLLAMA_BASE_URL}/api/tags"
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            _ollama_last_ok = resp.status == 200
            return _ollama_last_ok
    except Exception:
        logger.debug("Ollama health check failed", exc_info=True)
        _ollama_last_ok = False
        return False


def _maybe_start_ollama() -> None:
    if not OLLAMA_AUTO_START:
        return
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.DETACHED_PROCESS if os.name == "nt" else 0,
        )
        time.sleep(0.5)
    except Exception:
        logger.debug("Failed to auto-start Ollama", exc_info=True)


# ══════════════════════════════════════════════════════════════════
# Provider connectors
# ══════════════════════════════════════════════════════════════════

# ── Ollama (local) ────────────────────────────────────────────────

def _connect_ollama(
    model: str, prompt: str, text_only: bool, timeout_s: float, *,
    disable_cache: bool = False, max_output_tokens: int | None = None,
    temperature: float | None = None,
) -> Any:
    variant = _cache_variant(max_output_tokens=max_output_tokens, temperature=temperature)
    if not disable_cache:
        cached = _cache.get("ollama", model, prompt, variant=variant)
        if cached and "data" in cached:
            return cached["data"]

    last_exc: Exception | None = None
    for attempt in range(1, OLLAMA_MAX_ATTEMPTS + 1):
        try:
            url = f"{OLLAMA_BASE_URL}/api/generate"
            payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
            if max_output_tokens is not None:
                payload["num_predict"] = max_output_tokens
            if temperature is not None:
                payload.setdefault("options", {})["temperature"] = float(temperature)
            data = json.dumps(payload).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
            obj = json.loads(body)
            if isinstance(obj, dict) and obj.get("error"):
                raise RuntimeError(f"Ollama error: {obj['error']}")
            text = obj.get("response", obj) if isinstance(obj, dict) else obj
            data_out = text if text_only else obj
            if isinstance(data_out, str):
                data_out = strip_json_wrappers(data_out)
            if not disable_cache:
                _cache.set("ollama", model, prompt, {"data": data_out}, variant=variant)
            return data_out
        except Exception as e:
            last_exc = e
            if attempt >= OLLAMA_MAX_ATTEMPTS or not _is_retryable_generic_error(e):
                raise
            _maybe_start_ollama()
            _sleep_with_backoff(attempt)
    if last_exc is not None:
        raise last_exc


# ── OpenAI ────────────────────────────────────────────────────────

def _get_openai_client(force_new: bool = False) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai SDK not installed. Run: pip install openai")
    with _clients_lock:
        ent = _clients["openai"]
        if force_new or ent["client"] is None or _expired(ent["created_at"]):
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY not set")
            ent["client"] = OpenAI(api_key=key)
            ent["created_at"] = _now()
        return ent["client"]


def _connect_openai(
    model: str, prompt: str, text_only: bool, timeout_s: float, *,
    disable_cache: bool = False, max_output_tokens: int | None = None,
    temperature: float | None = None,
) -> Any:
    variant = _cache_variant(max_output_tokens=max_output_tokens, temperature=temperature)
    if not disable_cache:
        cached = _cache.get("openai", model, prompt, variant=variant)
        if cached and "data" in cached:
            return cached["data"]

    last_exc: Exception | None = None
    for attempt in range(1, OPENAI_MAX_ATTEMPTS + 1):
        try:
            client = _get_openai_client()
            try:
                idem_key = hashlib.sha256(f"openai|{model}|{prompt}".encode()).hexdigest()
                c2 = client.with_options(timeout=timeout_s, headers={"Idempotency-Key": idem_key})
            except Exception:
                c2 = client
            create_kwargs: dict[str, Any] = {"model": model, "input": prompt}
            if max_output_tokens is not None:
                create_kwargs["max_output_tokens"] = max_output_tokens
            if temperature is not None:
                create_kwargs["temperature"] = float(temperature)
            resp = c2.responses.create(**create_kwargs)
            data = getattr(resp, "output_text", resp) if text_only else resp
            if not disable_cache:
                _cache.set("openai", model, prompt, {"data": data}, variant=variant)
            return data
        except Exception as e:
            last_exc = e
            retryable = isinstance(e, (APIConnectionError, APITimeoutError, RateLimitError))
            if isinstance(e, APIStatusError):
                sc = getattr(e, "status_code", None)
                retryable = retryable or (sc is not None and int(sc) >= 500)
            if not retryable or attempt == OPENAI_MAX_ATTEMPTS:
                raise
            _sleep_with_backoff(attempt)
    if last_exc is not None:
        raise last_exc


# ── Gemini (Google) ───────────────────────────────────────────────

def _connect_gemini(
    model: str, prompt: str, text_only: bool, timeout_s: float, *,
    disable_cache: bool = False, max_output_tokens: int | None = None,
    temperature: float | None = None,
) -> Any:
    if not _GEMINI_SDK:
        raise RuntimeError("google-generativeai not installed. Run: pip install google-generativeai")

    variant = _cache_variant(max_output_tokens=max_output_tokens, temperature=temperature)
    if not disable_cache:
        cached = _cache.get("gemini", model, prompt, variant=variant)
        if cached and "data" in cached:
            return cached["data"]

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) not set")

    last_exc: Exception | None = None
    for attempt in range(1, GEMINI_MAX_ATTEMPTS + 1):
        try:
            genai.configure(api_key=api_key)
            gen_config: dict[str, Any] = {}
            if max_output_tokens is not None:
                gen_config["max_output_tokens"] = max_output_tokens
            if temperature is not None:
                gen_config["temperature"] = float(temperature)
            gm = genai.GenerativeModel(model, generation_config=gen_config or None)
            resp = gm.generate_content(prompt, request_options={"timeout": timeout_s})
            text = resp.text if hasattr(resp, "text") else str(resp)
            data = strip_json_wrappers(text) if text_only else resp
            if not disable_cache:
                _cache.set("gemini", model, prompt, {"data": data}, variant=variant)
            return data
        except Exception as e:
            last_exc = e
            if attempt >= GEMINI_MAX_ATTEMPTS or not _is_retryable_generic_error(e):
                raise
            _sleep_with_backoff(attempt)
    if last_exc is not None:
        raise last_exc


# ── xAI (Grok) ───────────────────────────────────────────────────

def _connect_xai(
    model: str, prompt: str, text_only: bool, timeout_s: float, *,
    disable_cache: bool = False, max_output_tokens: int | None = None,
    temperature: float | None = None,
) -> Any:
    variant = _cache_variant(max_output_tokens=max_output_tokens, temperature=temperature)
    if not disable_cache:
        cached = _cache.get("xai", model, prompt, variant=variant)
        if cached and "data" in cached:
            return cached["data"]

    api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY (or GROK_API_KEY) not set")

    # xAI uses an OpenAI-compatible API
    if OpenAI is None:
        raise RuntimeError("openai SDK required for xAI. Run: pip install openai")

    last_exc: Exception | None = None
    for attempt in range(1, XAI_MAX_ATTEMPTS + 1):
        try:
            client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
            create_kwargs: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if max_output_tokens is not None:
                create_kwargs["max_tokens"] = max_output_tokens
            if temperature is not None:
                create_kwargs["temperature"] = float(temperature)
            resp = client.chat.completions.create(**create_kwargs)
            text = resp.choices[0].message.content if resp.choices else ""
            data = text if text_only else resp
            if not disable_cache:
                _cache.set("xai", model, prompt, {"data": data}, variant=variant)
            return data
        except Exception as e:
            last_exc = e
            if attempt >= XAI_MAX_ATTEMPTS or not _is_retryable_generic_error(e):
                raise
            _sleep_with_backoff(attempt)
    if last_exc is not None:
        raise last_exc


# ── Anthropic (Claude) ────────────────────────────────────────────

_ANTHROPIC_AVAILABLE = True
try:
    from anthropic import Anthropic  # type: ignore[import-untyped]
except Exception:
    _ANTHROPIC_AVAILABLE = False


def _connect_anthropic(
    model: str, prompt: str, text_only: bool, timeout_s: float, *,
    disable_cache: bool = False, max_output_tokens: int | None = None,
    temperature: float | None = None,
) -> Any:
    if not _ANTHROPIC_AVAILABLE:
        raise RuntimeError("anthropic SDK not installed. Run: pip install anthropic")

    variant = _cache_variant(max_output_tokens=max_output_tokens, temperature=temperature)
    if not disable_cache:
        cached = _cache.get("anthropic", model, prompt, variant=variant)
        if cached and "data" in cached:
            return cached["data"]

    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    # Resolve short-name aliases
    m_lower = model.lower()
    if "claude" not in m_lower:
        if "opus" in m_lower:
            model = os.getenv("ANTHROPIC_OPUS_MODEL", "claude-3-opus-latest")
        elif "sonnet" in m_lower:
            model = os.getenv("ANTHROPIC_SONNET_MODEL", "claude-3-5-sonnet-latest")
        elif "haiku" in m_lower:
            model = os.getenv("ANTHROPIC_HAIKU_MODEL", "claude-3-haiku-latest")

    max_tokens = max_output_tokens or int(os.getenv("ANTHROPIC_MAX_TOKENS", "64000"))

    last_exc: Exception | None = None
    for attempt in range(1, ANTHROPIC_MAX_ATTEMPTS + 1):
        try:
            client = Anthropic(api_key=key)
            msg_kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if temperature is not None:
                msg_kwargs["temperature"] = float(temperature)
            msg = client.messages.create(**msg_kwargs)
            parts = [getattr(p, "text", "") for p in (getattr(msg, "content", []) or [])]
            full_text = "".join(parts).strip() or str(msg)
            data = strip_json_wrappers(full_text) if text_only else msg
            if not disable_cache:
                _cache.set("anthropic", model, prompt, {"data": data}, variant=variant)
            return data
        except Exception as e:
            last_exc = e
            if attempt >= ANTHROPIC_MAX_ATTEMPTS or not _is_retryable_generic_error(e):
                raise
            _sleep_with_backoff(attempt)
    if last_exc is not None:
        raise last_exc


# ══════════════════════════════════════════════════════════════════
# Main dispatcher
# ══════════════════════════════════════════════════════════════════

def connect(
    model: str,
    prompt: str,
    text_only: bool | None = None,
    per_request_timeout: float | None = None,
    *,
    operation: str | None = None,
    stage: str | None = None,
    disable_cache: bool = False,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
) -> Any:
    """Connect to an AI model and return its response.

    This is the single entry point for all AI calls in the manager.

    Parameters
    ----------
    model:
        Model identifier.  Prefix with ``ollama:`` for local models.
    prompt:
        The prompt text.
    text_only:
        If True, return only text (strip SDK response objects).
    per_request_timeout:
        HTTP timeout in seconds.
    operation / stage:
        Optional metadata for logging.
    disable_cache:
        Skip the persistent cache.
    max_output_tokens:
        Limit response length.
    temperature:
        Sampling temperature (0.0-2.0).
    """
    text_only = DEFAULT_TEXT_ONLY if text_only is None else bool(text_only)
    per_request_timeout = DEFAULT_PER_REQUEST_TIMEOUT_S if per_request_timeout is None else float(per_request_timeout)
    m = model.lower().strip()

    try:
        if m.startswith("ollama:") or m.startswith("ollama/"):
            if not _is_ollama_running():
                _maybe_start_ollama()
            if not _is_ollama_running():
                raise RuntimeError("Ollama server not running. Start with 'ollama serve'.")
            local_model = model.split(":", 1)[1] if ":" in model else model.split("/", 1)[1]
            return _connect_ollama(local_model, prompt, text_only, per_request_timeout,
                                   disable_cache=disable_cache, max_output_tokens=max_output_tokens, temperature=temperature)
        elif "gpt" in m or "o1" in m or "o3" in m or "o4" in m:
            return _connect_openai(model, prompt, text_only, per_request_timeout,
                                   disable_cache=disable_cache, max_output_tokens=max_output_tokens, temperature=temperature)
        elif "gemini" in m:
            return _connect_gemini(model, prompt, text_only, per_request_timeout,
                                   disable_cache=disable_cache, max_output_tokens=max_output_tokens, temperature=temperature)
        elif "grok" in m:
            return _connect_xai(model, prompt, text_only, per_request_timeout,
                                disable_cache=disable_cache, max_output_tokens=max_output_tokens, temperature=temperature)
        elif any(k in m for k in ("claude", "opus", "sonnet", "haiku")):
            return _connect_anthropic(model, prompt, text_only, per_request_timeout,
                                      disable_cache=disable_cache, max_output_tokens=max_output_tokens, temperature=temperature)
        else:
            raise ValueError(f"Unsupported model '{model}'. Supported: GPT, Gemini, Grok, Claude, ollama:<model>")
    except Exception as e:
        if recovery_hints is not None:
            try:
                error = recovery_hints.format_error_with_hint(str(e), stage=stage or operation)
                raise RuntimeError(error) from e
            except Exception:
                pass
        raise


def prompt_all(
    prompt: str,
    models: list[str] | None = None,
    text_only: bool | None = None,
    per_request_timeout: float | None = None,
) -> list[dict[str, Any]]:
    """Prompt multiple models in parallel and collect results."""
    text_only = DEFAULT_TEXT_ONLY if text_only is None else text_only
    per_request_timeout = DEFAULT_PER_REQUEST_TIMEOUT_S if per_request_timeout is None else float(per_request_timeout)
    models = models or ALL_MODELS

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(models)) as ex:
        futures = {ex.submit(connect, m, prompt, text_only, per_request_timeout): m for m in models}
        for fut in as_completed(futures):
            model = futures[fut]
            try:
                data = fut.result(timeout=per_request_timeout + 30.0)
                results.append({"model": model, "ok": True, "data": data})
            except FuturesTimeoutError:
                results.append({"model": model, "ok": False, "error": "timeout"})
            except Exception as e:
                results.append({"model": model, "ok": False, "error": str(e)})

    order = {m: i for i, m in enumerate(models)}
    results.sort(key=lambda r: order.get(r["model"], 999))
    return results
