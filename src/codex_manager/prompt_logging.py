"""Helpers for prompt logging with default secret-safe behavior."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from collections import Counter
from typing import Final

_PROMPT_DEBUG_ENV: Final[str] = "CODEX_MANAGER_PROMPT_DEBUG"
_PROMPT_DEBUG_HINT: Final[str] = f"set {_PROMPT_DEBUG_ENV}=1 to include full prompt text"

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}

_ASSIGNMENT_SECRET_RE = re.compile(
    r"(?i)\b("
    r"api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|"
    r"private[_-]?key|authorization|auth[_-]?token|token|secret|password|passwd"
    r")\b(\s*[:=]\s*)([^\s,;]+)"
)
_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|secret|password|key)=)([^&\s]+)"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._\-+/=]{10,})")
_JWT_RE = re.compile(r"\b[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")

_KNOWN_SECRET_TOKEN_RES = (
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
)

_HIGH_ENTROPY_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_\-+/=]{24,}\b")


def _shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    length = len(value)
    counts = Counter(value)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def _looks_like_secret_token(token: str) -> bool:
    if len(token) < 24:
        return False
    lower = token.lower()
    if lower.startswith(("http", "www")):
        return False
    if re.fullmatch(r"[0-9a-f]{24,}", lower):
        return len(token) >= 32
    if not any(ch.isalpha() for ch in token):
        return False
    if not any(ch.isdigit() for ch in token):
        return False
    entropy = _shannon_entropy(token)
    unique_ratio = len(set(token)) / len(token)
    return entropy >= 3.3 and unique_ratio >= 0.35


def is_prompt_debug_enabled() -> bool:
    """Return true when full prompt logging is explicitly enabled."""
    raw = os.getenv(_PROMPT_DEBUG_ENV, "").strip().lower()
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    return logging.getLogger().isEnabledFor(logging.DEBUG)


def redact_sensitive_text(text: str) -> tuple[str, int]:
    """Redact likely secrets and token-like substrings in text."""
    redacted = str(text or "")
    if not redacted:
        return "", 0

    hits = 0
    redacted, count = _ASSIGNMENT_SECRET_RE.subn(r"\1\2[REDACTED]", redacted)
    hits += count

    redacted, count = _QUERY_SECRET_RE.subn(r"\1[REDACTED]", redacted)
    hits += count

    redacted, count = _BEARER_RE.subn("Bearer [REDACTED]", redacted)
    hits += count

    redacted, count = _JWT_RE.subn("[REDACTED_TOKEN]", redacted)
    hits += count

    for pattern in _KNOWN_SECRET_TOKEN_RES:
        redacted, count = pattern.subn("[REDACTED_TOKEN]", redacted)
        hits += count

    def _entropy_replace(match: re.Match[str]) -> str:
        nonlocal hits
        token = match.group(0)
        if _looks_like_secret_token(token):
            hits += 1
            return "[REDACTED_TOKEN]"
        return token

    redacted = _HIGH_ENTROPY_TOKEN_RE.sub(_entropy_replace, redacted)
    return redacted, hits


def prompt_metadata(prompt: str) -> dict[str, int | str]:
    """Return compact metadata used for prompt-safe runtime logging."""
    text = str(prompt or "")
    _, redaction_hits = redact_sensitive_text(text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return {
        "length_chars": len(text),
        "sha256": digest,
        "redaction_hits": redaction_hits,
    }


def format_prompt_log_line(prompt: str, *, label: str = "Prompt", debug: bool | None = None) -> str:
    """Format a runtime log line for prompt logging."""
    text = str(prompt or "")
    debug_enabled = is_prompt_debug_enabled() if debug is None else bool(debug)
    if debug_enabled:
        return f"{label}: {text}"
    meta = prompt_metadata(text)
    return (
        f"{label} metadata: len={meta['length_chars']}, sha256={meta['sha256']}, "
        f"redaction_hits={meta['redaction_hits']} ({_PROMPT_DEBUG_HINT})"
    )


def format_prompt_preview(prompt: str, *, debug: bool | None = None) -> str:
    """Format follow-up prompt preview text for logs and records."""
    text = str(prompt or "")
    debug_enabled = is_prompt_debug_enabled() if debug is None else bool(debug)
    if debug_enabled:
        return text
    meta = prompt_metadata(text)
    return (
        "[metadata-only] "
        f"len={meta['length_chars']}, sha256={meta['sha256']}, "
        f"redaction_hits={meta['redaction_hits']}"
    )
