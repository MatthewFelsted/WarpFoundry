"""Tests for secret-safe prompt logging helpers."""

from __future__ import annotations

from codex_manager.prompt_logging import (
    format_prompt_log_line,
    format_prompt_preview,
    prompt_metadata,
    redact_sensitive_text,
)


def test_redact_sensitive_text_masks_representative_secret_patterns() -> None:
    openai_like = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz123456"
    github_like = "ghp_abcdefghijklmnopqrstuvwxyz123456"
    jwt_like = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkFsaWNlIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    raw = (
        f"api_key={openai_like} password=hunter2 "
        f"Authorization: Bearer {github_like} token={jwt_like}"
    )

    redacted, hits = redact_sensitive_text(raw)

    assert "hunter2" not in redacted
    assert openai_like not in redacted
    assert github_like not in redacted
    assert jwt_like not in redacted
    assert "[REDACTED]" in redacted
    assert "[REDACTED_TOKEN]" in redacted
    assert hits >= 3


def test_format_prompt_log_line_defaults_to_metadata_only() -> None:
    secret = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz123456"
    prompt = f"Deploy with api_key={secret}"

    line = format_prompt_log_line(prompt, debug=False)
    meta = prompt_metadata(prompt)

    assert line.startswith("Prompt metadata: ")
    assert secret not in line
    assert f"len={meta['length_chars']}" in line
    assert f"sha256={meta['sha256']}" in line
    assert f"redaction_hits={meta['redaction_hits']}" in line
    assert meta["redaction_hits"] >= 1


def test_format_prompt_helpers_allow_debug_opt_in_raw_text() -> None:
    prompt = "Use token=sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz123456 for auth."

    assert prompt in format_prompt_log_line(prompt, debug=True)
    assert format_prompt_preview(prompt, debug=True) == prompt

    safe_preview = format_prompt_preview(prompt, debug=False)
    assert safe_preview.startswith("[metadata-only] ")
    assert prompt not in safe_preview
