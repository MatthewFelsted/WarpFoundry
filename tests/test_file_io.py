"""Tests for resilient file I/O helpers used by tracker and ledger writes."""

from __future__ import annotations

from pathlib import Path

import pytest

import codex_manager.file_io as file_io

pytestmark = pytest.mark.unit


def test_path_lock_reuses_same_lock_for_resolved_aliases(tmp_path: Path) -> None:
    primary = tmp_path / "logs" / "PROGRESS.md"
    alias = tmp_path / "logs" / ".." / "logs" / "PROGRESS.md"

    assert file_io._path_lock(primary) is file_io._path_lock(alias)


def test_replace_file_with_retry_retries_permission_denied_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("new-content", encoding="utf-8")
    dst.write_text("old-content", encoding="utf-8")

    attempts = {"count": 0}
    original_replace = Path.replace

    def flaky_replace(self: Path, target: Path) -> Path:
        if self == src and Path(target) == dst and attempts["count"] < 2:
            attempts["count"] += 1
            raise PermissionError("file locked")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)

    file_io._replace_file_with_retry(src, dst)

    assert attempts["count"] == 2
    assert dst.read_text(encoding="utf-8") == "new-content"


def test_replace_file_with_retry_raises_non_permission_oserror(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("x", encoding="utf-8")

    def fail_replace(_self: Path, _target: Path) -> Path:
        raise OSError(5, "io error")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError) as exc_info:
        file_io._replace_file_with_retry(src, dst)

    assert exc_info.value.errno == 5


def test_replace_file_with_retry_raises_last_permission_error_after_max_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("x", encoding="utf-8")

    sleep_calls: list[float] = []

    def always_locked(_self: Path, _target: Path) -> Path:
        raise PermissionError("still locked")

    monkeypatch.setattr(Path, "replace", always_locked)
    monkeypatch.setattr(file_io.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    with pytest.raises(PermissionError):
        file_io._replace_file_with_retry(src, dst)

    assert len(sleep_calls) == file_io._ATOMIC_REPLACE_MAX_RETRIES - 1


def test_atomic_write_text_replaces_existing_content(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "value.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("old", encoding="utf-8")

    file_io.atomic_write_text(path, "new")

    assert path.read_text(encoding="utf-8") == "new"


def test_atomic_write_text_cleans_temp_file_when_replace_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "logs" / "PROGRESS.md"

    monkeypatch.setattr(file_io, "_replace_file_with_retry", lambda _src, _dst: (_ for _ in ()).throw(PermissionError("busy")))

    with pytest.raises(PermissionError):
        file_io.atomic_write_text(path, "content")

    leftovers = list(path.parent.glob(f"{path.name}.*.tmp"))
    assert leftovers == []


def test_read_text_utf8_resilient_returns_empty_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"

    result = file_io.read_text_utf8_resilient(missing)

    assert result.text == ""
    assert result.used_fallback is False


@pytest.mark.parametrize(
    ("payload", "expected_text", "expected_decoder"),
    [
        (b"alpha\x97omega", "alpha\u2014omega", "cp1252"),
    ],
)
def test_read_text_utf8_resilient_decodes_and_normalizes_legacy_text(
    tmp_path: Path,
    payload: bytes,
    expected_text: str,
    expected_decoder: str,
) -> None:
    target = tmp_path / "legacy.txt"
    target.write_bytes(payload)

    result = file_io.read_text_utf8_resilient(target)

    assert result.text == expected_text
    assert result.used_fallback is True
    assert result.decoder == expected_decoder
    assert result.normalized_to_utf8 is True
    assert target.read_text(encoding="utf-8") == expected_text


def test_read_text_utf8_resilient_preserves_bom_when_plain_utf8_decode_succeeds(tmp_path: Path) -> None:
    target = tmp_path / "with-bom.txt"
    target.write_bytes(b"\xef\xbb\xbfhello")

    result = file_io.read_text_utf8_resilient(target)

    assert result.text == "\ufeffhello"
    assert result.used_fallback is False
    assert result.decoder == "utf-8"


def test_read_text_utf8_resilient_skips_rewrite_when_normalization_disabled(tmp_path: Path) -> None:
    target = tmp_path / "legacy.txt"
    payload = b"alpha\x97omega"
    target.write_bytes(payload)

    result = file_io.read_text_utf8_resilient(target, normalize_to_utf8=False)

    assert result.text == "alpha\u2014omega"
    assert result.normalized_to_utf8 is False
    assert target.read_bytes() == payload


def test_read_text_utf8_resilient_uses_replacement_when_all_fallbacks_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "bad.txt"
    target.write_bytes(b"\xff")

    class _RawBytes:
        def decode(self, encoding: str, errors: str = "strict") -> str:
            if encoding in {"utf-8-sig", "cp1252", "latin-1"}:
                raise UnicodeDecodeError(encoding, b"\xff", 0, 1, "bad sequence")
            if encoding == "utf-8" and errors == "replace":
                return "x\ufffdy"
            raise AssertionError(f"unexpected decode call: {encoding=} {errors=}")

    monkeypatch.setattr(Path, "read_bytes", lambda _self: _RawBytes())

    captured: dict[str, str] = {}

    def _capture_atomic_write(path: Path, content: str, *, encoding: str = "utf-8") -> None:
        captured["path"] = str(path)
        captured["content"] = content
        captured["encoding"] = encoding

    monkeypatch.setattr(file_io, "atomic_write_text", _capture_atomic_write)

    result = file_io.read_text_utf8_resilient(target)

    assert result.decoder == "utf-8-replace"
    assert result.used_replacement is True
    assert result.normalized_to_utf8 is True
    assert result.text == "x\ufffdy"
    assert captured["path"] == str(target)
    assert captured["content"] == "x\ufffdy"
    assert captured["encoding"] == "utf-8"
