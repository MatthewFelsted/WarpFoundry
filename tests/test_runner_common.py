"""Tests for shared runner helpers."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from codex_manager.runner_common import coerce_int, resolve_binary


def _make_executable(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    if os.name == "nt":
        path.write_text("@echo off\r\nexit /b 0\r\n", encoding="utf-8")
    else:
        path.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_coerce_int_handles_non_finite_floats() -> None:
    assert coerce_int(float("nan")) == 0
    assert coerce_int(float("inf")) == 0
    assert coerce_int(float("-inf")) == 0


def test_resolve_binary_expands_environment_variables(monkeypatch, tmp_path: Path) -> None:
    var_name = "CODEX_MANAGER_TEST_BIN_DIR"
    monkeypatch.setenv(var_name, str(tmp_path))

    if os.name == "nt":
        tool = _make_executable(tmp_path, "codex-tool.cmd")
        configured = f"%{var_name}%\\{tool.name}"
    else:
        tool = _make_executable(tmp_path, "codex-tool")
        configured = f"${var_name}/{tool.name}"

    resolved = resolve_binary(configured)
    assert Path(resolved).resolve() == tool.resolve()

