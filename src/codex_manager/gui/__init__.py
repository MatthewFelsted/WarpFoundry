"""Codex Manager GUI - web-based task-chain builder and execution monitor."""

from __future__ import annotations


def main(port: int = 5088, open_browser: bool = True) -> None:
    """Launch the Codex Manager GUI."""
    from codex_manager.gui.app import run_gui

    run_gui(port=port, open_browser_=open_browser)
