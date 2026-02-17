"""Shared pytest configuration for marker registration and execution ordering."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "unit: fast isolated unit tests")
    config.addinivalue_line("markers", "integration: filesystem/subprocess integration tests")
    config.addinivalue_line("markers", "slow: expensive tests that may call external APIs")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Run fast unit tests first, integration tests second, slow tests last."""

    def sort_key(item: pytest.Item) -> tuple[int, str]:
        if item.get_closest_marker("slow"):
            return (2, item.nodeid)
        if item.get_closest_marker("integration"):
            return (1, item.nodeid)
        return (0, item.nodeid)

    items.sort(key=sort_key)
