"""Tests for project vector-memory fallback behavior."""

from __future__ import annotations

from pathlib import Path

import codex_manager.memory.vector_store as vector_store_module
from codex_manager.memory.vector_store import ProjectVectorMemory


def test_vector_memory_fallback_add_and_search(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    memory = ProjectVectorMemory(
        repo,
        enabled=True,
        backend="unsupported-backend",
        collection_name="test-memory",
        default_top_k=5,
    )
    assert memory.enabled is True
    assert memory.available is False

    memory.add_note(
        "Implemented robust retry logic for flaky network tests.",
        category="pipeline_phase",
        source="pipeline:testing",
    )
    memory.add_note(
        "Added onboarding visuals and improved README hero messaging.",
        category="marketing",
        source="chain:marketing_mode",
    )

    hits = memory.search("network retry strategy for tests", top_k=3)
    assert hits
    top = hits[0]
    assert "retry" in top.document.lower()
    assert top.score > 0


def test_deep_research_cache_reuse_lookup(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    memory = ProjectVectorMemory(repo, enabled=True, backend="unsupported")

    memory.record_deep_research(
        topic="pricing strategy for API usage tiers",
        summary="Found strong preference for usage-tier ladders with clear overage caps.",
        providers="both",
    )
    hit = memory.lookup_recent_deep_research(
        "api usage pricing tiers and overage strategy",
        max_age_hours=72,
        min_similarity=0.2,
    )
    assert hit is not None
    assert "pricing" in str(hit.get("topic", "")).lower()


def test_deep_research_lookup_reuses_cached_rows_without_reparsing(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    memory = ProjectVectorMemory(repo, enabled=True, backend="unsupported")

    memory.record_deep_research(
        topic="latency reduction for API retries",
        summary="Use jittered backoff plus per-endpoint timeout budgets.",
        providers="both",
    )

    first = memory.lookup_recent_deep_research(
        "api retry latency reduction",
        max_age_hours=72,
        min_similarity=0.1,
    )
    assert first is not None

    def _raise_on_json_loads(*_args, **_kwargs):
        raise AssertionError("json.loads should not run for cached lookup")

    monkeypatch.setattr(vector_store_module.json, "loads", _raise_on_json_loads)

    second = memory.lookup_recent_deep_research(
        "api retry latency reduction",
        max_age_hours=72,
        min_similarity=0.1,
    )
    assert second is not None
    assert second.get("id") == first.get("id")


def test_vector_search_reuses_cached_event_rows_without_reparsing(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    memory = ProjectVectorMemory(repo, enabled=True, backend="unsupported")

    memory.add_note("Implemented retry timeout and circuit breaker policy for network calls.")
    first_hits = memory.search("retry timeout policy", top_k=3)
    assert first_hits

    def _raise_on_json_loads(*_args, **_kwargs):
        raise AssertionError("json.loads should not run for cached fallback search")

    monkeypatch.setattr(vector_store_module.json, "loads", _raise_on_json_loads)

    second_hits = memory.search("retry timeout policy", top_k=3)
    assert second_hits
