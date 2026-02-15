"""Tests for project vector-memory fallback behavior."""

from __future__ import annotations

from pathlib import Path

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
