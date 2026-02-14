"""Tests for prompt optimizer scoring, branching, and catalog updates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from codex_manager.prompts.optimizer import OptimizationResult, PromptOptimizer


class _StubCatalog:
    def __init__(
        self,
        *,
        prompts: list[dict[str, str]],
        raw: dict[str, Any],
        optimizer_prompts: dict[str, str] | None = None,
    ) -> None:
        self._prompts = prompts
        self.raw = raw
        self._optimizer_prompts = optimizer_prompts or {}
        self.saved = False

    def all_prompts(self) -> list[dict[str, str]]:
        return list(self._prompts)

    def optimizer(self, key: str) -> str:
        return self._optimizer_prompts.get(key, "")

    def save(self) -> Path:
        self.saved = True
        return Path("saved.yaml")


def _single_prompt_catalog(
    original: str = "Original prompt content that should be improved for better quality.",
) -> _StubCatalog:
    return _StubCatalog(
        prompts=[{"path": "pipeline.ideation", "name": "Ideation", "content": original}],
        raw={"pipeline": {"ideation": {"prompt": original}}},
    )


@pytest.mark.parametrize(
    ("raw", "expected_overall"),
    [
        ('{"overall": 8.5, "scores": {"clarity": 9}}', 8.5),
        ('```json\n{"overall": 7}\n```', 7),
        ('prefix text\n{"overall": 6}\nsuffix', 6),
        ("not valid json", 5.0),
    ],
)
def test_parse_scores_handles_json_variants(raw: str, expected_overall: float) -> None:
    parsed = PromptOptimizer._parse_scores(raw)
    assert parsed["overall"] == expected_overall


def test_apply_to_catalog_updates_nested_and_direct_fields() -> None:
    raw = {
        "pipeline": {"ideation": {"prompt": "old pipeline prompt"}},
        "brain": {"plan_step": {"system": "old brain system"}},
        "presets": {"testing": {"prompt": "old prompt", "ai_prompt": "old ai"}},
    }
    catalog = _StubCatalog(prompts=[], raw=raw)
    optimizer = PromptOptimizer(catalog=catalog)

    optimizer._apply_to_catalog("pipeline.ideation", "new pipeline prompt")
    optimizer._apply_to_catalog("brain.plan_step", "new brain system")
    optimizer._apply_to_catalog("presets.testing.prompt", "new testing prompt")
    optimizer._apply_to_catalog("presets.testing.ai_prompt", "new testing ai")
    optimizer._apply_to_catalog("missing.path", "ignored")

    assert raw["pipeline"]["ideation"]["prompt"] == "new pipeline prompt"
    assert raw["brain"]["plan_step"]["system"] == "new brain system"
    assert raw["presets"]["testing"]["prompt"] == "new testing prompt"
    assert raw["presets"]["testing"]["ai_prompt"] == "new testing ai"


def test_optimize_all_skips_when_evaluation_fails(monkeypatch) -> None:
    catalog = _single_prompt_catalog()

    def fake_connect(*_args, **_kwargs):
        raise RuntimeError("evaluation blew up")

    monkeypatch.setattr("codex_manager.brain.connector.connect", fake_connect)
    optimizer = PromptOptimizer(catalog=catalog, threshold=7.5)
    results = optimizer.optimize_all(dry_run=True)

    assert len(results) == 1
    assert results[0].skipped is True
    assert "Evaluation failed" in results[0].skip_reason
    assert catalog.saved is False


def test_optimize_all_skips_when_already_above_threshold(monkeypatch) -> None:
    catalog = _single_prompt_catalog()
    calls = {"count": 0}

    def fake_connect(*_args, **_kwargs):
        calls["count"] += 1
        return '{"overall": 8.1, "scores": {"clarity": 8}}'

    monkeypatch.setattr("codex_manager.brain.connector.connect", fake_connect)
    optimizer = PromptOptimizer(catalog=catalog, threshold=7.5)
    results = optimizer.optimize_all(dry_run=False)

    assert calls["count"] == 1
    assert results[0].skipped is True
    assert "threshold" in results[0].skip_reason
    assert results[0].optimized == results[0].original
    assert catalog.saved is True


def test_optimize_all_skips_when_optimization_fails(monkeypatch) -> None:
    catalog = _single_prompt_catalog()
    calls = {"count": 0}

    def fake_connect(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return '{"overall": 4.0, "scores": {}, "suggestions": ["be more specific"]}'
        raise RuntimeError("optimizer backend failed")

    monkeypatch.setattr("codex_manager.brain.connector.connect", fake_connect)
    optimizer = PromptOptimizer(catalog=catalog)
    results = optimizer.optimize_all(dry_run=True)

    assert results[0].skipped is True
    assert "Optimization failed" in results[0].skip_reason


def test_optimize_all_skips_when_optimized_response_is_too_short(monkeypatch) -> None:
    catalog = _single_prompt_catalog()
    responses = iter(
        [
            '{"overall": 4.5, "scores": {}, "suggestions": ["expand constraints"]}',
            "short",
        ]
    )

    def fake_connect(*_args, **_kwargs):
        return next(responses)

    monkeypatch.setattr("codex_manager.brain.connector.connect", fake_connect)
    optimizer = PromptOptimizer(catalog=catalog)
    results = optimizer.optimize_all(dry_run=True)

    assert results[0].skipped is True
    assert results[0].skip_reason == "Optimizer returned inadequate response"


def test_optimize_all_applies_improved_prompt_and_saves(monkeypatch) -> None:
    original = "Old prompt that is too vague and should be optimized by the model."
    optimized = "Optimized prompt with explicit structure, constraints, and expected output format."
    catalog = _single_prompt_catalog(original)
    responses = iter(
        [
            '{"overall": 4.2, "scores": {"clarity": 4}, "suggestions": ["add structure"]}',
            optimized,
            '{"overall": 8.8, "scores": {"clarity": 9}}',
        ]
    )

    def fake_connect(*_args, **_kwargs):
        return next(responses)

    monkeypatch.setattr("codex_manager.brain.connector.connect", fake_connect)
    optimizer = PromptOptimizer(catalog=catalog, threshold=7.5)
    results = optimizer.optimize_all(dry_run=False)

    assert results[0].improved is True
    assert results[0].optimized == optimized
    assert catalog.raw["pipeline"]["ideation"]["prompt"] == optimized
    assert catalog.saved is True


def test_optimize_all_dry_run_does_not_apply_or_save(monkeypatch) -> None:
    original = "Prompt before dry run optimization."
    optimized = "Prompt after optimization with enough details to pass the length threshold."
    catalog = _single_prompt_catalog(original)
    responses = iter(
        [
            '{"overall": 3.9, "scores": {}, "suggestions": []}',
            optimized,
            '{"overall": 8.0, "scores": {}}',
        ]
    )

    def fake_connect(*_args, **_kwargs):
        return next(responses)

    monkeypatch.setattr("codex_manager.brain.connector.connect", fake_connect)
    optimizer = PromptOptimizer(catalog=catalog)
    results = optimizer.optimize_all(dry_run=True)

    assert results[0].improved is True
    assert catalog.raw["pipeline"]["ideation"]["prompt"] == original
    assert catalog.saved is False


def test_optimize_all_handles_re_evaluation_failure_with_assumed_gain(monkeypatch) -> None:
    catalog = _single_prompt_catalog()
    calls = {"count": 0}
    optimized = "Long optimized prompt text that should pass minimum response length."

    def fake_connect(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return '{"overall": 4.0, "scores": {}}'
        if calls["count"] == 2:
            return optimized
        raise RuntimeError("re-evaluation failed")

    monkeypatch.setattr("codex_manager.brain.connector.connect", fake_connect)
    optimizer = PromptOptimizer(catalog=catalog)
    results = optimizer.optimize_all(dry_run=False)

    assert results[0].improved is True
    assert results[0].scores_after["overall"] == 5.0


def test_summary_reports_counts_and_status_labels() -> None:
    catalog = _single_prompt_catalog()
    optimizer = PromptOptimizer(catalog=catalog)
    optimizer._results = [
        OptimizationResult(
            path="pipeline.ideation",
            name="Ideation",
            original="a",
            optimized="b",
            scores_before={"overall": 4},
            scores_after={"overall": 8},
            improved=True,
        ),
        OptimizationResult(
            path="presets.testing.prompt",
            name="Testing",
            original="a",
            optimized="a",
            scores_before={"overall": 9},
            skipped=True,
            skip_reason="Already excellent",
        ),
    ]

    summary = optimizer.summary()
    assert "Total prompts: 2" in summary
    assert "Improved:      1" in summary
    assert "Skipped:       1" in summary
    assert "IMPROVED" in summary
    assert "SKIPPED" in summary
    assert "Already excellent" in summary
