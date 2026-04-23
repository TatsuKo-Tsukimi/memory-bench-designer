"""Tests for RandomRetrievalAdapter-derived baselines and normalized scores."""

from __future__ import annotations

import pytest

from memory_bench.adapters.recency import RecencyAdapter
from memory_bench.metrics.adaptation import CrossSessionLearning, Personalization
from memory_bench.metrics.exploration import Coverage, NoveltyGuarantee
from memory_bench.metrics.maintenance import ForgettingQuality, UpdateCoherence
from memory_bench.metrics.ranking import FrequencyGain, RelevanceAtK
from memory_bench.runner import run_benchmark
from memory_bench.scenario.generator import ProtocolConfig, ScenarioConfig


def _mini_protocol() -> ProtocolConfig:
    return ProtocolConfig(seed=7, pool_size=60, sessions=2, steps_per_session=8, top_k=3)


def _mini_scenario(noise_ratio: float = 0.10) -> ScenarioConfig:
    return ScenarioConfig(
        name="mini",
        archetypes={
            "core": 0.30,
            "evolving": 0.20,
            "episode": 0.40 - noise_ratio + 0.10,
            "noise": noise_ratio,
        },
        themes=[
            {"name": "t1", "vocab": ["alpha", "beta", "gamma", "delta", "eps"]},
            {"name": "t2", "vocab": ["one", "two", "three", "four", "five"]},
        ],
        context_evolution={"type": "random", "drift_rate": 0.1},
        arrivals={"episode": "streaming", "noise": "streaming"},
    )


ALL_METRICS = [
    NoveltyGuarantee,
    Coverage,
    RelevanceAtK,
    FrequencyGain,
    Personalization,
    CrossSessionLearning,
    UpdateCoherence,
    ForgettingQuality,
]


def _run():
    protocol = _mini_protocol()
    scenario = _mini_scenario()
    return run_benchmark(protocol, scenario, [RecencyAdapter()], ALL_METRICS)


def test_baselines_emitted_for_all_dimensions():
    results = _run()
    dims = {r.dimension for r in results.per_adapter["recency"]}
    assert dims == set(results.baseline_scores.keys())
    for score in results.baseline_scores.values():
        assert 0.0 <= score <= 1.0


def test_coverage_is_normalized_like_other_metrics():
    results = _run()
    cov = next(r for r in results.per_adapter["recency"] if r.dimension == "coverage")
    # Coverage is treated uniformly with every other metric — it has a
    # baseline and a clamped normalized score. The random adapter typically
    # saturates coverage, so most user adapters land at or near -1.0.
    assert cov.baseline_score is not None
    assert cov.normalized_score is not None
    assert -1.0 <= cov.normalized_score <= 1.0


def test_every_metric_has_normalized_score():
    results = _run()
    for r in results.per_adapter["recency"]:
        assert r.baseline_score is not None, r.dimension
        assert r.normalized_score is not None, r.dimension
        assert -1.0 <= r.normalized_score <= 1.0, (r.dimension, r.normalized_score)


def test_baseline_scores_non_degenerate():
    results = _run()
    # All baselines are finite floats within [0, 1] — the structural guarantee.
    for dim, score in results.baseline_scores.items():
        assert isinstance(score, float), dim
        assert 0.0 <= score <= 1.0, (dim, score)


def test_baselines_reproducible_under_same_seed():
    a = _run()
    b = _run()
    assert a.baseline_scores == b.baseline_scores
    assert a.protocol_hash == b.protocol_hash
    assert a.scenario_hash == b.scenario_hash
