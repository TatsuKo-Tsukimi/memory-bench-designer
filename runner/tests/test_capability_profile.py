from __future__ import annotations

from memory_bench.adapters.recency import RecencyAdapter
from memory_bench.metrics.exploration import Coverage, NoveltyGuarantee
from memory_bench.profile import ScenarioProfile
from memory_bench.runner import run_benchmark_suite
from memory_bench.scenario.generator import ProtocolConfig, ScenarioConfig


def _scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="profile-mini",
        archetypes={"core": 0.25, "evolving": 0.25, "episode": 0.35, "noise": 0.15},
        themes=[
            {"name": "a", "vocab": ["alpha", "beta", "gamma", "delta"]},
            {"name": "b", "vocab": ["one", "two", "three", "four"]},
        ],
        context_evolution={"type": "stable", "drift_rate": 0.1},
        arrivals={"episode": "streaming", "noise": "streaming"},
    )


def test_multiseed_suite_emits_capability_profile_and_manifest():
    protocol = ProtocolConfig(seed=1, pool_size=40, sessions=2, steps_per_session=5, top_k=3)
    profile = ScenarioProfile(name="mini", priorities={"exploration": 1.0})
    results = run_benchmark_suite(
        protocol=protocol,
        scenario=_scenario(),
        adapter_factory=lambda: [RecencyAdapter()],
        metric_factories=[NoveltyGuarantee, Coverage],
        seeds=[1, 2, 3],
        profile=profile,
    )

    assert results.seeds == [1, 2, 3]
    assert len(results.per_seed) == 3
    assert "recency" in results.family_stats
    assert "exploration" in results.family_stats["recency"]
    assert results.family_stats["recency"]["exploration"].n == 3
    assert results.scenario_fit["recency"].n == 3
    assert results.manifest["adapters"][0]["name"] == "recency"
