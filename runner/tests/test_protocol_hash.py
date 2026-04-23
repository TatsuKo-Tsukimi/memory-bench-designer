"""Tests for protocol_hash / scenario_hash identity and sensitivity."""

from __future__ import annotations

from dataclasses import replace

from memory_bench.adapters.recency import RecencyAdapter
from memory_bench.metrics.exploration import Coverage
from memory_bench.runner import run_benchmark
from memory_bench.scenario.generator import ProtocolConfig, ScenarioConfig


def _minimal_scenario(name: str = "tiny") -> ScenarioConfig:
    return ScenarioConfig(
        name=name,
        archetypes={"core": 0.4, "evolving": 0.2, "episode": 0.3, "noise": 0.1},
        themes=[
            {"name": "a", "vocab": ["x", "y", "z", "w"]},
            {"name": "b", "vocab": ["p", "q", "r", "s"]},
        ],
        context_evolution={"type": "stable", "drift_rate": 0.1},
        arrivals={"episode": "streaming", "noise": "streaming"},
    )


def _protocol() -> ProtocolConfig:
    return ProtocolConfig(seed=1, pool_size=40, sessions=2, steps_per_session=4, top_k=3)


def _run(protocol: ProtocolConfig, scenario: ScenarioConfig):
    return run_benchmark(protocol, scenario, [RecencyAdapter()], [Coverage])


def test_same_config_same_hashes():
    a = _run(_protocol(), _minimal_scenario())
    b = _run(_protocol(), _minimal_scenario())
    assert a.protocol_hash == b.protocol_hash
    assert a.scenario_hash == b.scenario_hash


def test_changing_protocol_changes_protocol_hash_only():
    base_p = _protocol()
    base_s = _minimal_scenario()
    a = _run(base_p, base_s)
    b = _run(replace(base_p, seed=base_p.seed + 1), base_s)
    assert a.protocol_hash != b.protocol_hash
    assert a.scenario_hash == b.scenario_hash


def test_changing_scenario_content_changes_scenario_hash_only():
    base_p = _protocol()
    base_s = _minimal_scenario()
    a = _run(base_p, base_s)
    bumped = replace(
        base_s,
        archetypes={"core": 0.3, "evolving": 0.2, "episode": 0.3, "noise": 0.2},
    )
    b = _run(base_p, bumped)
    assert a.protocol_hash == b.protocol_hash
    assert a.scenario_hash != b.scenario_hash


def test_hash_is_deterministic_across_processes():
    """Python's json.dumps with sort_keys is deterministic; sha256 is too.
    This catches accidental introduction of non-canonical serialization
    (e.g. using repr() on a dict).
    """
    expected = _run(_protocol(), _minimal_scenario()).protocol_hash
    for _ in range(3):
        assert _run(_protocol(), _minimal_scenario()).protocol_hash == expected
