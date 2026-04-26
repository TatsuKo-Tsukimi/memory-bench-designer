from __future__ import annotations

from pathlib import Path

from memory_bench.adapters.bm25 import BM25Adapter
from memory_bench.profile import ScenarioProfile
from memory_bench.trace import TraceConfig, replay_trace


def test_trace_replay_scores_update_and_forgetting_fixture():
    trace_path = (
        Path(__file__).parents[1]
        / "configs"
        / "traces"
        / "coding-architecture-update.yaml"
    )
    trace = TraceConfig.from_yaml(trace_path)
    profile = ScenarioProfile(
        name="trace-test",
        priorities={"update_coherence": 1.0, "selective_forgetting": 1.0},
    )

    results = replay_trace(trace, adapter_factory=lambda: [BM25Adapter()], profile=profile)

    assert "bm25" in results.per_adapter
    assert "retrieval_recall" in results.per_adapter["bm25"]
    assert "update_coherence" in results.per_adapter["bm25"]
    assert "selective_forgetting" in results.per_adapter["bm25"]
    assert results.scenario_fit["bm25"].n == 1
    assert results.pareto_frontier == ["bm25"]
