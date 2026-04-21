"""Benchmark orchestrator — session x step x adapter main loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

from memory_bench.adapters.base import Adapter
from memory_bench.metrics.base import MetricAccumulator, MetricResult
from memory_bench.scenario.generator import ScenarioConfig, generate_scenario


@dataclass
class BenchmarkResults:
    scenario_name: str
    scenario_meta: Dict[str, Any]
    per_adapter: Dict[str, List[MetricResult]] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "meta": self.scenario_meta,
            "results": {
                adapter: [
                    {
                        "family": r.family,
                        "dimension": r.dimension,
                        "score": round(r.score, 4),
                        "n_samples": r.n_samples,
                    }
                    for r in results
                ]
                for adapter, results in self.per_adapter.items()
            },
        }


def run_benchmark(
    cfg: ScenarioConfig,
    adapters: List[Adapter],
    metric_factories: List[Type[MetricAccumulator]],
    hit_threshold: float = 0.5,
) -> BenchmarkResults:
    pool, stream, cfg = generate_scenario(cfg)

    metrics_per_adapter: Dict[str, List[MetricAccumulator]] = {
        a.name: [M() for M in metric_factories] for a in adapters
    }

    for s in range(cfg.sessions):
        for t in range(cfg.steps_per_session):
            global_step = s * cfg.steps_per_session + t
            for item in pool.arrivals_at(s, t):
                for adapter in adapters:
                    adapter.observe(item, global_step)

            query = stream.at(s, t)
            for adapter in adapters:
                retrieval = adapter.retrieve(query, global_step)
                hits = [
                    pool.get(iid).affinity_to(query.theme) >= hit_threshold
                    for iid in retrieval.item_ids
                ]
                aff = [
                    pool.get(iid).affinity_to(query.theme) for iid in retrieval.item_ids
                ]
                adapter.record_retrieval(retrieval, global_step)
                adapter.record_feedback(retrieval, hits)
                for metric in metrics_per_adapter[adapter.name]:
                    metric.observe(adapter, retrieval, hits, aff, pool, global_step)

    results = BenchmarkResults(
        scenario_name=cfg.name,
        scenario_meta={
            "pool_size": cfg.pool_size,
            "sessions": cfg.sessions,
            "steps_per_session": cfg.steps_per_session,
            "archetypes": cfg.archetypes,
            "context_evolution": cfg.context_evolution,
            "seed": cfg.seed,
        },
    )
    for adapter in adapters:
        results.per_adapter[adapter.name] = [
            m.result() for m in metrics_per_adapter[adapter.name]
        ]
    return results
