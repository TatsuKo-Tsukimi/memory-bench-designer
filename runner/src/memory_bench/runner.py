"""Benchmark orchestrator — session x step x adapter main loop.

v0.2 runs a :class:`RandomRetrievalAdapter` as a shadow adapter alongside
the user's adapters. Every metric — including Coverage — gets an empirical
baseline from the shadow's score, and each user adapter's :class:`MetricResult`
is populated with ``baseline_score`` and a clamped ``normalized_score``.

Each run emits ``protocol_hash`` and ``scenario_hash`` so two result files
can be compared at a glance — matching hashes mean the numbers are on the
same evaluation harness.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

from memory_bench.adapters.base import Adapter
from memory_bench.adapters.random_retrieval import RandomRetrievalAdapter
from memory_bench.metrics.base import MetricAccumulator, MetricResult
from memory_bench.scenario.generator import (
    ProtocolConfig,
    ScenarioConfig,
    generate_scenario,
)


@dataclass
class BenchmarkResults:
    scenario_name: str
    protocol_hash: str
    scenario_hash: str
    protocol_meta: Dict[str, Any]
    scenario_meta: Dict[str, Any]
    baseline_scores: Dict[str, float]
    per_adapter: Dict[str, List[MetricResult]] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "protocol_hash": self.protocol_hash,
            "scenario_hash": self.scenario_hash,
            "protocol": self.protocol_meta,
            "scenario_meta": self.scenario_meta,
            "baseline_scores": {k: round(v, 4) for k, v in self.baseline_scores.items()},
            "results": {
                adapter: [_result_dict(r) for r in results]
                for adapter, results in self.per_adapter.items()
            },
        }


def _result_dict(r: MetricResult) -> Dict[str, Any]:
    return {
        "family": r.family,
        "dimension": r.dimension,
        "score": round(r.score, 4),
        "baseline_score": None if r.baseline_score is None else round(r.baseline_score, 4),
        "normalized_score": None if r.normalized_score is None else round(r.normalized_score, 4),
        "n_samples": r.n_samples,
    }


def _config_hash(obj: Any) -> str:
    canonical = json.dumps(dataclasses.asdict(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _normalize(score: float, baseline: float) -> float:
    denom = 1.0 - baseline
    if denom <= 1e-9:
        return 0.0
    return max(-1.0, min(1.0, (score - baseline) / denom))


def run_benchmark(
    protocol: ProtocolConfig,
    scenario: ScenarioConfig,
    adapters: List[Adapter],
    metric_factories: List[Type[MetricAccumulator]],
    hit_threshold: float = 0.5,
) -> BenchmarkResults:
    pool, stream = generate_scenario(protocol, scenario)

    # Shadow adapter sourcing empirical baselines for every metric.
    # Seed offset keeps its RNG independent of scenario-generation RNG.
    random_adapter = RandomRetrievalAdapter(seed=protocol.seed + 1)
    all_adapters: List[Adapter] = list(adapters) + [random_adapter]

    metrics_per_adapter: Dict[str, List[MetricAccumulator]] = {
        a.name: [M() for M in metric_factories] for a in all_adapters
    }

    for s in range(protocol.sessions):
        for t in range(protocol.steps_per_session):
            global_step = s * protocol.steps_per_session + t
            for item in pool.arrivals_at(s, t):
                for adapter in all_adapters:
                    adapter.observe(item, global_step)

            query = stream.at(s, t)
            for adapter in all_adapters:
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

    baseline_scores: Dict[str, float] = {
        m.dimension: m.score() for m in metrics_per_adapter[random_adapter.name]
    }

    per_adapter: Dict[str, List[MetricResult]] = {}
    for adapter in adapters:
        adapter_results: List[MetricResult] = []
        for metric in metrics_per_adapter[adapter.name]:
            r = metric.result()
            baseline = baseline_scores.get(r.dimension, 0.0)
            r.baseline_score = baseline
            r.normalized_score = _normalize(r.score, baseline)
            adapter_results.append(r)
        per_adapter[adapter.name] = adapter_results

    protocol_meta = dataclasses.asdict(protocol)
    scenario_meta = {
        "archetypes": scenario.archetypes,
        "context_evolution": scenario.context_evolution,
        "n_themes": len(scenario.themes),
        "arrivals": scenario.arrivals,
    }

    return BenchmarkResults(
        scenario_name=scenario.name,
        protocol_hash=_config_hash(protocol),
        scenario_hash=_config_hash(scenario),
        protocol_meta=protocol_meta,
        scenario_meta=scenario_meta,
        baseline_scores=baseline_scores,
        per_adapter=per_adapter,
    )
