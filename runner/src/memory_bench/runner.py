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
import math
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Type

from memory_bench.adapters.base import Adapter
from memory_bench.adapters.random_retrieval import RandomRetrievalAdapter
from memory_bench.metrics.base import MetricAccumulator, MetricResult
from memory_bench.profile import ScenarioProfile
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


@dataclass
class StatSummary:
    mean: float
    std: float
    ci95: float
    min: float
    max: float
    n: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "ci95": round(self.ci95, 4),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "n": self.n,
        }


@dataclass
class BenchmarkSuiteResults:
    scenario_name: str
    protocol_template_hash: str
    scenario_hash: str
    profile_hash: str
    seeds: List[int]
    protocol_meta: Dict[str, Any]
    scenario_meta: Dict[str, Any]
    profile_meta: Dict[str, Any]
    manifest: Dict[str, Any]
    per_seed: List[BenchmarkResults]
    metric_stats: Dict[str, Dict[str, StatSummary]]
    family_stats: Dict[str, Dict[str, StatSummary]]
    scenario_fit: Dict[str, StatSummary]
    winner_stability: Dict[str, Dict[str, Any]]
    pareto_frontier: List[str]
    dominated: Dict[str, str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "protocol_template_hash": self.protocol_template_hash,
            "scenario_hash": self.scenario_hash,
            "profile_hash": self.profile_hash,
            "seeds": self.seeds,
            "protocol": self.protocol_meta,
            "scenario_meta": self.scenario_meta,
            "scenario_profile": self.profile_meta,
            "manifest": self.manifest,
            "per_seed": [r.as_dict() for r in self.per_seed],
            "metric_stats": {
                adapter: {dim: stat.as_dict() for dim, stat in dims.items()}
                for adapter, dims in self.metric_stats.items()
            },
            "family_stats": {
                adapter: {family: stat.as_dict() for family, stat in fams.items()}
                for adapter, fams in self.family_stats.items()
            },
            "scenario_fit": {
                adapter: stat.as_dict() for adapter, stat in self.scenario_fit.items()
            },
            "winner_stability": self.winner_stability,
            "pareto_frontier": self.pareto_frontier,
            "dominated": self.dominated,
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


def _dict_hash(obj: Dict[str, Any]) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
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


def run_benchmark_suite(
    protocol: ProtocolConfig,
    scenario: ScenarioConfig,
    adapter_factory: Callable[[], List[Adapter]],
    metric_factories: List[Type[MetricAccumulator]],
    seeds: List[int],
    profile: ScenarioProfile | None = None,
) -> BenchmarkSuiteResults:
    profile = profile or ScenarioProfile.unconditioned()
    per_seed: List[BenchmarkResults] = []
    adapter_configs: List[Dict[str, Any]] = []

    for idx, seed in enumerate(seeds):
        seeded_protocol = dataclasses.replace(protocol, seed=seed)
        adapters = adapter_factory()
        if idx == 0:
            adapter_configs = [a.config() for a in adapters]
        per_seed.append(run_benchmark(seeded_protocol, scenario, adapters, metric_factories))

    metric_stats = _aggregate_metric_stats(per_seed)
    family_stats = _aggregate_family_stats(per_seed)
    scenario_fit = _scenario_fit(per_seed, profile)
    winner_stability = _winner_stability(per_seed)
    pareto_frontier, dominated = _pareto(metric_stats, family_stats, profile)

    protocol_meta = dataclasses.asdict(protocol)
    protocol_template = dict(protocol_meta)
    protocol_template["seed"] = "<multi>" if len(seeds) > 1 else seeds[0]
    scenario_meta = per_seed[0].scenario_meta if per_seed else {}
    profile_meta = dataclasses.asdict(profile)

    return BenchmarkSuiteResults(
        scenario_name=scenario.name,
        protocol_template_hash=_dict_hash(protocol_template),
        scenario_hash=_config_hash(scenario),
        profile_hash=profile.hash(),
        seeds=list(seeds),
        protocol_meta=protocol_meta,
        scenario_meta=scenario_meta,
        profile_meta=profile_meta,
        manifest=_manifest(seeds, adapter_configs),
        per_seed=per_seed,
        metric_stats=metric_stats,
        family_stats=family_stats,
        scenario_fit=scenario_fit,
        winner_stability=winner_stability,
        pareto_frontier=pareto_frontier,
        dominated=dominated,
    )


def _aggregate_metric_stats(
    results: List[BenchmarkResults],
) -> Dict[str, Dict[str, StatSummary]]:
    values: Dict[str, Dict[str, List[float]]] = {}
    for result in results:
        for adapter, adapter_results in result.per_adapter.items():
            for metric in adapter_results:
                score = metric.normalized_score
                if score is None:
                    score = metric.score
                values.setdefault(adapter, {}).setdefault(metric.dimension, []).append(score)
    return {
        adapter: {dim: _summary(scores) for dim, scores in dims.items()}
        for adapter, dims in values.items()
    }


def _aggregate_family_stats(
    results: List[BenchmarkResults],
) -> Dict[str, Dict[str, StatSummary]]:
    values: Dict[str, Dict[str, List[float]]] = {}
    for result in results:
        for adapter, adapter_results in result.per_adapter.items():
            by_family: Dict[str, List[float]] = {}
            for metric in adapter_results:
                score = metric.normalized_score
                if score is None:
                    score = metric.score
                by_family.setdefault(metric.family, []).append(score)
            for family, scores in by_family.items():
                values.setdefault(adapter, {}).setdefault(family, []).append(
                    sum(scores) / len(scores)
                )
    return {
        adapter: {family: _summary(scores) for family, scores in fams.items()}
        for adapter, fams in values.items()
    }


def _scenario_fit(
    results: List[BenchmarkResults], profile: ScenarioProfile
) -> Dict[str, StatSummary]:
    per_adapter: Dict[str, List[float]] = {}
    axes = profile.weighted_axes()
    for result in results:
        metric_by_adapter = {
            adapter: {
                r.dimension: (
                    r.normalized_score if r.normalized_score is not None else r.score
                )
                for r in adapter_results
            }
            for adapter, adapter_results in result.per_adapter.items()
        }
        family_by_adapter: Dict[str, Dict[str, float]] = {}
        for adapter, adapter_results in result.per_adapter.items():
            by_family: Dict[str, List[float]] = {}
            for r in adapter_results:
                score = r.normalized_score if r.normalized_score is not None else r.score
                by_family.setdefault(r.family, []).append(score)
            family_by_adapter[adapter] = {
                family: sum(scores) / len(scores)
                for family, scores in by_family.items()
                if scores
            }
        for adapter in result.per_adapter:
            total = 0.0
            denom = 0.0
            for axis, weight in axes.items():
                score = metric_by_adapter[adapter].get(axis)
                if score is None:
                    score = family_by_adapter[adapter].get(axis)
                if score is None:
                    continue
                total += score * weight
                denom += weight
            per_adapter.setdefault(adapter, []).append(total / denom if denom else 0.0)
    return {adapter: _summary(scores) for adapter, scores in per_adapter.items()}


def _winner_stability(results: List[BenchmarkResults]) -> Dict[str, Dict[str, Any]]:
    axes: Dict[str, Dict[str, int]] = {}
    for result in results:
        values: Dict[str, Dict[str, float]] = {}
        for adapter, adapter_results in result.per_adapter.items():
            family_scores: Dict[str, List[float]] = {}
            for metric in adapter_results:
                score = metric.normalized_score if metric.normalized_score is not None else metric.score
                values.setdefault(metric.dimension, {})[adapter] = score
                family_scores.setdefault(metric.family, []).append(score)
            for family, scores in family_scores.items():
                values.setdefault(family, {})[adapter] = sum(scores) / len(scores)
        for axis, adapter_scores in values.items():
            winner = max(adapter_scores.items(), key=lambda kv: kv[1])[0]
            axes.setdefault(axis, {}).setdefault(winner, 0)
            axes[axis][winner] += 1

    out: Dict[str, Dict[str, Any]] = {}
    n = len(results)
    for axis, counts in axes.items():
        out[axis] = {
            "winner_counts": counts,
            "winner_rates": {k: round(v / n, 4) for k, v in counts.items()},
        }
    return out


def _pareto(
    metric_stats: Dict[str, Dict[str, StatSummary]],
    family_stats: Dict[str, Dict[str, StatSummary]],
    profile: ScenarioProfile,
    tolerance: float = 0.01,
) -> tuple[List[str], Dict[str, str]]:
    adapters = list(family_stats.keys())
    axes = profile.high_priority_axes()
    dominated: Dict[str, str] = {}
    for candidate in adapters:
        for challenger in adapters:
            if candidate == challenger:
                continue
            if _dominates(challenger, candidate, axes, metric_stats, family_stats, tolerance):
                dominated[candidate] = challenger
                break
    frontier = [adapter for adapter in adapters if adapter not in dominated]
    return frontier, dominated


def _dominates(
    challenger: str,
    candidate: str,
    axes: Dict[str, float],
    metric_stats: Dict[str, Dict[str, StatSummary]],
    family_stats: Dict[str, Dict[str, StatSummary]],
    tolerance: float,
) -> bool:
    saw_axis = False
    strictly_better = False
    for axis in axes:
        c_score = _axis_score(candidate, axis, metric_stats, family_stats)
        h_score = _axis_score(challenger, axis, metric_stats, family_stats)
        if c_score is None or h_score is None:
            continue
        saw_axis = True
        if h_score + tolerance < c_score:
            return False
        if h_score > c_score + tolerance:
            strictly_better = True
    return saw_axis and strictly_better


def _axis_score(
    adapter: str,
    axis: str,
    metric_stats: Dict[str, Dict[str, StatSummary]],
    family_stats: Dict[str, Dict[str, StatSummary]],
) -> float | None:
    if axis in metric_stats.get(adapter, {}):
        return metric_stats[adapter][axis].mean
    if axis in family_stats.get(adapter, {}):
        return family_stats[adapter][axis].mean
    return None


def _summary(values: List[float]) -> StatSummary:
    n = len(values)
    if n == 0:
        return StatSummary(mean=0.0, std=0.0, ci95=0.0, min=0.0, max=0.0, n=0)
    mean = sum(values) / n
    if n == 1:
        std = 0.0
    else:
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    ci95 = 1.96 * std / math.sqrt(n) if n else 0.0
    return StatSummary(
        mean=mean,
        std=std,
        ci95=ci95,
        min=min(values),
        max=max(values),
        n=n,
    )


def _manifest(seeds: List[int], adapter_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "seeds": seeds,
        "adapters": adapter_configs,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()
