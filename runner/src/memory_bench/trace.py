"""Trace replay for real-world agent-memory failure modes."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

import yaml

from memory_bench.adapters.base import Adapter
from memory_bench.profile import ScenarioProfile
from memory_bench.runner import StatSummary, _manifest, _pareto, _summary
from memory_bench.scenario.context import Query
from memory_bench.scenario.item_pool import Item


@dataclass
class TraceEvent:
    id: str
    session: str
    role: str
    content: str
    tags: List[str] = field(default_factory=list)
    supersedes: List[str] = field(default_factory=list)


@dataclass
class TraceQuerySpec:
    id: str
    after: str
    text: str
    expected_include: List[str] = field(default_factory=list)
    expected_exclude: List[str] = field(default_factory=list)
    top_k: int = 5


@dataclass
class TraceConfig:
    trace_version: int
    name: str
    domain: str
    events: List[TraceEvent]
    queries: List[TraceQuerySpec]

    @staticmethod
    def from_yaml(path: str | Path) -> "TraceConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        version = int(raw.get("trace_version", 1))
        if version != 1:
            raise ValueError(f"unsupported trace_version: {version}")
        return TraceConfig(
            trace_version=version,
            name=str(raw["name"]),
            domain=str(raw.get("domain", "unknown")),
            events=[TraceEvent(**e) for e in raw.get("events", [])],
            queries=[TraceQuerySpec(**q) for q in raw.get("queries", [])],
        )


@dataclass
class TraceReplayResults:
    trace_name: str
    domain: str
    profile_meta: Dict[str, Any]
    manifest: Dict[str, Any]
    per_adapter: Dict[str, Dict[str, StatSummary]]
    scenario_fit: Dict[str, StatSummary]
    pareto_frontier: List[str]
    dominated: Dict[str, str]
    query_results: Dict[str, List[Dict[str, Any]]]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "trace": self.trace_name,
            "domain": self.domain,
            "scenario_profile": self.profile_meta,
            "manifest": self.manifest,
            "capability_profile": {
                adapter: {dim: stat.as_dict() for dim, stat in dims.items()}
                for adapter, dims in self.per_adapter.items()
            },
            "scenario_fit": {
                adapter: stat.as_dict() for adapter, stat in self.scenario_fit.items()
            },
            "pareto_frontier": self.pareto_frontier,
            "dominated": self.dominated,
            "query_results": self.query_results,
        }


def replay_trace(
    trace: TraceConfig,
    adapter_factory: Callable[[], List[Adapter]],
    profile: ScenarioProfile | None = None,
) -> TraceReplayResults:
    profile = profile or ScenarioProfile.unconditioned()
    adapters = adapter_factory()
    adapter_configs = [a.config() for a in adapters]
    by_after: Dict[str, List[TraceQuerySpec]] = {}
    for query in trace.queries:
        by_after.setdefault(query.after, []).append(query)

    event_sessions = {event.id: event.session for event in trace.events}
    scores: Dict[str, Dict[str, List[float]]] = {a.name: {} for a in adapters}
    query_results: Dict[str, List[Dict[str, Any]]] = {a.name: [] for a in adapters}

    for step, event in enumerate(trace.events):
        item = _event_to_item(event, step)
        for adapter in adapters:
            adapter.observe(item, step)

        for spec in by_after.get(event.id, []):
            query = Query(
                session=_session_index(trace.events, event.session),
                step=step,
                theme=0,
                text=spec.text,
                top_k=spec.top_k,
            )
            for adapter in adapters:
                retrieval = adapter.retrieve(query, step)
                adapter.record_retrieval(retrieval, step)
                result = _score_query(spec, retrieval.item_ids, event.session, event_sessions)
                for dim, value in result["scores"].items():
                    scores[adapter.name].setdefault(dim, []).append(value)
                query_results[adapter.name].append(
                    {
                        "query_id": spec.id,
                        "retrieved": retrieval.item_ids,
                        "scores": result["scores"],
                    }
                )

    per_adapter = {
        adapter: {dim: _summary(values) for dim, values in dims.items()}
        for adapter, dims in scores.items()
    }
    fit = _trace_scenario_fit(per_adapter, profile)
    frontier, dominated = _pareto(per_adapter, per_adapter, profile)
    return TraceReplayResults(
        trace_name=trace.name,
        domain=trace.domain,
        profile_meta=dataclasses.asdict(profile),
        manifest=_manifest([], adapter_configs),
        per_adapter=per_adapter,
        scenario_fit=fit,
        pareto_frontier=frontier,
        dominated=dominated,
        query_results=query_results,
    )


def _event_to_item(event: TraceEvent, step: int) -> Item:
    is_noise = bool({"noise", "error", "injection"} & set(event.tags))
    return Item(
        id=event.id,
        archetype="noise" if is_noise else "trace",
        content=event.content,
        primary_theme=None if is_noise else 0,
        theme_affinity={} if is_noise else {0: 1.0},
        importance=0.1 if is_noise else 1.0,
        created_session=0,
        created_step=step,
    )


def _score_query(
    spec: TraceQuerySpec,
    retrieved: List[str],
    current_session: str,
    event_sessions: Dict[str, str],
) -> Dict[str, Any]:
    include = set(spec.expected_include)
    exclude = set(spec.expected_exclude)
    retrieved_set = set(retrieved)
    hits = [iid for iid in retrieved if iid in include]
    bad = [iid for iid in retrieved if iid in exclude]

    recall = len(retrieved_set & include) / len(include) if include else 1.0
    precision = len(hits) / len(retrieved) if retrieved else 0.0
    mrr = 0.0
    for idx, iid in enumerate(retrieved, 1):
        if iid in include:
            mrr = 1.0 / idx
            break
    ndcg = _ndcg(retrieved, include)
    update = _update_score(include, exclude, retrieved_set)
    forgetting = 1.0 - (len(bad) / len(retrieved) if retrieved else 0.0)
    cross_session = _cross_session_recall(include, retrieved_set, current_session, event_sessions)
    memory_evolution = update if exclude else recall
    provenance = 1.0 if all(iid in event_sessions for iid in retrieved) else 0.0

    return {
        "scores": {
            "retrieval_recall": recall,
            "retrieval_mrr": mrr,
            "retrieval_ndcg": ndcg,
            "update_coherence": update,
            "selective_forgetting": forgetting,
            "cross_session_learning": cross_session,
            "context_efficiency": precision,
            "memory_evolution": memory_evolution,
            "provenance": provenance,
        }
    }


def _ndcg(retrieved: List[str], include: set[str]) -> float:
    if not include:
        return 1.0
    dcg = 0.0
    for idx, iid in enumerate(retrieved, 1):
        if iid in include:
            dcg += 1.0 / idx
    ideal = sum(1.0 / idx for idx in range(1, min(len(include), len(retrieved)) + 1))
    return dcg / ideal if ideal else 0.0


def _update_score(include: set[str], exclude: set[str], retrieved: set[str]) -> float:
    if not include and not exclude:
        return 1.0
    includes_ok = bool(retrieved & include) if include else True
    excludes_bad = bool(retrieved & exclude)
    if includes_ok and not excludes_bad:
        return 1.0
    if includes_ok and excludes_bad:
        return 0.5
    return 0.0


def _cross_session_recall(
    include: set[str],
    retrieved: set[str],
    current_session: str,
    event_sessions: Dict[str, str],
) -> float:
    prior = {iid for iid in include if event_sessions.get(iid) != current_session}
    if not prior:
        return 1.0
    return len(retrieved & prior) / len(prior)


def _trace_scenario_fit(
    per_adapter: Dict[str, Dict[str, StatSummary]], profile: ScenarioProfile
) -> Dict[str, StatSummary]:
    axes = profile.weighted_axes()
    out: Dict[str, StatSummary] = {}
    for adapter, dims in per_adapter.items():
        total = 0.0
        denom = 0.0
        for axis, weight in axes.items():
            stat = dims.get(axis)
            if stat is None:
                continue
            total += stat.mean * weight
            denom += weight
        if denom == 0.0:
            means = [stat.mean for stat in dims.values()]
            total = sum(means)
            denom = len(means) or 1
        out[adapter] = _summary([total / denom])
    return out


def _session_index(events: List[TraceEvent], session: str) -> int:
    seen: List[str] = []
    for event in events:
        if event.session not in seen:
            seen.append(event.session)
    return seen.index(session)
