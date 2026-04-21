"""Adaptation-family metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, List, Set

from memory_bench.adapters.base import Adapter, Retrieval
from memory_bench.metrics.base import MetricAccumulator
from memory_bench.scenario.item_pool import ItemPool


class Personalization(MetricAccumulator):
    """Does the adapter re-surface items that previously hit the same theme?

    For each query of theme T after the first, measure how many of the items
    that have hit T in the past are present in the current top_k.
    Normalized by min(|history|, top_k).
    """

    family = "adaptation"
    dimension = "personalization"

    def __init__(self) -> None:
        super().__init__()
        self._theme_history: DefaultDict[int, Set[str]] = defaultdict(set)

    def observe(
        self,
        adapter: Adapter,
        retrieval: Retrieval,
        hits: List[bool],
        affinity_scores: List[float],
        pool: ItemPool,
        global_step: int,
    ) -> None:
        theme = retrieval.query.theme
        history = self._theme_history[theme]
        if history and retrieval.item_ids:
            denom = min(len(history), len(retrieval.item_ids))
            coverage = len(set(retrieval.item_ids) & history) / denom if denom else 0.0
            self._push(coverage)
        for iid, hit in zip(retrieval.item_ids, hits):
            if hit:
                history.add(iid)


class CrossSessionLearning(MetricAccumulator):
    """Does the adapter retain items that hit across multiple sessions?

    An item is "cross-session useful" if it has hit in at least 2 distinct sessions
    on the same theme. When that theme queries again in later sessions, does the
    adapter surface those multi-session-hitters?
    """

    family = "adaptation"
    dimension = "cross_session_learning"

    def __init__(self) -> None:
        super().__init__()
        # theme -> item -> set of sessions where it hit
        self._theme_item_sessions: DefaultDict[int, DefaultDict[str, Set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )

    def observe(
        self,
        adapter: Adapter,
        retrieval: Retrieval,
        hits: List[bool],
        affinity_scores: List[float],
        pool: ItemPool,
        global_step: int,
    ) -> None:
        theme = retrieval.query.theme
        current_session = retrieval.query.session
        theme_hits = self._theme_item_sessions[theme]

        multi = {
            iid
            for iid, sessions in theme_hits.items()
            if len(sessions) >= 2 and max(sessions) < current_session
        }
        if multi and retrieval.item_ids:
            denom = min(len(multi), len(retrieval.item_ids))
            coverage = len(set(retrieval.item_ids) & multi) / denom if denom else 0.0
            self._push(coverage)

        for iid, hit in zip(retrieval.item_ids, hits):
            if hit:
                theme_hits[iid].add(current_session)
