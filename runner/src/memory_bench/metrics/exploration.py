"""Exploration-family metrics."""

from __future__ import annotations

from typing import List, Set

from memory_bench.adapters.base import Adapter, Retrieval
from memory_bench.metrics.base import MetricAccumulator
from memory_bench.scenario.item_pool import ItemPool


class NoveltyGuarantee(MetricAccumulator):
    """Fraction of top_k items that are BOTH relevant to the current theme AND
    have not been retrieved in the last `cooldown` steps.

    Novelty of noise is not valuable — an adapter that pumps never-before-seen
    garbage should not win Exploration. This metric rewards only useful novelty.
    """

    family = "exploration"
    dimension = "novelty_guarantee"

    def __init__(self, cooldown: int = 10, relevance_threshold: float = 0.5) -> None:
        super().__init__()
        self.cooldown = cooldown
        self.relevance_threshold = relevance_threshold

    def observe(
        self,
        adapter: Adapter,
        retrieval: Retrieval,
        hits: List[bool],
        affinity_scores: List[float],
        pool: ItemPool,
        global_step: int,
    ) -> None:
        if not retrieval.item_ids:
            return
        useful = 0
        for iid, aff in zip(retrieval.item_ids, affinity_scores):
            if aff < self.relevance_threshold:
                continue
            prev = adapter._retr_times.get(iid, [])
            prior = [t for t in prev if t < global_step]
            if not prior or (global_step - prior[-1]) > self.cooldown:
                useful += 1
        self._push(useful / len(retrieval.item_ids))


class Coverage(MetricAccumulator):
    """Fraction of the observed item pool that the adapter has surfaced at least once.

    Higher = adapter explores broadly; lower = adapter concentrates on a small subset.
    This is a run-level metric: each retrieval only updates the coverage set, the
    scalar is computed on `.score()`.
    """

    family = "exploration"
    dimension = "coverage"

    def __init__(self) -> None:
        super().__init__()
        self._ever_retrieved: Set[str] = set()
        self._ever_observed: Set[str] = set()

    def observe(
        self,
        adapter: Adapter,
        retrieval: Retrieval,
        hits: List[bool],
        affinity_scores: List[float],
        pool: ItemPool,
        global_step: int,
    ) -> None:
        for iid in retrieval.item_ids:
            self._ever_retrieved.add(iid)
        for iid in adapter.known_item_ids():
            self._ever_observed.add(iid)

    def score(self) -> float:
        if not self._ever_observed:
            return 0.0
        return len(self._ever_retrieved & self._ever_observed) / len(self._ever_observed)

    def result(self):
        r = super().result()
        r.n_samples = len(self._ever_observed)
        return r
