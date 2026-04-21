"""Ranking-family metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, List

from memory_bench.adapters.base import Adapter, Retrieval
from memory_bench.metrics.base import MetricAccumulator
from memory_bench.scenario.item_pool import ItemPool


class RelevanceAtK(MetricAccumulator):
    """Rank-discounted relevance of the retrieved items (NDCG-like, simplified).

    score = sum_i (affinity_i / (i+1)) / sum_i (1 / (i+1))

    Captures "quality ranking" — you're rewarded both for retrieving items with
    high theme-affinity and for putting them at the top. A recency-ordered
    adapter that returns noise scores poorly; an adapter that surfaces relevant
    items in a sensible order scores high.
    """

    family = "ranking"
    dimension = "relevance_at_k"

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
        total = 0.0
        disc_sum = 0.0
        for i, aff in enumerate(affinity_scores):
            disc = 1.0 / (i + 1)
            total += aff * disc
            disc_sum += disc
        if disc_sum:
            self._push(total / disc_sum)


class FrequencyGain(MetricAccumulator):
    """Does the adapter boost items that have previously hit the current theme?

    For each query, among retrieved items, compute correlation between per-theme
    past-hit count and retrieval rank. High score = items that keep hitting this
    theme get promoted; low score = adapter ignores the recurrence signal.
    """

    family = "ranking"
    dimension = "frequency_gain"

    def __init__(self) -> None:
        super().__init__()
        self._theme_hit_count: DefaultDict[int, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
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
        counts = self._theme_hit_count[theme]

        ids = retrieval.item_ids
        if len(ids) >= 2 and any(counts.get(iid, 0) > 0 for iid in ids):
            n = len(ids)
            hit_counts = [counts.get(iid, 0) for iid in ids]
            concordant = 0
            total = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if hit_counts[i] == hit_counts[j]:
                        continue
                    total += 1
                    if (i < j) == (hit_counts[i] > hit_counts[j]):
                        concordant += 1
            if total:
                self._push(concordant / total)

        for iid, hit in zip(ids, hits):
            if hit:
                counts[iid] += 1
