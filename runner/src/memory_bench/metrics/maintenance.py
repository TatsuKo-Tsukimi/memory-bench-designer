"""Maintenance-family metrics."""

from __future__ import annotations

from typing import List

from memory_bench.adapters.base import Adapter, Retrieval
from memory_bench.metrics.base import MetricAccumulator
from memory_bench.scenario.item_pool import ItemPool


class UpdateCoherence(MetricAccumulator):
    """When an evolving item has been superseded, does the adapter suppress the stale
    version? Full credit requires SUPPRESSING the old version — half credit if both
    appear with the new ranked higher (the adapter at least prioritizes correctly but
    wastes top_k on the stale version).

    Scoring per (old -> new) pair, filtered to pairs where either version is relevant
    to the current theme:
      - only new in top_k:             +1.0 (ideal: stale suppressed)
      - both in top_k, new ranked up:  +0.5 (correct order but wastes slot on stale)
      - both in top_k, old ranked up:  +0.0 (wrong)
      - only old in top_k:             +0.0 (stale served to user)
      - neither:                       +0.0 (missed opportunity)
    """

    family = "maintenance"
    dimension = "update_coherence"

    def __init__(self, relevance_threshold: float = 0.5) -> None:
        super().__init__()
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
        known = set(adapter.known_item_ids())
        theme = retrieval.query.theme
        pairs = []
        for iid in known:
            item = pool.get(iid)
            if (
                item.archetype == "evolving"
                and item.supersedes is None
                and item.superseded_by
                and item.superseded_by in known
            ):
                new_item = pool.get(item.superseded_by)
                if (
                    item.affinity_to(theme) >= self.relevance_threshold
                    or new_item.affinity_to(theme) >= self.relevance_threshold
                ):
                    pairs.append((item.id, item.superseded_by))
        if not pairs:
            return

        in_retr = set(retrieval.item_ids)
        pos = {iid: i for i, iid in enumerate(retrieval.item_ids)}

        total_score = 0.0
        for old, new in pairs:
            old_in = old in in_retr
            new_in = new in in_retr
            if old_in and new_in:
                total_score += 0.5 if pos[new] < pos[old] else 0.0
            elif new_in:
                total_score += 1.0
        self._push(total_score / len(pairs))


class ForgettingQuality(MetricAccumulator):
    """Does the adapter keep noise OUT of retrieval?

    score = 1 - (noise_items_retrieved / total_retrieved).
    Adapters that systematically filter noise score high; ones that don't score low.
    """

    family = "maintenance"
    dimension = "forgetting_quality"

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
        noise = sum(
            1 for iid in retrieval.item_ids if pool.get(iid).archetype == "noise"
        )
        self._push(1.0 - (noise / len(retrieval.item_ids)))
